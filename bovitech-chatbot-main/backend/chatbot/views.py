"""
views.py — HTTP boundary only.

Endpoints:
  POST /chatbot/          → chat (agent JSON | text JSON | streaming text)
  POST /chatbot/stt/      → speech-to-text
  POST /chatbot/tts/      → text-to-speech
  POST /chatbot/skin/     → image skin disease classification  ← NEW

Response contracts:
  agent   → {"type":"agent",  "agent":"vet|meteo|feed|skin", "data":{...}}
  text    → {"type":"text",   "content":"..."}
  stream  → text/plain streaming
  error   → {"error":{"code":"...", "message":"..."}}
"""
import json
import logging
import time

from django.http import FileResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from chatbot.contracts import agent_response, make_error, text_response
from chatbot.models import Conversation
from chatbot.services import memory, retrieval, router
from chatbot.services.agents.feed import FeedAgent
from chatbot.services.agents.llm_agent import stream as llm_stream
from chatbot.services.agents.meteo import MeteoAgent
from chatbot.services.agents.skin import SkinAgent
from chatbot.services.agents.vet import VetAgent
from chatbot.stt.corrector import correct as stt_correct
from chatbot.stt.transcriber import transcribe
from chatbot.tts.synthesizer import delete_after_send, synthesize
from chatbot.utils.text import clean_text, is_arabic, is_valid_text

logger = logging.getLogger(__name__)

_vet_agent   = VetAgent()
_meteo_agent = MeteoAgent()
_feed_agent  = FeedAgent()
_skin_agent  = SkinAgent()

AGENT_MAP = {
    "vet_agent":   ("vet",   _vet_agent),
    "meteo_agent": ("meteo", _meteo_agent),
    "feed_agent":  ("feed",  _feed_agent),
}

LOCATION_REQUIRED = {"vet", "meteo"}

NO_LOCATION_MSG = {
    "vet":   {"fr": "Veuillez activer la localisation pour trouver un vétérinaire proche.",
              "ar": "يرجى تفعيل الموقع الجغرافي للعثور على طبيب بيطري قريب."},
    "meteo": {"fr": "Veuillez activer la localisation pour obtenir la météo.",
              "ar": "يرجى تفعيل الموقع الجغرافي للحصول على توقعات الطقس."},
}

# Max image size accepted: 10 MB
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _bytes_look_like_image(header: bytes) -> bool:
    """Si le client envoie application/octet-stream (souvent RN/Expo), accepter si les octets ressemblent à une image."""
    if len(header) < 12:
        return False
    if header.startswith(b"\xff\xd8\xff"):
        return True
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if header.startswith((b"GIF87a", b"GIF89a")):
        return True
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return True
    return False


def frontend(request):
    return render(request, "index.html")


# ------------------------------------------------------------------
# CHAT
# ------------------------------------------------------------------
@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    t_start = time.monotonic()

    try:
        body = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return make_error("INVALID_JSON", "Request body must be valid JSON.")

    message    = body.get("message", "")
    session_id = body.get("session_id", "default")
    raw_lang   = body.get("lang", "fr")
    lang       = raw_lang if raw_lang in ("fr", "ar") else "fr"
    lat        = body.get("lat")
    lon        = body.get("lon")

    if not isinstance(message, str) or not message.strip():
        return make_error("EMPTY_MESSAGE", "Message must be a non-empty string.")

    if lang == "fr" and is_arabic(message):
        return JsonResponse(text_response("Veuillez écrire en français 🇫🇷 ou changez la langue"))
    if lang == "ar" and not is_arabic(message):
        return JsonResponse(text_response("يرجى الكتابة باللغة العربية 🇸🇦 أو تغيير اللغة"))

    Conversation.objects.create(session_id=session_id, role="user", message=message)
    history = memory.get_history(session_id, lang)
    chunks  = retrieval.search(message)
    context = "\n\n".join(chunks)

    logger.debug("session=%s retrieval_hits=%d query=%r", session_id, len(chunks), message[:60])

    action = router.decide(message, history, lang)
    logger.info("session=%s action=%s lang=%s", session_id, action, lang)

    if action in AGENT_MAP:
        agent_key, agent_obj = AGENT_MAP[action]
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None

        if agent_key in LOCATION_REQUIRED and (
            lat_f is None or lon_f is None
        ):
            return JsonResponse(text_response(NO_LOCATION_MSG[agent_key][lang]))

        data = agent_obj.run(lat_f, lon_f, lang)
        if data is None:
            msg = ("تعذّر الحصول على بيانات الطقس." if lang == "ar"
                   else "Impossible de récupérer la météo en ce moment.")
            return JsonResponse(text_response(msg))

        elapsed = (time.monotonic() - t_start) * 1000
        logger.info("session=%s action=%s latency_ms=%.0f", session_id, action, elapsed)
        return JsonResponse(agent_response(agent_key, data))

    def _generate():
        full = yield from llm_stream(message, context, history, lang)
        memory.save_assistant_message(session_id, full)
        elapsed = (time.monotonic() - t_start) * 1000
        logger.info("session=%s action=answer latency_ms=%.0f tokens=%d", session_id, elapsed, len(full))

    return StreamingHttpResponse(_generate(), content_type="text/plain")


# ------------------------------------------------------------------
# SKIN — image disease classification  ← NEW
# ------------------------------------------------------------------
@csrf_exempt
@require_http_methods(["POST"])
def skin(request):
    """
    Reçoit une image (multipart/form-data, champ "image") et retourne
    le résultat de classification EfficientNet-B3.

    Réponse succès :
      {"type":"agent","agent":"skin","data":{
        "predicted_class":"Lumpy",
        "confidence":0.923,
        "probabilities":{...},
        "level":"high",
        "is_healthy":false,
        "description":"..."
      }}
    """
    t_start = time.monotonic()

    lang_raw = request.POST.get("lang", "fr")
    lang     = lang_raw if lang_raw in ("fr", "ar") else "fr"

    image_file = request.FILES.get("image")
    if not image_file:
        return make_error("NO_IMAGE", "No image file provided. Send the image in the 'image' field.")

    if image_file.size > MAX_IMAGE_BYTES:
        return make_error("IMAGE_TOO_LARGE", "Image must be under 10 MB.")

    image_bytes = b"".join(image_file.chunks())

    content_type = image_file.content_type or ""
    if not content_type.startswith("image/"):
        if not _bytes_look_like_image(image_bytes[:16]):
            return make_error(
                "INVALID_IMAGE_TYPE",
                "File must be an image (jpeg, png, webp...).",
            )

    data = _skin_agent.run(image_bytes, lang)

    if data is None:
        msg = ("تعذّر قراءة الصورة. تأكد من أنها واضحة وبصيغة صحيحة."
               if lang == "ar"
               else "Impossible de lire l'image. Assurez-vous qu'elle est nette et dans un format valide (jpg, png).")
        return JsonResponse(text_response(msg))

    elapsed = (time.monotonic() - t_start) * 1000
    logger.info(
        "action=skin predicted=%s confidence=%.3f latency_ms=%.0f",
        data["predicted_class"], data["confidence"], elapsed,
    )

    return JsonResponse(agent_response("skin", data))


# ------------------------------------------------------------------
# STT
# ------------------------------------------------------------------
@csrf_exempt
@require_http_methods(["POST"])
def stt(request):
    t_start = time.monotonic()
    lang    = request.POST.get("lang", "fr")
    if lang not in ("fr", "ar"):
        lang = "fr"

    audio_file = request.FILES.get("audio")
    if not audio_file:
        return make_error("NO_AUDIO", "No audio file provided.")

    audio_bytes = b"".join(audio_file.chunks())
    if len(audio_bytes) < 100:
        return JsonResponse({"text": "", "status": "incomprehensible"})

    raw_text = transcribe(audio_bytes, lang)
    if not raw_text:
        logger.warning("STT transcription returned empty (lang=%s)", lang)
        return JsonResponse({"text": "", "status": "incomprehensible"})

    cleaned = clean_text(raw_text, lang)
    if not is_valid_text(cleaned, lang):
        logger.info(
            "STT text invalid after cleaning (lang=%s raw=%r cleaned=%r)",
            lang,
            raw_text[:120],
            cleaned[:120],
        )
        return JsonResponse({"text": "", "status": "incomprehensible"})

    try:
        corrected = stt_correct(cleaned, lang)
    except ValueError:
        # LLM often returns INVALID on short / noisy clips; still return Whisper output
        logger.info(
            "STT LLM correction flagged as noise; using cleaned transcript (lang=%s len=%d)",
            lang,
            len(cleaned),
        )
        return JsonResponse({"text": cleaned, "status": "ok"})

    elapsed = (time.monotonic() - t_start) * 1000
    logger.info("STT success lang=%s latency_ms=%.0f", lang, elapsed)
    return JsonResponse({"text": corrected, "status": "ok"})


# ------------------------------------------------------------------
# TTS
# ------------------------------------------------------------------
@csrf_exempt
@require_http_methods(["POST"])
def tts(request):
    try:
        body = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return make_error("INVALID_JSON", "Request body must be valid JSON.")

    text = body.get("text", "").strip()
    lang = body.get("lang", "fr")

    if not text:
        return make_error("NO_TEXT", "text field is required.")

    try:
        audio_path, content_type = synthesize(text, lang)
    except Exception as exc:
        logger.error("TTS synthesis failed: %s", exc)
        return make_error("TTS_FAILED", "Audio synthesis failed.", status=500)

    ext = "mp3" if "mpeg" in content_type else "wav"
    response = FileResponse(open(audio_path, "rb"), content_type=content_type)
    response["Content-Disposition"] = f"inline; filename=tts.{ext}"
    delete_after_send(audio_path)
    return response