"""
views.py — HTTP boundary only.

Response contracts (documented here so the frontend team knows what to expect):

  POST /chatbot/
    agent answer   → 200 JsonResponse  {"type":"agent",  "agent":"vet|meteo|feed", "data":{...}}
    text answer    → 200 JsonResponse  {"type":"text",   "content":"..."}          (language gate)
    streamed text  → 200 StreamingHttpResponse  text/plain                         (normal LLM)
    error          → 4xx JsonResponse  {"error":{"code":"...","message":"..."}}

  POST /chatbot/stt/
    success        → 200 {"text": str, "status": "ok"}
    noise/invalid  → 200 {"text": "",  "status": "incomprehensible"}
    error          → 4xx {"error": {"code": str, "message": str}}

  POST /chatbot/tts/
    success        → 200 audio/wav stream
    error          → 4xx/5xx {"error": {"code": str, "message": str}}

NOTE on mixed response style (/chatbot/):
  Agent answers return JSON immediately (small payload, no need to stream).
  Normal LLM answers are streamed token-by-token for better UX.
  This is intentional and documented above. The frontend checks Content-Type
  to decide which path to take.
"""
import json
import logging
import time

from django.http import FileResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Single import point for all response helpers — no inline dict building
from chatbot.contracts import agent_response, make_error, text_response
from chatbot.models import Conversation
from chatbot.services import memory, retrieval, router
from chatbot.services.agents.feed import FeedAgent
from chatbot.services.agents.llm_agent import stream as llm_stream
from chatbot.services.agents.meteo import MeteoAgent
from chatbot.services.agents.vet import VetAgent
from chatbot.stt.corrector import correct as stt_correct
from chatbot.stt.transcriber import transcribe
from chatbot.tts.synthesizer import delete_after_send, synthesize
from chatbot.utils.text import clean_text, is_arabic, is_valid_text

logger = logging.getLogger(__name__)

# Agent singletons
_vet_agent   = VetAgent()
_meteo_agent = MeteoAgent()
_feed_agent  = FeedAgent()

AGENT_MAP = {
    "vet_agent":   ("vet",   _vet_agent),
    "meteo_agent": ("meteo", _meteo_agent),
    "feed_agent":  ("feed",  _feed_agent),
}

# Location is required for vet and meteo — not for feed
LOCATION_REQUIRED = {"vet", "meteo"}

NO_LOCATION_MSG = {
    "vet":   {"fr": "Veuillez activer la localisation pour trouver un vétérinaire proche.",
              "ar": "يرجى تفعيل الموقع الجغرافي للعثور على طبيب بيطري قريب."},
    "meteo": {"fr": "Veuillez activer la localisation pour obtenir la météo.",
              "ar": "يرجى تفعيل الموقع الجغرافي للحصول على توقعات الطقس."},
}


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

    # Language gate — returns JSON text_response, not an error
    if lang == "fr" and is_arabic(message):
        return JsonResponse(
            text_response("Veuillez écrire en français 🇫🇷 ou changez la langue")
        )
    if lang == "ar" and not is_arabic(message):
        return JsonResponse(
            text_response("يرجى الكتابة باللغة العربية 🇸🇦 أو تغيير اللغة")
        )

    Conversation.objects.create(session_id=session_id, role="user", message=message)
    history = memory.get_history(session_id, lang)

    chunks  = retrieval.search(message)
    context = "\n\n".join(chunks)
    logger.debug(
        "session=%s retrieval_hits=%d query=%r",
        session_id, len(chunks), message[:60],
    )

    action = router.decide(message, history, lang)
    logger.info("session=%s action=%s lang=%s", session_id, action, lang)

    # Agent path → JSON
    if action in AGENT_MAP:
        agent_key, agent_obj = AGENT_MAP[action]
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None

        if agent_key in LOCATION_REQUIRED and lat_f is None:
            return JsonResponse(text_response(NO_LOCATION_MSG[agent_key][lang]))

        data = agent_obj.run(lat_f, lon_f, lang)

        # MeteoAgent returns None on weather-fetch failure
        if data is None:
            msg = ("تعذّر الحصول على بيانات الطقس." if lang == "ar"
                   else "Impossible de récupérer la météo en ce moment.")
            return JsonResponse(text_response(msg))

        elapsed = (time.monotonic() - t_start) * 1000
        logger.info("session=%s action=%s latency_ms=%.0f", session_id, action, elapsed)
        return JsonResponse(agent_response(agent_key, data))

    # LLM streaming path
    def _generate():
        full = yield from llm_stream(message, context, history, lang)
        memory.save_assistant_message(session_id, full)
        elapsed = (time.monotonic() - t_start) * 1000
        logger.info(
            "session=%s action=answer latency_ms=%.0f tokens=%d",
            session_id, elapsed, len(full),
        )

    return StreamingHttpResponse(_generate(), content_type="text/plain")


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
        logger.info("STT text invalid after cleaning (lang=%s raw=%r)", lang, raw_text[:40])
        return JsonResponse({"text": "", "status": "incomprehensible"})

    try:
        corrected = stt_correct(cleaned, lang)
    except ValueError:
        logger.info("STT LLM correction flagged as noise (lang=%s)", lang)
        return JsonResponse({"text": "", "status": "incomprehensible"})

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
        audio_path = synthesize(text, lang)
    except Exception as exc:
        logger.error("TTS synthesis failed: %s", exc)
        return make_error("TTS_FAILED", "Audio synthesis failed.", status=500)

    response = FileResponse(open(audio_path, "rb"), content_type="audio/wav")
    response["Content-Disposition"] = "inline; filename=tts.wav"
    delete_after_send(audio_path)
    return response