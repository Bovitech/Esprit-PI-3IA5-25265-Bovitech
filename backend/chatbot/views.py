"""
views.py — HTTP boundary only.

Each view does exactly three things:
  1. Parse the request
  2. Call the appropriate service
  3. Return a response

Zero business logic here.
"""
import json
import logging

from django.http import FileResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

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

# Agent singletons — instantiated once
_vet_agent   = VetAgent()
_meteo_agent = MeteoAgent()
_feed_agent  = FeedAgent()

AGENT_MAP = {
    "vet_agent":   _vet_agent,
    "meteo_agent": _meteo_agent,
    "feed_agent":  _feed_agent,
}


def frontend(request):
    return render(request, "index.html")


# ------------------------------------------------------------------
# CHAT
# ------------------------------------------------------------------
@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    # 1. Parse
    try:
        body = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message    = body.get("message", "")
    session_id = body.get("session_id", "default")
    lang       = body.get("lang", "fr") if body.get("lang") in ("fr", "ar") else "fr"
    lat        = body.get("lat")
    lon        = body.get("lon")

    if not isinstance(message, str) or not message.strip():
        return JsonResponse({"error": "Empty or invalid message"}, status=400)

    # Language gate
    if lang == "fr" and is_arabic(message):
        return StreamingHttpResponse(
            iter(["Veuillez écrire en français 🇫🇷 ou changez la langue"]),
            content_type="text/plain",
        )
    if lang == "ar" and not is_arabic(message):
        return StreamingHttpResponse(
            iter(["يرجى الكتابة باللغة العربية 🇸🇦 أو تغيير اللغة"]),
            content_type="text/plain",
        )

    # 2. Save user message + load history
    Conversation.objects.create(session_id=session_id, role="user", message=message)
    history = memory.get_history(session_id, lang)

    # RAG context
    chunks  = retrieval.search(message)
    context = "\n\n".join(chunks)

    # Route
    action = router.decide(message, history, lang)

    # 3. Agent or LLM
    if action in AGENT_MAP:
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
        result = AGENT_MAP[action].run(lat_f, lon_f, lang)
        return StreamingHttpResponse(iter([result]), content_type="text/plain")

    # Streaming LLM
    def _generate():
        full = yield from llm_stream(message, context, history, lang)
        memory.save_assistant_message(session_id, full)

    return StreamingHttpResponse(_generate(), content_type="text/plain")


# ------------------------------------------------------------------
# STT
# ------------------------------------------------------------------
@csrf_exempt
@require_http_methods(["POST"])
def stt(request):
    lang = request.POST.get("lang", "fr")
    if lang not in ("fr", "ar"):
        lang = "fr"

    audio_file = request.FILES.get("audio")
    if not audio_file:
        return JsonResponse({"error": "No audio file"}, status=400)

    audio_bytes = b"".join(audio_file.chunks())

    raw_text = transcribe(audio_bytes, lang)
    if not raw_text:
        return JsonResponse({"text": "", "status": "incomprehensible"})

    cleaned = clean_text(raw_text, lang)
    if not is_valid_text(cleaned, lang):
        return JsonResponse({"text": "", "status": "incomprehensible"})

    try:
        corrected = stt_correct(cleaned, lang)
    except ValueError:
        return JsonResponse({"text": "", "status": "incomprehensible"})

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
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    text = body.get("text", "").strip()
    lang = body.get("lang", "fr")

    if not text:
        return JsonResponse({"error": "No text provided"}, status=400)

    audio_path = synthesize(text, lang)
    response   = FileResponse(open(audio_path, "rb"), content_type="audio/wav")
    response["Content-Disposition"] = "inline; filename=tts.wav"
    delete_after_send(audio_path)
    return response