"""
Piper TTS when binaries and models are present; otherwise Microsoft Edge TTS (edge-tts).
Paths are read from Django settings so they can be overridden per environment.
"""
import asyncio
import logging
import os
import subprocess
import tempfile
import threading

from django.conf import settings

logger = logging.getLogger(__name__)

# edge-tts voices (neural), used when Piper is not installed locally
_EDGE_VOICE_FR = "fr-FR-DeniseNeural"
_EDGE_VOICE_AR = "ar-SA-HamedNeural"


def _piper_paths(lang: str) -> tuple[str, str]:
    piper_path = settings.PIPER_PATH
    model_path = settings.PIPER_MODEL_AR if lang == "ar" else settings.PIPER_MODEL_FR
    return piper_path, model_path


def _synthesize_piper(text: str, lang: str) -> str:
    piper_path, model_path = _piper_paths(lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = tmp.name
    tmp.close()

    process = subprocess.Popen(
        [piper_path, "--model", model_path, "--output_file", output_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.communicate(input=text.encode("utf-8"))

    if process.returncode != 0:
        logger.error("Piper TTS failed (returncode=%d)", process.returncode)
        raise RuntimeError("Piper TTS failed")

    return output_path


def _synthesize_edge_tts(text: str, lang: str) -> str:
    import edge_tts

    voice = _EDGE_VOICE_AR if lang == "ar" else _EDGE_VOICE_FR
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    output_path = tmp.name
    tmp.close()

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(output_path)

    try:
        asyncio.run(_run())
    except Exception:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise

    return output_path


def synthesize(text: str, lang: str) -> tuple[str, str]:
    """
    Generate audio from text.
    Returns (path_to_temp_file, content_type).
    Caller is responsible for deleting the file after sending it.
    """
    piper_path, model_path = _piper_paths(lang)
    if os.path.isfile(piper_path) and os.path.isfile(model_path):
        path = _synthesize_piper(text, lang)
        return path, "audio/wav"

    logger.info("Piper absent ou modèle manquant; utilisation de edge-tts (fr/ar).")
    path = _synthesize_edge_tts(text, lang)
    return path, "audio/mpeg"


def delete_after_send(path: str) -> None:
    """Delete a temp file in a background thread so the response isn't delayed."""
    def _delete():
        try:
            os.remove(path)
        except OSError:
            pass
    threading.Thread(target=_delete, daemon=True).start()