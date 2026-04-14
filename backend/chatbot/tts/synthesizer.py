"""
Piper TTS wrapper.
Paths are read from Django settings so they can be overridden per environment.
"""
import logging
import os
import subprocess
import tempfile
import threading

from django.conf import settings

logger = logging.getLogger(__name__)


def synthesize(text: str, lang: str) -> str:
    """
    Generate a WAV file from text using Piper.
    Returns the path to the temporary WAV file.
    Caller is responsible for deleting the file after sending it.
    """
    piper_path = settings.PIPER_PATH
    model_path = settings.PIPER_MODEL_AR if lang == "ar" else settings.PIPER_MODEL_FR

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

    return output_path


def delete_after_send(path: str) -> None:
    """Delete a temp file in a background thread so the response isn't delayed."""
    def _delete():
        try:
            os.remove(path)
        except OSError:
            pass
    threading.Thread(target=_delete, daemon=True).start()