"""
Whisper-based audio transcription.
Model is loaded lazily on first use, not at server start.
"""
import logging
import os
import queue
import subprocess
import tempfile
import threading

logger = logging.getLogger(__name__)

_whisper_model = None
_model_lock    = threading.Lock()


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                import whisper
                logger.info("Loading Whisper model...")
                _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe(audio_bytes: bytes, lang: str) -> str | None:
    """
    Convert raw audio bytes → text.
    Returns None on failure or timeout.
    """
    tmp_dir     = tempfile.gettempdir()
    input_path  = os.path.join(tmp_dir, "bovitech_input.webm")
    output_path = os.path.join(tmp_dir, "bovitech_output.wav")

    with open(input_path, "wb") as f:
        f.write(audio_bytes)

    ffmpeg_result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ffmpeg_result.returncode != 0:
        logger.error("ffmpeg conversion failed")
        return None

    result_queue: queue.Queue = queue.Queue()

    def _transcribe():
        try:
            model  = _get_model()
            result = model.transcribe(output_path, language=lang)
            result_queue.put(result["text"])
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            result_queue.put(None)

    t = threading.Thread(target=_transcribe, daemon=True)
    t.start()
    t.join(timeout=30)

    return result_queue.get() if not result_queue.empty() else None