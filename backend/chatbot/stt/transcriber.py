"""
Whisper-based audio transcription.
- Unique temp files per call (no collision between concurrent requests)
- Always cleaned up in finally block
- Model loaded lazily with thread lock
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
                logger.info("Loading Whisper model (first call)...")
                _whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded.")
    return _whisper_model


def transcribe(audio_bytes: bytes, lang: str) -> str | None:
    """
    Convert raw audio bytes → text.
    Returns None on failure, timeout, or empty result.
    Temp files are always deleted regardless of outcome.
    """
    # Unique file names per call — no collisions under concurrent load
    input_tmp  = tempfile.NamedTemporaryFile(
        delete=False, suffix=".webm", prefix="bovitech_stt_in_"
    )
    output_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav",  prefix="bovitech_stt_out_"
    )
    input_path  = input_tmp.name
    output_path = output_tmp.name
    input_tmp.close()
    output_tmp.close()

    try:
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ar", "16000", "-ac", "1", output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if result.returncode != 0:
            logger.error(
                "ffmpeg failed (code=%d): %s",
                result.returncode,
                result.stderr.decode(errors="replace")[:200],
            )
            return None

        file_size = os.path.getsize(output_path)
        logger.debug("STT wav file size=%d bytes", file_size)
        if file_size < 1000:
            logger.warning("STT wav too small (%d bytes), likely empty audio", file_size)
            return None

        result_queue: queue.Queue = queue.Queue()

        def _run():
            try:
                model  = _get_model()
                result = model.transcribe(output_path, language=lang)
                result_queue.put(result["text"])
            except Exception as exc:
                logger.error("Whisper transcription error: %s", exc)
                result_queue.put(None)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=30)

        if result_queue.empty():
            logger.warning("Whisper transcription timed out")
            return None

        return result_queue.get()

    finally:
        # Always clean up — even if an exception occurs
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass