"""
Whisper-based audio transcription.
- Unique temp files per call (no collision between concurrent requests)
- Always cleaned up in finally block
- Model loaded lazily with thread lock
- ffmpeg: FFMPEG_PATH, then PATH, then imageio_ffmpeg (bundled binary)
"""
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_ffmpeg() -> str | None:
    """Return path to ffmpeg, or None if unavailable."""
    env_path = (os.environ.get("FFMPEG_PATH") or "").strip()
    if env_path and Path(env_path).is_file():
        return env_path
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.error(
            "ffmpeg introuvable (PATH / FFMPEG_PATH / imageio_ffmpeg): %s",
            exc,
        )
        return None

_whisper_model = None
_model_lock    = threading.Lock()


def _get_model_name() -> str:
    return (os.environ.get("BOVITECH_WHISPER_MODEL") or "base").strip() or "base"


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                import whisper
                name = _get_model_name()
                logger.info("Loading Whisper model %r (first call)...", name)
                _whisper_model = whisper.load_model(name)
                logger.info("Whisper model loaded (%s).", name)
    return _whisper_model


# First Whisper run may download ~139MB model; allow enough time on slow links.
_TRANSCRIBE_TIMEOUT_SEC = 180


def _wav_path_to_float_audio(wav_path: str) -> np.ndarray | None:
    """
    Read 16 kHz mono PCM WAV into float32 [-1, 1] for Whisper.
    Avoids whisper.audio.load_audio(), which spawns 'ffmpeg' by name — on Windows
    imageio_ffmpeg installs ffmpeg-win-x86_64-*.exe, so 'ffmpeg' is not found (WinError 2).
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            if wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                logger.error(
                    "STT: unexpected wav (rate=%s, width=%s)",
                    wf.getframerate(),
                    wf.getsampwidth(),
                )
                return None
            n_ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, OSError) as exc:
        logger.error("STT: cannot read wav: %s", exc)
        return None

    audio = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    return audio.astype(np.float32) / 32768.0


def _gain_normalize(audio: np.ndarray) -> np.ndarray:
    """Boost quiet mic / WebM paths so Whisper gets usable levels (float32 [-1, 1])."""
    if audio is None or audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    logger.debug("STT audio levels peak=%.5f rms=%.5f", peak, rms)
    if peak < 1e-5:
        return audio
    # Target ~0.9 peak when recording is very quiet (common on laptop / phone WebM)
    if peak < 0.18:
        audio = audio * (0.95 / peak)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio


def _initial_prompt_for_lang(lang: str) -> str:
    if lang == "ar":
        return (
            "محادثة عن تربية الأبقار، الحليب، صحة الحيوان، المزرعة، العلف، الحلابة."
        )
    return (
        "Discussion sur l'élevage bovin, la traite, la vache, l'étable, "
        "le pâturage, la santé du troupeau, l'alimentation."
    )


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

        ffmpeg_exe = _resolve_ffmpeg()
        if not ffmpeg_exe:
            logger.error("STT: install ffmpeg or: pip install imageio-ffmpeg")
            return None

        result = subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-f",
                "wav",
                output_path,
            ],
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

        audio_np = _wav_path_to_float_audio(output_path)
        if audio_np is None:
            return None
        audio_np = _gain_normalize(audio_np)

        result_queue: queue.Queue = queue.Queue()

        def _run():
            try:
                model = _get_model()
                # no_speech_threshold=None: do not skip windows as "silent" (quiet mics / WebM
                # often trigger false positives and yield empty transcripts).
                # logprob_threshold=None: avoid aggressive decode retries that end with no text.
                result = model.transcribe(
                    audio_np,
                    language=lang,
                    no_speech_threshold=None,
                    logprob_threshold=None,
                    condition_on_previous_text=False,
                    initial_prompt=_initial_prompt_for_lang(lang),
                    temperature=0.0,
                )
                text = (result.get("text") or "").strip()
                if not text:
                    logger.info(
                        "Whisper returned empty text (samples=%d lang=%s)",
                        len(audio_np),
                        lang,
                    )
                result_queue.put(text or None)
            except Exception as exc:
                logger.error("Whisper transcription error: %s", exc)
                result_queue.put(None)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=_TRANSCRIBE_TIMEOUT_SEC)

        if result_queue.empty():
            logger.warning(
                "Whisper transcription timed out (>%ds); first run may still be downloading the model",
                _TRANSCRIBE_TIMEOUT_SEC,
            )
            return None

        return result_queue.get()

    finally:
        # Always clean up — even if an exception occurs
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass