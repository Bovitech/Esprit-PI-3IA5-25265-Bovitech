"""Pre-process cow vocal audio → (128, 128, 1) float32 for model_audio_classification.h5."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


def _target_sample_rate() -> int:
    return int(os.environ.get("VOCAL_SAMPLE_RATE", "22050"))


def _normalize_channels(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        return a
    return a.mean(axis=1).astype(np.float32)


def _sniff_suffix(raw: bytes) -> str:
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return ".wav"
    if len(raw) >= 12 and raw[:4] == b"FORM" and raw[8:12] in (b"AIFF", b"AIFC"):
        return ".aiff"
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        return ".mp4"
    if len(raw) >= 4 and raw[0] == 0x1A and raw[1:4] == b"\x45\xdf\xa3":
        return ".webm"
    if len(raw) >= 4 and raw[:4] == b"OggS":
        return ".ogg"
    if len(raw) >= 3 and raw[:3] == b"ID3":
        return ".mp3"
    if len(raw) >= 2 and (raw[:2] == b"\xff\xfb" or raw[:2] == b"\xff\xf3"):
        return ".mp3"
    return ".bin"


def _ffmpeg_executable() -> Optional[str]:
    """
    WinGet/App Execution Alias often updates PATH after install; Cursor/Python may miss it until restart.
    FFMPEG_PATH may point directly to ffmpeg.exe.
    """
    explicit = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    if explicit and os.path.isfile(explicit):
        return explicit

    for name in ("ffmpeg", "ffmpeg.exe"):
        w = shutil.which(name)
        if w:
            return w

    localappdata = os.environ.get("LOCALAPPDATA", "").strip()
    if localappdata:
        local = Path(localappdata)
        links = local / "Microsoft" / "WinGet" / "Links"
        for exe in ("ffmpeg.exe", "ffmpeg.EXE"):
            p = links / exe
            if p.is_file():
                return str(p)

        pkg_root = local / "Microsoft" / "WinGet" / "Packages"
        if pkg_root.is_dir():
            candidates = sorted(pkg_root.glob("Gyan.FFmpeg*")) + sorted(
                pkg_root.glob("ffmpeg-*")
            )
            for pkg in candidates:
                for exe_path in sorted(pkg.rglob("ffmpeg.exe")):
                    return str(exe_path)
    return None


def _is_ebml_webm_magic(raw: bytes) -> bool:
    return len(raw) >= 4 and raw[0] == 0x1A and raw[1:4] == b"\x45\xdf\xa3"


def _ffmpeg_stderr_tail(stderr: Optional[bytes], limit: int = 600) -> str:
    if not stderr:
        return ""
    text = stderr.decode("utf-8", errors="replace").strip().replace("\r", "")
    return text[-limit:] if len(text) > limit else text


def _ffmpeg_bytes_to_pcm_wav_mono(
    raw: bytes,
    target_sr: int,
) -> tuple[Optional[bytes], list[str]]:
    """
    Try FFmpeg stdin decode with sensible format hints for Expo WebM (EBML).
    Returns (wav_bytes_or_none, log_lines_for_errors).
    """
    ffmpeg_exe = _ffmpeg_executable()
    details: list[str] = []

    def run(flags_before_i: list[str], label: str) -> tuple[Optional[bytes], Optional[str]]:
        if ffmpeg_exe is None:
            return None, None
        try:
            args = [
                ffmpeg_exe,
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                *flags_before_i,
                "-i",
                "pipe:0",
                "-vn",
                "-f",
                "wav",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "pipe:1",
            ]
            proc = subprocess.run(
                args,
                input=raw,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=int(os.environ.get("VOCAL_FFMPEG_TIMEOUT_SEC", "90")),
            )
            stderr_s = _ffmpeg_stderr_tail(proc.stderr)
            ok = proc.returncode == 0 and proc.stdout and len(proc.stdout) >= 100
            if ok:
                return proc.stdout, None
            hint = stderr_s or f"exit_code={proc.returncode}"
            return None, hint
        except FileNotFoundError:
            details.append(f"ffmpeg({label}): exécutable manquant après résolution PATH")
            return None, ""
        except subprocess.SubprocessTimeoutError:
            details.append(f"ffmpeg({label}): timeout après {int(os.environ.get('VOCAL_FFMPEG_TIMEOUT_SEC', '90'))} s")
            return None, ""

    attempts: list[tuple[list[str], str]] = []
    if _is_ebml_webm_magic(raw):
        attempts.append((["-fflags", "+genpts", "-f", "webm"], "webm"))
        attempts.append((["-fflags", "+genpts", "-f", "matroska"], "matroska"))
    attempts.append(([], "auto_detect"))

    if ffmpeg_exe is None:
        details.append(
            "ffmpeg introuvable: définissez FFMPEG_PATH vers ffmpeg.exe OU redémarrez Cursor/le PC "
            "après winget pour que PATH soit vu par Python."
        )
        return None, details

    for flags, lab in attempts:
        blob, stderr_msg = run(flags, lab)
        if blob is not None:
            return blob, details
        if stderr_msg:
            details.append(f"ffmpeg[{lab}]: {stderr_msg}")
    return None, details


def _pcm_from_wav_bytes(wav_blob: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    y, sr = sf.read(io.BytesIO(wav_blob), dtype="float32", always_2d=False)
    y = _normalize_channels(y)
    return y, int(sr)


def decode_audio_bytes_to_mono_pcm(raw: bytes, target_sr: int) -> np.ndarray:
    """
    Decode arbitrary audio bytes → mono float32 `target_sr` Hz.
    WAV/FLAC: soundfile BytesIO / scipy WAV. Else: ffmpeg pipe, then tempfile+librosa.
    """
    if not raw:
        raise ValueError("Empty audio.")

    errs: list[str] = []

    try:
        import soundfile as sf  # noqa: PLC0415

        bio = io.BytesIO(raw)
        y, sr = sf.read(bio, dtype="float32", always_2d=False)
        y = _normalize_channels(y)
        if sr != target_sr:
            import librosa  # noqa: PLC0415

            y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=target_sr).astype(
                np.float32
            )
        return np.asarray(y, dtype=np.float32)
    except Exception as exc:
        errs.append(f"soundfile(BytesIO): {exc}")

    try:
        from scipy.io import wavfile  # noqa: PLC0415

        sr, data = wavfile.read(io.BytesIO(raw))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.asarray(data)
        maxv = np.max(np.abs(data)) if data.size else 0.0
        if np.issubdtype(data.dtype, np.floating):
            y = data.astype(np.float32)
        elif maxv > np.iinfo(np.int16).max:
            y = (data.astype(np.float64) / np.float64(np.iinfo(data.dtype).max)).astype(np.float32)
        else:
            y = (data.astype(np.float32) / 32768.0).astype(np.float32)
        y = np.clip(y.astype(np.float32), -1.0, 1.0).astype(np.float32)
        if sr != target_sr:
            import librosa  # noqa: PLC0415

            y = librosa.resample(y.astype(np.float64), orig_sr=int(sr), target_sr=target_sr).astype(
                np.float32
            )
        return y
    except Exception as exc:
        errs.append(f"scipy.wavfile: {exc}")

    wav_blob: Optional[bytes]
    ffmpeg_meta: list[str]
    wav_blob, ffmpeg_meta = _ffmpeg_bytes_to_pcm_wav_mono(raw, target_sr)
    errs.extend(ffmpeg_meta)
    if wav_blob is not None:
        try:
            y, _sr = _pcm_from_wav_bytes(wav_blob)
            if _sr != target_sr:
                import librosa  # noqa: PLC0415

                y = librosa.resample(
                    np.asarray(y, dtype=np.float64), orig_sr=_sr, target_sr=target_sr
                ).astype(np.float32)
            return np.asarray(y, dtype=np.float32)
        except Exception as exc:
            errs.append(f"ffmpeg→wav→soundfile: {exc}")

    import librosa  # noqa: PLC0415

    suffix = _sniff_suffix(raw)
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(raw)
        try:
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            return np.asarray(y, dtype=np.float32)
        except Exception as exc:
            errs.append(f"tempfile({suffix}) librosa.load: {exc!r} ({type(exc).__name__})")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    hint = (
        "Décodage impossible pour ce flux audio (WebM/M4A). "
        "1) Vérifiez ffmpeg: ouvrez un terminal et lancez `where ffmpeg`; "
        "2) Ou définissez FFMPEG_PATH=C:\\chemin\\vers\\ffmpeg.exe; "
        "3) Redémarrez Cursor / le terminal serveur après installation WinGet pour que PATH soit lu par Python."
    )
    raise RuntimeError(hint + " Détails: " + "; ".join(errs[:14]))


def audio_bytes_to_model_input(raw: bytes) -> np.ndarray:
    """
    Load audio bytes, build log-mel, min-max normalize, resize to 128×128×1.

    If labels seem swapped versus l’étiquetage à l’entraînement, définissez VOCAL_CLASS_ORDER.
    """
    try:
        import librosa  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Install librosa for vocal classification (pip install librosa)") from exc

    sr = _target_sample_rate()
    y = decode_audio_bytes_to_mono_pcm(raw, target_sr=sr)
    if y.size < 2048:
        y = np.pad(y, (0, max(0, 2048 - len(y))), mode="constant")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        fmin=20.0,
        fmax=sr // 2,
    )
    mel_db = librosa.power_to_db(mel + 1e-10)
    x = mel_db.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn + 1e-9:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x, dtype=np.float32)

    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("tensorflow is required for vocal resize/inference.") from exc

    t = tf.constant(x[..., np.newaxis], dtype=tf.float32)
    t = tf.image.resize(t, [128, 128])
    out = np.asarray(t.numpy(), dtype=np.float32)
    if out.ndim == 2:
        out = out[..., np.newaxis]
    return out


def parse_class_labels_env() -> list[str]:
    raw = os.environ.get(
        "VOCAL_CLASS_ORDER",
        "toux,normal,ovulation,besoin_nourriture",
    ).strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts if len(parts) == 4 else ["toux", "normal", "ovulation", "besoin_nourriture"]
