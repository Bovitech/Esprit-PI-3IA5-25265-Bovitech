from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import threading
import warnings
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

from vocal_preprocess import audio_bytes_to_model_input, parse_class_labels_env
from build_rl_state import build_state_vector
from illness.health_score import build_temporal_health_score

MODEL_DIR = ROOT_DIR / "finale_model"
DEFAULT_BEHAVIOR_MODEL = MODEL_DIR / "behavior_rf_multimodal.joblib"
DEFAULT_MILK_MODEL = MODEL_DIR / "milk_xgb_pred_behavior_daily_milkhist_pipeline.joblib"
DEFAULT_MILK_METRICS = MODEL_DIR / "milk_xgb_pred_behavior_daily_milkhist_metrics.json"
DEFAULT_FAKE_IMU_CSV = ROOT_DIR / "T10_0725.csv"

# Trained in stress_sensor; copy StressDetectionV3_trained.pt here or set STRESS_V3_CHECKPOINT
DEFAULT_STRESS_V3 = MODEL_DIR / "StressDetectionV3_trained.pt"
STRESS_V3_WINDOW = 60
STRESS_CLASS_NAMES: tuple[str, str, str] = ("Normal", "At risk", "Stressed")
ILLNESS_ACTION_NAMES: tuple[str, str, str] = ("Healthy", "At-risk", "Ill")
ILLNESS_ACTION_NAMES_FR: dict[str, str] = {
    "Healthy": "Bon état",
    "At-risk": "À risque",
    "Ill": "Alerte santé",
}
EIGHT_H_SECONDS = 8 * 60 * 60

BEHAVIOR_LABELS: dict[int, str] = {
    1: "Walking",
    2: "Standing",
    3: "Feeding head up",
    4: "Feeding head down",
    5: "Licking",
    6: "Drinking",
    7: "Lying",
}


def _parse_cow_id_to_index(cow_id: str, num_cows: int) -> int:
    s = (cow_id or "C1").upper().strip()
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return 0
    idx = int(digits) - 1
    if idx < 0:
        return 0
    return int(min(max(0, idx), num_cows - 1))


def _resolve_stress_checkpoint() -> Path:
    raw = os.environ.get("STRESS_V3_CHECKPOINT", str(DEFAULT_STRESS_V3))
    return Path(raw).expanduser().resolve()


class Lying8hBuffer:
    """
    One binary sample per second (1 = cow classified as Lying) for the last 8 h.
    Used with live /simulate/tick to approximate lying minutes in the last 8 h.
    """

    def __init__(self) -> None:
        self._q: deque[int] = deque(maxlen=EIGHT_H_SECONDS)

    def push(self, is_lying: bool) -> None:
        self._q.append(1 if is_lying else 0)

    def lying_minutes_8h(self) -> float:
        if not self._q:
            return 4.0 * 60.0
        if len(self._q) < 60:
            return 4.0 * 60.0
        return float(sum(self._q)) / 60.0

    def clear(self) -> None:
        self._q.clear()


class StressV3Runtime:
    """
    StressDetectionV3: inputs THI, CBT (neck/body T), and lying (8 h minutes proxy).
    Window length must match training (60 × 1 min resolution; live uses a flat window from current point).
    """

    def __init__(self) -> None:
        self._ok = False
        self._err: str = ""
        self._model: Any = None
        self._num_cows: int = 20
        self._window: int = STRESS_V3_WINDOW
        self._device: str = "cpu"
        # Defaults if checkpoint has no normalisation stats
        self._thi_m = 70.0
        self._thi_s = 6.0
        self._neck_m = 38.5
        self._neck_s = 0.45
        self._ly_m = 0.35
        self._ly_s = 0.25
        self._path: Optional[Path] = None
        self._load()

    def _load(self) -> None:
        try:
            import torch

            from stress_v3_model import StressDetectionV3
        except Exception as exc:  # noqa: BLE001
            self._ok = False
            self._err = f"torch or stress_v3_model unavailable: {exc}"
            return

        path = _resolve_stress_checkpoint()
        self._path = path
        if not path.is_file():
            self._ok = False
            self._err = f"Checkpoint not found: {path} (set STRESS_V3_CHECKPOINT or place file in finale_model/)"
            return
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            try:
                raw: Any = torch.load(path, map_location=self._device, weights_only=False)  # type: ignore[call-overload]
            except TypeError:
                raw = torch.load(path, map_location=self._device)
        except Exception as exc:  # noqa: BLE001
            self._ok = False
            self._err = f"Failed to load {path}: {exc}"
            return

        if not isinstance(raw, dict):
            self._ok = False
            self._err = "Unrecognised checkpoint (expected a dict with training metadata and weights)."
            return
        meta = raw
        state: Any = None
        for key in ("model_state_dict", "model_state", "state_dict", "state"):
            cand = raw.get(key)
            if isinstance(cand, dict) and any(k.startswith("thi_enc") for k in cand):
                state = cand
                break
        if state is None and any(
            k.startswith("thi_enc") or k.startswith("neck_enc") or k.startswith("cow_emb")
            for k in raw
        ):
            state = {k: v for k, v in raw.items() if torch.is_tensor(v)}
        if state is None or not state:
            self._ok = False
            self._err = "Unrecognised checkpoint format (no thi_enc/cow_emb tensors found)."
            return

        try:
            w = state["cow_emb.weight"]
            self._num_cows = int(w.shape[0]) if hasattr(w, "shape") else 20
        except Exception:
            self._num_cows = 20

        self._window = int(meta.get("window", STRESS_V3_WINDOW))
        for a, b in (
            ("thi_mean", "_thi_m"),
            ("thi_std", "_thi_s"),
            ("neck_mean", "_neck_m"),
            ("neck_std", "_neck_s"),
            ("lying_mean", "_ly_m"),
            ("lying_std", "_ly_s"),
        ):
            if a in meta and meta[a] is not None:
                try:
                    setattr(self, b, float(meta[a]))
                except (TypeError, ValueError):
                    pass

        try:
            model = StressDetectionV3(num_cows=self._num_cows).to(self._device)
            model.load_state_dict(state, strict=True)
        except Exception as exc:  # noqa: BLE001
            self._ok = False
            self._err = f"State dict load failed: {exc}"
            return
        model.eval()
        self._model = model
        self._ok = True
        self._err = ""

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def error(self) -> str:
        return self._err

    @property
    def num_cows(self) -> int:
        return self._num_cows

    @property
    def window(self) -> int:
        return self._window

    @property
    def checkpoint_path(self) -> Optional[Path]:
        return self._path

    def predict(
        self,
        thi: float,
        cbt: float,
        lying_minutes_8h: float,
        cow_id: str = "C01",
    ) -> dict[str, Any]:
        if not self._ok or self._model is None:
            return {"ok": False, "error": self._err or "Stress model not loaded."}

        import torch

        t = (float(thi) - self._thi_m) / (self._thi_s if self._thi_s > 1e-9 else 1.0)
        n = (float(cbt) - self._neck_m) / (self._neck_s if self._neck_s > 1e-9 else 1.0)
        f = min(1.0, max(0.0, float(lying_minutes_8h) / 480.0))
        ly = (f - self._ly_m) / (self._ly_s if self._ly_s > 1e-9 else 1.0)
        t_seq = t
        n_seq = n
        ly_seq = ly
        t_tensor = (
            torch.full(
                (1, self._window, 1), float(t_seq), device=self._device, dtype=torch.float32
            )
        )
        n_tensor = (
            torch.full(
                (1, self._window, 1), float(n_seq), device=self._device, dtype=torch.float32
            )
        )
        ly_tensor = (
            torch.full(
                (1, self._window, 1), float(ly_seq), device=self._device, dtype=torch.float32
            )
        )
        cow_i = _parse_cow_id_to_index(cow_id, self._num_cows)
        cow_t = torch.tensor([cow_i], device=self._device, dtype=torch.long)
        with torch.inference_mode():
            logits = self._model(t_tensor, n_tensor, ly_tensor, cow_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        k = int(np.argmax(probs))
        return {
            "ok": True,
            "pred_stress": k,
            "pred_stress_name": STRESS_CLASS_NAMES[k] if 0 <= k < 3 else str(k),
            "probabilities": {STRESS_CLASS_NAMES[i]: float(probs[i]) for i in range(min(3, len(probs)))},
            "inputs_used": {
                "thi": float(thi),
                "cbt_temp_c": float(cbt),
                "lying_minutes_8h": float(lying_minutes_8h),
                "cow_id": cow_id,
            },
        }


def _sb3_zip_from_unpacked_dir(unzipped: Path) -> Optional[Path]:
    """Stable-Baselines3 loads a .zip; materialize `{dir}.zip` from an extracted save once."""
    if not unzipped.is_dir():
        return None
    zip_path = unzipped.parent / f"{unzipped.name}.zip"
    if zip_path.is_file():
        return zip_path
    if not (unzipped / "policy.pth").is_file():
        return None
    import zipfile

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in unzipped.iterdir():
                if f.is_file():
                    zf.write(f, arcname=f.name)
    except OSError:
        return None
    return zip_path if zip_path.is_file() else None


def _resolve_illness_ppo_zip() -> Optional[Path]:
    raw = os.environ.get("ILLNESS_PPO_PATH", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".zip":
            return p
        z = _sb3_zip_from_unpacked_dir(p)
        if z is not None:
            return z
    base = MODEL_DIR / "artifacts" / "illness_model" / "illness_ppo"
    return _sb3_zip_from_unpacked_dir(base)


class IllnessPpoRuntime:
    """PPO policy for illness classes Healthy / At-risk / Ill (34-d Box state)."""

    def __init__(self) -> None:
        self._ok = False
        self._err = ""
        self._model: Any = None
        self._path: Optional[Path] = None
        self._load()

    def _load(self) -> None:
        zp = _resolve_illness_ppo_zip()
        self._path = zp
        if zp is None or not zp.is_file():
            self._ok = False
            self._err = (
                "Illness PPO not found: place illness_ppo.zip or unpack SB3 save under "
                f"{MODEL_DIR / 'artifacts' / 'illness_model' / 'illness_ppo'} "
                "(or set ILLNESS_PPO_PATH)."
            )
            return
        try:
            from stable_baselines3 import PPO  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._ok = False
            self._err = f"stable_baselines3 import failed: {exc}"
            return
        try:
            self._model = PPO.load(str(zp), device="cpu")
            self._model.policy.set_training_mode(False)
            self._ok = True
            self._err = ""
        except Exception as exc:  # noqa: BLE001
            self._ok = False
            self._err = f"Failed to load illness PPO from {zp}: {exc}"

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def error(self) -> str:
        return self._err

    @property
    def model_path(self) -> Optional[Path]:
        return self._path

    def policy_probs(self, obs: np.ndarray) -> np.ndarray:
        if not self._ok or self._model is None:
            raise RuntimeError(self._err or "Illness model not loaded.")
        import torch  # noqa: PLC0415

        obs_vec = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32)
        with torch.no_grad():
            features = self._model.policy.extract_features(obs_tensor)
            latent_pi, _ = self._model.policy.mlp_extractor(features)
            dist = self._model.policy._get_action_dist_from_latent(latent_pi)
            probs = dist.distribution.probs
            if hasattr(probs, "detach"):
                probs = probs.detach()
            return probs.cpu().numpy().astype(np.float32).ravel()


def _resolve_vocal_model_path() -> Path:
    raw = os.environ.get("VOCAL_MODEL_PATH", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            return p
    cand = MODEL_DIR / "model_audio_classification.h5"
    if cand.is_file():
        return cand.resolve()
    matches = sorted(MODEL_DIR.glob("model_audio_classification*.h5"))
    if matches:
        return matches[0].resolve()
    return cand


class VocalClassificationRuntime:
    """
    Keras CNN on 128×128×1 mel-spectrogram (see vocal_preprocess).
    Class names default to toux, normal, ovulation, besoin_nourriture — override with VOCAL_CLASS_ORDER.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._path_checked: Optional[Path] = None
        self._ok = False
        self._err = ""
        self._labels = parse_class_labels_env()

    def refresh_labels(self) -> None:
        self._labels = parse_class_labels_env()

    def probe_path(self) -> tuple[Optional[Path], bool]:
        self._path_checked = _resolve_vocal_model_path()
        return self._path_checked, self._path_checked.is_file()

    def _load(self) -> None:
        if self._model is not None:
            return
        path = _resolve_vocal_model_path()
        self._path_checked = path
        if not path.is_file():
            self._ok = False
            self._err = f"Vocal model not found: {path} (copy .h5 to finale_model/ or set VOCAL_MODEL_PATH)"
            return
        try:
            import tensorflow as tf  # noqa: PLC0415
        except ImportError as exc:
            self._ok = False
            self._err = f"tensorflow not installed: {exc}"
            return
        try:
            self._model = tf.keras.models.load_model(path)
            self._ok = True
            self._err = ""
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._ok = False
            self._err = f"Failed to load vocal model: {exc}"

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def error(self) -> str:
        return self._err

    def predict_from_audio_bytes(self, raw: bytes) -> dict[str, Any]:
        if len(raw) < 200:
            return {"ok": False, "error": "Audio trop court ou fichier invalide."}
        self._load()
        if not self._ok or self._model is None:
            return {"ok": False, "error": self._err or "Vocal model unavailable."}
        if len(self._labels) != 4:
            return {"ok": False, "error": "VOCAL_CLASS_ORDER must list exactly 4 comma-separated labels."}
        try:
            x = audio_bytes_to_model_input(raw)
            batch = np.expand_dims(x, 0).astype(np.float32)
            probs_arr = np.asarray(self._model.predict(batch, verbose=0), dtype=np.float32).ravel()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if probs_arr.size != 4:
            return {"ok": False, "error": f"Unexpected model output size {probs_arr.size} (need 4)."}
        probs_arr = probs_arr.astype(np.float64)
        s = float(np.sum(probs_arr))
        if s > 1e-12:
            probs_arr = probs_arr / s
        probs_arr = probs_arr.astype(np.float32)
        k = int(np.argmax(probs_arr))
        return {
            "ok": True,
            "pred_vocal_class": k,
            "pred_vocal_name": self._labels[k] if 0 <= k < len(self._labels) else str(k),
            "probabilities": {self._labels[i]: float(probs_arr[i]) for i in range(4)},
        }


def _install_sklearn_pickle_compat() -> None:
    """
    Compatibility shim for older sklearn-pickled pipelines.

    Some models trained/saved on sklearn<=1.6 reference private symbols
    removed/renamed in newer versions (e.g. _RemainderColsList). We create
    a light fallback symbol so joblib.load can resolve it.
    """
    try:
        from sklearn.compose import _column_transformer as ct_mod
    except Exception:
        return
    if not hasattr(ct_mod, "_RemainderColsList"):
        # A simple list subtype is sufficient for unpickling older artifacts.
        class _RemainderColsList(list):
            pass

        ct_mod._RemainderColsList = _RemainderColsList


def _repair_legacy_sklearn_state(model: Any) -> None:
    """
    Patch legacy sklearn estimator internals missing in newer versions.

    Current known issue:
    - SimpleImputer missing `_fill_dtype` after unpickling old artifacts.
    """
    try:
        from sklearn.impute import SimpleImputer
    except Exception:
        return

    seen_ids: set[int] = set()
    stack: list[Any] = [model]
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)

        if isinstance(obj, SimpleImputer) and not hasattr(obj, "_fill_dtype"):
            stats = getattr(obj, "statistics_", None)
            if isinstance(stats, np.ndarray):
                obj._fill_dtype = stats.dtype
            else:
                obj._fill_dtype = np.dtype("float64")

        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
            continue
        if hasattr(obj, "__dict__"):
            stack.extend(obj.__dict__.values())


def _thi_from_temp_rh_celsius(temp_c: float, rh_pct: float) -> float:
    t = float(temp_c)
    rh = float(rh_pct)
    return (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0)


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None or raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _load_milk_features(metrics_path: Path) -> list[str]:
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        feats = data.get("features")
        if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
            return feats
    # Safe fallback if metrics file is not present.
    return [
        "DIM",
        "milk_lag1",
        "milk_roll3_mean",
        "cbt_temp_mean",
        "cbt_temp_std",
        "cbt_temp_min",
        "cbt_temp_max",
        "behavior_n",
        "behavior_mean",
        "behavior_std",
        "thi_mean",
        "thi_std",
        "thi_max",
        "env_temp_mean",
        "env_humidity_mean",
        "dow",
        "month",
    ]


def _build_behavior_feature_row(
    payload: dict[str, Any],
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Map raw IMU manual inputs to the model expected feature names.

    Expected inputs from mobile app:
    - accel_x_mps2, accel_y_mps2, accel_z_mps2
    - mag_x_uT, mag_y_uT, mag_z_uT
    """
    ax = _safe_float(payload.get("accel_x_mps2"), 0.0)
    ay = _safe_float(payload.get("accel_y_mps2"), 0.0)
    az = _safe_float(payload.get("accel_z_mps2"), 0.0)
    mx = _safe_float(payload.get("mag_x_uT"), 0.0)
    my = _safe_float(payload.get("mag_y_uT"), 0.0)
    mz = _safe_float(payload.get("mag_z_uT"), 0.0)
    acc_norm = float(np.sqrt(ax * ax + ay * ay + az * az))
    mag_norm = float(np.sqrt(mx * mx + my * my + mz * mz))

    values = {
        "accel_x_mps2": ax,
        "accel_y_mps2": ay,
        "accel_z_mps2": az,
        "mag_x_uT": mx,
        "mag_y_uT": my,
        "mag_z_uT": mz,
        "acc_norm": acc_norm,
        "mag_norm": mag_norm,
        # legacy aliases
        "acc_mean": acc_norm,
        "acc_std": 0.0,
        "acc_min": acc_norm,
        "acc_max": acc_norm,
        "roll_mean": 0.0,
        "pitch_mean": 0.0,
        "yaw_mean": 0.0,
    }

    row: dict[str, float] = {}
    for col in feature_cols:
        c = col.lower()
        if col in values:
            row[col] = float(values[col])
            continue
        if "accel_x" in c:
            row[col] = ax
        elif "accel_y" in c:
            row[col] = ay
        elif "accel_z" in c:
            row[col] = az
        elif "mag_x" in c:
            row[col] = mx
        elif "mag_y" in c:
            row[col] = my
        elif "mag_z" in c:
            row[col] = mz
        elif "acc_norm" in c:
            row[col] = acc_norm
        elif "mag_norm" in c:
            row[col] = mag_norm
        elif c.endswith("_std"):
            row[col] = 0.0
        else:
            row[col] = 0.0
    return row


class ModelRuntime:
    def __init__(self) -> None:
        self.behavior_model_path = DEFAULT_BEHAVIOR_MODEL
        self.milk_model_path = DEFAULT_MILK_MODEL
        self.milk_metrics_path = DEFAULT_MILK_METRICS

        if not self.behavior_model_path.exists():
            raise FileNotFoundError(f"Behavior model not found: {self.behavior_model_path}")
        if not self.milk_model_path.exists():
            raise FileNotFoundError(f"Milk model not found: {self.milk_model_path}")

        _install_sklearn_pickle_compat()
        # Keep startup logs readable; models are still loaded as-is.
        warnings.filterwarnings(
            "once",
            category=UserWarning,
            module="sklearn.base",
        )
        self.behavior_model = joblib.load(self.behavior_model_path)
        self.milk_model = joblib.load(self.milk_model_path)
        _repair_legacy_sklearn_state(self.milk_model)
        self.milk_feature_cols = _load_milk_features(self.milk_metrics_path)
        self.behavior_feature_cols = self._resolve_behavior_feature_columns()

    def _resolve_behavior_feature_columns(self) -> list[str]:
        estimator = self.behavior_model
        if hasattr(estimator, "feature_names_in_"):
            return list(estimator.feature_names_in_)
        return [
            "acc_mean",
            "acc_std",
            "acc_min",
            "acc_max",
            "roll_mean",
            "pitch_mean",
            "yaw_mean",
        ]

    def predict_behavior(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = _build_behavior_feature_row(payload, self.behavior_feature_cols)
        x_df = pd.DataFrame([row])
        pred = int(self.behavior_model.predict(x_df)[0])
        return {
            "pred_behavior": pred,
            "pred_behavior_name": BEHAVIOR_LABELS.get(pred, str(pred)),
        }

    def predict_milk(self, payload: dict[str, Any]) -> dict[str, Any]:
        date_str = str(payload.get("date", "")).strip()
        dt = pd.to_datetime(date_str, errors="coerce")
        dow = int(dt.dayofweek) if pd.notna(dt) else int(_safe_float(payload.get("dow"), 0))
        month = int(dt.month) if pd.notna(dt) else int(_safe_float(payload.get("month"), 1))

        env_temp = _safe_float(payload.get("env_temp_mean"), _safe_float(payload.get("temp_c"), 25.0))
        env_hum = _safe_float(payload.get("env_humidity_mean"), _safe_float(payload.get("humidity_per"), 60.0))
        thi = _safe_float(payload.get("thi_mean"), _thi_from_temp_rh_celsius(env_temp, env_hum))

        cbt = _safe_float(payload.get("cbt_temp_mean"), _safe_float(payload.get("cbt_temp_c"), 38.4))
        behavior_mean = _safe_float(payload.get("behavior_mean"), _safe_float(payload.get("behavior_id"), 7))

        full_payload = {
            "DIM": _safe_float(payload.get("DIM"), 220.0),
            "milk_lag1": _safe_float(payload.get("milk_lag1"), np.nan),
            "milk_roll3_mean": _safe_float(payload.get("milk_roll3_mean"), np.nan),
            "cbt_temp_mean": cbt,
            "cbt_temp_std": _safe_float(payload.get("cbt_temp_std"), 0.0),
            "cbt_temp_min": _safe_float(payload.get("cbt_temp_min"), cbt),
            "cbt_temp_max": _safe_float(payload.get("cbt_temp_max"), cbt),
            "behavior_n": _safe_float(payload.get("behavior_n"), 86400.0),
            "behavior_mean": behavior_mean,
            "behavior_std": _safe_float(payload.get("behavior_std"), 0.0),
            "thi_mean": thi,
            "thi_std": _safe_float(payload.get("thi_std"), 0.0),
            "thi_max": _safe_float(payload.get("thi_max"), thi),
            "env_temp_mean": env_temp,
            "env_humidity_mean": env_hum,
            "dow": dow,
            "month": month,
        }

        x_df = pd.DataFrame([{c: full_payload.get(c, np.nan) for c in self.milk_feature_cols}])
        pred = float(self.milk_model.predict(x_df)[0])
        return {"prediction_milk_kg_day": pred, "thi_mean": thi}


class FakeSensorSimulator:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._idx = 0
        self._df = self._load_source()

    def _load_source(self) -> pd.DataFrame:
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            needed = [
                "accel_x_mps2",
                "accel_y_mps2",
                "accel_z_mps2",
                "mag_x_uT",
                "mag_y_uT",
                "mag_z_uT",
            ]
            if all(c in df.columns for c in needed):
                return df[needed].copy().reset_index(drop=True)
        # Fallback small seed if CSV not found.
        return pd.DataFrame(
            [
                {
                    "accel_x_mps2": -7.8,
                    "accel_y_mps2": -5.9,
                    "accel_z_mps2": -3.7,
                    "mag_x_uT": -35.6,
                    "mag_y_uT": -19.1,
                    "mag_z_uT": -16.5,
                }
            ]
        )

    def next_payload(self) -> dict[str, float]:
        if len(self._df) == 0:
            raise RuntimeError("Fake IMU data source is empty.")
        row = self._df.iloc[self._idx % len(self._df)]
        t = float(self._idx)
        self._idx += 1

        # Smooth synthetic context changing every second.
        temp_c = 25.0 + 2.2 * np.sin(t / 35.0)
        humidity = 58.0 + 8.0 * np.cos(t / 48.0)
        cbt = 38.4 + 0.18 * np.sin(t / 25.0)
        behavior_mean = 4.0 + 1.5 * np.sin(t / 30.0)

        return {
            "accel_x_mps2": float(row["accel_x_mps2"]),
            "accel_y_mps2": float(row["accel_y_mps2"]),
            "accel_z_mps2": float(row["accel_z_mps2"]),
            "mag_x_uT": float(row["mag_x_uT"]),
            "mag_y_uT": float(row["mag_y_uT"]),
            "mag_z_uT": float(row["mag_z_uT"]),
            "temp_c": float(temp_c),
            "humidity_per": float(humidity),
            "cbt_temp_c": float(cbt),
            "DIM": 220.0,
            "milk_lag1": 18.0 + 0.5 * np.sin(t / 70.0),
            "milk_roll3_mean": 18.5 + 0.4 * np.cos(t / 75.0),
            "behavior_mean": float(behavior_mean),
            "behavior_n": 86400.0,
            "behavior_std": 1.0 + 0.2 * np.cos(t / 28.0),
            "cow_id": "C01",
            "date": "2023-08-03",
        }


RUNTIME = ModelRuntime()
SIMULATOR = FakeSensorSimulator(DEFAULT_FAKE_IMU_CSV)
STRESS_V3 = StressV3Runtime()
ILLNESS_PPO = IllnessPpoRuntime()
VOCAL_CLF = VocalClassificationRuntime()
LYING8H = Lying8hBuffer()

# Dernière mesure DHT/THI (ferme) — alimentée par POST /barn_sensor (ex. ESP8266) ou relais USB.
BARN_LOCK = threading.Lock()
BARN_LATEST: dict[str, Any] = {
    "ok": False,
    "temp_c": None,
    "humidity": None,
    "thi": None,
    "updated_iso": None,
}


def _resolve_stress_for_illness(payload: dict[str, Any], barn: dict[str, Any]) -> dict[str, Any]:
    st = payload.get("stress")
    if isinstance(st, dict) and st.get("ok") is not False and "pred_stress" in st:
        return st
    thi = _resolve_thi_for_stress(payload, barn)
    cbt = _safe_float(payload.get("cbt_temp_c"), _safe_float(payload.get("cbt"), 38.4))
    ly8 = _safe_float(payload.get("lying_minutes_8h"), float("nan"))
    if ly8 != ly8 and payload.get("use_server_lying_buffer"):
        ly8 = LYING8H.lying_minutes_8h()
    if ly8 != ly8:
        b = payload.get("pred_behavior")
        if b is not None and int(b) == 7:
            ly8 = 0.45 * 480.0
        elif b is not None:
            ly8 = 0.25 * 480.0
        else:
            ly8 = 4.0 * 60.0
    cow = str(payload.get("cow_id", "C01"))
    if not STRESS_V3.ok:
        return {"ok": False, "error": STRESS_V3.error or "Stress model not loaded."}
    return STRESS_V3.predict(thi, cbt, float(ly8), cow)


def _predict_illness_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not ILLNESS_PPO.ok:
        return {"ok": False, "error": ILLNESS_PPO.error or "Illness model not loaded."}
    with BARN_LOCK:
        barn = dict(BARN_LATEST)
    stress = _resolve_stress_for_illness(payload, barn)
    milk_out = RUNTIME.predict_milk(payload if isinstance(payload, dict) else {})
    milk_kg = float(milk_out.get("prediction_milk_kg_day", 18.0))

    merged = dict(payload) if isinstance(payload, dict) else {}
    merged.setdefault("prediction_milk_kg_day", milk_kg)

    obs, meta = build_state_vector(merged, barn=barn, stress=stress)
    probs_vec = ILLNESS_PPO.policy_probs(obs)
    k = int(np.argmax(probs_vec))
    names = ILLNESS_ACTION_NAMES
    class_name = names[k] if 0 <= k < len(names) else str(k)
    illness_probs = {names[i]: float(probs_vec[i]) for i in range(min(len(names), len(probs_vec)))}

    s_probs = stress.get("probabilities") if isinstance(stress.get("probabilities"), dict) else {}
    stress_p2 = float(s_probs.get("Stressed", 0.0)) if stress.get("ok") is not False else 0.0

    model_predictions = {
        "pred_behavior": float(_safe_float(merged.get("pred_behavior"), 0.0)),
        "prediction_milk_kg_day": milk_kg,
        "pred_stress": float(
            _safe_float(stress.get("pred_stress"), 0.0) if stress.get("ok") is not False else 0.0
        ),
        "stress_prob_2": stress_p2,
        "cbt_temp_c": float(_safe_float(merged.get("cbt_temp_c"), _safe_float(merged.get("cbt"), 38.5))),
    }

    temporal = build_temporal_health_score(
        cow_id=merged.get("cow_id", "C01"),
        timestamp=merged.get("timestamp"),
        predicted_action=k,
        confidence=float(probs_vec[k]),
        illness_probs=illness_probs,
        model_predictions=model_predictions,
        top_features=None,
        inrae_row=None,
    )
    return {
        "ok": True,
        "predicted_action": k,
        "class_name": class_name,
        "class_name_fr": ILLNESS_ACTION_NAMES_FR.get(class_name, class_name),
        "probabilities": illness_probs,
        "confidence": float(probs_vec[k]),
        "health_score": temporal.get("health_score"),
        "health_status": temporal.get("health_status"),
        "health_detail": temporal,
        "state_meta": meta,
    }


def _herdtrack_dashboard_dir() -> Path:
    raw = os.environ.get("HERDTRACK_DASHBOARD_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT_DIR / "gps_tracking" / "dashboard").resolve()


def _serve_herdtrack_dashboard(handler: BaseHTTPRequestHandler, path: str) -> bool:
    """
    Serve HerdTrack static files from Bovitech-GPS-style dashboard (index.html, app.js, JSON).
    GET /gps/dashboard → redirect to /gps/dashboard/index.html
    """
    prefix = "/gps/dashboard"
    if path != prefix and not path.startswith(prefix + "/"):
        return False

    if path in (prefix, prefix + "/"):
        handler.send_response(HTTPStatus.FOUND)
        handler.send_header("Location", prefix + "/index.html")
        handler.send_header("Content-Length", "0")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        return True

    rel = unquote(path[len(prefix) + 1 :]).strip().replace("\\", "/")
    if not rel or rel.endswith("/"):
        rel = "index.html"
    if ".." in rel.split("/"):
        handler._write_json(HTTPStatus.FORBIDDEN, {"error": "Invalid path"})
        return True

    base = _herdtrack_dashboard_dir()
    file_path = (base / rel).resolve()
    try:
        file_path.relative_to(base)
    except ValueError:
        handler._write_json(HTTPStatus.FORBIDDEN, {"error": "Invalid path"})
        return True

    if not file_path.is_file():
        handler._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        return True

    ctype, _ = mimetypes.guess_type(str(file_path))
    if not ctype:
        ctype = "application/octet-stream"

    body = file_path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)
    return True


def _resolve_thi_for_stress(payload: dict[str, Any], barn: dict[str, Any]) -> float:
    thi = _safe_float(payload.get("thi"), float("nan"))
    if thi == thi:
        return float(thi)
    if bool(barn.get("ok")) and barn.get("thi") is not None:
        return float(barn["thi"])
    t = _safe_float(payload.get("temp_c"), 25.0)
    h = _safe_float(
        payload.get("humidity_per"), _safe_float(payload.get("humidity"), 60.0)
    )
    return float(_thi_from_temp_rh_celsius(t, h))


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "BovitechModelHTTP/1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._write_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if _serve_herdtrack_dashboard(self, path):
            return
        if path == "/health":
            with BARN_LOCK:
                barn = dict(BARN_LATEST)
            vpath, vfound = VOCAL_CLF.probe_path()
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "behavior_model": str(RUNTIME.behavior_model_path),
                    "milk_model": str(RUNTIME.milk_model_path),
                    "barn_thi_ok": bool(barn.get("ok") and barn.get("thi") is not None),
                    "stress_v3": {
                        "ok": bool(STRESS_V3.ok),
                        "path": str(p) if (p := STRESS_V3.checkpoint_path) else None,
                        "error": STRESS_V3.error or None,
                    },
                    "vocal_classifier": {
                        "model_path": str(vpath) if vpath else None,
                        "model_found": bool(vfound),
                        "labels": parse_class_labels_env(),
                        "loaded": bool(VOCAL_CLF.ok),
                        "error": VOCAL_CLF.error or None,
                    },
                    "illness_ppo": {
                        "ok": bool(ILLNESS_PPO.ok),
                        "path": str(p) if (p := ILLNESS_PPO.model_path) else None,
                        "error": ILLNESS_PPO.error or None,
                    },
                },
            )
            return
        if path == "/simulate/tick":
            sensor_payload = SIMULATOR.next_payload()
            behavior = RUNTIME.predict_behavior(sensor_payload)
            milk = RUNTIME.predict_milk(sensor_payload)
            with BARN_LOCK:
                barn = dict(BARN_LATEST)
            thi = _resolve_thi_for_stress(sensor_payload, barn)
            cbt = _safe_float(sensor_payload.get("cbt_temp_c"), 38.4)
            is_lying = int(behavior.get("pred_behavior", 0)) == 7
            LYING8H.push(is_lying)
            ly8 = LYING8H.lying_minutes_8h()
            cow = str(sensor_payload.get("cow_id", "C01"))
            if STRESS_V3.ok:
                stress = STRESS_V3.predict(thi, cbt, ly8, cow)
            else:
                stress = {"ok": False, "error": STRESS_V3.error, "model_missing": True}
            self._write_json(
                HTTPStatus.OK,
                {
                    "sensor": sensor_payload,
                    "behavior": behavior,
                    "milk": milk,
                    "stress": stress,
                },
            )
            return
        if path == "/barn_sensor":
            with BARN_LOCK:
                out = dict(BARN_LATEST)
            self._write_json(HTTPStatus.OK, out)
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload"})
            return

        try:
            if path == "/predict/behavior":
                out = RUNTIME.predict_behavior(payload if isinstance(payload, dict) else {})
                self._write_json(HTTPStatus.OK, out)
                return
            if path == "/predict/milk":
                out = RUNTIME.predict_milk(payload if isinstance(payload, dict) else {})
                self._write_json(HTTPStatus.OK, out)
                return
            if path == "/predict/vocal":
                VOCAL_CLF.refresh_labels()
                p = payload if isinstance(payload, dict) else {}
                b64 = p.get("audio_wav_base64") or p.get("audio_base64")
                if not b64 or not isinstance(b64, str):
                    self._write_json(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "ok": False,
                            "error": "Champ audio_wav_base64 requis (WAV ou M4A en base64, mono recommandé).",
                        },
                    )
                    return
                try:
                    raw = base64.b64decode(b64, validate=False)
                except Exception as exc:
                    self._write_json(
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "error": f"Base64 invalide: {exc}"},
                    )
                    return
                out = VOCAL_CLF.predict_from_audio_bytes(raw)
                self._write_json(HTTPStatus.OK, out)
                return
            if path == "/predict/stress":
                p = payload if isinstance(payload, dict) else {}
                with BARN_LOCK:
                    barn = dict(BARN_LATEST)
                thi = _resolve_thi_for_stress(p, barn)
                cbt = _safe_float(
                    p.get("cbt_temp_c"), _safe_float(p.get("cbt"), 38.4)
                )
                ly8 = _safe_float(p.get("lying_minutes_8h"), float("nan"))
                if ly8 != ly8 and p.get("use_server_lying_buffer"):
                    ly8 = LYING8H.lying_minutes_8h()
                if ly8 != ly8:
                    b = p.get("pred_behavior")
                    if b is not None and int(b) == 7:
                        ly8 = 0.45 * 480.0
                    elif b is not None:
                        ly8 = 0.25 * 480.0
                    else:
                        ly8 = 4.0 * 60.0
                cow = str(p.get("cow_id", "C01"))
                if not STRESS_V3.ok:
                    self._write_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"ok": False, "error": STRESS_V3.error or "Stress model not loaded."},
                    )
                    return
                out = STRESS_V3.predict(thi, cbt, float(ly8), cow)
                self._write_json(HTTPStatus.OK, out)
                return
            if path == "/predict/illness":
                p = payload if isinstance(payload, dict) else {}
                out = _predict_illness_payload(p)
                if out.get("ok") is False:
                    self._write_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        out,
                    )
                    return
                self._write_json(HTTPStatus.OK, out)
                return
            if path == "/barn_sensor":
                if not isinstance(payload, dict):
                    self._write_json(HTTPStatus.BAD_REQUEST, {"error": "expected JSON object"})
                    return
                temp_c = _safe_float(
                    payload.get("temp_c"), _safe_float(payload.get("T"), float("nan"))
                )
                rh = _safe_float(
                    payload.get("humidity"),
                    _safe_float(payload.get("rh"), _safe_float(payload.get("RH"), float("nan"))),
                )
                if temp_c != temp_c or rh != rh:  # NaN check
                    self._write_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "temp_c and humidity (or rh) are required as numbers"},
                    )
                    return
                thi = _safe_float(payload.get("thi"), float("nan"))
                if thi != thi:
                    thi = _thi_from_temp_rh_celsius(temp_c, rh)
                now = datetime.now(timezone.utc).isoformat()
                with BARN_LOCK:
                    BARN_LATEST["ok"] = True
                    BARN_LATEST["temp_c"] = float(temp_c)
                    BARN_LATEST["humidity"] = float(rh)
                    BARN_LATEST["thi"] = float(thi)
                    BARN_LATEST["updated_iso"] = now
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "temp_c": float(temp_c),
                        "humidity": float(rh),
                        "thi": float(thi),
                        "updated_iso": now,
                    },
                )
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> None:
    skip = os.getenv("SKIP_MODEL_DOWNLOAD", "").strip().lower() in ("1", "true", "yes")
    if not skip:
        print("Checking ML artifacts (models.manifest.json)...")
        try:
            from model_download import ensure_models_for_api

            if not ensure_models_for_api(quiet=False):
                print(
                    "ERROR: Required models missing. Fix models.manifest.json URLs or run:\n"
                    "  python scripts/download_models.py",
                    flush=True,
                )
                raise SystemExit(1)
        except ImportError:
            import subprocess

            rc = subprocess.call([sys.executable, str(ROOT_DIR / "scripts" / "download_models.py"), "--required-only"])
            if rc != 0:
                raise SystemExit(rc)
    else:
        print("SKIP_MODEL_DOWNLOAD set — using existing files under finale_model/")

    host = "0.0.0.0"
    port = 8008
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Model API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
