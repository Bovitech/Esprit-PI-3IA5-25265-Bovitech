from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from gps_tracking.dataset import last_window_scaled
from gps_tracking.load_data import list_tag_ids, load_one_cow
from gps_tracking.model_lstm import CowTrajectoryLSTM
from gps_tracking.preprocess import preprocess_uwb

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
SEQ_LENGTH = 10


def _default_uwb_dir() -> Path:
    raw = os.environ.get("UWB_DATA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT_DIR / "gps_tracking" / "uwb_data").resolve()


def _default_checkpoint() -> Path | None:
    if os.environ.get("TRAJECTORY_CHECKPOINT"):
        return Path(os.environ["TRAJECTORY_CHECKPOINT"]).expanduser().resolve()
    models = ROOT_DIR / "gps_tracking" / "models"
    for name in ("lstm_all.pth", "lstm_t01.pth", "lstm.pth"):
        p = models / name
        if p.is_file():
            return p
    return None


def _default_scaler() -> Path | None:
    if os.environ.get("TRAJECTORY_SCALER"):
        return Path(os.environ["TRAJECTORY_SCALER"]).expanduser().resolve()
    models = ROOT_DIR / "gps_tracking" / "models"
    for name in ("scaler_all.pkl", "scaler_t01.pkl", "scaler.pkl"):
        p = models / name
        if p.is_file():
            return p
    return None


def _inverse_xy_cm(scaler, arr_2d: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(arr_2d.reshape(-1, 2))


class TrajectoryRuntime:
    """
    Loads CowTrajectoryLSTM + MinMaxScaler; serves /gps/herd and /predict/trajectory.
    """

    def __init__(self) -> None:
        self.uwb_dir = _default_uwb_dir()
        self.checkpoint_path: str | None = None
        self.scaler_path: str | None = None
        self.error: str | None = None
        self.ok = False
        self._model: CowTrajectoryLSTM | None = None
        self._scaler: Any = None

        ck = _default_checkpoint()
        sc = _default_scaler()
        if ck is None or not ck.is_file():
            self.error = f"No checkpoint (set TRAJECTORY_CHECKPOINT or add lstm_*.pth under {ROOT_DIR / 'gps_tracking' / 'models'})"
            return
        if sc is None or not sc.is_file():
            self.error = f"No scaler (set TRAJECTORY_SCALER or add scaler_*.pkl under {ROOT_DIR / 'gps_tracking' / 'models'})"
            return

        try:
            import torch

            self._model = CowTrajectoryLSTM()
            try:
                state = torch.load(ck, map_location=torch.device("cpu"), weights_only=False)
            except TypeError:
                state = torch.load(ck, map_location=torch.device("cpu"))
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise TypeError("Checkpoint must be a state_dict or contain state_dict")
            self._model.load_state_dict(state, strict=True)
            self._model.eval()

            with open(sc, "rb") as f:
                self._scaler = pickle.load(f)

            self.checkpoint_path = str(ck)
            self.scaler_path = str(sc)
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._scaler = None
            self.error = str(exc)

    def _predict_window(self, window_1_seq_2: np.ndarray) -> tuple[float, float]:
        import torch

        assert self._model is not None and self._scaler is not None
        t = torch.from_numpy(window_1_seq_2)
        with torch.no_grad():
            pred_scaled = self._model(t).numpy()
        xy = _inverse_xy_cm(self._scaler, pred_scaled[0])
        return float(xy[0, 0]), float(xy[0, 1])

    def _df_for_tag(self, tag_id: str) -> tuple[pd.DataFrame, str | None]:
        raw, path = load_one_cow(tag_id, self.uwb_dir)
        if raw.empty:
            return pd.DataFrame(), path
        clean = preprocess_uwb(raw, tag_id)
        return clean, path

    def predict_for_tag(self, tag_id: str) -> dict[str, Any]:
        if not self.ok:
            return {"ok": False, "error": self.error or "Trajectory model not loaded."}
        tag = tag_id.strip().upper()
        clean, path = self._df_for_tag(tag)
        if clean.empty or len(clean) < SEQ_LENGTH:
            return {
                "ok": False,
                "error": f"Not enough UWB rows for {tag} (need ≥{SEQ_LENGTH}). dir={self.uwb_dir}",
                "tag_id": tag,
                "csv_path": path,
            }
        assert self._scaler is not None
        w = last_window_scaled(clean, SEQ_LENGTH, self._scaler)
        if w is None:
            return {"ok": False, "error": "Could not build window", "tag_id": tag}
        x_n, y_n = self._predict_window(w)
        last = clean.iloc[-1]
        return {
            "ok": True,
            "tag_id": tag,
            "csv_path": path,
            "last_known": {
                "x_cm": float(last["x_cm"]),
                "y_cm": float(last["y_cm"]),
                "timestamp": float(last["timestamp"]) if "timestamp" in last and pd.notna(last["timestamp"]) else None,
            },
            "predicted_next": {"x_cm": x_n, "y_cm": y_n},
        }

    def predict_from_records(self, records: list[Any]) -> dict[str, Any]:
        if not self.ok:
            return {"ok": False, "error": self.error or "Trajectory model not loaded."}
        if not isinstance(records, list) or not records:
            return {"ok": False, "error": "records must be a non-empty list"}
        rows = []
        for r in records:
            if not isinstance(r, dict):
                continue
            x = r.get("x_cm", r.get("coord_x_cm"))
            y = r.get("y_cm", r.get("coord_y_cm"))
            ts = r.get("timestamp", 0)
            if x is None or y is None:
                continue
            rows.append({"timestamp": float(ts), "x_cm": float(x), "y_cm": float(y), "z_cm": float(r.get("z_cm", r.get("coord_z_cm", 0.0)) or 0.0)})
        if len(rows) < SEQ_LENGTH:
            return {"ok": False, "error": f"Need at least {SEQ_LENGTH} points with x_cm/y_cm"}
        df = pd.DataFrame(rows)
        df = preprocess_uwb(df, "INLINE")
        assert self._scaler is not None
        w = last_window_scaled(df, SEQ_LENGTH, self._scaler)
        if w is None:
            return {"ok": False, "error": "Could not build window from records"}
        x_n, y_n = self._predict_window(w)
        last = df.iloc[-1]
        return {
            "ok": True,
            "tag_id": None,
            "last_known": {"x_cm": float(last["x_cm"]), "y_cm": float(last["y_cm"])},
            "predicted_next": {"x_cm": x_n, "y_cm": y_n},
            "n_points": len(df),
        }

    def herd_snapshot(self) -> dict[str, Any]:
        if not self.uwb_dir.is_dir():
            return {
                "ok": False,
                "error": f"UWB data directory missing: {self.uwb_dir} (set UWB_DATA_DIR)",
                "uwb_dir": str(self.uwb_dir),
                "tags": [],
            }
        tags = list_tag_ids(self.uwb_dir)
        out_tags: list[dict[str, Any]] = []
        for tid in tags:
            pred = self.predict_for_tag(tid) if self.ok else None
            entry: dict[str, Any] = {"tag_id": tid}
            raw, path = load_one_cow(tid, self.uwb_dir)
            clean = preprocess_uwb(raw, tid) if not raw.empty else pd.DataFrame()
            if not clean.empty:
                last = clean.iloc[-1]
                entry["last_known"] = {
                    "x_cm": float(last["x_cm"]),
                    "y_cm": float(last["y_cm"]),
                    "timestamp": float(last["timestamp"]) if "timestamp" in last and pd.notna(last["timestamp"]) else None,
                }
                entry["csv_path"] = path
            else:
                entry["last_known"] = None
                entry["csv_path"] = path
            if pred and pred.get("ok"):
                entry["predicted_next"] = pred["predicted_next"]
            else:
                entry["predicted_next"] = None
                if pred and pred.get("error"):
                    entry["predict_error"] = pred["error"]
            out_tags.append(entry)

        return {
            "ok": bool(self.ok),
            "uwb_dir": str(self.uwb_dir),
            "model_ok": bool(self.ok),
            "model_error": self.error,
            "tags": out_tags,
        }
