from __future__ import annotations

import json
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR.parent / "finale_model"
DEFAULT_BEHAVIOR_MODEL = MODEL_DIR / "behavior_rf_multimodal.joblib"
DEFAULT_MILK_MODEL = MODEL_DIR / "milk_xgb_pred_behavior_daily_milkhist_pipeline.joblib"
DEFAULT_MILK_METRICS = Path(r"C:\sensor_data\model_outputs\milk_xgb_pred_behavior_daily_milkhist_metrics.json")
DEFAULT_FAKE_IMU_CSV = ROOT_DIR.parent / "T10_0725.csv"

BEHAVIOR_LABELS: dict[int, str] = {
    1: "Walking",
    2: "Standing",
    3: "Feeding head up",
    4: "Feeding head down",
    5: "Licking",
    6: "Drinking",
    7: "Lying",
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
        if path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "behavior_model": str(RUNTIME.behavior_model_path),
                    "milk_model": str(RUNTIME.milk_model_path),
                },
            )
            return
        if path == "/simulate/tick":
            sensor_payload = SIMULATOR.next_payload()
            behavior = RUNTIME.predict_behavior(sensor_payload)
            milk = RUNTIME.predict_milk(sensor_payload)
            self._write_json(
                HTTPStatus.OK,
                {
                    "sensor": sensor_payload,
                    "behavior": behavior,
                    "milk": milk,
                },
            )
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
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> None:
    host = "0.0.0.0"
    port = 8008
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Model API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
