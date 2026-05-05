"""
Run the saved milk model on one day / one cow (or batch via your own loop).

The training script saves:
  {out_dir}/{prefix}_pipeline.joblib
  {out_dir}/{prefix}_metrics.json   ← has the exact 'features' column order

Inference builds a single-row DataFrame with those columns; missing values become
NaN and the Pipeline's imputers fill them — but accuracy is way better if you
pass real DIM, lag milk, THI, behavior, etc.

Default paths assume defaults from train_milk_xgboost.py (same model_outputs folder).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "model_outputs"
DEFAULT_PREFIX = os.environ.get("MILK_MODEL_PREFIX", "milk_production")


def default_pipeline_path() -> Path:
    return OUT_DIR / f"{DEFAULT_PREFIX}_pipeline.joblib"


def default_metrics_path() -> Path:
    return OUT_DIR / f"{DEFAULT_PREFIX}_metrics.json"


@dataclass
class MilkPredictionInput:
    """Typed-ish reference — you can also just use a plain dict (predict_from_payload does)."""

    cow_id: str
    date: str  # YYYY-MM-DD
    DIM: Optional[float] = None
    milk_lag1: Optional[float] = None
    milk_roll3_mean: Optional[float] = None
    cbt_temp_mean: Optional[float] = None
    cbt_temp_std: Optional[float] = None
    cbt_temp_min: Optional[float] = None
    cbt_temp_max: Optional[float] = None
    behavior_n: Optional[float] = None
    behavior_mean: Optional[float] = None
    behavior_std: Optional[float] = None
    thi_mean: Optional[float] = None
    thi_std: Optional[float] = None
    thi_max: Optional[float] = None
    env_temp_mean: Optional[float] = None
    env_humidity_mean: Optional[float] = None
    weather_temp_f: Optional[float] = None
    weather_humidity_pct: Optional[float] = None
    weather_wind_speed_mph: Optional[float] = None
    weather_pressure_in: Optional[float] = None
    weather_uv: Optional[float] = None
    weather_solar_wm2: Optional[float] = None


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"no model file here: {model_path}")
    return joblib.load(model_path)


def expected_feature_columns(metrics_path: Path) -> list[str]:
    """
    Training stores the raw feature names in metrics JSON.
    I need that list because after OneHotEncoder the sklearn feature names are different.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics file missing: {metrics_path}\n"
            "train first, or pass --metrics pointing at *_metrics.json"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or "features" not in metrics:
        raise ValueError("metrics.json needs a 'features' list from training")
    return list(metrics["features"])


def predict_from_payload(
    payload: Dict[str, Any],
    *,
    pipeline_path: Path,
    metrics_path: Path,
) -> Dict[str, Any]:
    """
    payload: dict (e.g. from JSON). Required keys: cow_id, date.
    I add dow/month from the date string to match training.
    """
    pipeline = load_model(pipeline_path)
    expected_cols = expected_feature_columns(metrics_path)

    if "cow_id" not in payload or "date" not in payload:
        raise ValueError("payload must have cow_id and date (YYYY-MM-DD)")

    date = pd.to_datetime(payload["date"], errors="raise").floor("D")
    row: Dict[str, Any] = dict(payload)
    row["dow"] = int(date.dayofweek)
    row["month"] = int(date.month)

    # one row, every column the model saw at train time; missing → NaN
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in expected_cols}])
    pred = float(pipeline.predict(X)[0])
    return {
        "cow_id": payload["cow_id"],
        "date": str(date.date()),
        "prediction_milk_kg_day": pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict milk kg for one row (JSON).")
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=default_pipeline_path(),
        help="joblib from train_milk_xgboost.py",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=default_metrics_path(),
        help="metrics JSON with 'features' list",
    )
    parser.add_argument(
        "--json",
        required=True,
        help='JSON string, or @path/to/file.json (I borrowed the @file idea from curl)',
    )
    args = parser.parse_args()

    raw = args.json
    if raw.startswith("@"):
        p = Path(raw[1:])
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = json.loads(raw)

    out = predict_from_payload(payload, pipeline_path=args.pipeline, metrics_path=args.metrics)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
