from __future__ import annotations

"""
Load a trained sklearn model + metadata, build the exact feature matrix,
then run `.predict()` for each second.

Kept separate from the CLI (`predict_behavior.py`) so notebooks or FastAPI wrappers
can import `predict_from_immu` directly.
"""

# --- Persisted model bundle = joblib pickle + JSON sidecar with feature schema ---
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd

from imu_head_synthesis import synthesize_head_aggregate
from pipeline_utils import (
    aggregate_head,
    aggregate_immu_per_second,
    load_head_data,
    load_immu_csv,
    preprocess_immu_dataframe,
)


def resolve_model_paths(model_dir: Path) -> Tuple[Path, Path]:
    """
    Pick which joblib+json pair to load.

    train_model.py saves:
      - behavior_rf_immu.joblib + metadata_immu.json
      - behavior_rf_multimodal.joblib + metadata_multimodal.json

    Older setups might still use behavior_rf.joblib + metadata.json — we try that first
    so nothing breaks if someone renamed files manually.
    """
    # Try in fixed order; first fully existing pair wins.
    candidates = [
        (model_dir / "behavior_rf.joblib", model_dir / "metadata.json"),
        (model_dir / "behavior_rf_immu.joblib", model_dir / "metadata_immu.json"),
        (model_dir / "behavior_rf_multimodal.joblib", model_dir / "metadata_multimodal.json"),
    ]
    for model_path, meta_path in candidates:
        if model_path.exists() and meta_path.exists():
            return model_path, meta_path
    raise FileNotFoundError(
        f"Model artifacts not found in {model_dir}. "
        "Train first with train_model.py (--out-dir) or pass --model-dir to predict_behavior.py. "
        f"Checked: {[str(p[0]) for p in candidates]}"
    )


def load_model_bundle(model_dir: Path):
    """Loads sklearn model + metadata dict (feature list, flags, hyperparams snapshot)."""
    model_path, meta_path = resolve_model_paths(model_dir)
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, metadata


def predict_from_immu(
    *,
    model_dir: Path,
    immu_file: Optional[Path] = None,
    immu_df: Optional[pd.DataFrame] = None,
    use_multimodal: bool = False,
    head_file: Optional[Path] = None,
    smooth_window: int = 0,
    behavior_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Core prediction API.

    Pass either immu_file (path) XOR immu_df (already-loaded DataFrame).
    If use_multimodal: merge synthesized or file-based head features like training.
    """
    # Exactly one input mode: avoids silent "empty df" bugs.
    if (immu_file is None) == (immu_df is None):
        raise ValueError("Provide exactly one of immu_file or immu_df")

    # Reload training-time settings (feature list must match columns).
    model, metadata = load_model_bundle(model_dir)
    feature_cols = metadata["feature_columns"]
    include_mag = bool(metadata.get("include_mag", False))
    history_seconds = int(metadata.get("history_seconds", 0))

    # Training might have used real head CSVs; at inference we're allowed to synthesize,
    # but it's good to warn so we remember train/test mismatch is possible.
    if (
        use_multimodal
        and metadata.get("head_source") == "head_csv"
        and (head_file is None or not head_file.exists())
    ):
        import warnings

        warnings.warn(
            "Modèle entraîné avec des CSV head_direction ; sans --head-file, "
            "les features tête sont synthétisées — alignez train/prédiction ou fournissez un CSV tête.",
            UserWarning,
            stacklevel=2,
        )

    # --- Load / normalize raw IMU the same way as training ---
    if immu_file is not None:
        imu_raw = load_immu_csv(immu_file)
    else:
        imu_raw = preprocess_immu_dataframe(immu_df)

    immu_agg = aggregate_immu_per_second(imu_raw, include_mag=include_mag)

    # --- Optionally add head modality (merge on ts_sec) ---
    if use_multimodal:
        if head_file is not None and head_file.exists():
            head_df = aggregate_head(load_head_data(head_file))
        else:
            head_df = synthesize_head_aggregate(imu_raw)
        # Left merge: keep every IMU second; forward-fill head gaps then zero-fill remainder.
        features_df = immu_agg.merge(head_df, on="ts_sec", how="left").ffill().fillna(0.0)
    else:
        features_df = immu_agg

    # --- Lag features: repeat current value backwards at file start instead of stuffing zeros,
    # because zeros often look like “lying still” to the RF and confuse the first seconds.
    if history_seconds > 0:
        features_df = features_df.sort_values("ts_sec").reset_index(drop=True)
        base_cols = [c for c in features_df.columns if c != "ts_sec"]
        for lag in range(1, history_seconds + 1):
            shifted = features_df[base_cols].shift(lag)
            for c in base_cols:
                col_lag = f"{c}_lag{lag}"
                features_df[col_lag] = shifted[c].fillna(features_df[c])
        features_df = features_df.fillna(0.0)

    # Guarantee column order matches training even if something was missing upstream
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[["ts_sec"] + feature_cols]

    # --- Run sklearn predict (one class id per row) ---
    pred = model.predict(features_df[feature_cols])
    out = pd.DataFrame({"ts_sec": features_df["ts_sec"], "pred_behavior": pred})

    # Optional post-processing: majority vote over a sliding time window (seconds).
    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        out = out.sort_values("ts_sec").reset_index(drop=True)
        smoothed = (
            out["pred_behavior"]
            .rolling(window=w, center=True, min_periods=1)
            .apply(lambda x: pd.Series(x.astype("int64")).mode().iat[0], raw=False)
            .astype("int64")
        )
        out["pred_behavior_smooth"] = smoothed

    # Optional human-readable labels from JSON mapping id -> string.
    if behavior_map:
        out["pred_behavior_name"] = out["pred_behavior"].map(behavior_map).fillna("unknown")
        if "pred_behavior_smooth" in out.columns:
            out["pred_behavior_smooth_name"] = (
                out["pred_behavior_smooth"].map(behavior_map).fillna("unknown")
            )

    return out
