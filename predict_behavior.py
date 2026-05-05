from __future__ import annotations

"""
CLI: run inference on ONE new IMMU CSV and save predictions.

Typical uni flow after training:
  1. python train_model.py ... --multimodal-only  (or default IMU-only)
  2. python predict_behavior.py --immu-file path/to/new.csv --model-dir artifacts/model \\
        --use-multimodal   # only if you trained the multimodal model
"""

# --- CLI parsing + JSON for optional behavior name lookup ---
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from predict_core import predict_from_immu


def parse_args() -> argparse.Namespace:
    """Define every flag the user can pass from the terminal."""
    parser = argparse.ArgumentParser(description="Predict cow behavior from a new IMMU CSV file.")
    # Required: raw sensor file path.
    parser.add_argument("--immu-file", type=Path, required=True, help="Path to raw IMMU csv.")
    # Folder that contains behavior_rf_*.joblib + metadata_*.json.
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument(
        "--behavior-map",
        type=Path,
        default=None,
        help="Optional JSON { \"1\": \"Walking\", ... } for readable class names in output.",
    )
    parser.add_argument(
        "--head-file",
        type=Path,
        default=None,
        help="Optional head CSV if you trained with real head_direction files.",
    )
    parser.add_argument(
        "--use-multimodal",
        action="store_true",
        help="Use IMU + head pipeline (must match how the model was trained).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Seconds window for rolling majority vote smoothing (0 = off).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/predictions/predictions.csv"),
    )
    return parser.parse_args()


def load_behavior_map(path: Optional[Path]) -> Optional[Dict[int, str]]:
    """JSON keys are strings in JSON; we cast to int ids for pandas .map()."""
    if path is None:
        return None
    mapping_raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in mapping_raw.items()}


def main() -> None:
    """Entry point when you run: python predict_behavior.py ..."""
    args = parse_args()
    # Make sure write path exists (parents=True creates nested folders too).
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    behavior_map = load_behavior_map(args.behavior_map)

    # Delegate all feature engineering + sklearn call to predict_core.
    out = predict_from_immu(
        model_dir=args.model_dir,
        immu_file=args.immu_file,
        immu_df=None,
        use_multimodal=args.use_multimodal,
        head_file=args.head_file,
        smooth_window=args.smooth_window,
        behavior_map=behavior_map,
    )

    # Persist results then print tiny summary stats in the console.
    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions: {args.output_csv}")
    print(f"Predicted seconds: {len(out)}")
    print("Predicted class counts:")
    print(out["pred_behavior"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
