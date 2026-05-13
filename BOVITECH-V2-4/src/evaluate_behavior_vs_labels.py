"""
Compare behavior predictions to label CSV (ground truth): accuracy, F1, confusion matrix.

Usage (from project root):
  python src/evaluate_behavior_vs_labels.py ^
    --immu-file "C:/sensor_data/sensor_data/main_data/immu/T10/T10_0725.csv" ^
    --label-file "C:/sensor_data/sensor_data/behavior_labels/individual/C10_0725.csv" ^
    --model-dir "C:/bovitech_artifacts/model" ^
    --use-multimodal ^
    --head-file "C:/sensor_data/sensor_data/sub_data/head_direction/T10/T10_0725.csv" ^
    --smooth-window 7 ^
    --ts-start 1690271837 --ts-end 1690271881 ^
    --drop-unknown

Optional: score raw preds instead of smoothed:
  --pred-column pred_behavior
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from pipeline_utils import load_behavior_labels
from predict_core import predict_from_immu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate behavior predictions vs label CSV.")
    p.add_argument("--immu-file", type=Path, required=True)
    p.add_argument("--label-file", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--use-multimodal", action="store_true")
    p.add_argument("--head-file", type=Path, default=None)
    p.add_argument("--smooth-window", type=int, default=0)
    p.add_argument(
        "--behavior-map",
        type=Path,
        default=None,
        help="Optional JSON for printing names in confusion matrix labels.",
    )
    p.add_argument(
        "--ts-start",
        type=int,
        default=None,
        help="Only evaluate seconds >= this (inclusive), after merge with labels.",
    )
    p.add_argument(
        "--ts-end",
        type=int,
        default=None,
        help="Only evaluate seconds <= this (inclusive).",
    )
    p.add_argument(
        "--drop-unknown",
        action="store_true",
        help="Exclude rows where true behavior == 0 (Unknown).",
    )
    p.add_argument(
        "--pred-column",
        type=str,
        default="pred_behavior_smooth",
        choices=("pred_behavior", "pred_behavior_smooth"),
        help="Which prediction column to score (use pred_behavior if no smoothing).",
    )
    p.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional: save merged ts_sec, behavior, pred_* for dashboard / Excel.",
    )
    return p.parse_args()


def load_behavior_map(path: Path | None) -> dict[int, str] | None:
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in raw.items()}


def main() -> None:
    args = parse_args()
    bmap = load_behavior_map(args.behavior_map)

    labels_df = load_behavior_labels(args.label_file)

    pred_df = predict_from_immu(
        model_dir=args.model_dir,
        immu_file=args.immu_file,
        immu_df=None,
        use_multimodal=args.use_multimodal,
        head_file=args.head_file,
        smooth_window=args.smooth_window,
        behavior_map=bmap,
    )

    if args.pred_column == "pred_behavior_smooth" and "pred_behavior_smooth" not in pred_df.columns:
        raise SystemExit(
            "No pred_behavior_smooth column. Use --smooth-window > 1 or --pred-column pred_behavior"
        )

    merged = pred_df.merge(labels_df, on="ts_sec", how="inner", suffixes=("", "_true"))
    if args.ts_start is not None:
        merged = merged[merged["ts_sec"] >= args.ts_start]
    if args.ts_end is not None:
        merged = merged[merged["ts_sec"] <= args.ts_end]

    if args.drop_unknown:
        merged = merged[merged["behavior"] != 0].copy()

    if len(merged) == 0:
        raise SystemExit("No rows after merge / filters. Check paths and ts range.")

    y_true = merged["behavior"].astype("int64").values
    y_pred = merged[args.pred_column].astype("int64").values

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Rows evaluated: {len(merged)}")
    print(f"Prediction column: {args.pred_column}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 macro:   {f1_macro:.4f}")
    print(f"F1 weighted:{f1_weighted:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    header = [str(bmap.get(i, i)) if bmap else str(i) for i in labels]
    print("labels:", labels)
    if bmap:
        print("(" + ", ".join(f"{i}={header[j]}" for j, i in enumerate(labels)) + ")")
    print(cm)

    if args.export_csv:
        out = merged[["ts_sec", "behavior", args.pred_column]].copy()
        out = out.rename(columns={"behavior": "true_behavior", args.pred_column: "pred_behavior_eval"})
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.export_csv, index=False)
        print(f"\nSaved: {args.export_csv}")


if __name__ == "__main__":
    main()
