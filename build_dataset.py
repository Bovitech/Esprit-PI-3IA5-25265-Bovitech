from __future__ import annotations

"""
One-off export: build a single merged CSV (features + behavior) for one cow/day.

Useful to sanity-check features in Excel or to share a small slice with teammates.
Training itself usually calls pipeline_utils + imu_head_synthesis internally — you
don't have to run this script to train.
"""

# --- Stdlib ---
import argparse
from pathlib import Path

import pandas as pd

from imu_head_synthesis import synthesize_head_aggregate
from pipeline_utils import (
    aggregate_head,
    aggregate_immu_per_second,
    build_second_level_dataset,
    discover_pairs,
    drop_unknown_behavior,
    load_behavior_labels,
    load_head_data,
    load_immu_csv,
)


def parse_args() -> argparse.Namespace:
    """All command-line options for exporting one dataset CSV."""
    parser = argparse.ArgumentParser(description="Build second-level behavior dataset from IMMU + labels.")
    # Root that contains behavior_labels/, main_data/, optional sub_data/.
    parser.add_argument(
        "--sensor-root",
        type=Path,
        default=Path("sensor_data/sensor_data"),
        help="Root folder containing main_data and behavior_labels.",
    )
    parser.add_argument("--cow", type=str, default="C01", help="Cow ID, e.g., C01")
    parser.add_argument("--date", type=str, default="0725", help="Date code, e.g., 0725")
    parser.add_argument("--include-mag", action="store_true", help="Use magnetometer magnitude features.")
    parser.add_argument(
        "--use-multimodal",
        action="store_true",
        help="Merge IMU per-second + head (synth or CSV) + labels. Default without flag is IMU-only.",
    )
    parser.add_argument("--head-file", type=Path, default=None, help="Optional head CSV for this cow/day.")
    parser.add_argument(
        "--use-head-direction-csv",
        action="store_true",
        help="Use sub_data/head_direction/... instead of synthesizing head from IMU.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/datasets/dataset_C01_0725.csv"),
        help="Where to write the merged dataset.",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep behavior==0. Default drops unknown labels like training usually does.",
    )
    return parser.parse_args()


def main() -> None:
    """Build one merged table row-by-row logic lives here."""
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Discover every valid pair then pick the cow+date combo the user asked for.
    all_pairs = discover_pairs(args.sensor_root, cows=[args.cow])
    pair = next((p for p in all_pairs if p.date_code == args.date), None)
    if pair is None:
        raise FileNotFoundError(
            f"No matching pair found for cow={args.cow}, date={args.date}. "
            "Check files under behavior_labels/individual and main_data/immu."
        )

    if not args.use_multimodal:
        # --- Branch A: IMU-only features + labels (fastest, fewer columns) ---
        dataset, feature_cols = build_second_level_dataset(
            immu_file=pair.immu_file,
            label_file=pair.label_file,
            include_mag=args.include_mag,
        )
    else:
        # --- Branch B: IMU per-second + head + labels (matches multimodal training) ---
        imu_raw = load_immu_csv(pair.immu_file)
        immu_df = aggregate_immu_per_second(imu_raw, include_mag=args.include_mag)

        # Head table: explicit file > dataset head_direction > synthesize from IMU.
        if args.head_file is not None:
            head_path = args.head_file
            if not head_path.exists():
                raise FileNotFoundError(f"--head-file not found: {head_path}")
            head_df = aggregate_head(load_head_data(head_path))
        elif args.use_head_direction_csv:
            head_path = args.sensor_root / "sub_data" / "head_direction" / pair.tag_id / f"{pair.tag_id}_{args.date}.csv"
            if not head_path.exists():
                raise FileNotFoundError(f"Head CSV not found: {head_path}")
            head_df = aggregate_head(load_head_data(head_path))
        else:
            head_df = synthesize_head_aggregate(imu_raw)
        labels_df = load_behavior_labels(pair.label_file)

        # Inner join on labels so we only keep labeled seconds.
        dataset = (
            immu_df
            .merge(head_df, on="ts_sec", how="left")
            .merge(labels_df, on="ts_sec", how="inner")
        )

        dataset = dataset.sort_values("ts_sec").reset_index(drop=True)
        # Fill small gaps so we don't feed NaNs into a quick CSV inspection / some models.
        dataset = dataset.ffill().fillna(0)
        feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior"]]

    # --- Optional: drop unknown class 0 like default training does ---
    if not args.keep_unknown:
        before = len(dataset)
        dataset = drop_unknown_behavior(dataset, label_col="behavior", unknown_value=0)
        if before != len(dataset):
            print(f"Dropped behavior==0: {before - len(dataset)} rows")
        if len(dataset) == 0:
            raise ValueError("No rows left after dropping Unknown. Use --keep-unknown.")

    # --- Write CSV + print how many rows per behavior id ---
    dataset.to_csv(args.output_csv, index=False)
    class_dist = dataset["behavior"].value_counts().sort_index()

    print(f"Saved dataset: {args.output_csv}")
    print(f"Rows: {len(dataset)}, Features: {len(feature_cols)}")
    print("Class distribution:")
    print(class_dist.to_string())


if __name__ == "__main__":
    main()
