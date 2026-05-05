"""
Turn per-second behavior CSVs into one row per (cow_id, day).

Why milk code needs this:
  train_milk_xgboost.py can merge behavior features into the milk table.
  Those features can come from hand labels OR from a model that predicts
  behavior every second. This script aggregates seconds → daily stats.

Input CSVs should have something like:
  - timestamp (unix sec) OR ts_sec
  - behavior OR pred_behavior_smooth OR pred_behavior

Filename hack: I assume cow id is the part before the first underscore
(e.g. C01_0725.csv → C01, T01_0721_pred.csv → T01 then mapped to C01).

This is NOT the milk model — it only builds inputs the milk model can use.
(The batch_predict_behavior_all_immu.py file you have is different: that one
calls BOVITECH's predict_behavior.py over IMMU files; this file just summarizes
whatever CSVs you already have in a folder.)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _tag_to_cow_id(tag: str) -> str:
    """IMMU tags are T01, milk files use C01 — quick conversion."""
    m = re.match(r"^T(\d+)$", str(tag).strip(), flags=re.I)
    if m:
        return f"C{int(m.group(1)):02d}"
    return str(tag)


def build_behavior_daily_from_files(
    input_dir: Path,
    *,
    cow_id_from_filename: bool = True,
) -> pd.DataFrame:
    """
    Walk input_dir/*.csv, stack all days, aggregate.

    Returns columns like:
      cow_id, date, behavior_n, behavior_mean, behavior_std,
      behavior_class_0_count, behavior_class_0_ratio, ...
    """
    frames: list[pd.DataFrame] = []
    for p in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(p)

        ts_col = "timestamp" if "timestamp" in df.columns else ("ts_sec" if "ts_sec" in df.columns else None)
        if ts_col is None:
            continue

        beh_col = None
        for c in ("behavior", "pred_behavior_smooth", "pred_behavior"):
            if c in df.columns:
                beh_col = c
                break
        if beh_col is None:
            continue

        out = df[[ts_col, beh_col]].copy()
        out["ts_sec"] = pd.to_numeric(out[ts_col], errors="coerce")
        out["behavior"] = pd.to_numeric(out[beh_col], errors="coerce")
        out = out.dropna(subset=["ts_sec", "behavior"])

        out["datetime"] = pd.to_datetime(out["ts_sec"], unit="s", errors="coerce")
        out["date"] = out["datetime"].dt.floor("D")

        if cow_id_from_filename:
            stem = p.stem.replace("_pred", "")
            cow_id = stem.split("_")[0]
            out["cow_id"] = _tag_to_cow_id(cow_id)
        else:
            if "cow_id" not in df.columns:
                continue
            out["cow_id"] = df["cow_id"]

        frames.append(out[["cow_id", "date", "behavior"]])

    if not frames:
        return pd.DataFrame(
            columns=[
                "cow_id",
                "date",
                "behavior_n",
                "behavior_mean",
                "behavior_std",
            ]
        )

    all_b = pd.concat(frames, ignore_index=True)
    all_b["behavior"] = all_b["behavior"].round().astype(int)

    daily = all_b.groupby(["cow_id", "date"], as_index=False).agg(
        behavior_n=("behavior", "size"),
        behavior_mean=("behavior", "mean"),
        behavior_std=("behavior", "std"),
    )

    pivot = (
        all_b.pivot_table(
            index=["cow_id", "date"],
            columns="behavior",
            values="behavior",
            aggfunc="count",
            fill_value=0,
        ).reset_index()
    )
    class_cols = [c for c in pivot.columns if c not in {"cow_id", "date"}]
    for c in class_cols:
        pivot = pivot.rename(columns={c: f"behavior_class_{int(c)}_count"})
    count_cols = [c for c in pivot.columns if c.endswith("_count")]
    total = pivot[count_cols].sum(axis=1).replace(0, np.nan)
    for c in count_cols:
        pivot[c.replace("_count", "_ratio")] = pivot[c] / total

    return daily.merge(pivot, on=["cow_id", "date"], how="left").sort_values(["cow_id", "date"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build daily behavior feature CSV for milk model merge."
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder full of per-second behavior CSVs",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the daily summary",
    )
    args = ap.parse_args()

    df = build_behavior_daily_from_files(args.input_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"wrote {args.output_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
