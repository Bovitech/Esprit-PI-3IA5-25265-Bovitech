"""
Shared helpers for the behavior project.

We mostly: load IMMU CSVs, squash them to 1 row per second, load behavior labels,
and match files on disk (one cow + one day = one pair of files).

UWB helpers are still here for older experiments; the current train/predict scripts
do not use them (multimodal = IMU + head only).
"""

# --- Python version hint for type hints across the file ---
from __future__ import annotations

# --- Stdlib: typed containers, paths, dataclass for small structs ---
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --- Third party: arrays + tables ---
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Immu column names the rest of the pipeline assumes exist (after CSV read).
# -----------------------------------------------------------------------------
# Raw IMU rows must at least have time + accel; mag is optional but nice for extra features.
REQUIRED_IMMU_COLUMNS = [
    "timestamp",
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",
]


# -----------------------------------------------------------------------------
# Small struct: remembers which files belong to one recording session.
# -----------------------------------------------------------------------------
@dataclass
class PairSpec:
    """Links one cow/day to its IMU file + label file paths."""

    cow_id: str  # e.g. C01
    tag_id: str  # e.g. T01 (device id folder under main_data/immu)
    date_code: str  # e.g. 0725
    immu_file: Path
    label_file: Path


# -----------------------------------------------------------------------------
# Name mapping used by folder convention (C01 uses tag device folder T01).
# -----------------------------------------------------------------------------
def cow_to_tag(cow_id: str) -> str:
    """Cow C01 -> tag folder T01. Kept dumb on purpose so it matches the dataset layout."""
    # Reject weird ids early so we don't build wrong paths silently.
    if not cow_id.startswith("C"):
        raise ValueError(f"Invalid cow_id format: {cow_id}")
    # Strip the C and zero-pad so C1 -> T01 style matches file names.
    num = int(cow_id[1:])
    return f"T{num:02d}"


# -----------------------------------------------------------------------------
# IMMU cleaning: same steps for CSV load or in-memory DataFrame.
# -----------------------------------------------------------------------------
def preprocess_immu_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean IMU rows and add helper columns used later (ts_sec, vector magnitudes).

    This is the same path whether we read from CSV or already have a DataFrame in memory.
    """
    # Fail fast if columns are wrong (typos in CSV header, etc.).
    missing = [c for c in REQUIRED_IMMU_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"IMMU dataframe missing columns: {missing}")

    df = df.copy()
    # Force numeric so strings don't break groupby/agg later.
    for col in REQUIRED_IMMU_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional magnetometer: same numeric coercion if present.
    mag_cols = ["mag_x_uT", "mag_y_uT", "mag_z_uT"]
    if all(c in df.columns for c in mag_cols):
        for col in mag_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop junk rows where core signals are missing after coercion.
    df = df.dropna(subset=["timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]).copy()
    # Floor so all sub-second samples in the same second share one label second.
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")

    # Whole-vector length of acceleration (rotation-invariant-ish intensity).
    df["accel_mag"] = np.sqrt(
        df["accel_x_mps2"] ** 2 + df["accel_y_mps2"] ** 2 + df["accel_z_mps2"] ** 2
    )
    # Same idea for magnetometer magnitude if we have mag columns.
    if all(c in df.columns for c in mag_cols):
        df["mag_mag"] = np.sqrt(df["mag_x_uT"] ** 2 + df["mag_y_uT"] ** 2 + df["mag_z_uT"] ** 2)

    return df


def load_immu_csv(path: Path) -> pd.DataFrame:
    """Read a raw IMMU CSV and run the same preprocessing."""
    return preprocess_immu_dataframe(pd.read_csv(path))


# -----------------------------------------------------------------------------
# Per-second feature table from high-rate IMU.
# -----------------------------------------------------------------------------
def aggregate_immu_per_second(df: pd.DataFrame, include_mag: bool = True) -> pd.DataFrame:
    """
    Many IMU rows per second -> one row per second with simple stats.

    We use mean/std/min/max/median so the model sees both level and variability.
    """
    # Dict passed to pandas .agg(): which stats per sensor column.
    agg_dict: Dict[str, List[str]] = {
        "accel_x_mps2": ["mean", "std", "min", "max", "median"],
        "accel_y_mps2": ["mean", "std", "min", "max", "median"],
        "accel_z_mps2": ["mean", "std", "min", "max", "median"],
        "accel_mag": ["mean", "std", "min", "max", "median"],
    }
    # Only add mag features if we computed mag_mag earlier and caller wants them.
    if include_mag and "mag_mag" in df.columns:
        agg_dict["mag_mag"] = ["mean", "std", "min", "max", "median"]

    # One group = one second key.
    feat = df.groupby("ts_sec").agg(agg_dict)
    # Flatten MultiIndex columns to names like accel_x_mps2_mean.
    feat.columns = ["_".join(col) for col in feat.columns]
    feat = feat.reset_index()

    # Extra signal: did we actually have data that second or was it empty?
    sample_count = df.groupby("ts_sec").size().reset_index(name="samples_per_sec")
    feat = feat.merge(sample_count, on="ts_sec", how="left")

    return feat.fillna(0.0)


# -----------------------------------------------------------------------------
# Label filtering.
# -----------------------------------------------------------------------------
def drop_unknown_behavior(
    df: pd.DataFrame,
    label_col: str = "behavior",
    unknown_value: int = 0,
) -> pd.DataFrame:
    """Label 0 = unknown in our setup; we usually drop it for supervised learning."""
    # If no label column, just return (e.g. prediction-time frame).
    if label_col not in df.columns:
        return df
    out = df[df[label_col] != unknown_value].copy()
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Load expert / ground-truth behavior id per second.
# -----------------------------------------------------------------------------
def load_behavior_labels(path: Path) -> pd.DataFrame:
    """Load 1 Hz behavior ids aligned by timestamp (we convert to ts_sec)."""
    df = pd.read_csv(path)
    required = ["timestamp", "behavior"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing label columns: {missing}")

    # Numeric cleaning like IMU path.
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["behavior"] = pd.to_numeric(df["behavior"], errors="coerce")
    df = df.dropna(subset=["timestamp", "behavior"]).copy()
    # Labels are already per integer second in our dataset; still use same key name.
    df["ts_sec"] = df["timestamp"].astype("int64")
    df["behavior"] = df["behavior"].astype("int64")
    # One row per second to avoid duplicate keys on merge.
    return df[["ts_sec", "behavior"]].drop_duplicates(subset=["ts_sec"])


# -----------------------------------------------------------------------------
# IMU-only dataset: file path entry points.
# -----------------------------------------------------------------------------
def build_second_level_dataset(
    immu_file: Path,
    label_file: Optional[Path] = None,
    include_mag: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """IMMU-only path: aggregate IMU then optionally join labels."""
    imu = load_immu_csv(immu_file)
    return _build_second_level_from_imu(imu, label_file=label_file, include_mag=include_mag)


def build_second_level_dataset_from_dataframe(
    immu_df: pd.DataFrame,
    label_file: Optional[Path] = None,
    include_mag: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Same as build_second_level_dataset but you already have the IMU in a DataFrame."""
    imu = preprocess_immu_dataframe(immu_df)
    return _build_second_level_from_imu(imu, label_file=label_file, include_mag=include_mag)


def _build_second_level_from_imu(
    imu: pd.DataFrame,
    label_file: Optional[Path] = None,
    include_mag: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    # Step 1: second-level features from raw-ish IMU rows.
    feat = aggregate_immu_per_second(imu, include_mag=include_mag)
    feature_cols = [c for c in feat.columns if c != "ts_sec"]

    # No labels = return features only (rare; mostly for debugging).
    if label_file is None:
        return feat, feature_cols

    # Step 2: inner join keeps only seconds where we have both IMU summary and label.
    labels = load_behavior_labels(label_file)
    dataset = feat.merge(labels, on="ts_sec", how="inner").sort_values("ts_sec").reset_index(drop=True)
    return dataset, feature_cols


# -----------------------------------------------------------------------------
# Discover (immu_path, label_path) pairs on disk for training scripts.
# -----------------------------------------------------------------------------
def discover_pairs(sensor_root: Path, cows: Optional[Iterable[str]] = None) -> List[PairSpec]:
    """
    Walk the dataset folders and list (immu, label) pairs.

    Expected layout:
      sensor_root/main_data/immu/Txx/Txx_MMDD.csv
      sensor_root/behavior_labels/individual/Cxx_MMDD.csv
    """
    label_dir = sensor_root / "behavior_labels" / "individual"
    immu_dir = sensor_root / "main_data" / "immu"

    if not label_dir.exists() or not immu_dir.exists():
        raise FileNotFoundError(
            f"Expected folders not found under {sensor_root}. "
            "Need behavior_labels/individual and main_data/immu."
        )

    # If cows is set, only keep those ids (case-insensitive).
    allow = {c.upper() for c in cows} if cows else None
    pairs: List[PairSpec] = []

    # Every label file name encodes cow + date; we infer matching IMU path.
    for label_file in sorted(label_dir.glob("C*_*.csv")):
        stem = label_file.stem  # C01_0725
        cow_id, date_code = stem.split("_", maxsplit=1)
        cow_id = cow_id.upper()
        if allow and cow_id not in allow:
            continue
        tag_id = cow_to_tag(cow_id)
        immu_file = immu_dir / tag_id / f"{tag_id}_{date_code}.csv"
        # Only register pair if IMU side actually exists.
        if immu_file.exists():
            pairs.append(
                PairSpec(
                    cow_id=cow_id,
                    tag_id=tag_id,
                    date_code=date_code,
                    immu_file=immu_file,
                    label_file=label_file,
                )
            )
    return pairs


# -----------------------------------------------------------------------------
# Legacy UWB (ultra-wideband) helpers — unused by current behavior RF pipeline.
# -----------------------------------------------------------------------------
def load_uwb_data(path: Path) -> pd.DataFrame:
    """Legacy UWB loader (not used by current behavior RandomForest)."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"UWB file {path} is missing 'timestamp' column")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("float64")

    # Prefer obvious x/y/z style names; else fall back to any numeric columns.
    possible_cols = [c for c in df.columns if c.lower() in {"x","y","z","lat","lon","pos_x","pos_y","pos_z"}]
    if not possible_cols:
        possible_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]

    out = df[["timestamp"] + possible_cols].copy()
    return out


def process_uwb(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy: resample UWB-ish positions to one row per second."""
    if "timestamp" not in df.columns:
        raise ValueError("UWB DataFrame missing 'timestamp'")

    df = df.dropna(subset=["timestamp"]).copy()
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")

    value_cols = [c for c in df.columns if c not in {"timestamp", "ts_sec"}]
    if not value_cols:
        raise ValueError("UWB DataFrame has no position columns after timestamp")

    # Mean position within each second.
    agg = df.groupby("ts_sec")[value_cols].mean()

    # Fill gaps in the second index so timelines are contiguous.
    full_index = pd.RangeIndex(agg.index.min(), agg.index.max() + 1)
    agg = agg.reindex(full_index).ffill().bfill()
    agg = agg.reset_index().rename(columns={"index": "ts_sec"})

    # Optional planar speed from consecutive XY samples.
    if "x" in agg.columns and "y" in agg.columns:
        dx = agg["x"].diff().fillna(0.0)
        dy = agg["y"].diff().fillna(0.0)
        agg["uwb_speed"] = np.sqrt(dx ** 2 + dy ** 2)

    return agg


# -----------------------------------------------------------------------------
# Real head_direction CSV branch (alternative to synthesized head features).
# -----------------------------------------------------------------------------
def load_head_data(path: Path) -> pd.DataFrame:
    """Read a producer head_direction-style CSV (has timestamps + angle columns)."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Head file {path} missing 'timestamp' column")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("float64")

    return df


def aggregate_head(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse head traces to one mean row per second (same idea as IMU aggregation)."""
    if "timestamp" not in df.columns:
        raise ValueError("Head DataFrame missing 'timestamp'")

    df = df.dropna(subset=["timestamp"]).copy()
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")

    # Anything that isn't time becomes a signal to average.
    value_cols = [c for c in df.columns if c not in {"timestamp", "ts_sec"}]
    if not value_cols:
        raise ValueError("Head DataFrame has no signal columns")

    agg = df.groupby("ts_sec")[value_cols].mean().reset_index()
    return agg
