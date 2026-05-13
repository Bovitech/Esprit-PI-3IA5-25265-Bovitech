"""
data_pipeline.py — Cow Stress Detection Data Pipeline
======================================================
Matches your exact folder structure:

    data/
    └── visual_data/sensor_data/
        ├── main_data/
        │   ├── ankle/
        │   │   ├── C01/               ← one subfolder per cow
        │   │   │   ├── C01_0721.csv   ← one CSV per day (concatenated)
        │   │   │   ├── C01_0722.csv
        │   │   │   └── ...
        │   │   ├── C02/
        │   │   └── ...
        │   └── thi/
        │       └── average.csv        ← single shared file
        └── sub_data/
            └── neck_dev_temp/
                ├── T01.csv            ← one CSV per cow
                ├── T02.csv
                └── ...

Cow ID mapping:
    T01 (neck) ↔ C01 (ankle/lying)
    T02 (neck) ↔ C02 (ankle/lying)
    ...

Usage (standalone check):
    python data_pipeline.py

Usage (from train.py):
    from data_pipeline import build_pipeline
    datasets, num_cows, class_weights = build_pipeline()
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — only edit this section
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_data_dir() -> str:
    """
    Find repo `data/` from this file's location (not the process cwd).
    train.py is often run from `runs/stress_sensor/`, where relative `data/` is wrong.
    """
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        candidate = p / "data" / "visual_data" / "sensor_data"
        if candidate.is_dir():
            return str(p / "data")
    return "data"


DATA_DIR    = _resolve_data_dir()
SENSOR_ROOT = os.path.join(DATA_DIR, "visual_data", "sensor_data")

THI_FILE  = os.path.join(SENSOR_ROOT, "main_data", "thi",          "average.csv")
ANKLE_DIR = os.path.join(SENSOR_ROOT, "main_data", "ankle")         # subfolders: C01/, C02/...
NECK_DIR  = os.path.join(SENSOR_ROOT, "sub_data",  "neck_dev_temp") # flat: T01.csv, T02.csv...

# ── Stress thresholds (fill in your values) ───────────────────────────────
THI_AT_RISK   = 72.0    # THI >= this → at risk
THI_STRESSED  = 78.0    # THI >= this → stressed

NECK_AT_RISK  = 38.5    # body temp >= this → at risk
NECK_STRESSED = 39.5    # body temp >= this → stressed

LYING_MIN     = 8       # hours/day — below this → abnormal (too little rest)
LYING_MAX     = 14      # hours/day — above this → abnormal (too much lying)

# ── Windowing ────────────────────────────────────────────────────────────────
WINDOW  = 60    # minutes of past data the model sees
HORIZON = 120   # minutes ahead to predict (2 hours)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cow_number(name: str) -> str:
    """'T01' or 'C01' → '01'  (used to match neck ↔ ankle)"""
    return name.lstrip("TCc").lstrip("0") or "0"


def build_cow_map() -> dict:
    """
    Scans NECK_DIR for T*.csv and ANKLE_DIR for C*/ subfolders,
    matches them by number, returns a dict keyed by cow number.
    """
    mapping = {}

    for nf in sorted(glob.glob(os.path.join(NECK_DIR, "*.csv"))):
        name = os.path.basename(nf).replace(".csv", "")
        key  = cow_number(name)
        mapping.setdefault(key, {})
        mapping[key]["neck"]      = nf
        mapping[key]["neck_name"] = name

    for folder in sorted(glob.glob(os.path.join(ANKLE_DIR, "C*"))):
        if os.path.isdir(folder):
            name = os.path.basename(folder)
            key  = cow_number(name)
            mapping.setdefault(key, {})
            mapping[key]["ankle_dir"]  = folder
            mapping[key]["ankle_name"] = name

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_thi(path: str) -> pd.DataFrame:
    """Loads average.csv → timestamp, THI (already 1-min resolution)."""
    df = pd.read_csv(path)
    df = df[["timestamp", "THI"]].sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype(int)
    print(f"  [THI]   {len(df):>6} rows  |  "
          f"{pd.to_datetime(df.timestamp.min(), unit='s').strftime('%Y-%m-%d')} -> "
          f"{pd.to_datetime(df.timestamp.max(), unit='s').strftime('%Y-%m-%d')}")
    return df


def load_neck(path: str) -> pd.DataFrame:
    """Loads T01.csv → timestamp, neck_temp (resampled to 1-min mean)."""
    df = pd.read_csv(path)
    df = df[["timestamp", "temperature_C"]].sort_values("timestamp")
    df["timestamp"] = df["timestamp"].astype(int)
    df["minute"] = (df["timestamp"] // 60) * 60
    df = (df.groupby("minute")["temperature_C"]
            .mean().reset_index()
            .rename(columns={"minute": "timestamp", "temperature_C": "neck_temp"}))
    name = os.path.basename(path).replace(".csv", "")
    print(f"  [neck/{name}]  {len(df):>6} rows  |  "
          f"mean={df.neck_temp.mean():.2f}C  "
          f"min={df.neck_temp.min():.2f}  max={df.neck_temp.max():.2f}")
    return df


def load_lying(ankle_dir: str) -> pd.DataFrame:
    """
    Reads ALL daily CSVs in a cow's ankle folder (e.g. C01/)
    and concatenates them into one continuous time series.
    """
    daily_files = sorted(glob.glob(os.path.join(ankle_dir, "*.csv")))
    if not daily_files:
        raise FileNotFoundError(f"No CSV files in {ankle_dir}")

    frames = []
    for f in daily_files:
        try:
            day_df = pd.read_csv(f)
            if "lying" in day_df.columns and "timestamp" in day_df.columns:
                frames.append(day_df[["timestamp", "lying"]])
        except Exception as e:
            print(f"    Could not read {os.path.basename(f)}: {e}")

    if not frames:
        raise RuntimeError(f"No valid lying data in {ankle_dir}")

    df = (pd.concat(frames, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates(subset="timestamp")
            .reset_index(drop=True))
    df["timestamp"] = df["timestamp"].astype(int)

    cow  = os.path.basename(ankle_dir)
    print(f"  [ankle/{cow}] {len(df):>6} rows  |  "
          f"{len(daily_files)} day(s)  |  "
          f"lying {df.lying.mean()*100:.1f}% of time")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge_streams(thi_df, neck_df, lying_df) -> pd.DataFrame:
    """Inner join on timestamp — keeps only rows where all 3 sensors overlap."""
    df = thi_df.merge(neck_df,  on="timestamp", how="inner")
    df = df.merge(lying_df, on="timestamp", how="inner")
    return df.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    neck_slope  : rate of change of body temp (detects exponential rise)
    lying_hours : rolling 8h lying total
    thi_trend   : 30-min smoothed THI
    """
    df = df.copy()
    df["neck_slope"]  = df["neck_temp"].diff().fillna(0)
    df["lying_hours"] = df["lying"].rolling(window=480, min_periods=1).sum() / 60.0
    df["thi_trend"]   = df["THI"].rolling(window=30, min_periods=1).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_labels(df: pd.DataFrame) -> np.ndarray:
    """
    3-rule scoring system:
      Rule 1  THI trend    >= THI_STRESSED  +2  |  >= THI_AT_RISK  +1
      Rule 2  Lying hours  < MIN or > MAX        +1
      Rule 3  Neck temp    >= NECK_STRESSED +2  |  >= NECK_AT_RISK +1
              + slope > 0.05 C/min              +1 (exponential rise bonus)

      Score >= 4  ->  2 (Stressed)
      Score >= 2  ->  1 (At-Risk)
      else        ->  0 (Normal)
    """
    labels     = np.zeros(len(df), dtype=np.int64)
    thi_trend  = df["thi_trend"].values
    neck       = df["neck_temp"].values
    neck_slope = df["neck_slope"].values
    lying_h    = df["lying_hours"].values

    for i in range(len(df)):
        score = 0

        t = thi_trend[i]
        if   t >= THI_STRESSED: score += 2
        elif t >= THI_AT_RISK:  score += 1

        lh = lying_h[i]
        if lh < LYING_MIN or lh > LYING_MAX:
            score += 1

        nt = neck[i]
        if   nt >= NECK_STRESSED: score += 2
        elif nt >= NECK_AT_RISK:  score += 1

        if neck_slope[i] > 0.05:
            score += 1

        if   score >= 4: labels[i] = 2
        elif score >= 2: labels[i] = 1

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray) -> np.ndarray:
    mean, std = arr.mean(), arr.std()
    return (arr - mean) / std if std > 1e-6 else arr - mean


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CowStressDataset(Dataset):
    """
    Sliding window: input = WINDOW minutes of sensors,
    label = stress class HORIZON minutes after window ends (2h ahead).
    """
    def __init__(self, thi_arr, neck_arr, lying_arr, labels_arr,
                 cow_id: int, window: int = WINDOW, horizon: int = HORIZON):
        self.thi    = torch.FloatTensor(thi_arr).unsqueeze(-1)
        self.neck   = torch.FloatTensor(neck_arr).unsqueeze(-1)
        self.lying  = torch.FloatTensor(lying_arr).unsqueeze(-1)
        self.labels = torch.LongTensor(labels_arr)
        self.cow_id = cow_id
        self.window  = window
        self.horizon = horizon
        self.length  = len(thi_arr) - window - horizon

    def __len__(self):
        return max(0, self.length)

    def __getitem__(self, idx):
        s, e = idx, idx + self.window
        return (
            self.thi[s:e],
            self.neck[s:e],
            self.lying[s:e],
            torch.tensor(self.cow_id),
            self.labels[e + self.horizon - 1]
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    """
    Discovers all cows, loads and merges their data, generates labels,
    and returns ready-to-use PyTorch datasets.

    Returns:
        datasets      : list of CowStressDataset (one per cow)
        num_cows      : int
        class_weights : torch.Tensor shape (3,)
    """
    print("\n" + "=" * 55)
    print("  DATA PIPELINE")
    print("=" * 55)

    print("\n[1/4] Loading THI (shared across all cows)...")
    thi_df = load_thi(THI_FILE)

    print("\n[2/4] Discovering and matching cows...")
    cow_map  = build_cow_map()
    matched  = {k: v for k, v in cow_map.items() if "neck" in v and "ankle_dir" in v}
    skipped  = {k: v for k, v in cow_map.items() if "neck" not in v or "ankle_dir" not in v}

    pairs = [f"{v.get('neck_name','?')} <-> {v.get('ankle_name','?')}" for v in matched.values()]
    print(f"  Matched cows ({len(matched)}): {pairs}")
    for k, v in skipped.items():
        missing = "neck CSV" if "neck" not in v else "ankle folder"
        print(f"  Skipping cow {k}: missing {missing}")

    print(f"\n[3/4] Loading sensor data...")
    datasets   = []
    cow_index  = 0
    all_labels = []

    for key, info in sorted(matched.items()):
        print(f"\n  --- Cow {info['neck_name']} / {info['ankle_name']} ---")
        neck_df  = load_neck(info["neck"])
        lying_df = load_lying(info["ankle_dir"])
        merged   = merge_streams(thi_df, neck_df, lying_df)

        if len(merged) < WINDOW + HORIZON + 10:
            print(f"  Only {len(merged)} rows after merge — skipping")
            continue

        merged = add_features(merged)
        labels = generate_labels(merged)
        all_labels.extend(labels.tolist())

        ds = CowStressDataset(
            thi_arr   = normalize(merged["THI"].values),
            neck_arr  = normalize(merged["neck_temp"].values),
            lying_arr = normalize(merged["lying"].values),
            labels_arr= labels,
            cow_id    = cow_index
        )

        dist = dict(sorted(Counter(labels).items()))
        print(f"  Result: {len(merged)} rows -> {len(ds)} samples | labels: {dist}")
        datasets.append(ds)
        cow_index += 1

    if not datasets:
        raise RuntimeError(
            "No datasets built. Check paths:\n"
            f"  THI   : {THI_FILE}\n"
            f"  Neck  : {NECK_DIR}\n"
            f"  Ankle : {ANKLE_DIR}"
        )

    print(f"\n[4/4] Class distribution & weights...")
    counts = Counter(all_labels)
    total  = len(all_labels)
    names  = {0: "Normal", 1: "At-Risk", 2: "Stressed"}
    for i in range(3):
        n = counts.get(i, 0)
        print(f"  Class {i} ({names[i]:<8}): {n:>6} samples ({n/total*100:.1f}%)")

    weights = [total / (3 * max(counts.get(i, 1), 1)) for i in range(3)]
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"  Weights: Normal={weights[0]:.3f} | At-Risk={weights[1]:.3f} | Stressed={weights[2]:.3f}")

    print(f"\n{'='*55}")
    print(f"  Total cows    : {cow_index}")
    print(f"  Total samples : {sum(len(d) for d in datasets)}")
    print(f"{'='*55}\n")

    return datasets, cow_index, class_weights


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets, num_cows, class_weights = build_pipeline()

    from torch.utils.data import DataLoader, ConcatDataset
    loader = DataLoader(ConcatDataset(datasets), batch_size=4, shuffle=True)
    thi, neck, lying, cow_id, label = next(iter(loader))

    print("Sample batch shapes:")
    print(f"  THI    : {thi.shape}   (batch, window, 1)")
    print(f"  Neck   : {neck.shape}")
    print(f"  Lying  : {lying.shape}")
    print(f"  Cow ID : {cow_id}")
    print(f"  Labels : {label}  (0=Normal, 1=At-Risk, 2=Stressed)")
    print(f"\nPipeline is working correctly!")
