"""
illness_labels.py
-----------------
Ground-truth label extraction from INRAE dataset CSV.

Key fixes:
    1. oestrus/calving removed from At-risk — they are normal reproductive events
    2. Reward function rebalanced — much stronger penalty for missing Ill
    3. load_inrae accepts a list of paths or a single path (all INRAE datasets)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_inrae(path) -> pd.DataFrame:
    """Load one or more INRAE CSV files and return a combined DataFrame."""
    paths = [path] if isinstance(path, str) else list(path)
    dfs = []
    for p in paths:
        print(f"Loading {p}...")
        df = pd.read_csv(
            p,
            dtype={
                "cow": str,
                "date": str,
                "hour": np.int16,
                "IN_ALLEYS": np.float32,
                "REST": np.float32,
                "EAT": np.float32,
                "ACTIVITY_LEVEL": np.float32,
                "oestrus": np.float32,
                "calving": np.float32,
                "lameness": np.float32,
                "mastitis": np.float32,
                "LPS": np.float32,
                "acidosis": np.float32,
                "other_disease": np.float32,
                "accidents": np.float32,
                "disturbance": np.float32,
                "mixing": np.float32,
                "management_changes": np.float32,
                "OK": np.float32,
            },
            low_memory=False,
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(combined):,} rows, {combined['cow'].nunique()} cows")

    # Fill NAs in disease columns with 0
    disease_cols = [
        "oestrus", "calving", "lameness", "mastitis",
        "LPS", "acidosis", "other_disease", "accidents",
        "disturbance", "mixing", "management_changes", "OK",
    ]
    for col in disease_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0.0)

    # Pre-compute action_gt column once (vectorised — fast)
    combined = _add_action_gt(combined)

    combined = combined.sort_values(["cow", "date", "hour"]).reset_index(drop=True)
    print("INRAE data ready.")
    return combined


def _add_action_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised label assignment — O(n) instead of per-row Python calls."""
    severe = (
        (df.get("lameness", 0) > 0) |
        (df.get("mastitis", 0) > 0) |
        (df.get("LPS", 0) > 0) |
        (df.get("acidosis", 0) > 0) |
        (df.get("other_disease", 0) > 0)
    )
    moderate = (
        (df.get("accidents", 0) > 0) |
        (df.get("disturbance", 0) > 0)
    )
    ok = df["OK"] == 1

    # Default healthy
    action_gt = np.zeros(len(df), dtype=np.int8)
    # At-risk
    action_gt[moderate & ~severe] = 1
    # OK=0 but no specific flag → at-risk
    action_gt[~ok & ~severe & ~moderate] = 1
    # Ill
    action_gt[severe] = 2

    df = df.copy()
    df["action_gt"] = action_gt
    return df


# ---------------------------------------------------------------------------
# Episode index (lazy — no dicts in RAM)
# ---------------------------------------------------------------------------

def build_episode_dataset(inrae_df: pd.DataFrame, window_hours: int = 24):
    print("Building episode index...")
    episode_index = []

    # groupby returns (cow, date) groups — just store start/end row indices
    for (cow, date), group in inrae_df.groupby(["cow", "date"], sort=False):
        if len(group) == 0:
            continue
        episode_index.append({
            "cow": cow,
            "date": date,
            "start_idx": int(group.index[0]),
            "end_idx": int(group.index[-1]),
            "n_hours": len(group),
        })

    print(f"Indexed {len(episode_index):,} episodes (lazy, no dicts in RAM)")
    return episode_index


# ---------------------------------------------------------------------------
# Per-row label (used during step — O(1), reads pre-computed action_gt col)
# ---------------------------------------------------------------------------

def get_label_from_row(row) -> dict:
    """
    Fast label extraction from a single DataFrame row.
    Reads the pre-computed action_gt column — no conditionals.
    """
    action_gt = int(row.get("action_gt", 0))
    disease_flags = {
        k: int(row.get(k, 0))
        for k in ["lameness", "mastitis", "LPS", "acidosis",
                  "other_disease", "oestrus", "calving"]
    }
    return {
        "action_gt": action_gt,
        "disease_flags": disease_flags,
        "ok": bool(row.get("OK", 1)),
    }


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def build_reward(
    predicted_action: int,
    gt_action: int,
    early_bonus: bool = False,
    miss_streak: int = 0,
) -> float:
    if predicted_action == gt_action:
        base = {0: 1.0, 1: 2.5, 2: 8.0}[gt_action]
        bonus = 1.5 if early_bonus and gt_action in (1, 2) else 0.0
        return base + bonus

    # Missed illness — most costly, with an escalating penalty if the cow-day
    # keeps being missed across steps.
    if gt_action == 2 and predicted_action in (0, 1):
        base_penalty = -12.0 if predicted_action == 0 else -6.0
        return base_penalty - 1.5 * max(0, int(miss_streak) - 1)
    # False alarm
    if predicted_action == 2 and gt_action == 0:
        return -2.5
    # Wrong level
    return -0.5


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("data/inrae_activity/*.csv")
    df = load_inrae(paths)
    episodes = build_episode_dataset(df)

    print("\nLabel distribution:")
    counts = df["action_gt"].value_counts().sort_index()
    for cls, cnt in counts.items():
        pct = cnt / len(df) * 100
        names = {0: "Healthy", 1: "At-risk", 2: "Ill"}
        print(f"  Class {cls} ({names[cls]}): {cnt:>10,}  ({pct:.1f}%)")

    print(f"\nReward tests:")
    print(f"  Correct ill:    {build_reward(2,2)}")
    print(f"  Missed ill:     {build_reward(0,2)}")
    print(f"  False alarm:    {build_reward(2,0)}")
    print(f"  Correct healthy:{build_reward(0,0)}")
