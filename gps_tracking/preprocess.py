from __future__ import annotations

import pandas as pd


def preprocess_uwb(df: pd.DataFrame, tag_id: str) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(
        columns={
            "coord_x_cm": "x_cm",
            "coord_y_cm": "y_cm",
            "coord_z_cm": "z_cm",
        }
    )
    df["tag_id"] = tag_id.strip().upper()

    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        df["datetime"] = pd.NaT

    cols = [c for c in ("timestamp", "datetime", "tag_id", "x_cm", "y_cm", "z_cm") if c in df.columns]
    df = df[cols]
    df = df.dropna(subset=["x_cm", "y_cm"])
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df
