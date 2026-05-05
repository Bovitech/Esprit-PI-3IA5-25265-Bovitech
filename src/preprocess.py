import pandas as pd
from src.load_data import load_one_cow


def preprocess_uwb(df: pd.DataFrame, tag_id: str) -> pd.DataFrame:
    df = df.copy()

    # Rename columns for easier use
    df = df.rename(columns={
        "coord_x_cm": "x_cm",
        "coord_y_cm": "y_cm",
        "coord_z_cm": "z_cm"
    })

    # Add tag ID
    df["tag_id"] = tag_id

    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # Keep clean column order
    df = df[["timestamp", "datetime", "tag_id", "x_cm", "y_cm", "z_cm"]]

    # Drop rows with missing coordinates
    df = df.dropna(subset=["x_cm", "y_cm", "z_cm"])

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


if __name__ == "__main__":
    tag_id = "T01"

    raw_df, path = load_one_cow(tag_id)
    clean_df = preprocess_uwb(raw_df, tag_id)

    print(f"\nSource file: {path}")
    print(f"\nCleaned data shape: {clean_df.shape}")

    print("\nColumns:")
    print(clean_df.columns)

    print("\nFirst 5 rows:")
    print(clean_df.head())

    print("\nMissing values:")
    print(clean_df.isna().sum())

    print("\nCoordinate ranges:")
    print(f"x: {clean_df['x_cm'].min()} → {clean_df['x_cm'].max()} cm")
    print(f"y: {clean_df['y_cm'].min()} → {clean_df['y_cm'].max()} cm")
    print(f"z: {clean_df['z_cm'].min()} → {clean_df['z_cm'].max()} cm")