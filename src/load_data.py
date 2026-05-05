import os
import pandas as pd
from src.config import UWB_DIR

def load_one_cow(tag_id="T01"):
    cow_folder = os.path.join(UWB_DIR, tag_id)

    files = sorted(os.listdir(cow_folder))
    first_file = files[0]

    path = os.path.join(cow_folder, first_file)

    df = pd.read_csv(path)

    return df, path

if __name__ == "__main__":
    df, path = load_one_cow("T01")

    print(f"\nLoaded file: {path}\n")
    print(df.head())
    print("\nColumns:", df.columns)
    print("\nShape:", df.shape)