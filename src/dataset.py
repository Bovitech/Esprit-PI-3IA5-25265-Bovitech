import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.preprocess import preprocess_uwb
from src.load_data import load_one_cow


def create_sequences(df, seq_length=10, scaler=None, fit_scaler=True):
    data = df[["x_cm", "y_cm"]].values

    if scaler is None:
        scaler = MinMaxScaler()

    if fit_scaler:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    X = []
    y = []

    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length])

    return np.array(X), np.array(y), scaler


if __name__ == "__main__":
    tag_id = "T01"

    raw_df, _ = load_one_cow(tag_id)
    clean_df = preprocess_uwb(raw_df, tag_id)

    X, y, scaler = create_sequences(clean_df, seq_length=10)

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    print("\nExample input scaled:")
    print(X[0])

    print("\nExample target scaled:")
    print(y[0])

    print("\nScaler min:")
    print(scaler.data_min_)

    print("\nScaler max:")
    print(scaler.data_max_)