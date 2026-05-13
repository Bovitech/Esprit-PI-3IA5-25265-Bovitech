from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(
    df,
    seq_length: int = 10,
    scaler: MinMaxScaler | None = None,
    fit_scaler: bool = True,
):
    data = df[["x_cm", "y_cm"]].values.astype(np.float64)
    if scaler is None:
        scaler = MinMaxScaler()
    if fit_scaler:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i : i + seq_length])
        y.append(data_scaled[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler


def last_window_scaled(df, seq_length: int, scaler: MinMaxScaler) -> np.ndarray | None:
    """Shape (1, seq_length, 2) scaled with *existing* scaler, or None if too short."""
    data = df[["x_cm", "y_cm"]].values.astype(np.float64)
    if len(data) < seq_length:
        return None
    data_scaled = scaler.transform(data)
    window = data_scaled[-seq_length:].astype(np.float32)
    return window.reshape(1, seq_length, 2)
