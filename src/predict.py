import os
import pickle
import numpy as np
import torch

from src.load_data import load_one_cow
from src.preprocess import preprocess_uwb
from src.dataset import create_sequences
from src.model_lstm import CowTrajectoryLSTM
from src.config import BASE_DIR


def load_model_and_scaler():
    model_path = os.path.join(BASE_DIR, "models", "lstm_t01.pth")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler_t01.pkl")

    model = CowTrajectoryLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def inverse_scale(scaler, arr_2d):
    # arr_2d shape: (n, 2)
    return scaler.inverse_transform(arr_2d)


def run_prediction():
    tag_id = "T01"
    seq_length = 10

    # Load and preprocess
    raw_df, _ = load_one_cow(tag_id)
    clean_df = preprocess_uwb(raw_df, tag_id)

    # Create sequences WITH the same scaler (fit on full data once)
    X, y, scaler = create_sequences(clean_df, seq_length=seq_length)

    # Load trained model + saved scaler
    model, saved_scaler = load_model_and_scaler()

    # Take last sequence
    last_seq = X[-1]  # shape (10, 2)
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # (1, 10, 2)

    with torch.no_grad():
        pred_scaled = model(last_seq_tensor).numpy()  # (1, 2)

    # Inverse scaling to cm
    pred_cm = inverse_scale(saved_scaler, pred_scaled)[0]

    # Ground truth (last known real next point)
    true_scaled = y[-1].reshape(1, -1)
    true_cm = inverse_scale(saved_scaler, true_scaled)[0]

    # Euclidean error (cm)
    error = np.sqrt((pred_cm[0] - true_cm[0])**2 + (pred_cm[1] - true_cm[1])**2)

    print("\nPredicted next position (cm):", pred_cm)
    print("Real next position (cm):     ", true_cm)
    print(f"Error distance: {error:.2f} cm")


if __name__ == "__main__":
    run_prediction()