import os
import json
import pickle
import torch
import numpy as np

from src.load_data import load_one_cow
from src.preprocess import preprocess_uwb
from src.dataset import create_sequences
from src.model_lstm import CowTrajectoryLSTM
from src.config import BASE_DIR


def load_model_and_scaler():
    model_path = os.path.join(BASE_DIR, "models", "lstm_all.pth")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler_all.pkl")

    model = CowTrajectoryLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def export_predictions():
    tag_ids = ["T01", "T02", "T03", "T04", "T05","T06", "T07", "T08", "T09", "T10","T13", "T14"]  
    seq_length = 10

    model, scaler = load_model_and_scaler()

    all_records = []

    for tag_id in tag_ids:
        try:
            raw_df, _ = load_one_cow(tag_id)
            clean_df = preprocess_uwb(raw_df, tag_id)

            X, y, _ = create_sequences(clean_df, seq_length=seq_length)

            X_tensor = torch.tensor(X, dtype=torch.float32)

            with torch.no_grad():
                pred_scaled = model(X_tensor).numpy()

            pred_cm = scaler.inverse_transform(pred_scaled)
            true_cm = scaler.inverse_transform(y)

            for i in range(len(pred_cm)):
                row_index = i + seq_length
                row = clean_df.iloc[row_index]

                error_cm = float(
                    np.sqrt(
                        (pred_cm[i][0] - true_cm[i][0]) ** 2 +
                        (pred_cm[i][1] - true_cm[i][1]) ** 2
                    )
                )

                all_records.append({
                    "tag_id": tag_id,
                    "timestamp": float(row["timestamp"]),
                    "datetime": str(row["datetime"]),
                    "real_x_cm": float(true_cm[i][0]),
                    "real_y_cm": float(true_cm[i][1]),
                    "pred_x_cm": float(pred_cm[i][0]),
                    "pred_y_cm": float(pred_cm[i][1]),
                    "error_cm": error_cm
                })

        except Exception as e:
            print(f"Skipping {tag_id}: {e}")

    output_dir = os.path.join(BASE_DIR, "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "predictions_all.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nExported multi-cow data to: {output_path}")
    print(f"Total records: {len(all_records)}")
    print("Example:", all_records[0])

if __name__ == "__main__":
    export_predictions()