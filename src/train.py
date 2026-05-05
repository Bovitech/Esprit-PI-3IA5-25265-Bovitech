import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.load_data import load_one_cow
from src.preprocess import preprocess_uwb
from src.dataset import create_sequences
from src.model_lstm import CowTrajectoryLSTM
from src.config import BASE_DIR


def train():
    tag_ids = [
        "T01", "T02", "T03", "T04", "T05",
        "T06", "T07", "T08", "T09", "T10",
        "T13", "T14"
    ]

    seq_length = 10
    batch_size = 64
    epochs = 40
    learning_rate = 0.001
    val_ratio = 0.2

    all_X = []
    all_y = []
    scaler = None

    print("Loading multi-cow data...")

    for tag_id in tag_ids:
        try:
            raw_df, _ = load_one_cow(tag_id)
            clean_df = preprocess_uwb(raw_df, tag_id)

            if scaler is None:
                X, y, scaler = create_sequences(
                    clean_df,
                    seq_length=seq_length,
                    scaler=None,
                    fit_scaler=True
                )
            else:
                X, y, _ = create_sequences(
                    clean_df,
                    seq_length=seq_length,
                    scaler=scaler,
                    fit_scaler=False
                )

            all_X.append(X)
            all_y.append(y)

            print(f"{tag_id}: {X.shape[0]} sequences loaded")

        except Exception as e:
            print(f"Skipping {tag_id}: {e}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    full_dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CowTrajectoryLSTM(
        input_size=2,
        hidden_size=64,
        num_layers=2,
        output_size=2
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\nTraining started...")
    print("Total X shape:", X_tensor.shape)
    print("Total y shape:", y_tensor.shape)
    print("Train samples:", train_size)
    print("Validation samples:", val_size)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Train Loss: {avg_train_loss:.5f} "
            f"- Val Loss: {avg_val_loss:.5f}"
        )

    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "lstm_all.pth")
    scaler_path = os.path.join(model_dir, "scaler_all.pkl")

    torch.save(model.state_dict(), model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


if __name__ == "__main__":
    train()