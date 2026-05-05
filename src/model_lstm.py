import torch
import torch.nn as nn


class CowTrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(CowTrajectoryLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last time step
        last_output = lstm_out[:, -1, :]

        # Predict next x,y position
        output = self.fc(last_output)

        return output


if __name__ == "__main__":
    model = CowTrajectoryLSTM()

    dummy_input = torch.randn(4, 10, 2)  # batch=4, sequence=10, x/y
    output = model(dummy_input)

    print(model)
    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", output.shape)