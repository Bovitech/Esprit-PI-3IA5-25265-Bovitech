"""
StressDetectionV3 — inference module (PyTorch) for model_http_api.
See stress_sensor/stress_detection_v3.py in the training repo for the full file.
Output classes: 0=Normal, 1=At-Risk, 2=Stressed
"""

import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        x_aug = torch.cat([x, delta], dim=-1)
        out, _ = self.lstm(x_aug)
        last = out[:, -1, :]
        return self.proj(last)


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_sensors: int = 3):
        super().__init__()
        _ = num_sensors
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sensor_embeddings: list) -> torch.Tensor:
        x = torch.stack(sensor_embeddings, dim=1)
        attended, _ = self.attn(x, x, x)
        attended = self.norm(attended + x)
        return attended.mean(dim=1)


class StressDetectionV3(nn.Module):
    EMBED_DIM = 64

    def __init__(self, num_cows: int = 20, cow_embed_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        D = self.EMBED_DIM
        self.thi_enc = SensorEncoder(
            input_size=1, hidden_size=32, num_layers=2, output_size=D, dropout=dropout
        )
        self.neck_enc = SensorEncoder(
            input_size=1, hidden_size=64, num_layers=2, output_size=D, dropout=dropout
        )
        self.lying_enc = SensorEncoder(
            input_size=1, hidden_size=32, num_layers=2, output_size=D, dropout=dropout
        )
        self.cow_emb = nn.Embedding(num_cows, cow_embed_dim)
        self.fusion = AttentionFusion(embed_dim=D, num_sensors=3)
        fused_dim = D + cow_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 3),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, thi: torch.Tensor, neck_temp: torch.Tensor, lying: torch.Tensor, cow_id: torch.Tensor
    ) -> torch.Tensor:
        e_thi = self.thi_enc(thi)
        e_neck = self.neck_enc(neck_temp)
        e_lying = self.lying_enc(lying)
        fused = self.fusion([e_thi, e_neck, e_lying])
        cow_vec = self.cow_emb(cow_id)
        combined = torch.cat([fused, cow_vec], dim=-1)
        return self.classifier(combined)
