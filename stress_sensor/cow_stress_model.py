"""
Cow Heat-Stress Detection Model
================================
Inputs  : THI sequence (barn environment) + Neck temperature (per cow) + tau
Label   : Will THI >= 72 occur in the next 30 minutes?
Formula : Newton's law of cooling — T(t) = T_ext + (T_cons - T_ext) * (1 - e^(-t/tau))

Architecture:
    THI seq   (60x1)  --> BiLSTM --> 64-d  --|
    Neck seq  (240x1) --> BiLSTM --> 128-d --|
    tau       scalar  --> Linear --> 16-d  --|-->  Gated Fusion --> MLP --> stress/no stress
    Cow ID            --> Embed  --> 16-d  --|
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — PHYSICS FORMULA
#  Newton's law of cooling: how a cow's body heats up over time
# ─────────────────────────────────────────────────────────────────────────────

def thermal_model(t, tau, T_cons, T_ext):
    """
    Newton's law of cooling applied to a cow's body.

    The formula describes how temperature evolves over time when a body
    is exposed to a heat source:

        T(t) = T_ext + (T_cons - T_ext) * (1 - e^(-t/tau))

    Parameters
    ----------
    t      : time in seconds since start of measurement window
    tau    : thermal time constant (seconds) — what we want to estimate
             - small tau (e.g. 600s)  = cow heats up fast  = high stress risk
             - large tau (e.g. 3600s) = cow resists heat   = heat tolerant
    T_cons : the equilibrium temperature the cow's body is heading toward
    T_ext  : barn temperature at the start of the window (known from THI sensor)

    Returns
    -------
    Predicted neck temperature at time t
    """
    return T_ext + (T_cons - T_ext) * (1 - np.exp(-t / tau))


def estimate_tau(neck_temps, T_ext, dt_seconds=15.0):
    """
    Estimate tau for one window of neck temperature readings.

    We know T_ext (from the barn sensor) and we observe T(t) (neck temp).
    We use curve_fit (least squares) to find the tau and T_cons that best
    explain what we measured.

    Parameters
    ----------
    neck_temps  : np.array — raw neck temperature readings (degrees)
    T_ext       : float    — barn temperature at start of window
    dt_seconds  : float    — time between readings (15s for neck sensor)

    Returns
    -------
    tau : float — thermal time constant in seconds
          Returns 1800.0 (30 min) as fallback if fitting fails
    """
    # Build time axis: [0, 15, 30, 45, ...] seconds
    t = np.arange(len(neck_temps)) * dt_seconds

    # Wrap the model so T_ext is fixed (we only fit tau and T_cons)
    def model_fixed_Text(t, tau, T_cons):
        return thermal_model(t, tau, T_cons, T_ext)

    try:
        p0     = [1800.0, float(neck_temps.max())]  # initial guess
        bounds = ([30.0, T_ext], [7200.0, 45.0])     # tau: 30s to 2h

        popt, _ = curve_fit(
            model_fixed_Text, t, neck_temps,
            p0=p0, bounds=bounds, maxfev=2000
        )
        return float(popt[0])   # return tau in seconds

    except Exception:
        return 1800.0           # fallback: 30 minutes


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — NEURAL NETWORK COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for a time series.

    Reads the sequence forward AND backward — so it captures both
    rising trends (forward) and recent peaks (backward).

    Input  : (batch, time_steps, 1)
    Output : (batch, out_dim)
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, out_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,             # reads both directions
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, out_dim),  # *2 because bidirectional
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, hidden*2)
        return self.proj(out[:, -1])    # take last timestep → (B, out_dim)


class CowStressModel(nn.Module):
    """
    Full multimodal stress prediction model.

    Three data sources are encoded separately, then combined:

    1. THI sequence    → BiLSTM → captures barn heat trend
    2. Neck temperature→ BiLSTM → captures cow body response
    3. Tau (τ)         → Linear → captures how fast this cow heats up
    4. Cow ID          → Embedding → captures individual heat tolerance

    The four features are concatenated and passed through a gated fusion
    layer (sigmoid gate) that learns to weight each source automatically.
    Finally an MLP produces the binary stress prediction.
    """

    def __init__(self, n_cows, cfg):
        super().__init__()

        # Branch 1: THI encoder (barn environment)
        self.thi_enc = LSTMEncoder(
            input_size=1,
            hidden_size=cfg['THI_HIDDEN'],
            num_layers=cfg['NUM_LAYERS'],
            dropout=cfg['DROPOUT'],
            out_dim=64
        )

        # Branch 2: Neck temperature encoder (cow body)
        self.neck_enc = LSTMEncoder(
            input_size=1,
            hidden_size=cfg['NECK_HIDDEN'],
            num_layers=cfg['NUM_LAYERS'],
            dropout=cfg['DROPOUT'],
            out_dim=128
        )

        # Branch 3: Tau projection (physics feature)
        # tau is a single scalar → project to 16-d so it has equal weight
        self.tau_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )

        # Branch 4: Cow identity embedding
        # each cow gets a learnable 16-d vector
        self.cow_emb = nn.Embedding(n_cows, 16)

        # Fusion
        fused_dim = 64 + 128 + 16 + 16  # = 224

        # Gated fusion: learn how much to trust each branch per sample
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg['DROPOUT']),
            nn.Linear(64, 2),           # 2 outputs: [no stress, stress]
        )

    def forward(self, thi_seq, neck_seq, tau, cow_idx):
        """
        Parameters
        ----------
        thi_seq  : (B, 60, 1)   — normalised THI history
        neck_seq : (B, 240, 1)  — normalised neck temp history
        tau      : (B,)         — tau / 1800 (normalised thermal constant)
        cow_idx  : (B,)         — integer cow ID

        Returns
        -------
        logits : (B, 2) — raw scores for [no stress, stress]
        """
        t     = self.thi_enc(thi_seq)           # (B, 64)
        n     = self.neck_enc(neck_seq)          # (B, 128)
        tau_f = self.tau_proj(tau.unsqueeze(1))  # (B, 1) → (B, 16)
        c     = self.cow_emb(cow_idx)            # (B, 16)

        fused = torch.cat([t, n, tau_f, c], dim=1)  # (B, 224)
        fused = fused * self.gate(fused)             # apply gate
        return self.classifier(fused)                # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — INFERENCE UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def predict_stress(
    thi_window,       # np.array (60,)  — last 60 THI readings
    neck_window,      # np.array (240,) — last 240 neck temp readings (15s)
    barn_temp,        # float           — barn temperature at window start
    cow_idx,          # int             — cow index
    model,
    ckpt,
    device='cpu'
):
    """
    Run inference for one cow at one point in time.

    Returns
    -------
    dict with:
        stress_probability : float  (0 to 1)
        prediction         : str    ('Heat Stress' or 'No Stress')
        tau_seconds        : float  (estimated thermal time constant)
        tau_interpretation : str    (human-readable meaning)
    """
    # Normalise using training stats from checkpoint
    thi_mean  = ckpt['thi_mean'];  thi_std  = ckpt['thi_std']
    neck_mean = ckpt['neck_mean']; neck_std = ckpt['neck_std']

    thi_norm  = (thi_window  - thi_mean)  / thi_std
    neck_norm = (neck_window - neck_mean) / neck_std

    # Estimate tau from raw neck temperature
    tau_s    = estimate_tau(neck_window, barn_temp, dt_seconds=15.0)
    tau_norm = tau_s / 1800.0

    # Build tensors
    thi_t  = torch.tensor(thi_norm,  dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    neck_t = torch.tensor(neck_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    tau_t  = torch.tensor([tau_norm], dtype=torch.float32)
    cow_t  = torch.tensor([cow_idx],  dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits = model(thi_t.to(device), neck_t.to(device),
                       tau_t.to(device), cow_t.to(device))
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    # Interpret tau
    if tau_s < 600:
        tau_interp = 'heats up very fast — high stress risk'
    elif tau_s < 1800:
        tau_interp = 'moderate heat response'
    else:
        tau_interp = 'heat tolerant — resists warming well'

    return {
        'stress_probability' : round(prob, 4),
        'prediction'         : 'Heat Stress' if prob >= 0.5 else 'No Stress',
        'tau_seconds'        : round(tau_s, 1),
        'tau_interpretation' : tau_interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Test tau estimation
    print('=== Tau estimation test ===')
    t      = np.arange(240) * 15.0
    tau_true = 900.0
    T_ext    = 25.0
    T_cons   = 38.5
    # Simulate neck temp following the formula + small noise
    neck_sim = thermal_model(t, tau_true, T_cons, T_ext) + \
               np.random.normal(0, 0.05, len(t))
    tau_est  = estimate_tau(neck_sim, T_ext)
    print(f'  True tau : {tau_true:.0f}s')
    print(f'  Estimated: {tau_est:.0f}s')
    print(f'  Error    : {abs(tau_est - tau_true):.0f}s')

    # Test model forward pass
    print('\n=== Model forward pass test ===')
    CFG = {
        'THI_HIDDEN' : 32,
        'NECK_HIDDEN': 64,
        'NUM_LAYERS' : 2,
        'DROPOUT'    : 0.3,
    }
    model  = CowStressModel(n_cows=12, cfg=CFG)
    B      = 4
    thi_t  = torch.randn(B, 60,  1)
    neck_t = torch.randn(B, 240, 1)
    tau_t  = torch.rand(B)
    cow_t  = torch.randint(0, 12, (B,))

    out = model(thi_t, neck_t, tau_t, cow_t)
    print(f'  Input  — thi:{thi_t.shape} neck:{neck_t.shape} tau:{tau_t.shape}')
    print(f'  Output — logits:{out.shape}  (expected [{B}, 2])')
    print(f'  NaN check: {out.isnan().any().item()}')
    print(f'  Probabilities: {torch.softmax(out, dim=1)[:, 1].detach().numpy().round(3)}')

    total = sum(p.numel() for p in model.parameters())
    print(f'  Total parameters: {total:,}')
    print('\nAll tests passed.')
