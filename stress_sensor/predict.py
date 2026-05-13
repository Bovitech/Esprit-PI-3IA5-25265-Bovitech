"""
predict.py — Run inference on a new cow
========================================
Usage:
    python predict.py --neck path/to/T01.csv --thi path/to/average.csv

Outputs:
    - Stress probability at each timestep
    - Timeline plot saved to runs/thi_neck_stress/prediction_T01.png
    - Summary: how many stress alerts, when the first one occurs
"""

import argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from cow_stress_model import CowStressModel, estimate_tau
from train import load_thi, load_neck, align_signals

MODEL_PATH = 'runs/thi_neck_stress/best_model.pt'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_timeline(neck_csv, thi_csv, model, ckpt, cfg, cow_idx=0):
    """
    Run the model on a full CSV file and return a DataFrame
    with timestamp, stress probability, and prediction at each window.
    """
    thi_df   = load_thi(thi_csv)
    neck_df  = load_neck(neck_csv)
    merged   = align_signals(thi_df, neck_df)

    thi_raw  = merged['thi'].values.astype(np.float32)
    neck_raw = merged['neck_temp'].values.astype(np.float32)
    temp_col = 'temperature_f' if 'temperature_f' in merged.columns else 'thi'
    barn_raw = merged[temp_col].values.astype(np.float32)

    THI_MEAN  = ckpt['thi_mean'];  THI_STD  = ckpt['thi_std']
    NECK_MEAN = ckpt['neck_mean']; NECK_STD = ckpt['neck_std']

    thi_norm  = (thi_raw  - THI_MEAN)  / THI_STD
    neck_norm = (neck_raw - NECK_MEAN) / NECK_STD

    TW = cfg['THI_WINDOW']
    NW = cfg['NECK_WINDOW']
    PA = cfg['PREDICT_AHEAD']

    results = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(thi_raw) - TW - PA, 1):
            thi_end  = start + TW
            neck_end = start * 4 + NW
            if neck_end > len(neck_norm):
                break

            thi_seq  = thi_norm[start:thi_end].reshape(1,-1,1)
            neck_seq = neck_norm[start*4:neck_end].reshape(1,-1,1)

            T_ext = float(barn_raw[start])
            tau   = estimate_tau(neck_raw[start*4:neck_end], T_ext, 15.0)

            thi_t  = torch.tensor(thi_seq,  dtype=torch.float32).to(DEVICE)
            neck_t = torch.tensor(neck_seq, dtype=torch.float32).to(DEVICE)
            tau_t  = torch.tensor([tau/1800.0], dtype=torch.float32).to(DEVICE)
            cow_t  = torch.tensor([cow_idx], dtype=torch.long).to(DEVICE)

            logits = model(thi_t, neck_t, tau_t, cow_t)
            prob   = torch.softmax(logits, dim=1)[0,1].item()

            results.append({
                'timestamp'  : merged['timestamp'].iloc[thi_end - 1],
                'thi'        : float(thi_raw[thi_end - 1]),
                'neck_temp'  : float(neck_raw[thi_end*4 - 1]) if thi_end*4-1 < len(neck_raw) else np.nan,
                'tau_s'      : round(tau, 1),
                'stress_prob': round(prob, 4),
                'prediction' : 'Heat Stress' if prob >= 0.5 else 'No Stress',
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neck', required=True, help='Path to neck temp CSV (e.g. T01.csv)')
    parser.add_argument('--thi',  required=True, help='Path to THI CSV (average.csv)')
    parser.add_argument('--cow_idx', type=int, default=0,
                        help='Cow index (0-11). Check COW2IDX in checkpoint.')
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading model from {MODEL_PATH}...')
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    CFG   = ckpt['config']
    model = CowStressModel(ckpt['n_cows'], CFG).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f'Model loaded — trained for {ckpt["epoch"]} epochs')
    print(f'Cow ID mapping: {ckpt["cow2idx"]}')

    # ── Run prediction ────────────────────────────────────────────────────────
    print(f'\nRunning prediction on {args.neck}...')
    df = predict_timeline(args.neck, args.thi, model, ckpt, CFG, args.cow_idx)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_alerts = (df['prediction'] == 'Heat Stress').sum()
    print(f'\n=== Prediction Summary ===')
    print(f'  Total windows   : {len(df):,}')
    print(f'  Stress alerts   : {n_alerts} ({n_alerts/len(df)*100:.1f}%)')
    print(f'  Mean stress prob: {df["stress_prob"].mean():.3f}')
    print(f'  Mean tau        : {df["tau_s"].mean():.0f}s')

    if n_alerts > 0:
        first = df[df['prediction']=='Heat Stress'].iloc[0]
        print(f'  First alert at ts={first["timestamp"]:.0f}'
              f' (THI={first["thi"]:.1f}, prob={first["stress_prob"]:.3f})')

    print(f'\nFirst 5 predictions:')
    print(df[['timestamp','thi','neck_temp','tau_s','stress_prob','prediction']].head())

    # ── Plot ──────────────────────────────────────────────────────────────────
    cow_name = pathlib.Path(args.neck).stem
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df['timestamp'], df['thi'], lw=0.8, color='tomato')
    axes[0].axhline(CFG['THI_THRESHOLD'], color='red', ls='--', lw=1, label='THI threshold')
    axes[0].set_ylabel('THI'); axes[0].legend(fontsize=9)
    axes[0].set_title(f'Stress prediction — cow {cow_name}')

    axes[1].plot(df['timestamp'], df['neck_temp'], lw=0.8, color='steelblue')
    axes[1].set_ylabel('Neck temp (°C)')

    axes[2].fill_between(df['timestamp'], 0, df['stress_prob'],
                         alpha=0.4, color='tomato', label='Stress probability')
    axes[2].plot(df['timestamp'], df['stress_prob'], lw=0.6, color='tomato')
    axes[2].axhline(0.5, color='black', ls='--', lw=1, label='Decision threshold')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('P(stress)')
    axes[2].set_xlabel('Timestamp')
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    out = f'runs/thi_neck_stress/prediction_{cow_name}.png'
    plt.savefig(out, dpi=150)
    print(f'\nPlot saved: {out}')
    plt.show()

    # Save CSV
    csv_out = f'runs/thi_neck_stress/prediction_{cow_name}.csv'
    df.to_csv(csv_out, index=False)
    print(f'Results saved: {csv_out}')


if __name__ == '__main__':
    main()
