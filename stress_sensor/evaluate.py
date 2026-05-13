"""
evaluate.py — Full evaluation of a trained model
=================================================
Usage:
    python evaluate.py

Loads best_model.pt and produces:
    - Full classification report
    - Confusion matrix plot
    - Score distribution plot
    - Tau vs stress analysis (did tau actually help?)
"""

import os, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score, accuracy_score
)

from cow_stress_model import CowStressModel, estimate_tau
from train import (load_thi, load_neck, align_signals,
                   extract_windows, CowStressDataset, evaluate)
from torch.utils.data import DataLoader

SAVE_DIR   = 'runs/thi_neck_stress'
MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pt')
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f'Loading {MODEL_PATH}...')
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    CFG      = ckpt['config']
    COW2IDX  = ckpt['cow2idx']
    N_COWS   = ckpt['n_cows']
    THI_MEAN = ckpt['thi_mean'];  THI_STD  = ckpt['thi_std']
    NECK_MEAN= ckpt['neck_mean']; NECK_STD = ckpt['neck_std']

    model = CowStressModel(N_COWS, CFG).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Model from epoch {ckpt["epoch"]} | val PR-AUC={ckpt["val_metrics"]["pr_auc"]:.4f}')

    # ── Rebuild test set ──────────────────────────────────────────────────────
    thi_df     = load_thi(CFG['THI_PATH'])
    neck_dir   = pathlib.Path(CFG['NECK_DIR'])
    COW_IDS    = sorted(COW2IDX.keys())

    rng        = np.random.default_rng(CFG['SEED'])
    shuffled   = rng.permutation(COW_IDS).tolist()
    n_tr       = max(1, int(len(COW_IDS) * CFG['TRAIN_RATIO']))
    n_va       = max(1, int(len(COW_IDS) * CFG['VAL_RATIO']))
    TEST_COWS  = set(shuffled[n_tr+n_va:]) or set(shuffled[n_tr:n_tr+n_va])
    print(f'Test cows: {sorted(TEST_COWS)}')

    TEST_S = []
    for cow_id in TEST_COWS:
        f        = neck_dir / f'{cow_id}.csv'
        merged   = align_signals(thi_df, load_neck(f))
        samples  = extract_windows(cow_id, merged, CFG,
                                   COW2IDX, THI_MEAN, THI_STD, NECK_MEAN, NECK_STD)
        TEST_S.extend(samples)

    test_loader = DataLoader(
        CowStressDataset(TEST_S),
        batch_size=CFG['BATCH_SIZE'], shuffle=False
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    import torch.nn as nn
    y_train_pos = sum(1 for s in TEST_S if s['label']==1)
    y_train_neg = len(TEST_S) - y_train_pos
    criterion   = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, y_train_neg/max(1,y_train_pos)],
                            dtype=torch.float32, device=DEVICE)
    )

    test_m, y_true, y_prob = evaluate(model, test_loader, criterion, DEVICE)
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Print results ─────────────────────────────────────────────────────────
    print('\n' + '='*55)
    print('  TEST RESULTS')
    print('='*55)
    print(f'  Accuracy  : {accuracy_score(y_true, y_pred):.4f}')
    for k, v in test_m.items():
        print(f'  {k:<10}: {v:.4f}')
    print('='*55)
    print()
    print(classification_report(y_true, y_pred,
                                target_names=['No Stress', 'Heat Stress']))

    # ── Plot 1: Confusion matrix + score distribution ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Stress', 'Heat Stress'],
                yticklabels=['No Stress', 'Heat Stress'])
    axes[0].set_title('Confusion matrix')
    axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

    axes[1].hist(y_prob[y_true==0], bins=40, alpha=0.6,
                 label='No Stress', color='steelblue')
    axes[1].hist(y_prob[y_true==1], bins=40, alpha=0.6,
                 label='Heat Stress', color='tomato')
    axes[1].axvline(0.5, color='black', ls='--', lw=1.5, label='Threshold 0.5')
    axes[1].set_xlabel('Predicted probability')
    axes[1].set_title('Score distribution'); axes[1].legend()

    plt.tight_layout()
    out1 = os.path.join(SAVE_DIR, 'test_confusion.png')
    plt.savefig(out1, dpi=150)
    print(f'Saved: {out1}')
    plt.show()

    # ── Plot 2: Tau vs stress probability ─────────────────────────────────────
    # This is the key plot to show your professor:
    # does tau correlate with stress probability as expected?
    taus   = np.array([s['tau'] * 1800 for s in TEST_S])  # back to seconds
    labels = np.array([s['label'] for s in TEST_S])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # tau distribution: stressed vs non-stressed
    axes[0].hist(taus[labels==0], bins=40, alpha=0.6,
                 label='No Stress', color='steelblue')
    axes[0].hist(taus[labels==1], bins=40, alpha=0.6,
                 label='Heat Stress', color='tomato')
    axes[0].set_xlabel('τ (seconds)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Tau distribution by stress label')
    axes[0].legend()
    axes[0].axvline(np.median(taus[labels==0]), color='steelblue',
                    ls='--', lw=1.2, label=f'No-stress median')
    axes[0].axvline(np.median(taus[labels==1]), color='tomato',
                    ls='--', lw=1.2, label=f'Stress median')

    # tau vs predicted probability (scatter)
    axes[1].scatter(taus[labels==0], y_prob[labels==0],
                    alpha=0.3, s=8, color='steelblue', label='No Stress')
    axes[1].scatter(taus[labels==1], y_prob[labels==1],
                    alpha=0.3, s=8, color='tomato', label='Heat Stress')
    axes[1].axhline(0.5, color='black', ls='--', lw=1)
    axes[1].set_xlabel('τ (seconds) — lower = heats up faster')
    axes[1].set_ylabel('Predicted stress probability')
    axes[1].set_title('τ vs stress probability')
    axes[1].legend(markerscale=3)

    plt.suptitle('Thermal time constant (τ) analysis', fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(SAVE_DIR, 'tau_analysis.png')
    plt.savefig(out2, dpi=150)
    print(f'Saved: {out2}')
    plt.show()

    # ── Print tau summary ──────────────────────────────────────────────────────
    print('\n=== Tau analysis ===')
    print(f'No-stress windows  — tau mean: {taus[labels==0].mean():.0f}s'
          f'  median: {np.median(taus[labels==0]):.0f}s')
    print(f'Stressed windows   — tau mean: {taus[labels==1].mean():.0f}s'
          f'  median: {np.median(taus[labels==1]):.0f}s')
    print('\nInterpretation: if stressed tau < no-stress tau,')
    print('the formula is correctly capturing faster heating under stress.')


if __name__ == '__main__':
    main()
