"""
train.py — Train the cow stress detection model
================================================
Usage:
    python train.py

Outputs:
    runs/thi_neck_stress/best_model.pt   — best checkpoint (by val PR-AUC)
    runs/thi_neck_stress/training_curves.png
"""

import os, random, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

from cow_stress_model import CowStressModel, estimate_tau

# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    'THI_PATH'      : 'data/visual_data/sensor_data/main_data/thi/average.csv',
    'NECK_DIR'      : 'data/visual_data/sensor_data/sub_data/neck_dev_temp',
    'SAVE_DIR'      : 'runs/thi_neck_stress',
    'THI_THRESHOLD' : 72.0,
    'PREDICT_AHEAD' : 30,
    'THI_WINDOW'    : 60,
    'NECK_WINDOW'   : 240,
    'STRIDE'        : 30,
    'THI_HIDDEN'    : 32,
    'NECK_HIDDEN'   : 64,
    'NUM_LAYERS'    : 2,
    'DROPOUT'       : 0.3,
    'EPOCHS'        : 30,
    'BATCH_SIZE'    : 128,
    'LR'            : 1e-3,
    'WEIGHT_DECAY'  : 1e-4,
    'GRAD_CLIP'     : 1.0,
    'NUM_WORKERS'   : 0,
    'SEED'          : 42,
    'TRAIN_RATIO'   : 0.70,
    'VAL_RATIO'     : 0.15,
}


def _resolve_data_dir() -> pathlib.Path:
    """Locate repo `data/` from cwd (running from `runs/stress_sensor/` breaks bare `data/...`)."""
    here = pathlib.Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "visual_data" / "sensor_data").is_dir():
            return p / "data"
    return pathlib.Path("data")


_data = _resolve_data_dir()
CFG["THI_PATH"] = str(_data / "visual_data/sensor_data/main_data/thi/average.csv")
CFG["NECK_DIR"] = str(_data / "visual_data/sensor_data/sub_data/neck_dev_temp")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ─────────────────────────────────────────────────────────────────────────────
def load_thi(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.sort_values('timestamp').reset_index(drop=True)
    if "thi" in df.columns:
        thi_col = "thi"
    else:
        cands = [c for c in df.columns if "thi" in c and "humid" not in c]
        if not cands:
            raise ValueError(f"No THI column found. Columns: {list(df.columns)}")
        thi_col = cands[0]
    df = df.rename(columns={thi_col: "thi"})
    assert np.isfinite(df['thi'].values).all()
    return df

def load_neck(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.sort_values('timestamp').reset_index(drop=True)
    temp_col = [c for c in df.columns if 'temp' in c][0]
    df = df.rename(columns={temp_col: 'neck_temp'})
    df = df[np.isfinite(df['neck_temp'])]
    return df

def align_signals(thi_df, neck_df):
    thi_ts   = thi_df['timestamp'].values
    neck_ts  = neck_df['timestamp'].values
    neck_val = neck_df['neck_temp'].values
    idx      = np.searchsorted(neck_ts, thi_ts).clip(0, len(neck_ts)-1)
    matched  = neck_val[idx].astype(float)
    matched[np.abs(neck_ts[idx] - thi_ts) > 60] = np.nan
    merged = thi_df.copy()
    merged['neck_temp'] = matched
    merged['neck_temp'] = merged['neck_temp'].ffill().bfill()
    assert np.isfinite(merged['neck_temp'].values).all()
    return merged

# ─────────────────────────────────────────────────────────────────────────────
def extract_windows(cow_id, merged_df, cfg,
                    COW2IDX, THI_MEAN, THI_STD, NECK_MEAN, NECK_STD):
    thi_raw  = merged_df['thi'].values.astype(np.float32)
    neck_raw = merged_df['neck_temp'].values.astype(np.float32)
    temp_col = 'temperature_f' if 'temperature_f' in merged_df.columns else 'thi'
    barn_raw = merged_df[temp_col].values.astype(np.float32)

    TW, NW, PA, S = (cfg['THI_WINDOW'], cfg['NECK_WINDOW'],
                     cfg['PREDICT_AHEAD'], cfg['STRIDE'])
    cow_idx   = COW2IDX[cow_id]
    thi_norm  = (thi_raw  - THI_MEAN)  / THI_STD
    neck_norm = (neck_raw - NECK_MEAN) / NECK_STD
    labels    = (thi_raw >= cfg['THI_THRESHOLD']).astype(np.int64)

    samples, n = [], len(thi_raw)
    for start in range(0, n - TW - PA, S):
        thi_end  = start + TW
        neck_end = start * 4 + NW
        if neck_end > len(neck_norm):
            break

        thi_seq  = thi_norm[start:thi_end]
        neck_seq = neck_norm[start*4:neck_end]
        label    = int(labels[thi_end + PA - 1])

        if not (np.isfinite(thi_seq).all() and np.isfinite(neck_seq).all()):
            continue

        T_ext = float(barn_raw[start])
        tau   = estimate_tau(neck_raw[start*4:neck_end], T_ext, dt_seconds=15.0)

        samples.append({
            'thi_seq'  : thi_seq.reshape(-1,1).astype(np.float32),
            'neck_seq' : neck_seq.reshape(-1,1).astype(np.float32),
            'tau'      : np.float32(tau / 1800.0),
            'label'    : label,
            'cow_idx'  : cow_idx,
        })
    return samples

# ─────────────────────────────────────────────────────────────────────────────
class CowStressDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return (
            torch.from_numpy(s['thi_seq']),
            torch.from_numpy(s['neck_seq']),
            torch.tensor(s['tau'],     dtype=torch.float32),
            torch.tensor(s['cow_idx'], dtype=torch.long),
            torch.tensor(s['label'],   dtype=torch.long),
        )

# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    all_labels, all_probs, total_loss = [], [], 0.0
    with torch.no_grad():
        for thi_seq, neck_seq, tau, cow_idx, labels in loader:
            logits = model(thi_seq.to(device), neck_seq.to(device),
                           tau.to(device), cow_idx.to(device))
            loss   = criterion(logits, labels.to(device))
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    has_both = len(np.unique(y_true)) > 1
    return {
        'loss'   : total_loss / max(1, len(loader)),
        'f1'     : f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)           if has_both else float('nan'),
        'pr_auc' : average_precision_score(y_true, y_prob) if has_both else float('nan'),
    }, y_true, y_prob

# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(CFG['SEED'])
    pathlib.Path(CFG['SAVE_DIR']).mkdir(parents=True, exist_ok=True)
    print(f'Device: {DEVICE}')

    # ── Load data ─────────────────────────────────────────────────────────────
    thi_df     = load_thi(CFG['THI_PATH'])
    neck_dir   = pathlib.Path(CFG['NECK_DIR'])
    neck_files = sorted(neck_dir.glob('*.csv'))
    print(f'Found {len(neck_files)} cow files')

    COW_DATA = {}
    for f in neck_files:
        cow_id = f.stem
        COW_DATA[cow_id] = align_signals(thi_df, load_neck(f))

    COW_IDS = sorted(COW_DATA.keys())
    COW2IDX = {cid: i for i, cid in enumerate(COW_IDS)}
    N_COWS  = len(COW_IDS)

    # ── Split by cow ──────────────────────────────────────────────────────────
    rng       = np.random.default_rng(CFG['SEED'])
    shuffled  = rng.permutation(COW_IDS).tolist()
    n_tr      = max(1, int(N_COWS * CFG['TRAIN_RATIO']))
    n_va      = max(1, int(N_COWS * CFG['VAL_RATIO']))
    TRAIN_COWS = set(shuffled[:n_tr])
    VAL_COWS   = set(shuffled[n_tr:n_tr+n_va])
    TEST_COWS  = set(shuffled[n_tr+n_va:]) or VAL_COWS
    print(f'Train: {sorted(TRAIN_COWS)} | Val: {sorted(VAL_COWS)} | Test: {sorted(TEST_COWS)}')

    # ── Normalisation stats from train cows only ──────────────────────────────
    THI_MEAN  = np.concatenate([COW_DATA[c]['thi'].values      for c in TRAIN_COWS]).mean()
    THI_STD   = max(np.concatenate([COW_DATA[c]['thi'].values  for c in TRAIN_COWS]).std(), 1e-6)
    NECK_MEAN = np.concatenate([COW_DATA[c]['neck_temp'].values for c in TRAIN_COWS]).mean()
    NECK_STD  = max(np.concatenate([COW_DATA[c]['neck_temp'].values for c in TRAIN_COWS]).std(), 1e-6)

    # ── Extract windows ───────────────────────────────────────────────────────
    print('Extracting windows...')
    TRAIN_S, VAL_S, TEST_S = [], [], []
    for cow_id in COW_IDS:
        samples = extract_windows(cow_id, COW_DATA[cow_id], CFG,
                                  COW2IDX, THI_MEAN, THI_STD, NECK_MEAN, NECK_STD)
        pos = sum(s['label'] for s in samples)
        print(f'  {cow_id}: {len(samples):,} windows | stress={pos/max(1,len(samples))*100:.1f}%'
              f' | tau_mean={np.mean([s["tau"]*1800 for s in samples]):.0f}s')
        if cow_id in TRAIN_COWS: TRAIN_S.extend(samples)
        elif cow_id in VAL_COWS: VAL_S.extend(samples)
        else:                    TEST_S.extend(samples)

    print(f'Total — train:{len(TRAIN_S):,} val:{len(VAL_S):,} test:{len(TEST_S):,}')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    y_train      = np.array([s['label'] for s in TRAIN_S])
    class_counts = np.bincount(y_train)
    pos, neg     = int(class_counts[1]), int(class_counts[0])
    sw           = 1.0 / class_counts[y_train]
    sampler      = WeightedRandomSampler(sw, len(TRAIN_S), replacement=True)

    ldr_kw = dict(batch_size=CFG['BATCH_SIZE'], num_workers=CFG['NUM_WORKERS'],
                  pin_memory=(DEVICE == 'cuda'))
    train_loader = DataLoader(CowStressDataset(TRAIN_S), sampler=sampler, **ldr_kw)
    val_loader   = DataLoader(CowStressDataset(VAL_S),   shuffle=False,   **ldr_kw)
    test_loader  = DataLoader(CowStressDataset(TEST_S),  shuffle=False,   **ldr_kw)

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = CowStressModel(N_COWS, CFG).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, neg/max(1,pos)], dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4)
    scaler    = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    print(f'\nParameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Class pos_weight: {neg/max(1,pos):.2f}')

    # ── Training loop ─────────────────────────────────────────────────────────
    history     = {'train_loss':[], 'val_loss':[], 'val_f1':[], 'val_roc':[], 'val_pr':[]}
    _hist_val_k = {'val_loss': 'loss', 'val_f1': 'f1', 'val_roc': 'roc_auc', 'val_pr': 'pr_auc'}
    best_val_pr = -1.0
    save_path   = os.path.join(CFG['SAVE_DIR'], 'best_model.pt')

    for epoch in range(CFG['EPOCHS']):
        model.train()
        train_loss = 0.0

        for thi_seq, neck_seq, tau, cow_idx, labels in tqdm(
                train_loader, desc=f'Epoch {epoch+1}/{CFG["EPOCHS"]}', leave=False):
            thi_seq  = thi_seq.to(DEVICE,  non_blocking=True)
            neck_seq = neck_seq.to(DEVICE, non_blocking=True)
            tau      = tau.to(DEVICE,      non_blocking=True)
            cow_idx  = cow_idx.to(DEVICE,  non_blocking=True)
            labels   = labels.to(DEVICE,   non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                logits = model(thi_seq, neck_seq, tau, cow_idx)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['GRAD_CLIP'])
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()

        avg_train   = train_loss / len(train_loader)
        val_m, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_m['pr_auc'])

        for k in history:
            history[k].append(avg_train if k == 'train_loss' else val_m[_hist_val_k[k]])

        print(f'Epoch {epoch+1:02d} | train={avg_train:.4f} | val={val_m["loss"]:.4f} | '
              f'F1={val_m["f1"]:.4f} | ROC={val_m["roc_auc"]:.4f} | PR={val_m["pr_auc"]:.4f}')

        if val_m['pr_auc'] > best_val_pr:
            best_val_pr = val_m['pr_auc']
            torch.save({
                'epoch': epoch+1, 'model_state': model.state_dict(),
                'config': CFG, 'cow2idx': COW2IDX, 'n_cows': N_COWS,
                'thi_mean': THI_MEAN,   'thi_std':  THI_STD,
                'neck_mean': NECK_MEAN, 'neck_std': NECK_STD,
                'val_metrics': val_m,
            }, save_path)
            print(f'  ✅ Saved best (PR-AUC={best_val_pr:.4f})')

    # ── Training curves ───────────────────────────────────────────────────────
    ep = range(1, CFG['EPOCHS']+1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ep, history['train_loss'], label='Train')
    axes[0].plot(ep, history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(ep, history['val_roc'], label='ROC-AUC')
    axes[1].plot(ep, history['val_pr'],  label='PR-AUC')
    axes[1].set_title('AUC'); axes[1].legend()
    axes[2].plot(ep, history['val_f1'], color='green')
    axes[2].set_title('Val F1')
    for ax in axes: ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['SAVE_DIR'], 'training_curves.png'), dpi=150)
    print(f'\nCurves saved to {CFG["SAVE_DIR"]}/training_curves.png')

    # ── Quick test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    test_m, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    print('\n' + '='*50)
    print('  TEST RESULTS')
    print('='*50)
    for k, v in test_m.items():
        print(f'  {k:<10}: {v:.4f}')
    print('='*50)


if __name__ == '__main__':
    main()
