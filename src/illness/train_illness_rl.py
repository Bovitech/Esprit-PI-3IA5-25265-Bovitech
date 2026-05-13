from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Runtime logic lives in src/illness/; this file only trains the model.
from illness.illness_labels import build_episode_dataset, load_inrae
from illness.illness_rl_env import IllnessEnv

N_FEATURES = 34
N_ACTIONS = 3
ACTION_NAMES = {0: "Healthy", 1: "At-risk", 2: "Ill"}


# ---------------------------------------------------------------------------
# Warm-start MLP
# ---------------------------------------------------------------------------

class WarmStartMLP(nn.Module):
    def __init__(self, n_in: int = N_FEATURES, n_out: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Fast supervised sample collection
# ---------------------------------------------------------------------------

def _collect_supervised_samples(
    env: IllnessEnv,
    episodes: list,
    max_samples: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from INRAE matrix directly — no pandas per row.
    Uses env._inrae_matrix and env._col_idx for O(1) row access.
    """
    print(f"Collecting supervised samples from {len(episodes):,} episodes...")
    t0 = time.time()

    X_list, y_list = [], []
    total = 0

    for ep in episodes:
        cow_id = ep["cow"]
        n_hours = ep["n_hours"]

        for offset in range(n_hours):
            row_idx = ep["start_idx"] + offset
            row = env._inrae_matrix[row_idx]

            # Build a lightweight dict for _build_state_from_row.
            def _col(name, default=0.0):
                idx = env._col_idx.get(name, -1)
                return float(row[idx]) if idx >= 0 else default

            row_dict = {
                "hour": _col("hour", offset % 24),
                "ACTIVITY_LEVEL": _col("ACTIVITY_LEVEL"),
                "REST": _col("REST"),
                "EAT": _col("EAT"),
                "IN_ALLEYS": _col("IN_ALLEYS"),
                "oestrus": _col("oestrus"),
                "calving": _col("calving"),
                "lameness": _col("lameness"),
                "mastitis": _col("mastitis"),
                "other_disease": _col("other_disease"),
                "action_gt": _col("action_gt"),
            }

            hour_value = int(row_dict["hour"])
            state = env._build_state_from_row(row_dict, cow_id, hour_value)
            label = int(row_dict["action_gt"])

            X_list.append(state)
            y_list.append(label)
            total += 1

            if total >= max_samples:
                break
        if total >= max_samples:
            break

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Collected {len(X):,} samples in {time.time() - t0:.1f}s")
    return X, y


# ---------------------------------------------------------------------------
# Phase 1 — Supervised warm-start
# ---------------------------------------------------------------------------

def run_warmstart(
    env: IllnessEnv,
    train_episodes: list,
    out_dir: Path,
    n_epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    max_samples: int = 300_000,
) -> WarmStartMLP:

    X, y = _collect_supervised_samples(env, train_episodes, max_samples)

    # Save states for SHAP background and later inspection.
    np.save(out_dir / "warmstart_states.npy", X)
    print(f"Saved warm-start states to {out_dir / 'warmstart_states.npy'}")

    # Class weights help the warm-start model handle the heavy class imbalance.
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {dict(zip([0, 1, 2], weights.round(3)))}")

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model = WarmStartMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        acc = correct / len(X)
        print(f"Warm-start epoch {epoch + 1:02d}/{n_epochs} | "
              f"loss={total_loss / len(X):.4f} | train_acc={acc:.4f}")

    torch.save(model.state_dict(), out_dir / "warmstart_mlp.pt")
    print(f"Saved warm-start weights to {out_dir / 'warmstart_mlp.pt'}")

    # Quick sanity check before moving on to PPO training.
    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(1).numpy()

    print("\nWarm-start class accuracy:")
    print(f"{'class':>5} | {'support':>8} | {'accuracy':>8}")
    print("-" * 28)
    for cls in [0, 1, 2]:
        mask = y == cls
        if mask.sum() == 0:
            continue
        acc_cls = (preds[mask] == cls).mean()
        print(f"{cls:>5} | {mask.sum():>8,} | {acc_cls:>8.4f}")

    return model


# ---------------------------------------------------------------------------
# Phase 2 — PPO with warm-start weight transfer
# ---------------------------------------------------------------------------

def _copy_warmstart_to_ppo(warmstart: WarmStartMLP, ppo_model: PPO):
    """
    Copy warm-start MLP weights into the PPO actor network.
    SB3 MlpPolicy stores layers in policy.mlp_extractor.policy_net.
    """
    try:
        ws_layers = [
            l for l in warmstart.net
            if isinstance(l, nn.Linear)
        ]
        ppo_layers = [
            l for l in ppo_model.policy.mlp_extractor.policy_net
            if isinstance(l, nn.Linear)
        ]
        n_copy = min(len(ws_layers), len(ppo_layers))
        for i in range(n_copy):
            if ws_layers[i].weight.shape == ppo_layers[i].weight.shape:
                ppo_layers[i].weight.data.copy_(ws_layers[i].weight.data)
                ppo_layers[i].bias.data.copy_(ws_layers[i].bias.data)
        print(f"Copied {n_copy} warm-start layers into PPO policy network.")
    except Exception as exc:
        print(f"[warn] Weight copy failed ({exc}) — continuing with random init.")


def run_ppo(
    inrae_df,
    episodes: list,
    out_dir: Path,
    warmstart: WarmStartMLP,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    inrae_path: str = "",
):
    print(f"\nStarting PPO training ({total_timesteps:,} timesteps, {n_envs} envs)...")

    # Reuse the loaded dataframe so each vectorized env does not reload CSV files.
    def make_env():
        e = IllnessEnv.__new__(IllnessEnv)
        e._init_from_df(inrae_df, episodes, inrae_path=inrae_path)
        return e

    vec_env = make_vec_env(make_env, n_envs=n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=str(out_dir / "tb_logs"),
        device="cpu",
    )

    _copy_warmstart_to_ppo(warmstart, model)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    elapsed = time.time() - t0
    print(f"PPO training completed in {elapsed / 60:.1f} minutes.")

    save_path = str(out_dir / "illness_ppo")
    model.save(save_path)
    print(f"PPO model saved to {save_path}.zip")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: PPO,
    inrae_df,
    test_episodes: list,
    out_dir: Path,
    n_episodes: int = 100,
    inrae_path: str = "",
):
    print(f"\nEvaluating on {n_episodes} test episodes...")
    env = IllnessEnv.__new__(IllnessEnv)
    env._init_from_df(inrae_df, test_episodes[:n_episodes], inrae_path=inrae_path)

    all_preds, all_gt = [], []

    for _ in range(min(n_episodes, len(test_episodes))):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            all_preds.append(int(action))
            all_gt.append(info["gt_action"])
            done = terminated or truncated

    all_preds = np.array(all_preds)
    all_gt = np.array(all_gt)

    print("\nClassification report:")
    report = classification_report(
        all_gt,
        all_preds,
        target_names=["Healthy", "At-risk", "Ill"],
        digits=3,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(all_gt, all_preds)
    print("Confusion matrix (rows=GT, cols=Pred):")
    print("              Healthy  At-risk      Ill")
    for i, row in enumerate(cm):
        print(f"  {ACTION_NAMES[i]:>8}: {row}")

    ill_mask = all_gt == 2
    if ill_mask.sum() > 0:
        ill_recall = (all_preds[ill_mask] == 2).mean()
        print(f"\n>>> Ill class recall: {ill_recall:.3f} (target > 0.80)")
        if ill_recall < 0.80:
            print("    [!] Below target — consider increasing ent_coef or ill penalty")
    else:
        print("    [!] No ill samples in test set — check label distribution")

    report_dict = classification_report(
        all_gt,
        all_preds,
        target_names=["Healthy", "At-risk", "Ill"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "n_test_steps": len(all_preds),
        "ill_recall": float((all_preds[ill_mask] == 2).mean()) if ill_mask.sum() > 0 else None,
    }
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_dir / 'eval_metrics.json'}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BoviTech RL illness training")
    parser.add_argument(
        "--inrae-path",
        required=True,
        nargs="+",
        metavar="CSV",
        help="One or more INRAE CSV files (for example: data/inrae_activity/dataset1-1.csv ... dataset4-1.csv)",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--warmstart-epochs", type=int, default=10)
    parser.add_argument(
        "--warmstart-samples",
        type=int,
        default=300_000,
        help="Max samples for warm-start (reduce if OOM)",
    )
    parser.add_argument("--out-dir", default="artifacts/illness_model")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--test-split", type=float, default=0.20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the INRAE data once, then split episodes for training and evaluation.
    inrae_df = load_inrae(args.inrae_path)
    episodes = build_episode_dataset(inrae_df)

    n_total = len(episodes)
    n_test = int(n_total * args.test_split)
    n_train = n_total - n_test
    np.random.seed(42)
    idx = np.random.permutation(n_total)
    train_episodes = [episodes[i] for i in idx[:n_train]]
    test_episodes = [episodes[i] for i in idx[n_train:]]
    print(f"Loaded {n_total} episodes | train={n_train} | test={n_test}")

    # Phase 1: supervised warm-start pretraining.
    base_env = IllnessEnv.__new__(IllnessEnv)
    base_env._init_from_df(inrae_df, train_episodes, inrae_path=";".join(args.inrae_path))

    warmstart_model = run_warmstart(
        env=base_env,
        train_episodes=train_episodes,
        out_dir=out_dir,
        n_epochs=args.warmstart_epochs,
        max_samples=args.warmstart_samples,
    )

    # Phase 2: PPO reinforcement learning fine-tuning.
    ppo_model = run_ppo(
        inrae_df=inrae_df,
        episodes=train_episodes,
        out_dir=out_dir,
        warmstart=warmstart_model,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        inrae_path=";".join(args.inrae_path),
    )

    # Final evaluation on the held-out episodes.
    evaluate(
        model=ppo_model,
        inrae_df=inrae_df,
        test_episodes=test_episodes,
        out_dir=out_dir,
        n_episodes=100,
        inrae_path=";".join(args.inrae_path),
    )

    print("\nDone. Copy models to finale_model/:")
    print(f"  copy {out_dir}\\illness_ppo.zip finale_model\\")
    print(f"  copy {out_dir}\\warmstart_states.npy finale_model\\")


if __name__ == "__main__":
    main()
