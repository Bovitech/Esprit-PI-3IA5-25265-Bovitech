"""
check_overfit.py — run this after training to diagnose overfitting.
Reads outputs/history.json produced by train.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

HISTORY_PATH = "outputs/history.json"

def load_history(path):
    with open(path) as f:
        h = json.load(f)
    return h

def diagnose(h):
    train_acc = np.array(h["train_acc"])
    val_acc   = np.array(h["val_acc"])
    train_loss = np.array(h["train_loss"])
    val_loss   = np.array(h["val_loss"])
    epochs     = np.arange(1, len(train_acc) + 1)

    best_epoch    = int(np.argmax(val_acc)) + 1
    best_val_acc  = val_acc.max()
    best_train_acc = train_acc[np.argmax(val_acc)]
    gap           = best_train_acc - best_val_acc

    print("\n" + "="*52)
    print("  OVERFITTING DIAGNOSTIC REPORT")
    print("="*52)
    print(f"  Total epochs trained : {len(epochs)}")
    print(f"  Best epoch           : {best_epoch}")
    print(f"  Train acc @ best     : {best_train_acc:.4f}  ({best_train_acc*100:.1f}%)")
    print(f"  Val   acc @ best     : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
    print(f"  Gap (train - val)    : {gap:.4f}  ({gap*100:.1f}%)")
    print("-"*52)

    # Verdict
    if gap < 0.05:
        verdict = "✅ NO overfitting — train and val are very close."
    elif gap < 0.10:
        verdict = "⚠️  MILD overfitting — small gap, likely fine."
    elif gap < 0.20:
        verdict = "🔶 MODERATE overfitting — model memorizing some training data."
    else:
        verdict = "🔴 SEVERE overfitting — model is memorizing, not generalizing."

    print(f"\n  Verdict: {verdict}")

    # Loss trend check
    last_n = min(5, len(val_loss))
    val_loss_trend = val_loss[-last_n:].mean() - val_loss[-last_n*2:-last_n].mean() if len(val_loss) > last_n * 2 else 0
    if val_loss_trend > 0.05:
        print("  ⚠️  Val loss was rising at end — classic overfitting sign.")
    else:
        print("  ✅ Val loss was stable or decreasing — healthy training.")

    print("="*52 + "\n")
    return epochs, train_acc, val_acc, train_loss, val_loss, best_epoch

def plot(epochs, train_acc, val_acc, train_loss, val_loss, best_epoch, out_path):
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ── Accuracy ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, train_acc * 100, label="Train", color="#4C8EF5", linewidth=2)
    ax1.plot(epochs, val_acc   * 100, label="Validation", color="#F5824C", linewidth=2)
    ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax1.fill_between(epochs, train_acc * 100, val_acc * 100,
                     alpha=0.12, color="#F5824C", label="Overfit gap")
    ax1.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # ── Loss ──
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, train_loss, label="Train", color="#4C8EF5", linewidth=2)
    ax2.plot(epochs, val_loss,   label="Validation", color="#F5824C", linewidth=2)
    ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax2.set_title("Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Overfit Check", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"📈 Plot saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    h = load_history(HISTORY_PATH)
    epochs, tr_acc, vl_acc, tr_loss, vl_loss, best_epoch = diagnose(h)
    plot(epochs, tr_acc, vl_acc, tr_loss, vl_loss, best_epoch,
         out_path="outputs/overfit_check.png")