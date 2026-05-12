"""
Cow Disease Detection — train.py
Anti-overfitting version: heavy augmentation, dropout, mixup, label smoothing.
"""

import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_dataset_root(data_dir: str) -> Path:
    p = Path(data_dir)
    if p.is_absolute():
        return p.resolve()
    for base in (Path.cwd(), _SCRIPT_DIR, _SCRIPT_DIR.parent):
        candidate = (base / p).resolve()
        if (candidate / "train").is_dir():
            return candidate
    return (_SCRIPT_DIR.parent / p).resolve()

CFG = {
    "data_dir"      : "data",
    "output_dir"    : "outputs",
    "model_name"    : "efficientnet_b3",
    "image_size"    : 224,
    "batch_size"    : 32,
    "num_epochs"    : 60,
    "lr"            : 1e-3,
    "weight_decay"  : 5e-4,       # stronger weight decay
    "num_workers"   : 4,
    "patience"      : 10,
    "freeze_epochs" : 5,
    "dropout"       : 0.5,        # higher dropout before classifier
    "label_smoothing": 0.15,      # softer targets
    "mixup_alpha"   : 0.3,        # mixup regularization
    "seed"          : 42,
}


# ──────────────────────────────────────────────────────────
# SEED
# ──────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────
# TRANSFORMS  (heavier augmentation)
# ──────────────────────────────────────────────────────────
def get_transforms(image_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),   # more aggressive crop
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),                                  # wider rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.15),              # stronger jitter
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),    # new: perspective warp
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),     # new: blur
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),           # new: random erase
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ──────────────────────────────────────────────────────────
# MIXUP
# ──────────────────────────────────────────────────────────
def mixup_batch(imgs, labels, num_classes, alpha=0.3):
    """Blends two random images and their labels."""
    if alpha <= 0:
        return imgs, nn.functional.one_hot(labels, num_classes).float()
    lam    = np.random.beta(alpha, alpha)
    idx    = torch.randperm(imgs.size(0))
    mixed  = lam * imgs + (1 - lam) * imgs[idx]
    labels_oh = nn.functional.one_hot(labels, num_classes).float()
    mixed_labels = lam * labels_oh + (1 - lam) * labels_oh[idx]
    return mixed, mixed_labels


# ──────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────
def load_data(cfg):
    train_tf, val_tf = get_transforms(cfg["image_size"])
    root = resolve_dataset_root(cfg["data_dir"])
    for split in ("train", "valid", "test"):
        split_path = root / split
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {split_path}\n"
                f"Resolved data root: {root}\n"
                f"Expected: {root}/train, {root}/valid, {root}/test with class subfolders."
            )

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(root / "valid", transform=val_tf)
    test_ds  = datasets.ImageFolder(root / "test",  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    counts = np.bincount(train_ds.targets)
    print(f"\n📦 Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")
    print(f"   Classes : {train_ds.classes}")
    print(f"   Counts  : {counts.tolist()}\n")

    return train_loader, valid_loader, test_loader, train_ds.classes


# ──────────────────────────────────────────────────────────
# MODEL  (higher dropout)
# ──────────────────────────────────────────────────────────
def build_model(model_name, num_classes, dropout):
    if "efficientnet_b3" in model_name:
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif "efficientnet_b0" in model_name:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif "resnet50" in model_name:
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif "densenet121" in model_name:
        model = models.densenet121(weights="IMAGENET1K_V1")
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.classifier.in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def freeze_backbone(model, model_name):
    if "efficientnet" in model_name:
        for p in model.features.parameters():
            p.requires_grad = False
    elif "resnet" in model_name:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
    elif "densenet" in model_name:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ──────────────────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion_soft, criterion_hard, optimizer, device, scaler, cfg):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    num_classes = len(loader.dataset.classes)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # mixup
        mixed_imgs, mixed_labels = mixup_batch(imgs, labels, num_classes, cfg["mixup_alpha"])
        mixed_imgs   = mixed_imgs.to(device)
        mixed_labels = mixed_labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            out = model(mixed_imgs)
            # soft loss for mixed labels
            log_probs = torch.nn.functional.log_softmax(out, dim=1)
            loss = -(mixed_labels * log_probs).sum(dim=1).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        # accuracy on original (non-mixed) labels
        with torch.no_grad():
            out_orig = model(imgs)
        correct += (out_orig.argmax(1) == labels).sum().item()
        n       += imgs.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type=device.type):
            out  = model(imgs)
            loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def train(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader, class_names = load_data(cfg)
    num_classes = len(class_names)

    model          = build_model(cfg["model_name"], num_classes, cfg["dropout"]).to(device)
    criterion_hard = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    criterion_eval = nn.CrossEntropyLoss()
    scaler         = torch.amp.GradScaler(device.type)

    # ── Phase 1: head only ──
    freeze_backbone(model, cfg["model_name"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=1e-6)

    best_acc, best_weights = 0.0, None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"🚀 Training for up to {cfg['num_epochs']} epochs "
          f"(frozen for first {cfg['freeze_epochs']})\n")
    print(f"   Regularization: dropout={cfg['dropout']}  "
          f"mixup={cfg['mixup_alpha']}  "
          f"label_smoothing={cfg['label_smoothing']}  "
          f"weight_decay={cfg['weight_decay']}\n")

    for epoch in range(cfg["num_epochs"]):
        t0 = time.time()

        if epoch == cfg["freeze_epochs"]:
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=cfg["lr"] / 10,
                                    weight_decay=cfg["weight_decay"])
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg["num_epochs"] - cfg["freeze_epochs"],
                eta_min=1e-6
            )
            print(f"\n🔓 Epoch {epoch+1}: full fine-tuning (lr={cfg['lr']/10})\n")

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion_hard,
                                      criterion_hard, optimizer, device, scaler, cfg)
        vl_loss, vl_acc = eval_epoch(model, valid_loader, criterion_eval, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        gap  = tr_acc - vl_acc
        flag = ""
        if vl_acc > best_acc:
            best_acc     = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "state_dict" : best_weights,
                "class_names": class_names,
                "config"     : cfg,
                "val_acc"    : best_acc,
            }, out_dir / "best_model.pth")
            patience_counter = 0
            flag = "  ✅ best"
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:>3}/{cfg['num_epochs']}  "
              f"loss: {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc: {tr_acc:.4f}/{vl_acc:.4f}  "
              f"gap: {gap:+.4f}  "
              f"({time.time()-t0:.1f}s){flag}")

        if patience_counter >= cfg["patience"]:
            print(f"\n⏹  Early stopping (patience={cfg['patience']})")
            break

    model.load_state_dict(best_weights)
    te_loss, te_acc = eval_epoch(model, test_loader, criterion_eval, device)
    print(f"\n🎯 Test accuracy: {te_acc:.4f}  |  Test loss: {te_loss:.4f}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Done. Model saved to: {out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    train(CFG)