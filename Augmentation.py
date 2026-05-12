"""
augment_pediculosis.py
Physically generates and saves augmented images for Pediculosis (and any
other minority class you specify) directly into data/train/<class>/.

Run ONCE before retraining:
    python augment_pediculosis.py

After it finishes, just run train.py normally — the new images are already
in the folder and will be picked up automatically.
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms

# ──────────────────────────────────────────────────────────
# CONFIG — edit these if needed
# ──────────────────────────────────────────────────────────
DATA_DIR      = r'C:\Users\louka\Cow disease\data\train'
TARGET_CLASSES = {
    "Pediculosis": 213,   # generate until this class has 200 images total
    "Ringworm": 213,
    "Healthy": 1042,
    "Dermatophilosis": 212,
    "Lumpy": 1068,
    
}
SEED = 42
# ──────────────────────────────────────────────────────────

random.seed(SEED)


def resolve_data_dir(raw_path: str) -> Path:
    """
    Resolve data directory from either absolute path, current working directory,
    or script directory.
    """
    configured = Path(raw_path).expanduser()
    if configured.is_absolute():
        return configured

    cwd_candidate = (Path.cwd() / configured).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    script_dir = Path(__file__).resolve().parent
    script_candidate = (script_dir / configured).resolve()
    if script_candidate.exists():
        return script_candidate

    # Common project layout: script in subfolder, data at parent level.
    parent_candidate = (script_dir.parent / configured).resolve()
    return parent_candidate


# ──────────────────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ──────────────────────────────────────────────────────────

def pediculosis_aug(img):
    """
    Texture-aware augmentation for Pediculosis.
    Emphasizes lice/nit details in hair to distinguish from Ringworm.
    """
    aug = transforms.Compose([
        transforms.RandomApply([transforms.Lambda(
            lambda x: x.filter(ImageFilter.SHARPEN))], p=0.6),
        transforms.RandomApply([transforms.Lambda(
            lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(1.2, 1.8)))], p=0.5),
        transforms.RandomApply([transforms.Lambda(
            lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(1.5, 2.5)))], p=0.4),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.Resize((224, 224)),
    ])
    return aug(img)


def generic_aug(img):
    """Standard augmentation for any other minority class."""
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.Resize((224, 224)),
    ])
    return aug(img)


CLASS_AUGS = {
    "Pediculosis": pediculosis_aug,
    # default for everything else: generic_aug
}


# ──────────────────────────────────────────────────────────
# CORE
# ──────────────────────────────────────────────────────────
def get_images(class_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in Path(class_dir).iterdir()
            if p.suffix.lower() in exts and "_aug" not in p.stem]


def augment_class(class_name, target_count, data_dir):
    class_dir   = Path(data_dir) / class_name
    aug_fn      = CLASS_AUGS.get(class_name, generic_aug)

    if not class_dir.exists():
        print(f"⚠️  {class_name}: folder not found at {class_dir}, skipping.")
        return

    originals   = get_images(class_dir)
    current     = len(list(class_dir.iterdir()))  # includes any previously generated
    needed      = max(0, target_count - current)

    if needed == 0:
        print(f"✅ {class_name}: already has {current} images, nothing to do.")
        return

    print(f"\n🐄 {class_name}")
    print(f"   Original images : {len(originals)}")
    print(f"   Current total   : {current}")
    print(f"   Target          : {target_count}")
    print(f"   Generating      : {needed} new images...")

    generated = 0
    idx       = 0

    while generated < needed:
        src_path = originals[idx % len(originals)]
        img      = Image.open(src_path).convert("RGB")

        for attempt in range(10):   # try up to 10 variants per source image
            if generated >= needed:
                break
            try:
                aug_img  = aug_fn(img)
                out_name = f"{src_path.stem}_aug{generated:04d}{src_path.suffix}"
                out_path = class_dir / out_name
                aug_img.save(out_path, quality=95)
                generated += 1
            except Exception as e:
                print(f"   ⚠️  Error on {src_path.name}: {e}")

        idx += 1

    final_count = len(list(class_dir.iterdir()))
    print(f"   ✅ Done — folder now has {final_count} images")


def main():
    data_dir = resolve_data_dir(DATA_DIR)

    print("=" * 52)
    print("  PEDICULOSIS AUGMENTATION SCRIPT")
    print("=" * 52)
    print(f"  Data dir : {data_dir}\n")

    if not data_dir.exists():
        print("❌ Dataset folder not found.")
        print(f"   Checked path: {data_dir}")
        print("   Set DATA_DIR at the top of this file to your real train folder path.")
        print(r"   Example: DATA_DIR = r'C:\Users\louka\TP1\data\train'")
        return

    # Show current counts for all classes first
    print("📊 Current class counts in train/:")
    for cls in sorted(os.listdir(data_dir)):
        cls_path = data_dir / cls
        if cls_path.is_dir():
            n       = len(list(cls_path.iterdir()))
            bar     = "█" * (n // 10)
            target  = TARGET_CLASSES.get(cls)
            note    = f"  → target: {target}" if target else ""
            warning = "  ⚠️  low!" if n < 50 else ""
            print(f"  {cls:<20s} {n:>4d}  {bar}{note}{warning}")

    print()

    # Run augmentation for each target class
    for class_name, target in TARGET_CLASSES.items():
        augment_class(class_name, target, data_dir)

    # Final summary
    print("\n" + "=" * 52)
    print("  FINAL COUNTS")
    print("=" * 52)
    for cls in sorted(os.listdir(data_dir)):
        cls_path = data_dir / cls
        if cls_path.is_dir():
            orig = len([p for p in cls_path.iterdir()
                        if "_aug" not in p.stem and p.suffix.lower()
                        in {".jpg",".jpeg",".png",".bmp",".webp"}])
            total = len(list(cls_path.iterdir()))
            added = total - orig
            tag   = f"  (+{added} generated)" if added > 0 else ""
            print(f"  {cls:<20s} {total:>4d}{tag}")

    print("\n✅ Done. Now retrain with:  python train.py")


if __name__ == "__main__":
    main()