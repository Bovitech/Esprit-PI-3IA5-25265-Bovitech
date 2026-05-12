"""
Cow Disease Detection — pipeline.py
Inference pipeline: single image, batch folder, or webcam.

Usage:
    python pipeline.py --image   path/to/cow.jpg
    python pipeline.py --folder  path/to/images/
    python pipeline.py --webcam
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# ──────────────────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    ckpt        = torch.load(checkpoint_path, map_location=device)
    cfg         = ckpt["config"]
    class_names = ckpt["class_names"]
    model_name  = cfg["model_name"]
    num_classes = len(class_names)

    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    print(f"✅ Loaded  : {model_name}")
    print(f"   Classes : {class_names}")
    print(f"   Val acc : {ckpt.get('val_acc', 'N/A')}\n")

    return model, class_names, cfg["image_size"]


# ──────────────────────────────────────────────────────────
# TRANSFORM
# ──────────────────────────────────────────────────────────
def get_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────
# CORE PREDICT  (fixed AMP API)
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_image(pil_image: Image.Image, model, transform, class_names, device, top_k=3):
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type=device.type):
        logits = model(tensor)

    probs        = torch.softmax(logits, dim=1)[0].cpu()
    top          = probs.topk(min(top_k, len(class_names)))
    top_k_results = [(class_names[i], float(p)) for p, i in zip(top.values, top.indices)]

    return top_k_results[0][0], top_k_results[0][1], top_k_results


def print_result(label, pred_class, confidence, top_k_results):
    print(f"\n🐄  {label}")
    print(f"   ┌── Prediction : {pred_class}  ({confidence*100:.1f}%)")
    for rank, (cls, prob) in enumerate(top_k_results, 1):
        bar = "█" * int(prob * 30)
        print(f"   │  #{rank} {cls:<20s} {prob*100:5.1f}%  {bar}")
    print(f"   └{'─'*50}")


# ──────────────────────────────────────────────────────────
# MODES
# ──────────────────────────────────────────────────────────
def run_single(image_path, model, transform, class_names, device):
    img  = Image.open(image_path)
    pred, conf, topk = predict_image(img, model, transform, class_names, device)
    print_result(image_path, pred, conf, topk)
    return pred, conf


def run_folder(folder_path, model, transform, class_names, device):
    folder     = Path(folder_path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images     = [p for p in folder.rglob("*") if p.suffix.lower() in image_exts]

    if not images:
        print(f"⚠️  No images found in {folder_path}")
        return

    print(f"📂 Found {len(images)} image(s) in {folder_path}\n")
    results = []
    t0 = time.time()

    for img_path in images:
        img  = Image.open(img_path)
        pred, conf, topk = predict_image(img, model, transform, class_names, device)
        print_result(img_path.name, pred, conf, topk)
        results.append({"file": str(img_path), "prediction": pred, "confidence": conf})

    elapsed = time.time() - t0
    print(f"\n⏱  {len(images)} images in {elapsed:.2f}s  "
          f"({elapsed/len(images)*1000:.1f} ms/image)")

    from collections import Counter
    print("\n📊 Summary:")
    for cls, cnt in Counter(r["prediction"] for r in results).most_common():
        print(f"   {cls:<20s}: {cnt}")

    return results


def run_webcam(model, transform, class_names, device):
    try:
        import cv2
    except ImportError:
        print("❌ opencv-python required: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("📷 Webcam running — press Q to quit\n")
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_img      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pred, conf, topk = predict_image(pil_img, model, transform, class_names, device)

        color = (0, 200, 0) if pred == "Healthy" else (0, 60, 220)
        cv2.rectangle(frame, (10, 10), (400, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{pred}  {conf*100:.1f}%", (15, 32), FONT, 0.8, color, 2)
        for i, (cls, prob) in enumerate(topk):
            cv2.putText(frame, f"#{i+1} {cls}: {prob*100:.1f}%",
                        (15, 65 + i * 22), FONT, 0.55, (200, 200, 200), 1)

        cv2.imshow("Cow Disease Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Cow Disease Detection — Inference Pipeline")
    p.add_argument("--checkpoint", default="outputs/best_model.pth")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str)
    group.add_argument("--folder", type=str)
    group.add_argument("--webcam", action="store_true")

    p.add_argument("--top_k",  type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"⚡ Device: {device}")

    model, class_names, image_size = load_model(args.checkpoint, device)
    transform = get_transform(image_size)

    if args.image:
        run_single(args.image, model, transform, class_names, device)
    elif args.folder:
        run_folder(args.folder, model, transform, class_names, device)
    elif args.webcam:
        run_webcam(model, transform, class_names, device)


if __name__ == "__main__":
    main()