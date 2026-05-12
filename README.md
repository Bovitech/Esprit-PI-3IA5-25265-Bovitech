# Bovitech: Multimodal Smart Cattle Monitoring System

This repository contains a multimodal machine learning system for intelligent cattle monitoring, combining audio, sensor, and environmental data to model behavior, stress, and productivity in livestock. This project was developed as part of an academic program at **ESPRIT School of Engineering**.

---

## Overview

The system focuses on:

- Behavioral activity recognition from IMU sensor data
- Acoustic classification of bovine vocalizations
- Stress prediction using multimodal time-series modeling
- Milk yield prediction using historical and sensor-derived features
- Geospatial movement analysis for anomaly detection
- Computer vision for cattle skin and coat condition classification

The project combines classical machine learning and deep learning models depending on the data modality.

---

## Features

- Behavioral activity recognition from wearable IMU sensor data
- Acoustic classification using CNNs on Mel spectrograms
- Stress detection using multimodal time-series deep learning models
- Milk yield prediction using gradient boosting regression
- Geospatial tracking and movement pattern analysis
- Multimodal fusion of environmental and physiological signals
- Feature engineering from raw sensor streams and temporal aggregation
- Image classification for cattle dermatological conditions

---

## Requirements

- Python >= 3.9
- numpy
- pandas
- scikit-learn
- scipy
- tensorflow >= 2.10
- keras
- librosa
- soundfile
- xgboost
- fastapi
- uvicorn
- supabase
- matplotlib
- tqdm
- joblib
- shap
- ffmpeg
- libsndfile

---

### Frontend

- React
- NodeJs

---

### Backend

- Python
- FastAPI
- Supabase
- NumPy / Pandas pipeline processing

---

## Installation

```bash
git clone https://github.com/Malek-ami/Bovitech.git
cd Bovitech

python -m venv .venv

# Activate environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Computer Vision: Cow Disease Detection (V1)

Image classification pipeline for cattle skin and coat conditions. Training uses transfer learning on EfficientNet-B3 with strong regularization; inference runs through `pipeline.py` on a single image, a folder, or a webcam.

### Classes

The trained checkpoint predicts five labels:

- Dermatophilosis
- Healthy
- Lumpy
- Pediculosis
- Ringworm

### Project layout

```text
CowDiseaseV1/
├── train2.py           # Main training script
├── pipeline.py         # Inference: image, folder, or webcam
├── Augmentation.py     # Offline augmentation for minority classes
├── check_duplicate.py  # Perceptual-hash duplicate detection
├── Check_overfit.py    # Plots and diagnostics from training history
├── requirements.txt
└── outputs/
    ├── best_model.pth  # Best checkpoint by validation accuracy
    └── history.json    # Per-epoch train/val loss and accuracy
```

Dataset images are expected under a `data/` directory with `train/`, `valid/`, and `test/` splits. Each split should contain one subfolder per class. `train2.py` resolves `data/` from the current working directory, from `CowDiseaseV1/`, or from the parent project folder.

### Setup

From `CowDiseaseV1/`:

```bash
pip install -r requirements.txt
```

For webcam inference, install OpenCV separately:

```bash
pip install opencv-python
```

`check_duplicate.py` also needs `imagehash`:

```bash
pip install imagehash
```

### Training

```bash
python train2.py
```

#### Hyperparameters

| Setting | Value |
| --- | --- |
| Model | `efficientnet_b3` (ImageNet pretrained) |
| Image size | 224 |
| Batch size | 32 |
| Max epochs | 60 |
| Learning rate | 1e-3, then 1e-4 after unfreezing |
| Weight decay | 5e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Freeze backbone | First 5 epochs |
| Early stopping patience | 10 epochs |
| Dropout | 0.5 |
| Label smoothing | 0.15 |
| Mixup alpha | 0.3 |
| Seed | 42 |
| Data workers | 4 |

#### Training process

1. Load `train/`, `valid/`, and `test/` with `ImageFolder` and `DataLoader`.
2. Apply heavy augmentation on the training split (random resized crop, flips, rotation, color jitter, grayscale, perspective, blur, random erasing) and resize plus center crop on validation and test.
3. Build EfficientNet-B3 with a custom head: `Dropout(0.5)` then `Linear(1536 -> num_classes)`.
4. Phase 1 (epochs 1-5): freeze the backbone and train only the classifier head.
5. Phase 2 (epoch 6 onward): unfreeze all layers, reduce the learning rate by 10x, and fine-tune the full network.
6. Each training batch uses mixup with soft cross-entropy loss, mixed precision, and gradient clipping at norm 1.0.
7. Each epoch evaluates on the validation set, saves the best checkpoint to `outputs/best_model.pth`, and stops early if validation accuracy does not improve for 10 epochs.
8. After training, reload the best weights, evaluate on the test set, and write metrics to `outputs/history.json`.

#### Model architecture

- Backbone: EfficientNet-B3 feature extractor (`features[0]` through `features[8]`)
- Pooling: global average pooling
- Head: `Dropout(0.5)` + `Linear(1536 -> 5)`
- Input: RGB images at 224x224 after normalization with ImageNet mean and standard deviation

The saved checkpoint stores `state_dict`, `class_names`, `config`, and `val_acc`.

### Inference

Run from `CowDiseaseV1/` so the default checkpoint path resolves correctly.

Single image:

```bash
python pipeline.py --image path/to/cow.jpg
```

Folder of images:

```bash
python pipeline.py --folder path/to/images/
```

Webcam:

```bash
python pipeline.py --webcam
```

Optional arguments:

- `--checkpoint outputs/best_model.pth`
- `--top_k 3`
- `--device cuda` or `--device cpu`

### Supporting scripts

Offline class balancing before retraining:

```bash
python Augmentation.py
```

Duplicate check across splits:

```bash
python check_duplicate.py
```

Overfitting report from the last training run:

```bash
python Check_overfit.py
```

### Outputs

- `outputs/best_model.pth`: weights for the best validation epoch
- `outputs/history.json`: training curves used by `Check_overfit.py`
