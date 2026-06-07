# Datasets — Bovitech

Training data is **not in Git**. To **run** the app, use trained weights from Hugging Face (`python scripts/download_models.py` or Docker — see root [README.md](../README.md)).  
This folder is only for **retraining** models from raw data.

```
data/
├── README.md
├── raw/        ← download Kaggle datasets here
└── processed/  ← preprocessed training data
```

---

## Our dataset sources (Kaggle)

All training data used in this project comes from **two public Kaggle datasets**:

| Source | Link | Used for |
|--------|------|----------|
| **MMCows** | [kaggle.com/datasets/hienvuvg/mmcows](https://www.kaggle.com/datasets/hienvuvg/mmcows) | Behavior, stress, milk, vocal, GPS, illness *(everything except skin disease)* |
| **Cattle diseases datasets** | [kaggle.com/datasets/devang03mgr/cattle-diseases-datasets](https://www.kaggle.com/datasets/devang03mgr/cattle-diseases-datasets) | Skin disease EfficientNet-B3 ([Amiiimi/DiseaseClassification](https://huggingface.co/Amiiimi/DiseaseClassification)) |

### Per model

| Model | Dataset source | Raw folder (suggested) |
|-------|----------------|------------------------|
| Behavior (Random Forest) | [MMCows](https://www.kaggle.com/datasets/hienvuvg/mmcows) | `data/raw/mmcows/` |
| Stress (BiLSTM) | [MMCows](https://www.kaggle.com/datasets/hienvuvg/mmcows) | `data/raw/mmcows/` |
| Milk production (XGBoost) | [MMCows](https://www.kaggle.com/datasets/hienvuvg/mmcows) | `data/raw/mmcows/` |
| Vocalization (CNN) | [MMCows](https://www.kaggle.com/datasets/hienvuvg/mmcows) | `data/raw/mmcows/` |
| GPS / trajectory (LSTM) | [MMCows](https://www.kaggle.com/datasets/hienvuvg/mmcows) | `data/raw/mmcows/` |
| Illness (PPO) | [MMCows](https://www.kaggle.com/datasets/hienvuvg/mmcows) | `data/raw/mmcows/` |
| Skin / DiseaseClassification (EfficientNet-B3) | [Cattle diseases](https://www.kaggle.com/datasets/devang03mgr/cattle-diseases-datasets) | `data/raw/skin/` |

---

## Download and layout

1. Create a [Kaggle](https://www.kaggle.com/) account if needed.
2. Download each dataset from the links above.
3. Extract into `data/raw/` as below.

| Folder | Content |
|--------|---------|
| `data/raw/mmcows/` | MMCows files as downloaded from Kaggle |
| `data/raw/skin/` | Cattle diseases images as downloaded from Kaggle |
| `data/processed/` | Cleaned, split, and windowed data per model (after your preprocessing pipelines) |

Preprocessing notes (repo scripts):

- **Behavior** — IMU collar CSVs; cleaning, alignment, feature extraction, sliding windows
- **Stress** — sensor sequences (THI, temperature, posture); see `stress_sensor/data_pipeline.py`
- **Milk** — daily milk history + behavioral features
- **Vocal** — audio → spectrogram / MFCC, train/validation split
- **GPS** — position time series; scaler + LSTM sequences
- **Illness** — temporal herd states for PPO (Stable-Baselines3)
- **Skin** — resize, augmentation, EfficientNet-B3 training (chatbot module)

---

**Inference vs training:** Kaggle data is **not required** to use the app. Pre-trained weights are fetched separately via `scripts/models.manifest.json`. You only need these datasets if you **retrain** models locally.

More detail: [Datasets section in root README](../README.md#datasets).
