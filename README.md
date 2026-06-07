# Bovitech — Multimodal Cattle Monitoring

**GitHub repository:** `Esprit-PI-3IA5-25265-Bovitech`  
**Class:** 3IA5 · **Academic year:** 2025–2026  
**Institution:** [ESPRIT School of Engineering](https://esprit.tn/)

---

## Description

**Bovitech** is a smart cattle monitoring platform developed as part of the PIDEV project at ESPRIT. It combines signals from IMU collars, barn environment (THI index), acoustics, and herd context to assess behavior, stress, milk production, vocalizations, and certain health indicators.

A **React Native (Expo)** mobile app lets farmers interact with the system. Inference is handled by a dedicated **Python ML API**. An optional **Django REST backend** (PI_Backend) manages authentication and farm data.

> **Important:** trained model files are **not stored in this Git repository**. They are **downloaded separately** on first API startup (via `models.manifest.json` or `python scripts/download_models.py`).

---

## Objectives

- Monitor cattle health in real time.
- Detect abnormal behavior.
- Estimate animal stress levels.
- Predict milk production.
- Provide decision support for farmers.

---

## Technologies

**Frontend:** React Native, Expo

**Backend:** Python, Django REST Framework, ML inference API

**Database:** SQLite

**AI / ML:** PyTorch, TensorFlow, scikit-learn, XGBoost, Stable-Baselines3

**Models:** Random Forest, XGBoost, BiLSTM, CNN, PPO, EfficientNet-B3

**Embedded:** ESP32, DHT22

Python 3.10+ · Node.js 18+

---

## Prerequisites

- **Python 3.10+** and pip
- **Git**
- **Node.js 18+** (npm, for the mobile app)
- **ffmpeg** *(optional, vocal analysis)*

A single `pip install -r requirements.txt` installs all Python dependencies (ML API and PI_Backend).

**Trained models** and **training datasets** are **not included** in the repository; only download scripts and manifests are versioned.

---

## Datasets

Training data is **not versioned in Git**. It is only needed to **retrain** models, not to run the app (inference uses automatically downloaded weights).

### Behavior dataset

- **Source:** TBD
- **Link:** TBD
- **Preparation:** Multimodal IMU collar CSV files; cleaning, temporal alignment, feature extraction, and sliding windows. See the behavior training scripts.

### Stress dataset

- **Source:** TBD
- **Link:** TBD
- **Preparation:** Sensor sequences (THI, temperature, posture) per cow; sliding windows and shifted labels. See `stress_sensor/data_pipeline.py`.

### Milk production dataset

- **Source:** TBD
- **Link:** TBD
- **Preparation:** Daily milk history coupled with behavioral features; XGBoost pipeline with derived variables.

### Vocalization dataset

- **Source:** TBD
- **Link:** TBD
- **Preparation:** Bovine audio recordings; spectrogram / MFCC conversion, normalization, train/validation split.

### Illness dataset (PPO)

- **Source:** INRAE *(indicative)* — TBD
- **Link:** TBD
- **Preparation:** Temporal herd states, rewards per health class; RL training (Stable-Baselines3 PPO).

### GPS / trajectory dataset

- **Source:** UWB traces — TBD
- **Link:** TBD
- **Preparation:** Position time series; normalization (scaler) and LSTM sequences.

### Skin dataset (chatbot)

- **Source:** Bovine dermatology images — TBD
- **Link:** TBD
- **Preparation:** Resizing, augmentation, EfficientNet-B3 training (chatbot module).

---

## AI Models

**Trained models are not stored in the Git repository** (size, licensing, ESPRIT best practices). On first ML API startup, required artifacts are **downloaded automatically** from URLs defined in `models.manifest.json`.

Manual download:

```bash
python scripts/download_models.py
```

Set `SKIP_MODEL_DOWNLOAD=1` in `.env` to skip auto-download if files are already cached locally.

| Model | Algorithm | Hugging Face | Kaggle | Google Drive |
|-------|-----------|--------------|--------|--------------|
| Behavior | Random Forest | [Amiiimi/Behaviour](https://huggingface.co/Amiiimi/Behaviour) | TBD | TBD |
| Milk production | XGBoost | [Amiiimi/MilkProduction](https://huggingface.co/Amiiimi/MilkProduction) | TBD | TBD |
| Stress | BiLSTM | [Amiiimi/StressModel](https://huggingface.co/Amiiimi/StressModel) | TBD | TBD |
| Vocalization | CNN | [Amiiimi/AudioModel](https://huggingface.co/Amiiimi/AudioModel) | TBD | TBD |
| Illness | PPO | TBD | TBD | TBD |
| GPS / trajectory | LSTM | TBD | TBD | TBD |
| Skin (chatbot) | EfficientNet-B3 | TBD | TBD | TBD |

---

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| Behavior | Accuracy | TBD |
| Behavior | Macro F1 | TBD |
| Stress | F1 Score | TBD |
| Stress | ROC-AUC | TBD |
| Milk production | RMSE | TBD |
| Milk production | MAE | TBD |
| Vocalization | Accuracy | TBD |
| Illness (PPO) | Accuracy | TBD |

---

## Hardware

| Component | Role |
|-----------|------|
| **ESP32** | IoT microcontroller, data collection and transmission |
| **DHT22** | Temperature / humidity sensor (THI calculation) |
| **Smart collar** | IMU and onboard sensors on cattle |
| **Android / iOS smartphone** | Mobile client via Expo Go or native build |

Full bill of materials (BOM), wiring diagrams, and supplier references: **[docs/liste-materiel.md](docs/liste-materiel.md)**

---

## Schematics and Documentation

Additional documentation in the `docs/` folder:

```
docs/
├── schema-cablage.png    # ESP32 / DHT22 wiring diagram (to be added)
├── api.md                # HTTP endpoint reference
└── liste-materiel.md     # Hardware list (BOM)
```

---

## Installation

```bash
git clone https://github.com/Bovitech/Esprit-PI-3IA5-25265-Bovitech.git
cd Esprit-PI-3IA5-25265-Bovitech

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate

pip install -r requirements.txt

copy .env.example .env
cd main-bovitech-main && copy .env.example .env && cd ..
cd PI_Backend && copy .env.example .env && cd ..
```

On Linux / macOS, replace `copy` with `cp`.

Mobile app:

```bash
cd main-bovitech-main
npm install
cd ..
```

Optional manual model download: `python scripts/download_models.py`

---

## Launch

Use **multiple terminals**. Activate the Python venv (`.venv`) before any Python command.

**Terminal 1 — ML API (required)**

```bash
python src/model_http_api.py
```

→ **http://127.0.0.1:8008** · First startup: model download (2–10 min on CPU).

```bash
curl http://127.0.0.1:8008/health
```

**Terminal 2 — Mobile app (required)**

```bash
cd main-bovitech-main
npm run web
```

→ **http://localhost:8081** · On phone: `npx expo start` (Expo Go).

**Terminal 3 — PI_Backend Django (optional)**

```bash
cd PI_Backend
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

→ **http://127.0.0.1:8000** · Admin: `/admin/` · Auth: `/api/auth/` · Cows: `/api/cows/`

| Service | Port |
|---------|------|
| ML API | 8008 |
| PI_Backend | 8000 |
| Expo (web) | 8081 |

---

## Environment Variables

Copy each `.env.example` to `.env` (never commit `.env`). See those files for all variables (ML API, mobile app, PI_Backend, chatbot).

---

## Demo

**Local run:** start the ML API (`/health`), then the Expo app.

**Demo video:** TBD

**Screenshots:**

- Home screen
- Dashboard
- Herd monitoring
- Prediction analysis

**Deployment:** TBD

---

## Authors

| Member | Class | Year |
|--------|-------|------|
| Salah Ghanoui | 3IA5 | 2025–2026 |
| Melek Amimi | 3IA5 | 2025–2026 |
| Meryem Benani | 3IA5 | 2025–2026 |
| Zeineb Moujehed | 3IA5 | 2025–2026 |
| Maram Ben Farhat | 3IA5 | 2025–2026 |

---

## Supervision

- Ms. **Dorsaf Hrizi**
- Ms. **Oumayma Guasmi**

---

License: [MIT](LICENSE)
