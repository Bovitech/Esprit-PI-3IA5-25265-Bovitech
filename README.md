<div align="center">
🐄 Bovitech
AI-Powered Smart Cattle Management System
Transforming raw farm signals into actionable insights through multimodal data science
<br/>
Afficher l'image
Afficher l'image
Afficher l'image
Afficher l'image
Afficher l'image
<br/>
</div>

📖 Overview
Bovitech is a comprehensive cow-tech ecosystem designed to modernize livestock management through multimodal data science and intelligent automation. By combining wearable IoT sensors (smart collars) with advanced machine learning, Bovitech transforms raw signals from the farm into actionable insights for health, reproduction, and welfare.
The project integrates high-frequency acoustic analysis, geospatial tracking, and behavioral modeling to provide a 360-degree view of every animal in the herd.
<br/>

✨ Key Features
<br/>
FeatureDescription🎙️ Intention ClassificationReal-time analysis of bovine vocalizations and sounds🏃 Behavioral MonitoringAutomated tracking of standing, walking, feeding, and lying states📍 Geospatial TrackingGPS-based trajectory analysis for movement patterns and anomaly detection🚨 Health & Stress AlertsEarly warning systems for respiratory issues, heat periods, and nutritional stress🤖 Multimodal ChatbotAI assistant supporting text, voice (STT/TTS), and image-based diagnosis
<br/>

🛠️ Tech Stack
<br/>
🧠 Data Science & AI
CategoryToolsDeep LearningCNN (sound spectrograms), BiLSTM (sensor time-series), Llama-3.1-8B (conversational AI)Classic MLRandomForest, XGBoost for tabular behavioral featuresAudio ProcessingLibrosa for Mel spectrogram extractionExplainabilitySHAP values for model interpretabilityFrameworksTensorFlow, Keras, Scikit-learn
<br/>
☁️ Backend & Cloud
CategoryToolsCorePython 3.9+, FastAPIDatabaseSupabase (real-time data management)ArchitectureMulti-agent system — Vet · Feed · Meteo · Skin agents
<br/>

⚙️ System Architecture
<br/>
1. 🔊 Acoustic Classification Pipeline
The system processes .wav audio files into Mel Spectrograms (128×128) using a CNN to detect four critical states:
Raw .wav audio  →  Mel Spectrogram (128×128)  →  CNN  →  Classification
<br/>
ClassMeaning🤧 CoughEarly respiratory pathology indicator🌡️ EstrusIdentification of the reproductive window🌾 FoodFeeding behavior and welfare monitoring✅ NormalStandard ambient farm noise
<br/>
2. 📡 Behavioral & Sensor Fusion (Bovitech-V5)
The Bovitech-V5 pipeline uses a wearable collar (IMU + Head Direction) to categorize behaviors every second:
IMU Collar (40–100 Hz)
  ├── Tri-axial accelerometer
  ├── Magnetometer
  └── Head orientation (roll, pitch, yaw)
        │
        ▼
  Sliding window (3–5 s)
        │
        ▼
  Feature extraction + RandomForest
        │
        ▼
  Behavior classification (7 classes)
<br/>
Output behavior classes:
IDBehavior0Unknown1🚶 Walking2🧍 Standing3🌿 Feeding — head up4🌱 Feeding — head down5👅 Licking6💧 Drinking7🛌 Lying
<br/>
3. 🧬 StressDetectionV3
A multimodal deep learning model for early cow stress warning (2-hour prediction horizon):
THI (B,T,1) ──────┐
                   ├──▶  BiLSTM encoders  ──▶  AttentionFusion  ──▶  MLP  ──▶  Stress class
Neck temp (B,T,1) ─┤         (×3)              (4-head MHA)
                   │
Lying (B,T,1) ─────┘
                                                        +
                                               cow_id embedding (B,16)
Classes: 0 Normal · 1 At-Risk · 2 Stressed
Metrics: Accuracy 86.8% · F1 macro 0.665 · F1 weighted 0.876
<br/>
4. 🥛 Milk Yield Prediction
Daily regression model (XGBoost) estimating milk_weight_kg per cow per day:
Inputs (18 features)
  ├── Milk history    (lag-1, rolling mean 3d)
  ├── CBT temperatures (mean, std, min, max)
  ├── Behaviour aggregates (n, mean, std)
  ├── THI / environment (mean, std, max)
  ├── Calendar (day of week, month)
  ├── DIM (days in milk)
  └── cow_id (one-hot encoded)
        │
        ▼
  sklearn Pipeline  →  XGBoost Regressor
        │
        ▼
  milk_weight_kg  (scalar, kg/day)
Metrics: R² test 0.950 · MAPE 2.58% · MAE 1.15 kg
<br/>

🏁 Getting Started
<br/>
Prerequisites

Python 3.9+
TensorFlow 2.x & Keras
Librosa (for audio processing)

<br/>
Installation
1. Clone the repository
bashgit clone https://github.com/your-repo/bovitech.git
cd bovitech
2. Set up a virtual environment
bashpython -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
3. Install dependencies
bashpip install -r requirements.txt
<br/>
Running Inference
Behavior prediction from a new sensor file:
bashpython src/predict_behavior.py \
  --immu-file path/to/sensor.csv \
  --model-dir artifacts/model_v5 \
  --use-sliding-window \
  --window-size 3
Stress detection:
bashpython StressDetection/runs/stress_sensor/stress_detection_v3.py \
  --infer \
  --input path/to/thi_neck_lying.csv \
  --model StressDetectionV3_trained.pt
Milk yield prediction:
bashpython src/predict_milk.py \
  --cow-id C01 \
  --date 2024-08-05 \
  --model-dir artifacts/milk_model
<br/>

📁 Project Structure
bovitech/
├── src/
│   ├── pipeline_utils.py          # Core data loading & aggregation
│   ├── build_dataset.py           # Dataset construction
│   ├── train_model.py             # Behavior model training
│   ├── train_model_v5.py          # Cow-wise generalization (V5)
│   └── predict_behavior.py        # Inference entry point
│
├── StressDetection/
│   └── runs/stress_sensor/
│       ├── stress_detection_v3.py
│       └── StressDetectionV3_trained.pt
│
├── artifacts/
│   ├── model/                     # Trained behavior models
│   ├── model_v5/                  # V5 cow-wise split models
│   └── datasets/                  # Pre-built feature tables
│
├── sensor_data/                   # Raw IMU + label files
├── requirements.txt
└── README.md
<br/>

📊 Model Performance Summary
<br/>
ModelTaskKey MetricCNN (Mel spectrogram)Acoustic classification4-class sound detectionRandomForest V5Behavior classificationAccuracy 59.3% (cross-cow)StressDetectionV3Stress detection (BiLSTM)F1 weighted 0.876XGBoostMilk yield regressionR² 0.950 · MAPE 2.58%

Note: V5 behavior accuracy is intentionally lower than V4 (93.8%) — V5 tests on unseen cows, making it a genuine generalization benchmark rather than a memorization test.

<br/>

📜 Acknowledgments
Developed at Esprit School of Engineering, this project addresses the real challenges faced by farmers in regions like Zaghouan, Tunisia — providing technical solutions to age-old agricultural problems through the cow-tech startup vision.
Special thanks to the entire Bovitech team for the vision, data collection, and endless enthusiasm for making cattle farming smarter. 🐄
<br/>

<div align="center">
Built with ❤️ for farmers and their herds
</div>
