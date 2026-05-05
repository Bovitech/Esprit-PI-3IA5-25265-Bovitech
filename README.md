# Bovitech: Multimodal Smart Cattle Monitoring System

This repository contains a multimodal machine learning system for intelligent cattle monitoring, combining audio, sensor, and environmental data to model behavior, stress, and productivity in livestock.

---

## Overview

Bovitech is designed to modernize livestock monitoring by integrating wearable IoT sensor data, acoustic analysis, and environmental features into a unified machine learning pipeline.

The system focuses on:

- Behavioral activity recognition from IMU sensor data  
- Acoustic classification of bovine vocalizations  
- Stress prediction using multimodal time-series modeling  
- Milk yield prediction using historical and sensor-derived features  
- Geospatial movement analysis for anomaly detection  

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

## Installation

```bash
git clone https://github.com/yourusername/bovitech.git
cd bovitech

python -m venv .venv

# Activate environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
