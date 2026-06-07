# Bovitech

## Description

Bovitech is a multimodal cattle monitoring platform developed as part of the PIDEV project at ESPRIT (3IA5, 2025–2026).  
It analyzes behavior, stress, milk production, vocalizations, and certain health indicators using IMU collars, barn climate (THI), audio, and herd context.

A React Native (Expo) mobile app lets farmers interact with the system.  
Inference is handled by a dedicated Python API (port 8008).  
An optional Django backend (PI_Backend) manages authentication and farm data (port 8000).

Model weights are not stored in Git — they download on first API start.  
The SQLite database is created empty — users register through the app (no initial seed data).

## Technologies used

**Frontend:** React Native, Expo  

**Backend:** Python, Django REST Framework, ML inference API (PyTorch, TensorFlow, scikit-learn, XGBoost)  

**Database:** SQLite  

## Prerequisites

- Python 3.10+
- Node.js 18+
- Git
- Docker (recommended)

## Installation

```bash
git clone https://github.com/Bovitech/Esprit-PI-3IA5-25265-Bovitech.git
cd Esprit-PI-3IA5-25265-Bovitech

cp .env.example .env

pip install -r requirements.txt

cd main-bovitech-main
npm install
cd ..

python scripts/download_models.py
```

With Docker (ML API only):

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Launch

**Terminal 1 — ML API (port 8008)**

```bash
docker compose -f docker/docker-compose.yml up
```

Without Docker:

```bash
python src/model_http_api.py
```

**Terminal 2 — Mobile app (port 8081)**

```bash
cd main-bovitech-main
npm run web
```

**Terminal 3 — Authentication (PI_Backend, port 8000)**

```bash
cd PI_Backend
python manage.py migrate
python manage.py runserver
```

## Environment variables

See `.env.example`

## Demo

Video: https://sites.google.com/view/bovitechproject/demo?authuser=0 

Deployment: TBD  

## Authors

Salah Ghanoui — 3IA5 — 2025–2026  
Melek Amimi — 3IA5 — 2025–2026  
Meryem Benani — 3IA5 — 2025–2026  
Zeineb Moujehed — 3IA5 — 2025–2026  
Maram Ben Farhat — 3IA5 — 2025–2026  

**Supervisors:** Ms. Dorsaf Hrizi, Ms. Oumayma Guasmi
