# Documentation API — Bovitech

Référence des endpoints HTTP de l'API ML d'inférence.

**Base URL (local) :** `http://127.0.0.1:8008`

| Méthode | Chemin | Description |
|---------|--------|-------------|
| GET | `/health` | État du service et modèles chargés |
| POST | `/predict/stress` | Prédiction stress (JSON capteurs) |
| POST | `/predict/behavior` | Classification comportement |
| POST | `/predict/milk` | Estimation production laitière |
| POST | `/predict/vocal` | Classification vocalisation (fichier audio) |
| POST | `/predict/illness` | Score santé / maladie (PPO, si modèle disponible) |
| GET | `/gps/dashboard/` | Tableau de bord trajectoire GPS (si modèle LSTM disponible) |

## PI_Backend (Django REST)

**Base URL (local) :** `http://127.0.0.1:8000`

| Chemin | Description |
|--------|-------------|
| `/api/auth/register/` | Inscription |
| `/api/auth/login/` | Connexion (JWT) |
| `/api/auth/me/` | Profil utilisateur |
| `/api/cows/` | CRUD vaches |

Voir le [README](../README.md) pour l'installation et le lancement.
