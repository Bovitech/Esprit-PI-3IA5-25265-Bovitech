# Prédiction de la production laitière (kg / jour)

Petit projet Python pour **entraîner** un modèle qui estime la quantité de lait par vache et par jour, puis **l’utiliser** sur de nouvelles dates.

---

## Ce qu’il y a dans ce dossier

| Fichier | Rôle |
|--------|------|
| `train_milk_xgboost.py` | Lit les capteurs / fichiers CSV, assemble le tableau d’entraînement, entraîne le modèle (XGBoost ou forêt aléatoire), enregistre le modèle et les métriques. |
| `milk_model_inference.py` | Charge le modèle enregistré et prédit la production pour **une** ligne (identifiant vache + date), à partir d’un JSON. |
| `build_behavior_daily_features.py` | *Optionnel* : à partir de CSV “une ligne par seconde” (comportement), calcule des **résumés par jour** utilisables par le modèle lait. |
| `requirements.txt` | Liste des bibliothèques Python nécessaires. |

Les fichiers générés par l’entraînement vont par défaut dans le dossier `model_outputs/` (à côté des scripts).

---

## Installation

1. Python 3.9 ou plus récent conseillé.  
2. Dans ce dossier :

```bash
pip install -r requirements.txt
```

---

## Où mettre les données

Par défaut, le script suppose une arborescence du type :

```text
<racine_capteurs>/
├── behavior_labels/individual/   # CSV par vache (optionnel si tu passes un CSV agrégé)
├── main_data/
│   ├── milk/                     # fichiers C*.csv (timestamp, milk_weight_kg, DIM, …)
│   ├── cbt/                    # température collier (optionnel)
│   ├── thi/                    # THI / environnement (optionnel)
│   └── weather/                # météo (optionnel)
```

Tu peux indiquer la racine de deux façons :

- **Variable d’environnement** `SENSOR_DATA_ROOT` (chemin vers le dossier qui contient `behavior_labels` et `main_data`)
- **Ligne de commande** : `--sensor-root "C:\chemin\vers\sensor_data"`

---

## Utilisation rapide

### 1. Entraîner le modèle

```bash
python train_milk_xgboost.py --sensor-root "C:\chemin\vers\tes_donnees"
```

Options utiles (voir `--help` pour la liste complète) :

- `--out-dir` : où sauvegarder le modèle (défaut : `./model_outputs`)
- `--output-prefix` : préfixe des fichiers (`milk_production` par défaut)
- `--model-type xgb` ou `rf` : XGBoost ou Forêt aléatoire
- `--behavior-daily-csv chemin.csv` : utiliser un fichier comportement **déjà agrégé par jour** (produit par `build_behavior_daily_features.py`)
- `--include-weather` / `--no-include-thi` / `--no-include-behavior` etc. : activer ou couper des blocs de variables

À la fin tu obtiens notamment :

- `{prefix}_pipeline.joblib` — le modèle prêt à l’emploi  
- `{prefix}_metrics.json` — métriques + **liste des colonnes** attendues à l’inférence  

### 2. (Optionnel) Comportement au format “jour”

Si tu as des CSV avec une mesure par seconde (colonne `timestamp` ou `ts_sec`, et `behavior` ou `pred_behavior`, etc.) :

```bash
python build_behavior_daily_features.py --input-dir ".\dossier_des_csv" --output-csv ".\comportement_par_jour.csv"
```

Puis relance l’entraînement avec :

```bash
python train_milk_xgboost.py --sensor-root "..." --behavior-daily-csv ".\comportement_par_jour.csv"
```

### 3. Prédire pour un jour donné

Il faut au minimum `cow_id` et `date` (format `YYYY-MM-DD`). Plus tu remplis de champs (DIM, THI, comportement, historique lait…), plus la prédiction sera cohérente avec l’entraînement.

**Exemple Windows (PowerShell)** — JSON entre guillemets simples à l’extérieur :

```powershell
python milk_model_inference.py --json '{\"cow_id\":\"C01\",\"date\":\"2025-07-25\",\"DIM\":120}'
```

**Exemple avec un fichier** :

```powershell
python milk_model_inference.py --json @payload.json
```

Si tes fichiers modèle ne sont pas les noms par défaut :

```powershell
python milk_model_inference.py --pipeline .\model_outputs\mon_modele_pipeline.joblib --metrics .\model_outputs\mon_modele_metrics.json --json @payload.json
```

Tu peux aussi fixer le préfixe par défaut côté inférence avec la variable d’environnement `MILK_MODEL_PREFIX` (doit correspondre au `--output-prefix` utilisé à l’entraînement).

---

## Ce qui n’est pas dans ce dossier

Le script `batch_predict_behavior_all_immu.py` (s’il existe ailleurs sur ta machine) sert à lancer la **classification du comportement** sur des fichiers IMMU, pas le modèle lait directement. Pour la chaîne complète : comportement → agrégation journalière → entraînement lait avec `--behavior-daily-csv`.

---

## Dépôt GitHub

- Ajoute ce dossier (ou tout le dépôt) à Git **sans** `model_outputs/` ni les gros CSV si tu ne veux pas les versionner : pense à un `.gitignore` avec `model_outputs/`, `*.joblib`, `__pycache__/`, etc.

---

## Aide

```bash
python train_milk_xgboost.py --help
python milk_model_inference.py --help
python build_behavior_daily_features.py --help
```

Bon courage pour le repo.
