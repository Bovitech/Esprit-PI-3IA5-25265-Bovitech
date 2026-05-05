# Milk production prediction (kg / day)

Small Python project to **train** a model that estimates daily milk yield per cow, then **run** it on new dates.

---

## What’s in this folder

| File | Purpose |
|------|---------|
| `train_milk_xgboost.py` | Loads sensor / CSV data, builds the training table, trains the model (XGBoost or random forest), saves the model and metrics. |
| `milk_model_inference.py` | Loads the saved model and predicts production for **one** row (cow id + date) from JSON. |
| `build_behavior_daily_features.py` | *Optional:* from per-second behavior CSVs, builds **daily summaries** the milk model can use. |
| `requirements.txt` | Python dependencies. |

Training output goes by default into `model_outputs/` next to these scripts.

---

## Setup

1. Use Python 3.9 or newer.  
2. In this folder:

```bash
pip install -r requirements.txt
```

---

## Where to put your data

By default, scripts expect a layout like:

```text
<sensor_root>/
├── behavior_labels/individual/   # per-cow CSVs (optional if you pass a pre-aggregated CSV)
├── main_data/
│   ├── milk/                     # C*.csv (timestamp, milk_weight_kg, DIM, …)
│   ├── cbt/                      # collar / body temperature (optional)
│   ├── thi/                      # THI / barn environment (optional)
│   └── weather/                  # weather files (optional)
```

Point the scripts at that root in two ways:

- **Environment variable** `SENSOR_DATA_ROOT` (folder that contains `behavior_labels` and `main_data`)
- **CLI** : `--sensor-root "C:\path\to\sensor_data"`

---

## Quick usage

### 1. Train the model

```bash
python train_milk_xgboost.py --sensor-root "C:\path\to\your_data"
```

Useful flags (see `--help` for everything):

- `--out-dir` — where to save artifacts (default: `./model_outputs`)
- `--output-prefix` — file prefix (`milk_production` by default)
- `--model-type xgb` or `rf` — XGBoost vs random forest
- `--behavior-daily-csv path.csv` — use behavior **already rolled up by day** (from `build_behavior_daily_features.py`)
- `--include-weather`, `--no-include-thi`, `--no-include-behavior`, etc. — toggle feature blocks

You get:

- `{prefix}_pipeline.joblib` — fitted pipeline for inference  
- `{prefix}_metrics.json` — metrics plus the **feature column list** inference must use  

### 2. (Optional) Behavior as daily features

If you have per-second CSVs (`timestamp` or `ts_sec`, plus `behavior` or `pred_behavior`, etc.):

```bash
python build_behavior_daily_features.py --input-dir ".\folder_of_csvs" --output-csv ".\behavior_daily.csv"
```

Then train with:

```bash
python train_milk_xgboost.py --sensor-root "..." --behavior-daily-csv ".\behavior_daily.csv"
```

### 3. Predict for one day

You need at least `cow_id` and `date` (`YYYY-MM-DD`). Filling more fields (DIM, THI, behavior, milk history, …) keeps inference closer to how the model was trained.

**PowerShell** — outer single quotes, escaped inner quotes:

```powershell
python milk_model_inference.py --json '{\"cow_id\":\"C01\",\"date\":\"2025-07-25\",\"DIM\":120}'
```

**From a file:**

```powershell
python milk_model_inference.py --json @payload.json
```

**Custom artifact names:**

```powershell
python milk_model_inference.py --pipeline .\model_outputs\my_model_pipeline.joblib --metrics .\model_outputs\my_model_metrics.json --json @payload.json
```

Set `MILK_MODEL_PREFIX` so the default paths match the `--output-prefix` you used when training.

---

## Not included here

`batch_predict_behavior_all_immu.py` (if you keep it elsewhere) only runs **behavior** classification on IMMU files, not milk directly. Full pipeline: behavior → daily aggregation → milk training with `--behavior-daily-csv`.

---

## GitHub

A `.gitignore` is included so you don’t commit `model_outputs/`, `__pycache__/`, virtualenvs, etc. Add large raw CSV paths if needed.

---

## Help

```bash
python train_milk_xgboost.py --help
python milk_model_inference.py --help
python build_behavior_daily_features.py --help
```
