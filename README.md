# Cow behavior prediction (IMU → RandomForest)

This folder is a **self-contained Python pipeline** for the BOVITECH / MmCows-style workflow:

1. Read neck **IMU** CSVs (high rate) and optional **magnetometer**.
2. Aggregate to **one row per second** and align with **behavior labels**.
3. Optionally add **head-like features** (from a CSV or synthesized from IMU).
4. **Train** a `RandomForestClassifier` and save `joblib` + JSON metadata.
5. **Predict** behavior on a new IMU file.

## Contents

| File | Role |
|------|------|
| `pipeline_utils.py` | Load IMU/labels, per-second aggregation, file discovery, head CSV helpers. |
| `imu_head_synthesis.py` | Build proxy head features (roll, pitch, etc.) from IMU when no head file is used. |
| `train_model.py` | Train IMU-only and/or multimodal model; writes artifacts under `--out-dir`. |
| `predict_core.py` | Load model + metadata, rebuild features, `predict()`. |
| `predict_behavior.py` | CLI wrapper: new IMU CSV → predictions CSV. |
| `build_dataset.py` | Optional: export one merged training-style CSV for one cow/day. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
cd behavior_prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Working directory:** Paths like `sensor_data/sensor_data` and `artifacts/` are relative to how you run Python. From the **repository root** (`BOVITECH-V2-4`), run:

```bash
python behavior_prediction/train_model.py --sensor-root sensor_data/sensor_data ...
python behavior_prediction/predict_behavior.py --immu-file path/to/file.csv --model-dir artifacts/model ...
```

Or `cd behavior_prediction` and pass **absolute** paths to `--sensor-root`, `--immu-file`, and `--model-dir`.

## Expected data layout

Under `--sensor-root` (default `sensor_data/sensor_data`):

- IMU: `main_data/immu/Txx/Txx_MMDD.csv`
- Labels: `behavior_labels/individual/Cxx_MMDD.csv`

Optional multimodal head files from the dataset:

- `sub_data/head_direction/Txx/Txx_MMDD.csv`

## Train

**IMU-only (default)** — also writes `behavior_rf_immu.joblib` and `metadata_immu.json`:

```bash
python behavior_prediction/train_model.py \
  --sensor-root sensor_data/sensor_data \
  --cows C01 \
  --dates 0725 \
  --include-mag \
  --out-dir artifacts/model
```

**Multimodal only** (IMU + head synthesized from IMU unless you pass head flags):

```bash
python behavior_prediction/train_model.py \
  --sensor-root sensor_data/sensor_data \
  --cows C01 \
  --dates 0725 \
  --include-mag \
  --multimodal-only \
  --out-dir artifacts/model
```

**Compare** IMU-only vs multimodal in one run:

```bash
python behavior_prediction/train_model.py \
  --sensor-root sensor_data/sensor_data \
  --cows C01 C02 \
  --dates 0725 \
  --include-mag \
  --compare-multimodal \
  --out-dir artifacts/model
```

Useful flags: `--history-seconds N` (lag features), `--keep-unknown` (keep label `0`), `--holdout-cows` with `--multimodal-only` for cow-level evaluation.

## Predict

Must match training mode (IMU-only vs `--use-multimodal`).

```bash
python behavior_prediction/predict_behavior.py \
  --immu-file path/to/new_immu.csv \
  --model-dir artifacts/model \
  --output-csv artifacts/predictions/predictions.csv
```

If you trained **multimodal**:

```bash
python behavior_prediction/predict_behavior.py \
  --immu-file path/to/new_immu.csv \
  --model-dir artifacts/model \
  --use-multimodal \
  --output-csv artifacts/predictions/predictions.csv
```

Optional human-readable class names:

```bash
python behavior_prediction/predict_behavior.py \
  --immu-file path/to/new_immu.csv \
  --model-dir artifacts/model \
  --behavior-map path/to/behavior_map.json
```

## Export one dataset CSV (optional)

```bash
python behavior_prediction/build_dataset.py \
  --sensor-root sensor_data/sensor_data \
  --cow C01 \
  --date 0725 \
  --include-mag \
  --use-multimodal \
  --output-csv artifacts/datasets/dataset_C01_0725.csv
```

## Model artifacts

Training writes under `--out-dir` (default `artifacts/model`), for example:

- `behavior_rf_immu.joblib` / `metadata_immu.json`
- `behavior_rf_multimodal.joblib` / `metadata_multimodal.json`
- Confusion matrices and feature importance CSVs

`predict_core.py` picks the first matching pair it finds in that folder (see `resolve_model_paths` in code). If both IMU-only and multimodal models share one directory, prefer **separate** `--out-dir` folders to avoid ambiguity.

## License / project

Part of the BOVITECH project; use the license and citation policy of the parent repository.
