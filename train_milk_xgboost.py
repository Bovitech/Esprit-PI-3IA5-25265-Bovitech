"""
Train a regression model to predict daily milk (kg) per cow.

What this script does (high level):
- Reads milk logs, optional CBT, behavior summaries, THI, weather
- Merges everything by cow_id + date (and by date alone for herd-level env data)
- Trains XGBoost or RandomForest with a sklearn Pipeline (impute + one-hot)
- Saves the fitted pipeline, merged table, test predictions, and metrics JSON

I kept paths configurable because everyone's machine stores sensor_data differently.
Set SENSOR_DATA_ROOT or pass --sensor-root on the command line.

Note: batch_predict_behavior_all_immu.py is NOT part of milk training — it only
runs the behavior classifier. If you use predicted behavior, first export daily
features with build_behavior_daily_features.py, then pass that CSV here with
--behavior-daily-csv.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "need xgboost for this script — pip install xgboost (see requirements.txt)"
    ) from exc


# ---------------------------------------------------------------------------
# Paths: same folder as this file = "project" for outputs; data root is separate
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
_default_sensor = os.environ.get("SENSOR_DATA_ROOT", r"C:\sensor_data\sensor_data")
SENSOR_ROOT = Path(_default_sensor)
OUT_DIR = HERE / "model_outputs"


def _resolve_paths(sensor_root: Path) -> dict[str, Path]:
    """Little helper so all input locations stay in one place."""
    root = sensor_root.resolve()
    return {
        "behavior_labels": root / "behavior_labels" / "individual",
        "milk": root / "main_data" / "milk",
        "thi": root / "main_data" / "thi",
        "weather": root / "main_data" / "weather",
        "cbt": root / "main_data" / "cbt",
    }


def clean_numeric(series: pd.Series) -> pd.Series:
    """Weather files sometimes use commas / extra text — strip to plain numbers."""
    text = series.astype(str).str.replace(",", "", regex=False)
    text = text.str.replace(r"[^0-9.\-]+", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def parse_weather_date(file_name: str) -> pd.Timestamp:
    """Filename often has the day in it; I pull month-day-year with a regex."""
    match = re.search(r"(\d{1,2})[_-](\d{1,2})[_-](\d{4})", file_name)
    if match:
        month, day, year = match.groups()
        return pd.to_datetime(f"{year}-{int(month):02d}-{int(day):02d}", errors="coerce")
    return pd.NaT


def load_milk_daily(milk_dir: Path) -> pd.DataFrame:
    """
    One row per (cow, day): average milk that day + DIM + simple history features.

    milk_lag1 = yesterday's kg (same cow)
    milk_roll3_mean = rolling mean of last 3 days (pandas rolling, min_periods=1)
    """
    frames: list[pd.DataFrame] = []
    for file in sorted(milk_dir.glob("C*.csv")):
        cow_id = file.stem
        df = pd.read_csv(file)
        if "timestamp" not in df.columns or "milk_weight_kg" not in df.columns:
            continue
        df["cow_id"] = cow_id
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["date"] = df["datetime"].dt.floor("D")
        frames.append(df[["cow_id", "date", "milk_weight_kg", "DIM"]])

    if not frames:
        return pd.DataFrame()

    milk = pd.concat(frames, ignore_index=True)
    milk["milk_weight_kg"] = pd.to_numeric(milk["milk_weight_kg"], errors="coerce")
    milk["DIM"] = pd.to_numeric(milk["DIM"], errors="coerce")
    milk = (
        milk.groupby(["cow_id", "date"], as_index=False)
        .agg(milk_weight_kg=("milk_weight_kg", "mean"), DIM=("DIM", "mean"))
        .sort_values(["cow_id", "date"])
    )
    milk["milk_lag1"] = milk.groupby("cow_id")["milk_weight_kg"].shift(1)
    milk["milk_roll3_mean"] = (
        milk.groupby("cow_id")["milk_weight_kg"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return milk


def load_cbt_daily(cbt_dir: Path) -> pd.DataFrame:
    """Daily summary stats for collar/body temperature (CBT)."""
    frames: list[pd.DataFrame] = []
    for file in sorted(cbt_dir.glob("C*.csv")):
        cow_id = file.stem
        df = pd.read_csv(file)
        if "timestamp" not in df.columns:
            continue
        temp_col = "temperature_C" if "temperature_C" in df.columns else None
        if temp_col is None:
            continue

        df["cow_id"] = cow_id
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["date"] = df["datetime"].dt.floor("D")
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        frames.append(df[["cow_id", "date", temp_col]])

    if not frames:
        return pd.DataFrame()

    cbt = pd.concat(frames, ignore_index=True)
    cbt = cbt.groupby(["cow_id", "date"], as_index=False).agg(
        cbt_temp_mean=("temperature_C", "mean"),
        cbt_temp_std=("temperature_C", "std"),
        cbt_temp_min=("temperature_C", "min"),
        cbt_temp_max=("temperature_C", "max"),
    )
    return cbt


def load_behavior_daily(
    behavior_dir: Path,
    behavior_daily_csv: str | None = None,
) -> pd.DataFrame:
    """
    Per-(cow_id, date) behavior features.

    If behavior_daily_csv is set, I just read that file (must have the required cols).
    Otherwise I walk behavior_dir and aggregate raw per-second labels myself.

    Required columns for the CSV path:
      cow_id, date, behavior_n, behavior_mean, behavior_std
    """
    if behavior_daily_csv:
        p = Path(behavior_daily_csv)
        df = pd.read_csv(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
        required = {"cow_id", "date", "behavior_n", "behavior_mean", "behavior_std"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"behavior CSV missing columns: {missing}")
        return df.sort_values(["cow_id", "date"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for file in sorted(behavior_dir.glob("C*.csv")):
        cow_id = file.stem.split("_")[0]
        df = pd.read_csv(file)
        if "timestamp" not in df.columns or "behavior" not in df.columns:
            continue

        df["cow_id"] = cow_id
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["date"] = df["datetime"].dt.floor("D")
        df["behavior"] = pd.to_numeric(df["behavior"], errors="coerce")
        frames.append(df[["cow_id", "date", "behavior"]])

    if not frames:
        return pd.DataFrame()

    behavior = pd.concat(frames, ignore_index=True)
    count = behavior.groupby(["cow_id", "date"]).size().rename("behavior_n")
    mean = behavior.groupby(["cow_id", "date"])["behavior"].mean().rename("behavior_mean")
    std = behavior.groupby(["cow_id", "date"])["behavior"].std().rename("behavior_std")
    base = pd.concat([count, mean, std], axis=1).reset_index()

    proportions = (
        behavior.pivot_table(
            index=["cow_id", "date"],
            columns="behavior",
            values="behavior",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda c: f"behavior_class_{int(c)}_count")
        .reset_index()
    )
    prop_cols = [c for c in proportions.columns if c.endswith("_count")]
    total = proportions[prop_cols].sum(axis=1).replace(0, np.nan)
    for col in prop_cols:
        proportions[col.replace("_count", "_ratio")] = proportions[col] / total

    return base.merge(proportions, on=["cow_id", "date"], how="left")


def load_thi_daily(thi_dir: Path) -> pd.DataFrame:
    """THI is herd/environment scale — merge key is date only."""
    frames: list[pd.DataFrame] = []
    for file in sorted(thi_dir.glob("S*.csv")):
        df = pd.read_csv(file)
        if "timestamp" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["date"] = df["datetime"].dt.floor("D")
        keep_cols = [c for c in ["temperature_C", "humidity_per", "THI"] if c in df.columns]
        if not keep_cols:
            continue
        for col in keep_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df[["date"] + keep_cols])

    if not frames:
        return pd.DataFrame()

    thi = pd.concat(frames, ignore_index=True)
    agg = thi.groupby("date", as_index=False).agg(
        thi_mean=("THI", "mean"),
        thi_std=("THI", "std"),
        thi_max=("THI", "max"),
        env_temp_mean=("temperature_C", "mean"),
        env_humidity_mean=("humidity_per", "mean"),
    )
    return agg


def load_weather_daily(weather_dir: Path) -> pd.DataFrame:
    """Excel/CSV weather station files, averaged to one row per calendar date."""
    frames: list[pd.DataFrame] = []
    for file in sorted(weather_dir.glob("*")):
        if file.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
            continue

        if file.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        day = parse_weather_date(file.name)
        if pd.isna(day):
            continue

        if "Time" in df.columns:
            times = pd.to_datetime(
                df["Time"].astype(str), format="%H:%M:%S", errors="coerce"
            ).dt.time
            df["datetime"] = [
                pd.Timestamp.combine(day.date(), t) if pd.notna(t) else pd.NaT for t in times
            ]
            df["date"] = pd.to_datetime(df["datetime"]).dt.floor("D")
        else:
            df["date"] = day

        num_map = {
            "Temperature": "weather_temp_f",
            "Humidity": "weather_humidity_pct",
            "Speed": "weather_wind_speed_mph",
            "Pressure": "weather_pressure_in",
            "UV": "weather_uv",
            "Solar": "weather_solar_wm2",
        }
        out = pd.DataFrame({"date": df["date"]})
        for src, dst in num_map.items():
            if src in df.columns:
                out[dst] = clean_numeric(df[src])
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["date"])
    weather = pd.concat(frames, ignore_index=True)
    weather = weather.groupby("date", as_index=False).mean(numeric_only=True)
    return weather


def build_dataset_config(
    paths: dict[str, Path],
    *,
    include_weather: bool,
    include_thi: bool,
    include_behavior: bool,
    include_cbt: bool,
    include_milk_history: bool,
    behavior_daily_csv: str | None = None,
) -> pd.DataFrame:
    """Stack features onto the milk table — order of merges matches my old notebook."""
    milk = load_milk_daily(paths["milk"])
    if milk.empty:
        return pd.DataFrame()

    df = milk.copy()
    if not include_milk_history:
        df = df.drop(columns=[c for c in ["milk_lag1", "milk_roll3_mean"] if c in df.columns])

    if include_cbt:
        cbt = load_cbt_daily(paths["cbt"])
        if not cbt.empty:
            df = df.merge(cbt, on=["cow_id", "date"], how="left")

    if include_behavior:
        behavior = load_behavior_daily(paths["behavior_labels"], behavior_daily_csv=behavior_daily_csv)
        if not behavior.empty:
            df = df.merge(behavior, on=["cow_id", "date"], how="left")

    if include_thi:
        thi = load_thi_daily(paths["thi"])
        if not thi.empty:
            df = df.merge(thi, on="date", how="left")

    if include_weather:
        weather = load_weather_daily(paths["weather"])
        if not weather.empty:
            df = df.merge(weather, on="date", how="left")

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df.sort_values(["date", "cow_id"]).reset_index(drop=True)


def _make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Numeric cols → median impute. Object/category → frequent + one-hot."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    transformers: list[tuple] = [
        ("num", SimpleImputer(strategy="median"), num_cols),
    ]
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers)


def train_model(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    output_prefix: str = "milk_production",
    model_type: str = "xgb",
) -> dict:
    """
    Time-based split: last 20% of dates = test (quantile 0.8 on date column).
    I print RMSE / MAE / R2 + "% within 1kg" style checks because that's easy to explain.
    """
    target = "milk_weight_kg"
    df = df.dropna(subset=[target]).copy()
    split_date = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()

    feature_cols = [c for c in df.columns if c not in {"date", target}]
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    preprocessor = _make_preprocessor(X_train)

    model_type = (model_type or "xgb").lower().strip()
    if model_type in {"rf", "random_forest", "randomforest"}:
        model = RandomForestRegressor(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=5,
        )
    elif model_type in {"xgb", "xgboost"}:
        # slightly conservative hyperparams — I was overfitting with deeper trees
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.9,
            min_child_weight=10,
            reg_lambda=2.0,
            reg_alpha=0.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unknown model_type={model_type!r} — use xgb or rf")

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)

    rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    mae_train = float(mean_absolute_error(y_train, pred_train))
    r2_train = float(r2_score(y_train, pred_train))

    rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae_test = float(mean_absolute_error(y_test, pred_test))
    r2_test = float(r2_score(y_test, pred_test))

    abs_err = np.abs(y_test.to_numpy() - pred_test)
    within_1kg = float(np.mean(abs_err <= 1.0))
    within_2kg = float(np.mean(abs_err <= 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = abs_err / np.abs(y_test.to_numpy())
    mape = float(np.nanmean(ape) * 100.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = out_dir / f"{output_prefix}_pipeline.joblib"
    metrics_path = out_dir / f"{output_prefix}_metrics.json"

    joblib.dump(pipe, pipeline_path)
    df.to_csv(out_dir / f"{output_prefix}_merged_training_dataset.csv", index=False)
    test_out = test_df.copy()
    test_out["prediction"] = pred_test
    test_out.to_csv(out_dir / f"{output_prefix}_test_predictions.csv", index=False)

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "split_date": str(split_date),
        "model_type": model_type,
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "r2_train": r2_train,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "within_1kg_accuracy": within_1kg,
        "within_2kg_accuracy": within_2kg,
        "mape_test_pct": mape,
        "features": feature_cols,
        "pipeline_path": str(pipeline_path),
        "metrics_path": str(metrics_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train milk production regressor.")
    parser.add_argument(
        "--sensor-root",
        type=Path,
        default=SENSOR_ROOT,
        help="Root folder that contains behavior_labels/ and main_data/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Where to save pipeline.joblib, metrics.json, CSVs",
    )
    parser.add_argument(
        "--model-type",
        default="xgb",
        choices=["xgb", "rf"],
        help="xgb = XGBoost, rf = RandomForest",
    )
    # BooleanOptionalAction => you get --include-thi / --no-include-thi (py3.9+)
    parser.add_argument("--include-weather", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-thi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-behavior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-cbt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-milk-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--behavior-daily-csv",
        default="",
        help="Optional: pre-aggregated behavior features (from build_behavior_daily_features.py)",
    )
    parser.add_argument(
        "--output-prefix",
        default="milk_production",
        help="Filename prefix inside out-dir",
    )
    args = parser.parse_args()

    paths = _resolve_paths(args.sensor_root)
    df = build_dataset_config(
        paths,
        include_weather=bool(args.include_weather),
        include_thi=bool(args.include_thi),
        include_behavior=bool(args.include_behavior),
        include_cbt=bool(args.include_cbt),
        include_milk_history=bool(args.include_milk_history),
        behavior_daily_csv=(args.behavior_daily_csv.strip() or None),
    )
    if df.empty:
        raise RuntimeError(
            "merged dataset empty — check --sensor-root and that milk CSVs exist under main_data/milk"
        )
    metrics = train_model(
        df,
        out_dir=args.out_dir,
        output_prefix=str(args.output_prefix),
        model_type=str(args.model_type),
    )
    print("done training")
    print(json.dumps(metrics, indent=2))
    print(f"saved under: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
