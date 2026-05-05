from __future__ import annotations

"""
Train a RandomForest behavior classifier on MmCows-style files.

What this script actually does (high level):
  - discover matching IMU + label CSVs for the cows/dates you pass
  - build per-second feature tables (IMU-only and/or IMU+head)
  - optionally add lag columns so the model sees a little history
  - train/test split, fit sklearn RandomForestClassifier, print metrics
  - dump joblib model + json metadata that predict_core.py reads later

I left a bunch of CLI flags so we can fight overfitting (tree depth, leaf size, etc.)
without editing code every time.
"""

# --- CLI ---
import argparse
import json
import errno
from pathlib import Path

# --- Model IO + tables ---
import joblib
import numpy as np
import pandas as pd

# --- sklearn: classifier + metrics + split ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# --- Our feature pipeline (same code path as prediction) ---
from imu_head_synthesis import synthesize_head_aggregate
from pipeline_utils import (
    aggregate_head,
    aggregate_immu_per_second,
    build_second_level_dataset,
    discover_pairs,
    drop_unknown_behavior,
    load_behavior_labels,
    load_head_data,
    load_immu_csv,
)


def coerce_max_features_for_rf(val: object) -> object:
    """
    Clicking around argparse gives strings; RandomForest wants int/float/str/None weirdly.

    So we normalize: 'sqrt', 'log2', None-ish strings, ints, or a fraction 0<x<=1.
    """
    # Explicit None from code (not CLI) -> sklearn default-ish choice we like.
    if val is None:
        return "sqrt"
    # Int already usable (avoid bool subclass edge case).
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    # Float already usable (fraction of columns).
    if isinstance(val, float):
        return val
    # String branch: magic words vs numbers.
    s = str(val).strip().lower()
    if s in ("sqrt", "log2"):
        return s
    if s in ("none", ""):
        return None
    try:
        x = float(s)
        # Between 0 and 1 means "fraction of features per split"
        if 0.0 < x <= 1.0:
            return x
        # Larger values treated as integer count of features.
        if x >= 1.0:
            return int(round(x))
    except ValueError:
        pass
    # Last resort: return as-is (sklearn might still accept or will error clearly).
    return val


def parse_args() -> argparse.Namespace:
    """Reads sys.argv into a Namespace used everywhere below."""
    parser = argparse.ArgumentParser(description="Train behavior classifier on MmCows-style sensor data.")
    # Where CSVs live on disk (relative paths ok if cwd is repo root).
    parser.add_argument("--sensor-root", type=Path, default=Path("sensor_data/sensor_data"))
    parser.add_argument(
        "--cows",
        type=str,
        nargs="+",
        default=["C01"],
        help="One or more cow IDs, e.g. C01 C02 C03",
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["0725"],
        help="Date codes, e.g. 0725",
    )
    parser.add_argument("--include-mag", action="store_true", help="Use magnetometer magnitude features.")
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep label 0 (Unknown). Default drops behavior==0 for cleaner supervised learning.",
    )
    parser.add_argument("--compare-multimodal", action="store_true", help="Train IMU-only THEN multimodal and print comparison.")
    parser.add_argument(
        "--multimodal-only",
        action="store_true",
        help="Skip IMU-only baseline; train only multimodal (IMMU+Head, no UWB).",
    )
    parser.add_argument(
        "--history-seconds",
        type=int,
        default=0,
        help="Lag features across past N seconds within each cow/day (no leakage across cows).",
    )
    parser.add_argument(
        "--holdout-cows",
        type=str,
        nargs="*",
        default=[],
        help="Cows NEVER seen during fit; evaluated at the end if --multimodal-only.",
    )
    parser.add_argument(
        "--eval-smooth-window",
        type=int,
        default=0,
        help="Rolling majority vote window for holdout preds (reporting only).",
    )
    parser.add_argument(
        "--head-file",
        type=Path,
        default=None,
        help="Force one head CSV for every pair (mostly for debugging mismatched layouts).",
    )
    parser.add_argument(
        "--use-head-direction-csv",
        action="store_true",
        help="Multimodal: read dataset head_direction files instead of synthesizing from IMU.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=None, help="Tree depth cap (None = grow until pure / min_leaf).")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument(
        "--max-features",
        type=str,
        default="sqrt",
        help="RandomForest max_features (e.g. 'sqrt', 'log2', or a float like 0.5).",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        default=None,
        help="Bootstrap fraction per tree (e.g. 0.7) to regularize.",
    )
    parser.add_argument(
        "--ccp-alpha",
        type=float,
        default=0.0,
        help="Cost-complexity pruning; tiny values can shave overfit.",
    )
    parser.add_argument(
        "--model-compress",
        type=int,
        default=3,
        help="joblib compress=0..9 for smaller artifacts on disk.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/model"))
    return parser.parse_args()


def build_multimodal_dataset(pairs, args):
    """
    Loop each IMU/label pair, build per-second IMU + head table, attach labels.

    Head source priority:
      1) explicit --head-file
      2) dataset head_direction CSV if --use-head-direction-csv
      3) synthesize from raw IMU (what we usually want for deployment parity)
    """
    frames = []
    for p in pairs:
        # Same raw loading as predict: keeps columns consistent end-to-end.
        imu_raw = load_immu_csv(p.immu_file)
        immu_df = aggregate_immu_per_second(imu_raw, include_mag=args.include_mag)

        # Pick head feature source (CSV vs synthesized).
        if args.head_file is not None:
            head_path = args.head_file
            if not head_path.exists():
                raise FileNotFoundError(f"--head-file not found: {head_path}")
            head_df = aggregate_head(load_head_data(head_path))
        elif getattr(args, "use_head_direction_csv", False):
            head_path = args.sensor_root / "sub_data" / "head_direction" / p.tag_id / f"{p.tag_id}_{p.date_code}.csv"
            if not head_path.exists():
                raise FileNotFoundError(
                    f"Head CSV expected at {head_path}. Use default (synthesize from IMU) or fix path."
                )
            head_df = aggregate_head(load_head_data(head_path))
        else:
            head_df = synthesize_head_aggregate(imu_raw)
        labels_df = load_behavior_labels(p.label_file)

        # Outer-ish merge strategy: inner on labels guarantees supervised rows only.
        merged = immu_df.merge(head_df, on="ts_sec", how="left").merge(labels_df, on="ts_sec", how="inner")

        merged = merged.sort_values("ts_sec").reset_index(drop=True).ffill().fillna(0)
        # Extra metadata columns used only for grouping lag features correctly.
        merged["cow_id"] = p.cow_id
        merged["date_code"] = p.date_code
        frames.append(merged)

    if not frames:
        raise FileNotFoundError("No multimodal data found for requested cows/dates")
    return pd.concat(frames, axis=0, ignore_index=True)


def add_lag_features(dataset: pd.DataFrame, history_seconds: int) -> pd.DataFrame:
    """
    For each numeric feature column, add lag1..lagN within (cow_id, date_code) groups.

    That grouping part matters: we must not leak future seconds from cow A into cow B,
    or yesterday into today.
    """
    if history_seconds <= 0:
        return dataset

    dataset = dataset.sort_values(["cow_id", "date_code", "ts_sec"]).reset_index(drop=True)
    # Everything that's not meta / label becomes a lagged signal.
    base_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior", "cow_id", "date_code"]]

    grouped = dataset.groupby(["cow_id", "date_code"], sort=False)
    lag_frames = []
    for lag in range(1, history_seconds + 1):
        shifted = grouped[base_cols].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in base_cols]
        lag_frames.append(shifted)

    if lag_frames:
        dataset = pd.concat([dataset] + lag_frames, axis=1)
        # Forward-fill lag NaNs at segment start using same-second value (short history proxy).
        for lag in range(1, history_seconds + 1):
            for c in base_cols:
                col_lag = f"{c}_lag{lag}"
                dataset[col_lag] = dataset[col_lag].fillna(dataset[c])

    return dataset.fillna(0.0)


def filter_training_unknown(dataset: pd.DataFrame, args) -> pd.DataFrame:
    """Most reports treat class 0 as 'unknown' — dropping it tends to stabilize training."""
    if getattr(args, "keep_unknown", False):
        return dataset
    before = len(dataset)
    out = drop_unknown_behavior(dataset, label_col="behavior", unknown_value=0)
    dropped = before - len(out)
    if dropped:
        print(f"[Training] Dropped behavior==0 (Unknown): {dropped} rows; remaining {len(out)}")
    if len(out) == 0:
        raise ValueError("No training rows left after dropping Unknown. Use --keep-unknown or check labels.")
    return out


def evaluate_model(model, X_test, y_test):
    """Small wrapper so we don't copy/paste metric code in five places."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1_macro, report, cm


def smooth_pred_series(ts_sec: pd.Series, pred: pd.Series, window: int) -> pd.Series:
    """Rolling mode for integer predictions on a sorted 1Hz timeline."""
    if window <= 1:
        return pred.astype("int64")

    df = pd.DataFrame({"ts_sec": ts_sec.astype("int64"), "pred": pred.astype("int64")}).sort_values("ts_sec")
    sm = (
        df["pred"]
        .rolling(window=window, center=True, min_periods=1)
        .apply(lambda x: pd.Series(x.astype("int64")).mode().iat[0], raw=False)
        .astype("int64")
    )
    sm.index = df.index
    return sm.reindex(pred.index).fillna(pred).astype("int64")


def train_and_report(dataset, args, suffix):
    """
    Fit RF, print train/test gap hint, save artifacts.

    suffix is 'immu' or 'multimodal' -> controls filenames on disk.
    """
    # X = all columns except bookkeeping / target.
    feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior", "cow_id", "date_code"]]
    X = dataset[feature_cols]
    y = dataset["behavior"].astype("int64")

    if y.nunique() < 2:
        raise ValueError("Training data has <2 behavior classes. Add more cows/dates.")

    # Stratified split keeps rare behaviors in both train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    max_feat = coerce_max_features_for_rf(args.max_features)
    rf_kw = dict(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=max_feat,
        ccp_alpha=float(getattr(args, "ccp_alpha", 0.0) or 0.0),
    )
    if getattr(args, "max_samples", None) is not None:
        rf_kw["max_samples"] = float(args.max_samples)
    model = RandomForestClassifier(**rf_kw)
    model.fit(X_train, y_train)

    train_acc, train_f1_macro, _, _ = evaluate_model(model, X_train, y_train)
    acc, f1_macro, report, cm = evaluate_model(model, X_test, y_test)

    # Simple heuristic banner if train scores look suspiciously hugging-perfect.
    gap_acc = train_acc - acc
    gap_f1 = train_f1_macro - f1_macro
    if gap_acc > 0.08 or gap_f1 > 0.08 or train_acc >= 0.999:
        print(
            f"[Overfitting hint] train/test gap: Δacc={gap_acc:.4f}, ΔmacroF1={gap_f1:.4f}. "
            "Try: lower --max-depth (e.g. 12), raise --min-samples-leaf (25-50), "
            "--max-samples 0.7, or --max-features 0.3."
        )

    model_path = args.out_dir / f"behavior_rf_{suffix}.joblib"
    metadata_path = args.out_dir / f"metadata_{suffix}.json"
    cm_path = args.out_dir / f"confusion_matrix_{suffix}.csv"
    fi_path = args.out_dir / f"feature_importance_{suffix}.csv"

    # Serialize estimator (might be heavy; compression trades CPU for disk).
    try:
        joblib.dump(model, model_path, compress=args.model_compress)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise OSError(
                f"No space left on device while saving model to {model_path}. "
                "Free disk space or use a different --out-dir."
            ) from exc
        raise
    metadata = {
        "sensor_root": str(args.sensor_root),
        "cows": [c.upper() for c in args.cows],
        "dates": sorted({d.strip() for d in args.dates}),
        "include_mag": args.include_mag,
        "feature_columns": feature_cols,
        "history_seconds": int(getattr(args, "history_seconds", 0)),
        "model_params": {
            "n_estimators": int(args.n_estimators),
            "random_state": int(args.random_state),
            "class_weight": "balanced",
            "max_depth": args.max_depth,
            "min_samples_leaf": int(args.min_samples_leaf),
            "min_samples_split": int(args.min_samples_split),
            "max_features": max_feat,
            "max_samples": getattr(args, "max_samples", None),
            "ccp_alpha": float(getattr(args, "ccp_alpha", 0.0) or 0.0),
        },
        "metrics": {
            "train_accuracy": train_acc,
            "train_f1_macro": train_f1_macro,
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
        },
        "classes_seen": sorted(y.unique().tolist()),
        "drop_unknown_behavior": not getattr(args, "keep_unknown", False),
        "notes": "multimodal" if suffix == "multimodal" else "immu-only",
    }
    # Extra metadata only for multimodal runs (predict_core warns on mismatch).
    if suffix == "multimodal":
        metadata["head_source"] = (
            "head_csv"
            if (getattr(args, "use_head_direction_csv", False) or args.head_file is not None)
            else "synthesize_imu_mag"
        )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(cm).to_csv(cm_path, index=False)
    pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(fi_path, index=False)

    print(f"Model {suffix} saved: {model_path}")
    print(f"Train accuracy {suffix}: {train_acc:.4f}")
    print(f"Train macro F1 {suffix}: {train_f1_macro:.4f}")
    print(f"Accuracy {suffix}: {acc:.4f}")
    print(f"Macro F1 {suffix}: {f1_macro:.4f}")
    print("Confusion matrix:")
    print(pd.DataFrame(cm).to_string())
    print("Classification report:")
    print(pd.DataFrame(report).transpose().to_string())

    return acc, f1_macro


def main() -> None:
    """Top-level dispatcher: multimodal-only vs baseline + optional comparison."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted_cows = [c.upper() for c in args.cows]
    holdout_cows = [c.upper() for c in (args.holdout_cows or [])]
    wanted_dates = {d.strip() for d in args.dates}
    # Training file pairs filtered by cow list + requested date codes.
    train_pairs = [p for p in discover_pairs(args.sensor_root, cows=wanted_cows) if p.date_code in wanted_dates]
    holdout_pairs = (
        [p for p in discover_pairs(args.sensor_root, cows=holdout_cows) if p.date_code in wanted_dates]
        if holdout_cows
        else []
    )
    if not train_pairs:
        raise FileNotFoundError("No matching (IMMU,label) file pairs found for requested cows/dates.")

    # --- Branch 1: user only cares about multimodal (common for demos) ---
    if args.multimodal_only:
        print("\n=== Multimodal training (IMMU+Head, UWB removed) ===")
        if getattr(args, "use_head_direction_csv", False) or args.head_file is not None:
            print("[Head] source: head_direction CSV (--use-head-direction-csv or --head-file)")
        else:
            print("[Head] source: synthesized from IMMU (imu_head_synthesis) — recommended for prod alignment")
        multimodal_dataset = build_multimodal_dataset(train_pairs, args)
        multimodal_dataset = add_lag_features(multimodal_dataset, args.history_seconds)
        multimodal_dataset = filter_training_unknown(multimodal_dataset, args)
        train_and_report(multimodal_dataset, args, suffix="multimodal")

        # Optional: evaluate cows that never appeared in train_pairs.
        if holdout_pairs:
            print("\n=== Holdout evaluation (by cow) ===")
            holdout_dataset = build_multimodal_dataset(holdout_pairs, args)
            holdout_dataset = add_lag_features(holdout_dataset, args.history_seconds)
            feature_cols = [c for c in holdout_dataset.columns if c not in ["ts_sec", "behavior", "cow_id", "date_code"]]

            model = joblib.load(args.out_dir / "behavior_rf_multimodal.joblib")
            X_hold = holdout_dataset[feature_cols]
            y_hold = holdout_dataset["behavior"].astype("int64")
            h_acc, h_f1, h_report, h_cm = evaluate_model(model, X_hold, y_hold)
            print(f"Holdout cows: {holdout_cows}")
            print(f"Holdout accuracy: {h_acc:.4f}")
            print(f"Holdout macro F1: {h_f1:.4f}")
            print("Holdout confusion matrix:")
            print(pd.DataFrame(h_cm).to_string())
            print("Holdout classification report:")
            print(pd.DataFrame(h_report).transpose().to_string())

            # Extra experiment: smoothing only changes reported metrics / noise in preds.
            if args.eval_smooth_window and args.eval_smooth_window > 1:
                window = int(args.eval_smooth_window)
                dfh = holdout_dataset[["cow_id", "date_code", "ts_sec"]].copy()
                dfh["pred"] = model.predict(X_hold)
                dfh["pred_smooth"] = (
                    dfh.sort_values(["cow_id", "date_code", "ts_sec"])
                    .groupby(["cow_id", "date_code"], sort=False)
                    .apply(lambda g: smooth_pred_series(g["ts_sec"], g["pred"], window))
                    .reset_index(level=[0, 1], drop=True)
                )
                y_pred_s = dfh["pred_smooth"].astype("int64")
                s_acc = accuracy_score(y_hold, y_pred_s)
                s_f1 = f1_score(y_hold, y_pred_s, average="macro", zero_division=0)
                print(f"\nHoldout (smoothed, window={window}) accuracy: {s_acc:.4f}")
                print(f"Holdout (smoothed, window={window}) macro F1: {s_f1:.4f}")
        return

    # --- Branch 2: default = always train IMU-only baseline first ---
    immu_frames = []
    for p in train_pairs:
        ds, _ = build_second_level_dataset(immu_file=p.immu_file, label_file=p.label_file, include_mag=args.include_mag)
        ds["cow_id"] = p.cow_id
        ds["date_code"] = p.date_code
        immu_frames.append(ds)

    immu_dataset = pd.concat(immu_frames, axis=0, ignore_index=True)
    immu_dataset = add_lag_features(immu_dataset, args.history_seconds)
    immu_dataset = filter_training_unknown(immu_dataset, args)
    print("\n=== IMMU-only training ===")
    immu_acc, immu_f1 = train_and_report(immu_dataset, args, suffix="immu")

    # --- Optional third block: multimodal benchmark side-by-side ---
    if args.compare_multimodal:
        print("\n=== Multimodal training (IMMU+Head, UWB removed) ===")
        if getattr(args, "use_head_direction_csv", False) or args.head_file is not None:
            print("[Head] source: head_direction CSV")
        else:
            print("[Head] source: synthesized from IMMU (imu_head_synthesis)")
        multimodal_dataset = build_multimodal_dataset(train_pairs, args)
        multimodal_dataset = add_lag_features(multimodal_dataset, args.history_seconds)
        multimodal_dataset = filter_training_unknown(multimodal_dataset, args)
        multi_acc, multi_f1 = train_and_report(multimodal_dataset, args, suffix="multimodal")

        print("\n=== Comparison summary ===")
        print(f"IMMU-only: accuracy={immu_acc:.4f}, macro_f1={immu_f1:.4f}")
        print(f"Multimodal: accuracy={multi_acc:.4f}, macro_f1={multi_f1:.4f}")
    else:
        print("\nMultimodal comparison not requested. Run with --compare-multimodal.")


if __name__ == "__main__":
    main()
