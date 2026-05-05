"""
Fake / proxy "head" features from plain IMU (accel + optional magnetometer).

We don't always have a separate head_direction file. So we approximate roll/pitch/etc.
from the neck IMU signals, then average per second — same granularity as aggregate_head()
so training and prediction stay consistent.

Notes from comparing to real head CSVs (for our class project / dataset):
  - accel_norm matched almost exactly with sqrt(ax^2+ay^2+az^2).
  - roll matched atan2(ax, sqrt(ay^2+az^2)) pretty well (~0.03 deg RMSE in tests).
  - pitch was correlated but not identical (producer probably filtered more).
  - yaw didn't match naive tilt-compensated formulas; treat as rough only.

That's ok: the model treats these columns as extra inputs, not physics ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_head_columns_from_imu_mag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add roll, pitch, yaw, accel_norm, relative_angle to each IMU sample row.

    Expects accel_* columns; if mag_* exist we try a simple tilt-comp yaw.
    """
    # Guard: accel is mandatory for every formula below.
    missing = [c for c in ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2") if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes accéléromètre manquantes: {missing}")

    out = df.copy()
    # Work in numpy for speed over huge IMU dumps.
    ax = out["accel_x_mps2"].to_numpy(dtype=np.float64)
    ay = out["accel_y_mps2"].to_numpy(dtype=np.float64)
    az = out["accel_z_mps2"].to_numpy(dtype=np.float64)

    # --- accel_norm = length of accel vector ---
    accel_norm = np.sqrt(ax * ax + ay * ay + az * az)

    # --- roll / pitch from gravity vector geometry (approximate orientation) ---
    # Small epsilon avoids sqrt(0) weirdness when the cow is barely moving / noisy zeros
    yz = np.sqrt(np.maximum(ay * ay + az * az, 1e-18))
    xz = np.sqrt(np.maximum(ax * ax + az * az, 1e-18))
    roll = np.degrees(np.arctan2(ax, yz))
    pitch = np.degrees(np.arctan2(-ay, xz))

    # --- yaw: needs magnetometer for "compass"-like heading; otherwise zeros ---
    mag_cols = ("mag_x_uT", "mag_y_uT", "mag_z_uT")
    if all(c in out.columns for c in mag_cols):
        mx = out["mag_x_uT"].to_numpy(dtype=np.float64)
        my = out["mag_y_uT"].to_numpy(dtype=np.float64)
        mz = out["mag_z_uT"].to_numpy(dtype=np.float64)
        # Rotate mag into horizontal plane using roll/pitch estimate (classic tilt compensate sketch).
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        cr = np.cos(roll_rad)
        sr = np.sin(roll_rad)
        cp = np.cos(pitch_rad)
        sp = np.sin(pitch_rad)
        xh = mx * cp + my * sr * sp + mz * cr * sp
        yh = my * cr - mz * sr
        yaw = np.degrees(np.arctan2(-yh, xh))
    else:
        yaw = np.zeros_like(roll)

    # --- relative_angle: simple proxy; not identical to vendor column but usable as feature ---
    xy = np.sqrt(np.maximum(ax * ax + ay * ay, 1e-18))
    relative_angle = np.abs(np.degrees(np.arctan2(az, xy)))

    out["roll"] = roll
    out["pitch"] = pitch
    out["yaw"] = yaw
    out["accel_norm"] = accel_norm
    out["relative_angle"] = relative_angle
    return out


def synthesize_head_aggregate(imu_preprocessed: pd.DataFrame) -> pd.DataFrame:
    """
    From preprocessed IMU (several Hz), build per-second head-like features.

    Returns columns: ts_sec, roll, pitch, yaw, accel_norm, relative_angle
    """
    if "timestamp" not in imu_preprocessed.columns:
        raise ValueError("IMMU doit contenir une colonne timestamp")

    # Row-wise angles etc. at original sampling rate.
    df = add_head_columns_from_imu_mag(imu_preprocessed)
    df = df.dropna(subset=["timestamp"]).copy()

    # Same second bucket as everywhere else (floor to integer second).
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")

    # Average within each second so one row/sec lines up with IMU aggregates + labels.
    value_cols = ["roll", "pitch", "yaw", "accel_norm", "relative_angle"]
    agg = df.groupby("ts_sec")[value_cols].mean().reset_index()
    return agg
