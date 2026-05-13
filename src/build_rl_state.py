"""
34-d RL state for the illness PPO policy (see artifacts/illness_model/illness_ppo/illness.md).
Observation space at training: Box(-5, 5, (34,)). Values are scaled then clipped to that range.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np

# Canonical order — must match training export.
RL_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "accel_x_mean",
    "accel_y_mean",
    "accel_z_mean",
    "accel_mag_mean",
    "accel_mag_std",
    "roll_mean",
    "pitch_mean",
    "yaw_mean",
    "accel_mag_lag1",
    "accel_mag_lag2",
    "accel_mag_lag3",
    "cbt_temp_c",
    "barn_temp_c",
    "barn_humidity",
    "thi",
    "pred_behavior_norm",
    "pred_stress_norm",
    "milk_kg_day_zscore",
    "stress_prob_0",
    "stress_prob_1",
    "stress_prob_2",
    "inrae_in_alleys_per_hour",
    "inrae_rest_per_hour",
    "inrae_eat_per_hour",
    "inrae_activity_level_normalized",
    "inrae_oestrus",
    "inrae_calving",
    "inrae_lameness",
    "inrae_mastitis",
    "inrae_other_disease",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

# Alias for explainability modules (`illness_xai`) and external callers.
FEATURE_NAMES = RL_STATE_FEATURE_NAMES


def _clip_obs(x: float) -> float:
    return float(np.clip(x, -5.0, 5.0))


def _sf(payload: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    raw = payload.get(key, default)
    try:
        if raw is None or raw == "":
            return float(default)
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _parse_ts_iso(ts: Optional[str]) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)
    s = str(ts).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def _time_encodings(dt: datetime) -> tuple[float, float, float, float]:
    """Hour and weekday cyclic features (sin/cos)."""
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    hour_rad = 2.0 * math.pi * (h / 24.0)
    # Monday = 0 .. Sunday = 6
    wd = int(dt.weekday())
    dow_rad = 2.0 * math.pi * (wd / 7.0)
    return (
        math.sin(hour_rad),
        math.cos(hour_rad),
        math.sin(dow_rad),
        math.cos(dow_rad),
    )


_MAG_HISTORY: MutableMapping[str, deque[float]] = defaultdict(lambda: deque(maxlen=8))


def _accel_mag_lags(cow_key: str, mag: float) -> tuple[float, float, float]:
    q = _MAG_HISTORY[cow_key]
    lag1 = q[-1] if len(q) >= 1 else mag
    lag2 = q[-2] if len(q) >= 2 else lag1
    lag3 = q[-3] if len(q) >= 3 else lag2
    q.append(float(mag))
    return float(lag1), float(lag2), float(lag3)


def stress_probs_vector(stress: Optional[Mapping[str, Any]]) -> tuple[float, float, float]:
    """Order: class 0, 1, 2 (Normal, At risk, Stressed)."""
    if not stress or stress.get("ok") is False:
        return 0.33, 0.34, 0.33
    probs = stress.get("probabilities")
    if isinstance(probs, dict):
        keys = ("Normal", "At risk", "Stressed")
        out: list[float] = []
        for k in keys:
            try:
                out.append(float(probs.get(k, 0.0)))
            except (TypeError, ValueError):
                out.append(0.0)
        ssum = sum(out)
        if ssum > 1e-6:
            out = [x / ssum for x in out]
            return out[0], out[1], out[2]
    k = int(_sf(stress, "pred_stress", 0))
    k = max(0, min(2, k))
    p = [0.0, 0.0, 0.0]
    p[k] = 1.0
    return p[0], p[1], p[2]


def build_state_vector(
    payload: dict[str, Any],
    *,
    barn: Optional[dict[str, Any]] = None,
    stress: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Merge POST JSON with optional server-side barn + stress outputs.

    Returns:
        obs: float32 (34,)
        meta: debug scalars (raw lags, timestamp used)
    """
    cow = str(payload.get("cow_id", "C01"))
    ts = _parse_ts_iso(payload.get("timestamp") if isinstance(payload.get("timestamp"), str) else None)

    ax = _sf(payload, "accel_x_mean", _sf(payload, "accel_x_mps2"))
    ay = _sf(payload, "accel_y_mean", _sf(payload, "accel_y_mps2"))
    az = _sf(payload, "accel_z_mean", _sf(payload, "accel_z_mps2"))
    mag = float(np.sqrt(ax * ax + ay * ay + az * az))

    accel_std = _sf(payload, "accel_mag_std", 0.0)
    if accel_std == 0.0 and payload.get("accel_samples_std") is not None:
        accel_std = _sf(payload, "accel_samples_std")

    roll = _sf(payload, "roll_mean", _sf(payload, "roll_deg", 0.0) * math.pi / 180.0)
    pitch = _sf(payload, "pitch_mean", _sf(payload, "pitch_deg", 0.0) * math.pi / 180.0)
    yaw = _sf(payload, "yaw_mean", _sf(payload, "yaw_deg", 0.0) * math.pi / 180.0)

    lag1, lag2, lag3 = _accel_mag_lags(cow, mag)

    cbt = _sf(payload, "cbt_temp_c", _sf(payload, "cbt", 38.4))

    b = barn or {}
    if b.get("ok") and b.get("temp_c") is not None:
        barn_t = float(b["temp_c"])
        barn_h = float(b.get("humidity") or 60.0)
        thi_v = float(b["thi"]) if b.get("thi") is not None else float("nan")
    else:
        barn_t = _sf(payload, "barn_temp_c", _sf(payload, "temp_c", 25.0))
        barn_h = _sf(payload, "barn_humidity", _sf(payload, "humidity_per", 60.0))
        thi_v = _sf(payload, "thi", float("nan"))
    if thi_v != thi_v:
        thi_v = (1.8 * barn_t + 32.0) - (0.55 - 0.0055 * barn_h) * (1.8 * barn_t - 26.0)

    pb = int(_sf(payload, "pred_behavior", _sf(payload, "behavior_id", 4)))
    pb = max(1, min(7, pb))
    ps = int(_sf(payload, "pred_stress", 0))
    ps = max(0, min(2, ps))

    milk_kg = _sf(payload, "prediction_milk_kg_day", _sf(payload, "milk_kg_day", 18.0))
    milk_z = _sf(payload, "milk_kg_day_zscore", float("nan"))
    if milk_z != milk_z:
        ref = _sf(payload, "milk_roll3_mean", milk_kg)
        denom = max(1.0, abs(ref) * 0.15)
        milk_z = (milk_kg - ref) / denom
    milk_z = float(np.clip(milk_z, -3.0, 3.0))

    st = stress if stress is not None else None
    if st is None and isinstance(payload.get("stress"), dict):
        st = payload["stress"]
    s0, s1, s2 = stress_probs_vector(st)
    if payload.get("stress_prob_0") is not None:
        s0 = _sf(payload, "stress_prob_0", s0)
        s1 = _sf(payload, "stress_prob_1", s1)
        s2 = _sf(payload, "stress_prob_2", s2)
        sm = s0 + s1 + s2
        if sm > 1e-6:
            s0, s1, s2 = s0 / sm, s1 / sm, s2 / sm

    ia = _sf(payload, "inrae_in_alleys_per_hour")
    ir = _sf(payload, "inrae_rest_per_hour")
    ie = _sf(payload, "inrae_eat_per_hour")
    ial = _sf(payload, "inrae_activity_level_normalized")
    ioe = _sf(payload, "inrae_oestrus")
    ica = _sf(payload, "inrae_calving")
    ila = _sf(payload, "inrae_lameness")
    ims = _sf(payload, "inrae_mastitis")
    iod = _sf(payload, "inrae_other_disease")

    hs, hc, ds, dc = _time_encodings(ts)

    vec = np.array(
        [
            _clip_obs(ax / 12.0),
            _clip_obs(ay / 12.0),
            _clip_obs(az / 12.0),
            _clip_obs(mag / 12.0),
            _clip_obs(accel_std / 4.0),
            _clip_obs(roll / math.pi),
            _clip_obs(pitch / math.pi),
            _clip_obs(yaw / math.pi),
            _clip_obs(lag1 / 12.0),
            _clip_obs(lag2 / 12.0),
            _clip_obs(lag3 / 12.0),
            _clip_obs((cbt - 38.5) / 1.2),
            _clip_obs((barn_t - 18.0) / 14.0),
            _clip_obs((barn_h - 55.0) / 30.0),
            _clip_obs((thi_v - 62.0) / 22.0),
            _clip_obs((pb - 4.0) / 2.5),
            _clip_obs((ps - 1.0) / 1.0),
            _clip_obs(milk_z / 1.5),
            _clip_obs((s0 - 0.33) * 3.0),
            _clip_obs((s1 - 0.33) * 3.0),
            _clip_obs((s2 - 0.33) * 3.0),
            _clip_obs(ia / 6.0),
            _clip_obs(ir / 6.0),
            _clip_obs(ie / 6.0),
            _clip_obs(ial),
            _clip_obs(ioe),
            _clip_obs(ica),
            _clip_obs(ila),
            _clip_obs(ims),
            _clip_obs(iod),
            _clip_obs(hs),
            _clip_obs(hc),
            _clip_obs(ds),
            _clip_obs(dc),
        ],
        dtype=np.float32,
    )

    meta = {
        "cow_id": cow,
        "timestamp_source": payload.get("timestamp"),
        "accel_mag": mag,
        "accel_mag_lag1": lag1,
        "thi": float(thi_v),
    }
    return vec, meta
