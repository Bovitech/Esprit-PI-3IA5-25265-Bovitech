from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


_SCORE_HISTORY: dict[str, deque[dict[str, Any]]] = {}


def _cow_key(cow_id: Any) -> str:
    return str(cow_id or "C01").strip().upper()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp_to_datetime(timestamp: Any) -> datetime:
    if isinstance(timestamp, datetime):
        return timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    if isinstance(timestamp, (int, float, np.integer, np.floating)):
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    raw = str(timestamp).strip()
    if not raw:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _day_key(timestamp: Any) -> str:
    return _timestamp_to_datetime(timestamp).date().isoformat()


def _history_for(cow_id: Any) -> deque[dict[str, Any]]:
    key = _cow_key(cow_id)
    if key not in _SCORE_HISTORY:
        _SCORE_HISTORY[key] = deque(maxlen=14)
    return _SCORE_HISTORY[key]


def _hist_value(history: deque[dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not history:
        return default
    return _safe_float(history[-1].get(key), default)


def _format_value(value: float, suffix: str = "") -> str:
    if suffix == "%":
        return f"{value:.0f}%"
    if suffix == "°C":
        return f"{value:.1f}°C"
    if suffix == "z":
        return f"{value:+.1f} z-score"
    return f"{value:.1f}{suffix}"


def _score_band(score: float) -> str:
    if score >= 75:
        return "healthy"
    if score >= 55:
        return "watch"
    if score >= 35:
        return "risk"
    return "critical"


def _feature_names(top_features: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for item in top_features or []:
        feature = str(item.get("feature", "")).strip().lower()
        if feature:
            names.add(feature)
    return names


def _disease_hints(
    top_features: list[dict[str, Any]] | None,
    inrae_row: Mapping[str, Any] | None,
    cbt_temp_c: float,
    stress_prob: float,
) -> list[str]:
    feature_names = _feature_names(top_features)
    row = dict(inrae_row or {})
    hints: list[str] = []

    if _safe_float(row.get("lameness"), 0.0) > 0 or "lameness" in feature_names:
        hints.append("Lameness/Pain")
    if _safe_float(row.get("mastitis"), 0.0) > 0 or "mastitis" in feature_names:
        hints.append("Mastitis/Ketosis")
    if _safe_float(row.get("other_disease"), 0.0) > 0:
        hints.append("Possible infection")
    if cbt_temp_c >= 39.1 or "body temperature" in feature_names or "cbt temp c" in feature_names:
        hints.append("Possible infection")
    if stress_prob >= 0.6 or "stress level" in feature_names:
        hints.append("Confirm with vet")

    deduped: list[str] = []
    for item in hints:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _suggested_action(score: float, illness_prob: float, cbt_temp_c: float, hints: list[str]) -> str:
    if score < 40 or illness_prob >= 0.55 or cbt_temp_c >= 39.2:
        return "Veterinary check recommended"
    if score < 60 or hints:
        return "Increase monitoring and repeat check within 24h"
    return "Routine observation"


def _risk_signals(
    history: deque[dict[str, Any]],
    behavior_pred: float,
    milk_kg: float,
    stress_prob: float,
    cbt_temp_c: float,
) -> list[dict[str, str]]:
    previous_milk = _hist_value(history, "milk_kg", milk_kg)
    previous_temp = _hist_value(history, "cbt_temp_c", cbt_temp_c)
    milk_delta = milk_kg - previous_milk
    temp_delta = cbt_temp_c - previous_temp

    lying_pct = 78.0 if int(round(behavior_pred)) == 7 else 42.0
    milk_z = (milk_delta / max(abs(previous_milk), 1.0)) * 2.5

    return [
        {
            "label": "Lying time",
            "value": _format_value(lying_pct, "%") + " of last 24h",
            "trend": "↑ abnormal" if lying_pct >= 70 else "→ normal",
            "hint": "Lameness/Pain" if lying_pct >= 70 else "Monitor",
        },
        {
            "label": "Milk trend",
            "value": _format_value(milk_z, "z"),
            "trend": "↓ dropping" if milk_delta < 0 else "→ stable",
            "hint": "Mastitis/Ketosis" if milk_delta < 0 else "Routine",
        },
        {
            "label": "Stress level",
            "value": _format_value(stress_prob * 100.0, "%") + " probability",
            "trend": "↑ elevated" if stress_prob >= 0.55 else "→ normal",
            "hint": "Confirm with vet" if stress_prob >= 0.55 else "Observe",
        },
        {
            "label": "Body temp",
            "value": _format_value(cbt_temp_c, "°C"),
            "trend": "↑ elevated" if cbt_temp_c >= 39.0 else "→ normal",
            "hint": "Possible infection" if cbt_temp_c >= 39.0 or temp_delta > 0.2 else "Observe",
        },
    ]


def _trend_summary(history: deque[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"label": "no trend yet", "direction": "flat", "days": 0, "delta": 0.0}

    recent = list(history)[-3:]
    scores = [float(item.get("score", 0.0)) for item in recent]
    if len(scores) >= 3 and scores[0] > scores[1] > scores[2]:
        return {
            "label": "declining 3 days",
            "direction": "down",
            "days": 3,
            "delta": float(scores[-1] - scores[0]),
        }
    if len(scores) >= 2 and scores[-1] < scores[0]:
        return {
            "label": "declining 2 days",
            "direction": "down",
            "days": 2,
            "delta": float(scores[-1] - scores[0]),
        }
    if len(scores) >= 2 and scores[-1] > scores[0]:
        return {
            "label": "recovering",
            "direction": "up",
            "days": min(3, len(scores)),
            "delta": float(scores[-1] - scores[0]),
        }
    return {"label": "stable", "direction": "flat", "days": 1, "delta": 0.0}


def build_temporal_health_score(
    *,
    cow_id: Any,
    timestamp: Any,
    predicted_action: int,
    confidence: float,
    illness_probs: Mapping[str, Any],
    model_predictions: Mapping[str, Any],
    top_features: list[dict[str, Any]] | None = None,
    inrae_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse the illness, stress, milk, behavior, and rule signals into a temporal score."""
    dt = _timestamp_to_datetime(timestamp)
    day_key = dt.date().isoformat()
    history = _history_for(cow_id)

    illness_prob = _safe_float(illness_probs.get("Ill"), 0.0)
    at_risk_prob = _safe_float(illness_probs.get("At-risk"), 0.0)
    healthy_prob = _safe_float(illness_probs.get("Healthy"), 0.0)

    behavior_pred = _safe_float(model_predictions.get("pred_behavior"), 0.0)
    milk_kg = _safe_float(model_predictions.get("prediction_milk_kg_day"), 0.0)
    stress_prob_2 = _safe_float(model_predictions.get("stress_prob_2"), 0.0)
    stress_pred = _safe_float(model_predictions.get("pred_stress"), 0.0)
    cbt_temp_c = _safe_float(
        model_predictions.get("cbt_temp_c"),
        _safe_float(model_predictions.get("cbt"), 38.5),
    )

    behavior_risk = 1.0 if int(round(behavior_pred)) == 7 else 0.35
    stress_risk = max(stress_prob_2, stress_pred / 2.0)

    previous = history[-1] if history else None
    previous_milk = _safe_float(previous.get("milk_kg"), milk_kg) if previous else milk_kg
    previous_temp = _safe_float(previous.get("cbt_temp_c"), cbt_temp_c) if previous else cbt_temp_c
    milk_drop = max(0.0, previous_milk - milk_kg)
    temp_risk = max(0.0, cbt_temp_c - 38.7) / 1.0
    milk_risk = min(1.0, milk_drop / max(previous_milk, 1.0))

    rule_hints = _disease_hints(top_features, inrae_row, cbt_temp_c, stress_prob_2)
    rule_risk = 0.15 if rule_hints else 0.0

    weighted_risk = (
        0.32 * illness_prob
        + 0.18 * stress_risk
        + 0.18 * temp_risk
        + 0.16 * milk_risk
        + 0.10 * behavior_risk
        + 0.06 * rule_risk
    )
    score = float(np.clip(100.0 * (1.0 - weighted_risk), 0.0, 100.0))

    daily_record = {
        "date": day_key,
        "score": round(score, 1),
        "milk_kg": round(milk_kg, 3),
        "cbt_temp_c": round(cbt_temp_c, 3),
        "illness_prob": round(illness_prob, 4),
    }

    if history and history[-1]["date"] == day_key:
        prev = history[-1]
        blended_score = round(0.7 * float(prev.get("score", score)) + 0.3 * score, 1)
        history[-1] = {**prev, **daily_record, "score": blended_score}
    else:
        history.append(daily_record)

    trend = _trend_summary(history)
    signals = _risk_signals(history, behavior_pred, milk_kg, stress_prob_2, cbt_temp_c)

    if not rule_hints and illness_prob >= 0.35:
        rule_hints = ["Confirm with vet"]

    action = _suggested_action(score, illness_prob, cbt_temp_c, rule_hints)
    status = _score_band(score)

    return {
        "health_score": int(round(score)),
        "health_status": status,
        "health_trend": trend,
        "score_history": list(history),
        "risk_signals": signals,
        "suggested_action": action,
        "disease_hints": rule_hints,
        "score_components": {
            "illness_prob": round(illness_prob, 4),
            "at_risk_prob": round(at_risk_prob, 4),
            "healthy_prob": round(healthy_prob, 4),
            "stress_prob": round(stress_risk, 4),
            "milk_risk": round(milk_risk, 4),
            "behavior_risk": round(behavior_risk, 4),
            "temp_risk": round(temp_risk, 4),
            "rule_risk": round(rule_risk, 4),
        },
        "last_updated": dt.isoformat(),
        "current_day": day_key,
        "previous_day_score": _safe_float(previous.get("score"), score) if previous else score,
        "behavior_pred": int(round(behavior_pred)),
        "milk_kg_day": round(milk_kg, 3),
        "cbt_temp_c": round(cbt_temp_c, 3),
        "stress_prob_2": round(stress_prob_2, 4),
        "stress_pred": int(round(stress_pred)),
        "illness_prob": round(illness_prob, 4),
        "confidence": round(confidence, 4),
    }
