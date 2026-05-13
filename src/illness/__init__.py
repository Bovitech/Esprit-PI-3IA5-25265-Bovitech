"""Illness feature package (lazy: training modules import subpackages directly)."""

from .health_score import build_temporal_health_score

__all__ = ["build_temporal_health_score"]
