"""Illness feature package.

This package keeps the illness-specific runtime and training helpers together so
the main backend can import a single namespace instead of scattered root files.
"""

from .health_score import build_temporal_health_score
from .illness_labels import build_episode_dataset, build_reward, get_label_from_row, load_inrae
from .illness_rl_env import IllnessEnv
from .illness_xai import IllnessExplainer

__all__ = [
    "IllnessEnv",
    "IllnessExplainer",
    "build_episode_dataset",
    "build_reward",
    "build_temporal_health_score",
    "get_label_from_row",
    "load_inrae",
]
