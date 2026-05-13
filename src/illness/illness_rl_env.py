"""
illness_rl_env.py
-----------------
Gymnasium environment for RL-based illness prediction.

Speed optimisations vs original:
  - All INRAE data converted to numpy arrays at init — no pandas iloc per step
  - State vector built from numpy slices — no dict allocation per step
  - CSV loaded once and shared across vectorised envs via _init_from_df()
  - No date parsing at runtime
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .illness_labels import (
    build_episode_dataset,
    build_reward,
    get_label_from_row,
    load_inrae,
)
from build_rl_state import FEATURE_NAMES, RunningNormalizer


# ---------------------------------------------------------------------------
# Column index map — built once from INRAE df column order
# ---------------------------------------------------------------------------

_INRAE_FLOAT_COLS = [
    "hour",
    "IN_ALLEYS", "REST", "EAT", "ACTIVITY_LEVEL",
    "oestrus", "calving", "lameness", "mastitis", "other_disease",
    "LPS", "acidosis", "accidents", "disturbance",
    "OK", "action_gt",
]


class IllnessEnv(gym.Env):
    """
    One episode = one cow-day (up to 24 hourly steps).

    Observation: float32 vector of shape (34,)  — see FEATURE_NAMES
    Action:      Discrete(3)  — 0=Healthy  1=At-risk  2=Ill
    Reward:      from build_reward()
    """

    metadata = {"render_modes": ["human"]}

    N_FEATURES = 34
    N_ACTIONS = 3

    def __init__(self, inrae_path: str, mode: str = "offline"):
        super().__init__()
        self.inrae_path = inrae_path
        self.mode = mode

        inrae_df = load_inrae(inrae_path)
        episodes = build_episode_dataset(inrae_df)
        self._init_from_df(inrae_df, episodes, inrae_path=inrae_path)

    # ------------------------------------------------------------------
    # Shared initialisation (called by make_vec_env factory too)
    # ------------------------------------------------------------------

    def _init_from_df(self, inrae_df, episodes, inrae_path: str = ""):
        self.inrae_path = inrae_path
        self.episodes = episodes
        self.n_episodes = len(episodes)

        # ── Convert INRAE columns to a single numpy matrix for O(1) access ──
        cols_present = [c for c in _INRAE_FLOAT_COLS if c in inrae_df.columns]
        self._col_idx = {c: i for i, c in enumerate(cols_present)}
        self._inrae_matrix = inrae_df[cols_present].values.astype(np.float32)

        # ── Normalisation constants for INRAE activity columns ──
        self._act_max = max(float(inrae_df["ACTIVITY_LEVEL"].abs().max()), 1.0)
        self._rest_max = 3600.0  # seconds in an hour
        self._rest_mean = float(inrae_df["REST"].mean())
        self._rest_std = max(float(inrae_df["REST"].std()), 1.0)
        self._eat_mean = float(inrae_df["EAT"].mean())
        self._eat_std = max(float(inrae_df["EAT"].std()), 1.0)
        self._act_mean = float(inrae_df["ACTIVITY_LEVEL"].mean())
        self._act_std = max(float(inrae_df["ACTIVITY_LEVEL"].std()), 1.0)

        # ── Gym spaces ──
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.N_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        # ── Runtime state ──
        self.normalizer = RunningNormalizer(self.N_FEATURES)
        self._episode_order = np.arange(self.n_episodes)
        np.random.shuffle(self._episode_order)
        self._order_pos = 0
        self.episode_idx = 0
        self.step_idx = 0
        self._current_ep: Optional[dict] = None
        self._lag_buffer: deque = deque([0.0] * 3, maxlen=3)
        self._last_obs = np.zeros(self.N_FEATURES, dtype=np.float32)
        self._ill_miss_streak = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Advance episode pointer (shuffle when exhausted)
        if self._order_pos >= self.n_episodes:
            np.random.shuffle(self._episode_order)
            self._order_pos = 0

        self.episode_idx = int(self._episode_order[self._order_pos])
        self._order_pos += 1
        self.step_idx = 0
        self._lag_buffer = deque([0.0] * 3, maxlen=3)
        self._current_ep = self.episodes[self.episode_idx]
        self._ill_miss_streak = 0

        obs = self._get_obs()
        self._last_obs = obs
        info = self._step_info()
        return obs, info

    def step(self, action: int):
        ep = self._current_ep
        gt_action = self._get_gt_action()
        if gt_action == 2 and int(action) != 2:
            self._ill_miss_streak += 1
        else:
            self._ill_miss_streak = 0

        reward = float(
            build_reward(
                int(action),
                int(gt_action),
                miss_streak=self._ill_miss_streak,
            )
        )

        self.step_idx += 1
        terminated = self.step_idx >= ep["n_hours"]
        truncated = False

        if not terminated:
            obs = self._get_obs()
        else:
            obs = self._last_obs.copy()

        self._last_obs = obs
        info = {
            "gt_action": int(gt_action),
            "cow": ep["cow"],
            "date": ep["date"],
            "hour": self._current_hour(),
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        ep = self._current_ep
        gt = self._get_gt_action()
        print(
            f"Ep {self.episode_idx:5d} | Cow {ep['cow']} | "
            f"Step {self.step_idx:2d}/{ep['n_hours']} | GT={gt}"
        )

    # ------------------------------------------------------------------
    # Fast internal helpers — all numpy, no pandas
    # ------------------------------------------------------------------

    def _row_vec(self) -> np.ndarray:
        """Return the current INRAE row as a float32 numpy vector."""
        ep = self._current_ep
        row_idx = ep["start_idx"] + min(self.step_idx, ep["n_hours"] - 1)
        return self._inrae_matrix[row_idx]

    def _get_gt_action(self) -> int:
        vec = self._row_vec()
        idx = self._col_idx.get("action_gt", -1)
        if idx < 0:
            return 0
        return int(vec[idx])

    def _current_hour(self) -> int:
        vec = self._row_vec()
        idx = self._col_idx.get("hour", -1)
        if idx < 0:
            return int(self.step_idx % 24)
        return int(vec[idx])

    def _step_info(self) -> dict:
        ep = self._current_ep
        return {
            "gt_action": int(self._get_gt_action()),
            "cow": ep["cow"],
            "date": ep["date"],
            "hour": self._current_hour(),
        }

    def _get_obs(self) -> np.ndarray:
        """Build the 34-feature state vector from numpy — no pandas overhead."""
        vec = self._row_vec()

        def _col(name: str, default: float = 0.0) -> float:
            idx = self._col_idx.get(name, -1)
            return float(vec[idx]) if idx >= 0 else default

        # ── IMU (synthetic from activity level) ──
        activity = _col("ACTIVITY_LEVEL")
        rest = _col("REST")
        eat = _col("EAT")
        accel_mag_mean = max(0.1, 9.8 - abs(activity) * 0.0003)
        accel_mag_std = 0.3 + np.random.normal(0, 0.05)

        # lag buffer update
        self._lag_buffer.append(accel_mag_mean)
        lag1, lag2, lag3 = list(self._lag_buffer)

        # ── Temperature proxy (no gt leakage) ──
        rest_z = (rest - self._rest_mean) / self._rest_std
        act_z = (activity - self._act_mean) / self._act_std
        anomaly = float(np.clip(rest_z - act_z, -2.0, 2.0))
        cbt = 38.5 + 0.15 * anomaly + np.random.normal(0, 0.1)
        thi = 65.0
        barn_temp = 20.0
        barn_hum = 60.0

        # ── Model output proxies ──
        pred_behavior = 7.0 if rest > self._rest_mean else 2.0
        stress_raw = float(np.clip(-act_z, 0.0, 2.0)) / 2.0
        pred_stress = 2 if stress_raw > 0.55 else (1 if stress_raw > 0.25 else 0)
        stress_p0 = max(0.0, 1.0 - stress_raw * 1.5)
        stress_p1 = min(1.0, stress_raw * 1.0)
        stress_p2 = min(1.0, max(0.0, stress_raw - 0.3))
        milk_kg = max(0.0, 25.0 + act_z * 2.0 + np.random.normal(0, 0.5))

        # ── INRAE activity (normalised) ──
        in_alleys_n = _col("IN_ALLEYS") / self._rest_max
        rest_n = rest / self._rest_max
        eat_n = _col("EAT") / self._rest_max
        act_n = activity / self._act_max

        # ── Time context ──
        hour = float(self._current_hour() % 24)
        dow = float(int(self.step_idx / 24) % 7)
        hour_sin = math.sin(hour * 2 * math.pi / 24)
        hour_cos = math.cos(hour * 2 * math.pi / 24)
        dow_sin = math.sin(dow * 2 * math.pi / 7)
        dow_cos = math.cos(dow * 2 * math.pi / 7)

        # ── Assemble 34-feature vector ──
        state = np.array([
            # IMU (8)
            0.0, 0.0, -9.8,
            accel_mag_mean, accel_mag_std, 0.0, 0.0, 0.0,
            # Lag (3)
            lag1, lag2, lag3,
            # Temperature (4)
            cbt, barn_temp, barn_hum, thi,
            # Model outputs (6)
            pred_behavior / 7.0, pred_stress / 2.0,
            milk_kg / 40.0,
            stress_p0, stress_p1, stress_p2,
            # INRAE activity (9)
            in_alleys_n, rest_n, eat_n, act_n,
            _col("oestrus"), _col("calving"),
            _col("lameness"), _col("mastitis"), _col("other_disease"),
            # Time (4)
            hour_sin, hour_cos, dow_sin, dow_cos,
        ], dtype=np.float32)

        # Clip to observation space bounds
        np.clip(state, -5.0, 5.0, out=state)
        return state

    def _build_state_from_row(self, row, cow_id: str, hour: int) -> np.ndarray:
        """
        Compatibility shim for train_illness_rl._collect_supervised_samples.
        row can be a pandas Series or dict.
        """

        def _get(key, default=0.0):
            try:
                return float(row[key])
            except (KeyError, TypeError):
                return default

        activity = _get("ACTIVITY_LEVEL")
        rest = _get("REST")
        accel_mag_mean = max(0.1, 9.8 - abs(activity) * 0.0003)

        act_z = (activity - self._act_mean) / self._act_std
        rest_z = (rest - self._rest_mean) / self._rest_std
        anomaly = float(np.clip(rest_z - act_z, -2.0, 2.0))
        cbt = 38.5 + 0.15 * anomaly
        pred_behavior = 7.0 if rest > self._rest_mean else 2.0
        stress_raw = float(np.clip(-act_z, 0.0, 2.0)) / 2.0
        pred_stress = 2 if stress_raw > 0.55 else (1 if stress_raw > 0.25 else 0)
        milk_kg = max(0.0, 25.0 + act_z * 2.0)

        hour_f = float(hour % 24)
        dow_f = 0.0

        state = np.array([
            0.0, 0.0, -9.8,
            accel_mag_mean, 0.3, 0.0, 0.0, 0.0,
            accel_mag_mean, accel_mag_mean, accel_mag_mean,
            cbt, 20.0, 60.0, 65.0,
            pred_behavior / 7.0, pred_stress / 2.0, milk_kg / 40.0,
            1.0 - pred_stress * 0.4, pred_stress * 0.3, pred_stress * 0.1,
            _get("IN_ALLEYS") / 3600.0,
            rest / 3600.0,
            _get("EAT") / 3600.0,
            activity / self._act_max,
            _get("oestrus"), _get("calving"),
            _get("lameness"), _get("mastitis"), _get("other_disease"),
            math.sin(hour_f * 2 * math.pi / 24),
            math.cos(hour_f * 2 * math.pi / 24),
            math.sin(dow_f * 2 * math.pi / 7),
            math.cos(dow_f * 2 * math.pi / 7),
        ], dtype=np.float32)

        np.clip(state, -5.0, 5.0, out=state)
        return self.normalizer.transform(state)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("data/inrae_activity/*.csv")

    print("=== IllnessEnv smoke test ===")
    env = IllnessEnv(paths)

    for ep_n in range(3):
        obs, info = env.reset()
        total_r = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            done = terminated or truncated
        print(f"Episode {ep_n+1}: {steps} steps, total reward={total_r:.1f}, cow={info['cow']}")

    print("Smoke test passed.")
