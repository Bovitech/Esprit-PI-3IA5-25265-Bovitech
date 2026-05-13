from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch as th

try:
    import shap
except ImportError as exc:  # pragma: no cover - runtime dependency guard.
    shap = None  # type: ignore[assignment]
    _SHAP_IMPORT_ERROR = exc

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - runtime dependency guard.
    PPO = None  # type: ignore[assignment]
    _SB3_IMPORT_ERROR = exc

from build_rl_state import FEATURE_NAMES as RL_FEATURE_NAMES


ACTION_NAMES = {0: "Healthy", 1: "At-risk", 2: "Ill"}
PLAIN_ENGLISH_FEATURES = {
    "cbt_temp_c": "body temperature",
    "thi": "heat index",
    "pred_stress_norm": "stress level",
    "pred_behavior_norm": "behavior signal",
    "accel_mag_mean": "movement level",
    "REST": "resting time",
    "inrae_rest_per_hour": "resting time",
    "lameness": "lameness flag",
    "inrae_lameness": "lameness flag",
    "mastitis": "mastitis flag",
    "inrae_mastitis": "mastitis flag",
    "other_disease": "other disease flag",
    "inrae_other_disease": "other disease flag",
    "inrae_oestrus": "oestrus flag",
    "inrae_calving": "calving flag",
    "barn_temp_c": "barn temperature",
    "barn_humidity": "barn humidity",
    "milk_kg_day_zscore": "milk yield trend",
}


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        if np.isnan(value):
            return default  # type: ignore[arg-type]
        return float(value)
    except (TypeError, ValueError):
        return default


class IllnessExplainer:
    """SHAP-based explanation helper for the illness PPO agent.

    Parameters
    ----------
    ppo_model_path:
        Path to a saved Stable-Baselines3 PPO model.
    feature_names:
        Ordered list of the 34 RL feature names.
    training_states_array:
        Optional background sample shaped `(N, 34)`. If omitted, the class tries
        to load `artifacts/illness_model/warmstart_states.npy`.
    background_path:
        Optional explicit path to a `.npy` background file.
    """

    def __init__(
        self,
        ppo_model_path: str | Path,
        feature_names: Iterable[str],
        training_states_array: Optional[np.ndarray] = None,
        background_path: str | Path | None = None,
    ) -> None:
        if shap is None:
            raise ImportError(f"shap is required for IllnessExplainer: {_SHAP_IMPORT_ERROR}")
        if PPO is None:
            raise ImportError(f"stable-baselines3 is required for IllnessExplainer: {_SB3_IMPORT_ERROR}")

        self.ppo_model_path = Path(ppo_model_path)
        self.feature_names = list(feature_names)
        if len(self.feature_names) != 34:
            raise ValueError(f"Expected 34 feature names, got {len(self.feature_names)}")

        self.model = PPO.load(str(self.ppo_model_path), device="cpu")
        self.model.policy.set_training_mode(False)

        self.background = self._load_background(training_states_array, background_path)
        self.background_stats = self._background_statistics(self.background)
        background_sample = shap.sample(self.background, min(100, len(self.background)))
        self.explainer = shap.KernelExplainer(self.predict_fn, background_sample)

    def _load_background(
        self,
        training_states_array: Optional[np.ndarray],
        background_path: str | Path | None,
    ) -> np.ndarray:
        if training_states_array is not None:
            background = np.asarray(training_states_array, dtype=np.float32)
        else:
            candidate_paths = []
            if background_path is not None:
                candidate_paths.append(Path(background_path))
            candidate_paths.extend(
                [
                    self.ppo_model_path.with_name("warmstart_states.npy"),
                    Path("artifacts/illness_model/warmstart_states.npy"),
                ]
            )
            background = None
            for candidate in candidate_paths:
                if candidate.exists():
                    background = np.load(candidate).astype(np.float32)
                    break
            if background is None:
                raise FileNotFoundError(
                    "No SHAP background sample was provided and warmstart_states.npy was not found. "
                    "Pass training_states_array or background_path explicitly."
                )

        if background.ndim != 2 or background.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Background sample must have shape (N, {len(self.feature_names)}); got {background.shape}"
            )
        return background.astype(np.float32, copy=False)

    def _background_statistics(self, background: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "mean": np.mean(background, axis=0),
            "std": np.std(background, axis=0),
        }

    def _policy_action_probs(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected state vectors with {len(self.feature_names)} features; got {states.shape[1]}")

        obs_tensor = th.tensor(states, dtype=th.float32)
        with th.no_grad():
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi, _ = self.model.policy.mlp_extractor(features)
            dist = self.model.policy._get_action_dist_from_latent(latent_pi)
            probs = dist.distribution.probs
            if hasattr(probs, "detach"):
                probs = probs.detach()
            return probs.cpu().numpy().astype(np.float32)

    def predict_fn(self, states: np.ndarray) -> np.ndarray:
        """Return PPO action probabilities for SHAP (N, 34) -> (N, 3)."""
        return self._policy_action_probs(states)

    def _feature_display_name(self, feature_name: str) -> str:
        return PLAIN_ENGLISH_FEATURES.get(feature_name, feature_name.replace("_", " "))

    def _is_continuous_feature(self, feature_name: str) -> bool:
        disallowed = {
            "inrae_oestrus",
            "inrae_calving",
            "inrae_lameness",
            "inrae_mastitis",
            "inrae_other_disease",
        }
        return feature_name not in disallowed

    def _format_value(self, feature_name: str, value: float) -> str:
        if feature_name == "cbt_temp_c":
            return f"{value:.1f}°C"
        if feature_name in {"barn_temp_c"}:
            return f"{value:.1f}°C"
        if feature_name in {"barn_humidity"}:
            return f"{value:.1f}%"
        if feature_name == "thi":
            return f"{value:.1f}"
        if feature_name in {"pred_behavior_norm", "pred_stress_norm", "milk_kg_day_zscore"}:
            return f"{value:.3f}"
        return f"{value:.3f}"

    def _shap_values_for_state(self, state_vector: np.ndarray, nsamples: int = 200):
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)
        shap_values = self.explainer.shap_values(state, nsamples=nsamples)
        if isinstance(shap_values, list):
            return shap_values
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        return [np.asarray(shap_values)]

    def explain(self, state_vector: np.ndarray) -> dict[str, Any]:
        """Explain one RL state with SHAP and return a structured summary."""
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)
        probs = self.predict_fn(state)
        predicted_action = int(np.argmax(probs[0]))
        confidence = float(probs[0, predicted_action])

        shap_values = self._shap_values_for_state(state, nsamples=200)
        action_shap = np.asarray(shap_values[predicted_action])[0]
        ranking = np.argsort(np.abs(action_shap))[::-1]

        top_features: list[dict[str, Any]] = []
        for idx in ranking[:5]:
            feature_name = self.feature_names[int(idx)]
            value = float(state[0, int(idx)])
            shap_value = float(action_shap[int(idx)])
            top_features.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "shap": shap_value,
                    "direction": "+" if shap_value >= 0 else "-",
                }
            )

        positive = [item for item in top_features if item["shap"] > 0]
        if len(positive) < 2:
            positive = sorted(
                (
                    {
                        "feature": self.feature_names[int(idx)],
                        "value": float(state[0, int(idx)]),
                        "shap": float(action_shap[int(idx)]),
                        "direction": "+" if action_shap[int(idx)] >= 0 else "-",
                    }
                    for idx in ranking
                ),
                key=lambda item: item["shap"],
                reverse=True,
            )

        top1 = positive[0] if positive else top_features[0]
        top2 = positive[1] if len(positive) > 1 else top_features[1]
        human_explanation = (
            f"Alert triggered mainly by {self._feature_display_name(top1['feature'])} ({top1['direction']}) "
            f"and {self._feature_display_name(top2['feature'])} ({top2['direction']})."
        )

        return {
            "predicted_action": predicted_action,
            "action_name": ACTION_NAMES.get(predicted_action, str(predicted_action)),
            "confidence": confidence,
            "top_features": top_features,
            "human_explanation": human_explanation,
        }

    def counterfactual(self, state_vector: np.ndarray, target_action: int = 1) -> str:
        """Find a small one-feature perturbation that flips the PPO prediction.

        The search only touches continuous features and avoids binary illness flags.
        """
        state = np.asarray(state_vector, dtype=np.float32).reshape(-1).copy()
        target_action = int(target_action)
        current_probs = self.predict_fn(state.reshape(1, -1))[0]
        current_action = int(np.argmax(current_probs))
        if current_action == target_action:
            return f"The state already predicts {ACTION_NAMES.get(target_action, target_action)}."

        shap_values = self._shap_values_for_state(state, nsamples=200)
        target_shap = np.asarray(shap_values[target_action])[0]

        best: Optional[tuple[float, str, float, float]] = None
        fallback: Optional[tuple[float, str, float, float, float]] = None

        for idx, feature_name in enumerate(self.feature_names):
            if not self._is_continuous_feature(feature_name):
                continue

            direction = 1.0 if target_shap[idx] >= 0 else -1.0
            base_step = float(max(0.05, 0.25 * float(self.background_stats["std"][idx]), 0.1 * abs(state[idx]) if abs(state[idx]) > 0 else 0.05))
            for mult in (1, 2, 3, 4, 5, 6):
                candidate = state.copy()
                candidate[idx] = candidate[idx] + direction * base_step * mult
                candidate_probs = self.predict_fn(candidate.reshape(1, -1))[0]
                candidate_action = int(np.argmax(candidate_probs))
                delta = abs(float(candidate[idx] - state[idx]))
                if candidate_action == target_action:
                    if best is None or delta < best[0]:
                        best = (delta, feature_name, float(state[idx]), float(candidate[idx]))
                    break
                score = float(candidate_probs[target_action])
                if fallback is None or score > fallback[4] or (np.isclose(score, fallback[4]) and delta < fallback[0]):
                    fallback = (delta, feature_name, float(state[idx]), float(candidate[idx]), score)

        if best is not None:
            _, feature_name, old_value, new_value = best
            return (
                f"If {self._feature_display_name(feature_name)} were {self._format_value(feature_name, new_value)} "
                f"instead of {self._format_value(feature_name, old_value)}, prediction would be "
                f"{ACTION_NAMES.get(target_action, target_action)}."
            )

        if fallback is not None:
            _, feature_name, old_value, new_value, score = fallback
            return (
                f"A small change in {self._feature_display_name(feature_name)} from "
                f"{self._format_value(feature_name, old_value)} to {self._format_value(feature_name, new_value)} "
                f"would move the prediction toward {ACTION_NAMES.get(target_action, target_action)} (target prob ~{score:.3f})."
            )

        return f"No single-feature counterfactual found for {ACTION_NAMES.get(target_action, target_action)}."


def _demo() -> None:
    default_model = Path("artifacts/illness_model/illness_ppo.zip")
    default_background = Path("artifacts/illness_model/warmstart_states.npy")

    if not default_model.exists():
        print(f"Demo skipped: PPO model not found at {default_model}")
        return

    feature_names = list(RL_FEATURE_NAMES)
    if default_background.exists():
        background = np.load(default_background).astype(np.float32)
    else:
        rng = np.random.default_rng(7)
        background = rng.normal(size=(100, len(feature_names))).astype(np.float32)

    explainer = IllnessExplainer(default_model, feature_names, training_states_array=background)
    rng = np.random.default_rng(123)
    state = rng.normal(size=(len(feature_names),)).astype(np.float32)
    explanation = explainer.explain(state)
    counterfactual = explainer.counterfactual(state, target_action=1)

    print("Predicted action:", explanation["predicted_action"])
    print("Action name:", explanation["action_name"])
    print("Confidence:", explanation["confidence"])
    print("Top features:")
    for item in explanation["top_features"]:
        print(item)
    print("Human explanation:", explanation["human_explanation"])
    print("Counterfactual:", counterfactual)


if __name__ == "__main__":
    _demo()
