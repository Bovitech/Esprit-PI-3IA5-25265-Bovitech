# Illness Feature Guide

This folder is the single home for the illness feature.
It contains everything needed to understand the model, run inference, explain results, retrain the policy, and hand the feature to another developer.

## Overview

The illness system is a near-real-time decision support pipeline for cows.
It combines sensor, behavior, environment, and herd-health signals and classifies each cow into one of three states.

| Class | Meaning | Use |
|---|---|---|
| `0` | Healthy | Normal operating state |
| `1` | At-risk | Early warning state |
| `2` | Ill | Strong illness signal |

The goal is not to diagnose disease automatically.
The goal is to surface early warning signs, provide a clear explanation, and give operators enough context to review the result.

## Folder Contents

This folder now holds both the runtime code and the training code.

| File | Type | Purpose |
|---|---|---|
| `health_score.py` | Runtime | Builds the rolling health score and risk summary |
| `illness_labels.py` | Runtime + training support | Loads INRAE data and creates labels, rewards, and episode metadata |
| `illness_rl_env.py` | Training support | Defines the RL environment used for training and evaluation |
| `illness_xai.py` | Runtime | Generates SHAP-based explanations and counterfactual text |
| `train_illness_rl.py` | Training | Trains the illness model end to end |

## Runtime Architecture

The backend uses the illness feature in this order:

| Step | Component | Output |
|---|---|---|
| 1 | Backend receives a payload | Sensor and herd context |
| 2 | `build_state_vector(...)` | 34-feature illness state |
| 3 | PPO policy | One action: Healthy, At-risk, or Ill |
| 4 | Policy logits | Action probabilities |
| 5 | `IllnessExplainer` | Top features, human explanation, counterfactual |
| 6 | `build_temporal_health_score(...)` | Rolling health score and risk summary |

## Model Inputs

The illness model does not rely on one raw sensor alone.
It uses a fused state built from several signal groups.

| Signal Group | Examples | Why It Matters |
|---|---|---|
| IMU motion features | accelerometer summary, lagged movement, posture proxy | Captures behavior changes and short-term motion trends |
| Stress signal | predicted stress class and stress probabilities | Adds a higher-level physiological signal |
| Milk trend | daily production estimate and z-score | Helps detect production-related health changes |
| Barn environment | temperature, humidity, THI | Adds heat stress and environment context |
| Behavioral context | predicted behavior class | Helps separate resting, eating, and abnormal activity patterns |
| Health flags | lameness, mastitis, other disease, calving, oestrus | Adds explicit herd-health information |
| Time context | cyclical hour and day encoding | Preserves daily and weekly rhythm without discontinuities |

The canonical 34-feature state is defined in `src/build_rl_state.py`.

## Explainability

The illness feature is designed to be interpretable, not just predictive.

| Explainability Element | Description | Practical Meaning |
|---|---|---|
| SHAP attribution | `IllnessExplainer` computes SHAP values for the selected action | Shows which features pushed the prediction up or down |
| Positive SHAP | Feature supports the predicted class | Strengthens the current decision |
| Negative SHAP | Feature pushes away from the predicted class | Weakens the current decision |
| Human explanation | Short sentence built from the top drivers | Makes the result easy to read in the UI |
| Counterfactual | Small change that could shift the class | Helps a user understand what would alter the result |

## Health Score

`health_score.py` turns the current illness prediction into a more user-friendly temporal health score.

| Health Score Output | Purpose |
|---|---|
| 0-100 style risk level | Gives a quick intuitive state |
| Trend over time | Shows whether the cow is improving or declining |
| Risk signals | Surfaces issues like milk drop, elevated temperature, or declining behavior |
| Consolidated state | Helps the app show one summary instead of only a raw model action |

This score is operational context, not a veterinary diagnosis.

## Training Pipeline

The training script uses two phases.

| Phase | Name | What Happens | Outputs |
|---|---|---|---|
| 1 | Supervised warm-start | Loads INRAE activity data, builds training episodes, extracts state/label samples, and trains a small MLP | `warmstart_mlp.pt`, `warmstart_states.npy` |
| 2 | PPO fine-tuning | Creates the RL environment, copies warm-start weights into PPO, and trains the final policy | `illness_ppo.zip`, `eval_metrics.json` |

## Files to Keep or Deploy

| Use Case | Files |
|---|---|
| Required at runtime | `artifacts/illness_model/illness_ppo.zip` |
| Recommended for explainability | `artifacts/illness_model/warmstart_states.npy` |
| Useful for retraining | `src/illness/train_illness_rl.py`, `src/illness/illness_rl_env.py`, `src/illness/illness_labels.py` |

## How to Run Training

Run from the repository root:

```bash
python src/illness/train_illness_rl.py --inrae-path data/inrae_activity/dataset1-1.csv data/inrae_activity/dataset2-1.csv --timesteps 50000 --warmstart-epochs 5 --out-dir artifacts/illness_model
```

You can add more CSV files to `--inrae-path` if you want to train on more data.

## Backend Output Fields

The backend loads the illness runtime at startup and serves the feature through `POST /predict/illness`.

| Field | Meaning |
|---|---|
| `predicted_action` | Numeric class selected by the policy |
| `action_name` | Human-readable class name |
| `confidence` | Probability of the selected action |
| `top_features` | Ranked explanation features |
| `human_explanation` | Short readable explanation |
| `counterfactual` | What change could alter the result |
| `all_action_probs` | Full class probability distribution |
| `health_score` | Temporal health score |
| `health_trend` | Trend direction and change |
| `risk_signals` | Extra health-related observations |

## Example Request Shape

The app sends a payload with fields like:

| Field Group | Examples |
|---|---|
| Core identifiers | `cow_id`, `timestamp` |
| IMU features | motion and lag features |
| Model predictions | stress, milk, and behavior predictions |
| Barn data | temperature, humidity, THI |
| INRAE flags | lameness, mastitis, other disease, calving, oestrus |

The exact payload schema is documented in the app screens and backend request handling.

## Example Output Meaning

If the model returns `At-risk` with `confidence = 0.68` and the top features include stress probability and movement lag, that means the PPO policy saw a moderate risk pattern and the explanation shows which signals pushed the decision.

This is still decision support only.

## GitHub Sharing Guide

If you want the project to be reusable by your teammate, commit the runtime package and the trained model.

| Scenario | What to Include |
|---|---|
| App only | `src/illness/`, `artifacts/illness_model/illness_ppo.zip` |
| App + retraining support | `src/illness/`, `src/illness/train_illness_rl.py`, `artifacts/illness_model/illness_ppo.zip` |
| Full reproduction | Above items plus the INRAE CSV training data, if your team is allowed to share it |

Do not commit large generated logs, virtual environments, or temporary caches.

## Deployment Checklist

| Check | Expected Result |
|---|---|
| Model file exists | `artifacts/illness_model/illness_ppo.zip` is present |
| Backend import path | Backend can import from `illness.*` |
| Health endpoint | `/health` confirms the illness runtime is loaded |
| Prediction endpoint | `POST /predict/illness` returns a valid response |
| Response completeness | Prediction, probabilities, explanation, and health score are present |

## Limitations

| Limitation | Notes |
|---|---|
| Not a diagnosis | This is not a veterinary diagnosis |
| Explanation scope | SHAP explains model behavior, not biological causality |
| Data drift | Performance can drift if sensors, barns, or herd patterns change |
| History persistence | The health score is not persisted across restarts unless stored externally |

## Practical Summary

| Need | Minimum Files |
|---|---|
| Run the app | `src/illness/`, `artifacts/illness_model/illness_ppo.zip` |
| Retrain the model | `src/illness/train_illness_rl.py`, `src/illness/illness_rl_env.py`, `src/illness/illness_labels.py`, INRAE CSV data |

