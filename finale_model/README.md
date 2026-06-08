# ML runtime artifacts

This folder is **not in Git** (weights live on [Hugging Face](https://huggingface.co/Amiiimi)). After clone, download once:

```bash
python scripts/download_models.py
```

Files (see `scripts/models.manifest.json`):

| File | Hugging Face repo |
|------|-------------------|
| `behavior_rf_multimodal.joblib` | Amiiimi/Behaviour |
| `milk_xgb_pred_behavior_daily_milkhist_pipeline.joblib` | Amiiimi/MilkProduction |
| `StressDetectionV3_trained.pt` | Amiiimi/StressModel |
| `model_audio_classification (1).h5` | Amiiimi/AudioModel |
| `illness_ppo.zip` | *(optional — add URL when published)* |

To skip auto-download on API start: set `SKIP_MODEL_DOWNLOAD=1` in `.env`.
