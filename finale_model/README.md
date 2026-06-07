# ML runtime artifacts

This folder is **empty in Git**. On first run, the inference API downloads required files from `models.manifest.json` (Hugging Face).

Manual download:

```bash
python scripts/download_models.py
```

To skip auto-download: set `SKIP_MODEL_DOWNLOAD=1` in `.env`.
