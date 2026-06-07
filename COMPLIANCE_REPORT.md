# Repository Compliance Report

**Project:** Bovitech  
**Repository name:** `Esprit-PI-3IA5-25265-Bovitech`  
**Class:** 3IA5 · **Year:** 2025–2026  
**Guide:** ESPRIT GitHub Organization Publication Guide (2025–2026)  
**Date:** 2026-06-05

## Summary

| Area | Status |
|------|--------|
| Repository structure | ✅ Compliant |
| README quality | ✅ Compliant |
| Security audit | ✅ Compliant (fixes applied) |
| Dependencies | ✅ Compliant |
| Database readiness | ⚠️ Needs Attention |
| Documentation (`docs/`) | ✅ Compliant |
| Demo assets | ⚠️ Local run only (no screenshots/video by team choice) |
| Docker | ✅ Compliant (`docker/Dockerfile`, `docker/docker-compose.yml` — single service) |
| CI/CD | ✅ Compliant |
| AI / ML annex | ✅ Compliant |
| IoT / Embedded | ✅ Compliant |

---

## Detailed checklist

### 1. Repository structure

| Requirement | Status |
|-------------|--------|
| `README.md` | ✅ |
| `.gitignore` | ✅ |
| `.env.example` | ✅ (single file at repo root) |
| `docs/` | ✅ |
| `demo/` | ✅ |
| `src/` | ✅ |
| Dependency files | ✅ (`requirements.txt`, `PI_Backend/requirements.txt`, mobile `package.json`) |

### 2. README quality

| Section | Status |
|---------|--------|
| Description, Features, Architecture | ✅ |
| Technologies, Prerequisites, Installation | ✅ |
| Environment variables, Database, Running | ✅ |
| Demo, Project structure, Authors, License | ✅ |

### 3. Security audit

| Issue | Status |
|-------|--------|
| WiFi credentials in `thi/*.ino` | ✅ Replaced with placeholders |
| Django `SECRET_KEY` hardcoded in PI_Backend | ✅ Moved to `.env` |
| Password printed in `accounts/services.py` | ✅ Removed |
| `.env` gitignored | ✅ |
| `.env.example` without secrets | ✅ |

**Note:** If credentials were ever pushed to Git history, rotate WiFi password and regenerate Django `SECRET_KEY`.

### 4. Dependencies

| Item | Status |
|------|--------|
| Root `requirements.txt` pinned ranges | ✅ |
| `xgboost`, `python-dotenv` included | ✅ |
| PI_Backend `python-dotenv` | ✅ |
| Mobile `package.json` | ✅ |

### 5. Database readiness

| Item | Status |
|------|--------|
| SQLite + Django migrations (`PI_Backend`) | ✅ |
| Seed scripts / sample data | ⚠️ Manual: create superuser via `createsuperuser` |
| Documented in README | ✅ |

### 6. Documentation

| File | Status |
|------|--------|
| `docs/README.md` | ✅ Points to root README (single source of truth) |

### 7. Demo assets

| Item | Status |
|------|--------|
| `demo/README.md` checklist | ✅ |
| Screenshots / GIFs | ⚠️ Not provided — local README demo steps documented |

### Authors

| Status |
|--------|
| ✅ Salah Ghanoui, Melek Amimi, Meryem Benani, Zeineb Moujehed, Maram Ben Farhat — 3IA5, 2025–2026 |
| ✅ Tuteurs: Dorsaf Hrizi, Oumayma Guasmi |

### Repository naming

| Item | Status |
|------|--------|
| Target name `Esprit-PI-3IA5-25265-Bovitech` | ⚠️ Documented in README; rename on GitHub Settings (see below) |

### 8. CI/CD

| Item | Status |
|------|--------|
| `.github/workflows/ci.yml` | ✅ |

### 9. Project-type rules

| Rule | Status |
|------|--------|
| ML files >100MB gitignored | ✅ |
| `scripts/download_models.py` | ✅ |
| `data/README.md` | ✅ |
| Python/CUDA documented | ✅ (root README) |
| ESP32 BOM / flash | ✅ (root README) |
| No exploit code exposed | ✅ |

### 10. Acceptance (10-minute clone)

| Step | Status |
|------|--------|
| Clone + venv + pip | ✅ Documented |
| Models obtain path | ⚠️ Requires manifest URLs or team artifacts |
| `python src/model_http_api.py` + `/health` | ✅ (cold start noted) |

---

## Files created

- `.env.example` at repo root (ML API, Expo, PI_Backend, chatbot)
- `docs/README.md`
- `demo/README.md`, `demo/screenshots/.gitkeep`
- `data/README.md`, `gps_tracking/uwb_data/README.md`
- `scripts/download_models.py`, `models.manifest.json`
- `.github/workflows/ci.yml`
- `LICENSE`, `COMPLIANCE_REPORT.md`

## Files modified

- `README.md` (ESPRIT sections)
- `.gitignore`, `requirements.txt`
- `src/model_http_api.py` (dotenv, milk metrics path)
- `PI_Backend/config/settings.py`, `PI_Backend/requirements.txt`, `PI_Backend/accounts/services.py`
- `thi/thi.ino`, `thi/thi_esp32_http/thi_esp32_http.ino`

## Remaining manual tasks

1. **Rename on GitHub:** Settings → General → Repository name → `Esprit-PI-3IA5-25265-Bovitech`, then update remote:  
   `git remote set-url origin https://github.com/Bovitech/Esprit-PI-3IA5-25265-Bovitech.git`
2. Add final model metrics to root README (and Hugging Face model cards).
3. Run `PI_Backend`: ensure repo root `.env` exists, `migrate`, optional `createsuperuser`.
4. Purge secrets from Git history if they were previously committed (WiFi, old Django key).
5. Confirm GitHub org visibility and ESPRIT publication form with your supervisor.
