from pathlib import Path
from dotenv import load_dotenv
import os

# backend/backend/settings.py → parent.parent is the folder that contains manage.py and .env
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_ROOT / ".env")

BASE_DIR = _BACKEND_ROOT.parent

# ------------------------------------------------------------------
# SECURITY
# ------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required.")

DEBUG = os.getenv("DEBUG", "False") == "True"

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")

# ------------------------------------------------------------------
# APPLICATIONS
# ------------------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'chatbot',
    'corsheaders',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'backend' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# ------------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ------------------------------------------------------------------
# INTERNATIONALISATION
# ------------------------------------------------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'UTC'
USE_I18N      = True
USE_TZ        = True

# ------------------------------------------------------------------
# STATIC FILES
# ------------------------------------------------------------------
STATIC_URL = 'static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ------------------------------------------------------------------
# CORS
# In production replace this with the exact frontend origin.
# ------------------------------------------------------------------
CORS_ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000"
).split(",")

# ------------------------------------------------------------------
# THIRD-PARTY API KEYS
# ------------------------------------------------------------------
_GROQ_PLACEHOLDERS = frozenset(
    {
        "replace-with-your-real-groq-key",
        "your-api-key-here",
        "changeme",
    }
)
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is required. Add it to backend/.env (create a key at "
        "https://console.groq.com/keys - it must start with gsk_)."
    )
if GROQ_API_KEY.lower() in _GROQ_PLACEHOLDERS or not GROQ_API_KEY.startswith("gsk_"):
    raise RuntimeError(
        "GROQ_API_KEY in backend/.env is missing or not a real Groq key. "
        "Replace it with a key from https://console.groq.com/keys (starts with gsk_)."
    )

# ------------------------------------------------------------------
# PIPER TTS
# Env vars are required. No personal absolute paths in source code.
# Project-relative fallback for local dev convenience only.
# ------------------------------------------------------------------
_TOOLS_DIR  = BASE_DIR / "tools" / "piper"
_MODELS_DIR = BASE_DIR / "models"

PIPER_PATH     = os.getenv("PIPER_PATH",     str(_TOOLS_DIR  / "piper.exe"))
PIPER_MODEL_FR = os.getenv("PIPER_MODEL_FR", str(_MODELS_DIR / "fr_FR-upmc-medium.onnx"))
PIPER_MODEL_AR = os.getenv("PIPER_MODEL_AR", str(_MODELS_DIR / "ar_JO-kareem-medium.onnx"))

# ------------------------------------------------------------------
# DATA PATHS
# All reusable paths in one place — read from settings everywhere.
# ------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
QDRANT_PATH        = BASE_DIR / "qdrant_data"

# ------------------------------------------------------------------
# SKIN DISEASE MODEL
# ------------------------------------------------------------------
SKIN_MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
 
# Validation au démarrage
if not SKIN_MODEL_PATH.exists():
    import warnings
    warnings.warn(
        f"SKIN_MODEL_PATH not found at {SKIN_MODEL_PATH}. "
        "Skin disease classification will fail at runtime. "
        "Place best_model.pth in bovitech_chatbot/models/",
        RuntimeWarning,
        stacklevel=2,
    )

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "app": {
            "format": "{levelname} {asctime} {module} {message}",
            "style":  "{",
        },
    },
    "filters": {
        "suppress_hf": {
            "()": "django.utils.log.CallbackFilter",
            # Suppress noisy HuggingFace HEAD requests at INFO level
            "callback": lambda r: not (
                r.name in ("httpx._client", "_client")
                and r.levelno == 20
                and "huggingface.co" in (r.getMessage())
            ),
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "app",
            "filters":   ["suppress_hf"],
        },
    },
    "root": {
        "handlers": ["console"],
        "level":    "WARNING",   
    },
    "loggers": {
        "chatbot": {
            "handlers":  ["console"],
            "level":     "DEBUG",
            "propagate": False,
        },
    },
}

# ------------------------------------------------------------------
# AUTH PASSWORD VALIDATORS
# ------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]