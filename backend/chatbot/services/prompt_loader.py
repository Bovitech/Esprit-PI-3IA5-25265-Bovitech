"""
Loads prompt YAML files from chatbot/prompts/{lang}/{name}.yaml
Caches results so disk is only read once per prompt.
"""
import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(lang: str, name: str) -> dict:
    """
    Returns the parsed YAML dict for prompts/{lang}/{name}.yaml
    Falls back to 'fr' if the requested language file doesn't exist.
    """
    path = PROMPTS_DIR / lang / f"{name}.yaml"
    if not path.exists():
        logger.warning("Prompt %s/%s not found, falling back to fr", lang, name)
        path = PROMPTS_DIR / "fr" / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)