"""
Thin wrapper around Groq client.
- Single import point for the whole project
- Safe call() that never raises — returns None on failure
- Centralises model name so you change it in one place
"""
import logging
from groq import Groq
from django.conf import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in settings.")
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def call(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 200,
    stream: bool = False,
    model: str = DEFAULT_MODEL,
):
    """
    Wrapper around client.chat.completions.create.
    Returns the completion object, or None on error.
    """
    try:
        return get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=stream,
        )
    except Exception as exc:
        logger.error("Groq call failed: %s", exc)
        return None


def call_text(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 200,
    model: str = DEFAULT_MODEL,
) -> str | None:
    """Convenience: returns plain text or None."""
    result = call(messages, temperature=temperature, max_tokens=max_tokens, model=model)
    if result is None:
        return None
    try:
        return result.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Groq response parse failed: %s", exc)
        return None