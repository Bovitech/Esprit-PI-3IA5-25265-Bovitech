"""
Post-processes raw Whisper output with a small LLM call.
Returns corrected text, or raises ValueError if audio is invalid/noise.
"""
import logging

from chatbot.services import llm_client

logger = logging.getLogger(__name__)

_INVALID = "INVALID"


def correct(raw_text: str, lang: str) -> str:
    """
    Returns cleaned text.
    Raises ValueError if the transcription is noise / incomprehensible.
    """
    if lang == "ar":
        prompt = f"""أنت مصحح نصوص منطوقة بالعربية.
النص: "{raw_text}"
- إذا كان عربياً مفهوماً → صحح وأعد النص فقط
- إذا كان ضوضاء أو غير مفهوم → أعد بالضبط: INVALID"""
    else:
        prompt = f"""Tu es un correcteur de transcription vocale française.
Texte: "{raw_text}"
- Si c'est du français parlé → corrige et retourne UNIQUEMENT le texte corrigé
- Si c'est du bruit ou incompréhensible → retourne exactement: INVALID"""

    result = llm_client.call_text(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=100,
    )

    if result is None:
        logger.warning("STT correction LLM call failed, returning raw text")
        return raw_text  # graceful fallback

    cleaned = result.strip().upper().replace(".", "").replace("!", "")
    if cleaned == _INVALID:
        raise ValueError("Transcription invalid / noise")

    return result.strip()