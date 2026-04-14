"""
Normal (non-agent) LLM response: streams tokens back to the caller.
"""
import logging
from typing import Generator

from chatbot.services import llm_client
from chatbot.services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


def stream(
    question: str,
    context: str,
    history: list[dict],
    lang: str,
) -> Generator[str, None, str]:
    """
    Yields tokens one by one.
    Returns the full response text as the StopIteration value
    so the caller can save it: `full = yield from stream(...)`
    """
    prompt_data = load_prompt(lang, "system")
    system_text = prompt_data["system"].format(
        instruction=prompt_data["instruction"],
        context=context or "(aucun contexte trouvé)",
    )

    user_prefix = "أجب بالعربية فقط: " if lang == "ar" else "Réponds en français uniquement : "

    messages = [{"role": "system", "content": system_text}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prefix + question})

    full = ""
    result = llm_client.call(messages, temperature=0.3, max_tokens=300, stream=True)
    if result is None:
        msg = "عذراً، حدث خطأ." if lang == "ar" else "Erreur lors de la génération de la réponse."
        yield msg
        return msg

    try:
        for chunk in result:
            delta = chunk.choices[0].delta.content
            if delta:
                full += delta
                yield delta
    except Exception as exc:
        logger.error("Stream error: %s", exc)

    return full