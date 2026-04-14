"""
Classifies user intent into one of: vet_agent, meteo_agent, feed_agent, answer
Uses the classifier prompt from prompts/{lang}/classifier.yaml
"""
import logging

from chatbot.services import llm_client
from chatbot.services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

ACTION_MAP = {
    "VET":   "vet_agent",
    "METEO": "meteo_agent",
    "FEED":  "feed_agent",
}


def decide(question: str, history: list[dict], lang: str) -> str:
    """
    Returns one of: 'vet_agent', 'meteo_agent', 'feed_agent', 'answer'
    """
    prompt_data = load_prompt(lang, "classifier")
    system_msg  = prompt_data["system"]

    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history[-2:])
    messages.append({"role": "user", "content": question})

    raw = llm_client.call_text(messages, temperature=0, max_tokens=10)

    if raw is None:
        logger.warning("Router LLM call failed, defaulting to answer")
        return "answer"

    label = raw.strip().upper()
    for key, action in ACTION_MAP.items():
        if key in label:
            return action

    return "answer"