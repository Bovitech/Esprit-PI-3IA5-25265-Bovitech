"""
Conversation memory with automatic summarisation.

Summary is injected as a SYSTEM message — not fake user/assistant turns.
This gives the LLM correct context without polluting the conversation history.
"""
import logging
import threading
import time

from chatbot.services import llm_client
from chatbot.services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLD = 12
MESSAGES_TO_KEEP  = 4

_summarise_locks: dict[str, threading.Lock] = {}
_lock_map_lock = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    with _lock_map_lock:
        if session_id not in _summarise_locks:
            _summarise_locks[session_id] = threading.Lock()
        return _summarise_locks[session_id]


def _models():
    from chatbot.models import Conversation, ConversationSummary
    return Conversation, ConversationSummary


def _summarise(session_id: str, messages: list[dict], lang: str) -> str:
    Conversation, ConversationSummary = _models()

    conversation_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )

    template = load_prompt(lang, "summarizer")
    prompt   = template["prompt"].format(conversation=conversation_text)

    text = llm_client.call_text(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    summary_text = text or "Résumé non disponible."

    ConversationSummary.objects.create(session_id=session_id, summary=summary_text)

    old_ids = [m["id"] for m in messages if "id" in m]
    if old_ids:
        Conversation.objects.filter(id__in=old_ids).delete()

    logger.info(
        "session=%s action=summarise messages_compressed=%d",
        session_id, len(messages)
    )
    return summary_text


def get_history(session_id: str, lang: str) -> list[dict]:
    """
    Returns message list for the LLM:
      [optional {"role": "system", "content": "<summary>"},
       ...recent messages]

    Summary is a SYSTEM message — correct LLM context, no history pollution.
    """
    Conversation, ConversationSummary = _models()

    all_msgs = list(
        Conversation.objects.filter(session_id=session_id)
        .order_by("created_at")
        .values("id", "role", "message", "created_at")
    )

    if len(all_msgs) > SUMMARY_THRESHOLD:
        session_lock = _get_session_lock(session_id)
        if session_lock.acquire(blocking=False):
            try:
                old_formatted = [
                    {"id": m["id"], "role": m["role"], "content": m["message"]}
                    for m in all_msgs[:-MESSAGES_TO_KEEP]
                ]
                _summarise(session_id, old_formatted, lang)
                all_msgs = list(
                    Conversation.objects.filter(session_id=session_id)
                    .order_by("created_at")
                    .values("id", "role", "message")
                )
            finally:
                session_lock.release()

    history: list[dict] = []

    # ✅ Summary injected as SYSTEM message — not fake user/assistant turns
    latest_summary = ConversationSummary.objects.filter(session_id=session_id).first()
    if latest_summary:
        summary_content = (
            f"ملخص المحادثة السابقة (سياق فقط، لا تعيد ذكره): {latest_summary.summary}"
            if lang == "ar"
            else f"Résumé de la conversation précédente (contexte uniquement, ne pas répéter) : {latest_summary.summary}"
        )
        history.append({"role": "system", "content": summary_content})

    for m in all_msgs:
        history.append({"role": m["role"], "content": m["message"]})

    return history


def save_assistant_message(session_id: str, text: str) -> None:
    Conversation, _ = _models()
    if text.strip():
        Conversation.objects.create(
            session_id=session_id, role="assistant", message=text
        )