"""
Tests for conversation memory + summarisation.
Run: python manage.py test chatbot.tests.test_memory
"""
from unittest.mock import patch

from django.test import TestCase

from chatbot.models import Conversation, ConversationSummary
from chatbot.services import memory


class MemoryTest(TestCase):

    SESSION = "test-session-001"

    def _add_messages(self, n: int):
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            Conversation.objects.create(
                session_id=self.SESSION,
                role=role,
                message=f"Message {i}",
            )

    def test_history_returned_in_order(self):
        self._add_messages(4)
        history = memory.get_history(self.SESSION, "fr")
        roles = [m["role"] for m in history]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])

    def test_summary_injected_as_system_message(self):
        """Summary must be a SYSTEM message, not fake user/assistant."""
        ConversationSummary.objects.create(
            session_id=self.SESSION,
            summary="Les vaches sont malades."
        )
        self._add_messages(2)
        history = memory.get_history(self.SESSION, "fr")

        # First message must be system
        self.assertEqual(history[0]["role"], "system")
        self.assertIn("Résumé", history[0]["content"])
        self.assertIn("Les vaches", history[0]["content"])

        # Following messages are real conversation turns
        self.assertEqual(history[1]["role"], "user")
        self.assertEqual(history[2]["role"], "assistant")

    def test_no_fake_assistant_ack_message(self):
        """Old code injected a fake 'Compris' assistant message — must be gone."""
        ConversationSummary.objects.create(
            session_id=self.SESSION,
            summary="Résumé test."
        )
        self._add_messages(2)
        history = memory.get_history(self.SESSION, "fr")
        system_msgs   = [m for m in history if m["role"] == "system"]
        assistant_ack = [m for m in history if "Compris" in m.get("content", "")]
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(len(assistant_ack), 0)

    def test_summarisation_triggered_above_threshold(self):
        self._add_messages(memory.SUMMARY_THRESHOLD + 2)
        with patch(
            "chatbot.services.memory.llm_client.call_text",
            return_value="Résumé test"
        ):
            memory.get_history(self.SESSION, "fr")

        remaining = Conversation.objects.filter(session_id=self.SESSION).count()
        self.assertLessEqual(remaining, memory.MESSAGES_TO_KEEP)
        self.assertTrue(
            ConversationSummary.objects.filter(session_id=self.SESSION).exists()
        )

    def test_no_double_summarisation(self):
        self._add_messages(memory.SUMMARY_THRESHOLD + 2)
        with patch(
            "chatbot.services.memory.llm_client.call_text",
            return_value="Résumé"
        ):
            memory.get_history(self.SESSION, "fr")
            memory.get_history(self.SESSION, "fr")
        count = ConversationSummary.objects.filter(session_id=self.SESSION).count()
        self.assertEqual(count, 1)

    def test_save_assistant_message(self):
        memory.save_assistant_message(self.SESSION, "Bonjour !")
        msg = Conversation.objects.get(session_id=self.SESSION)
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.message, "Bonjour !")

    def test_empty_message_not_saved(self):
        memory.save_assistant_message(self.SESSION, "   ")
        self.assertFalse(
            Conversation.objects.filter(session_id=self.SESSION).exists()
        )

    def test_arabic_summary_label(self):
        """Arabic sessions should get Arabic summary label."""
        ConversationSummary.objects.create(
            session_id=self.SESSION, summary="ملخص"
        )
        history = memory.get_history(self.SESSION, "ar")
        system_content = history[0]["content"]
        self.assertIn("ملخص", system_content)