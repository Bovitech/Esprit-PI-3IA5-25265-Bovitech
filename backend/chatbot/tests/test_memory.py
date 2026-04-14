"""
Tests for conversation memory + summarisation.
Run with: python manage.py test chatbot.tests.test_memory
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

    def test_summary_injected_at_top(self):
        ConversationSummary.objects.create(session_id=self.SESSION, summary="Les vaches sont malades.")
        self._add_messages(2)
        history = memory.get_history(self.SESSION, "fr")
        self.assertIn("Résumé", history[0]["content"])
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_summarisation_triggered_above_threshold(self):
        self._add_messages(memory.SUMMARY_THRESHOLD + 2)
        with patch("chatbot.services.memory.llm_client.call_text", return_value="Résumé test"):
            history = memory.get_history(self.SESSION, "fr")

        # After summarisation, DB should have fewer messages
        remaining = Conversation.objects.filter(session_id=self.SESSION).count()
        self.assertLessEqual(remaining, memory.MESSAGES_TO_KEEP)

        # A summary should have been saved
        self.assertTrue(ConversationSummary.objects.filter(session_id=self.SESSION).exists())

    def test_no_double_summarisation(self):
        """Calling get_history twice should not create two summaries."""
        self._add_messages(memory.SUMMARY_THRESHOLD + 2)
        with patch("chatbot.services.memory.llm_client.call_text", return_value="Résumé"):
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
        self.assertFalse(Conversation.objects.filter(session_id=self.SESSION).exists())