"""
Tests for the intent router.
Uses unittest.mock to avoid real Groq API calls.
Run with: python manage.py test chatbot.tests.test_router
"""
from unittest.mock import MagicMock, patch

from django.test import TestCase

from chatbot.services import router


class RouterTest(TestCase):

    def _mock_llm(self, return_value: str):
        """Patch llm_client.call_text to return a fixed string."""
        return patch(
            "chatbot.services.router.llm_client.call_text",
            return_value=return_value,
        )

    # ---- expected classifications ---------------------------------- #

    def test_vet_keywords(self):
        cases = ["je cherche un vétérinaire", "vet proche", "clinique animaux",
                 "أريد طبيباً بيطرياً"]
        for q in cases:
            with self._mock_llm("VET"):
                self.assertEqual(router.decide(q, [], "fr"), "vet_agent", msg=q)

    def test_meteo_keywords(self):
        cases = ["est-ce que je peux sortir mes vaches", "quel temps fait-il",
                 "هل الطقس مناسب للخروج"]
        for q in cases:
            with self._mock_llm("METEO"):
                self.assertEqual(router.decide(q, [], "fr"), "meteo_agent", msg=q)

    def test_feed_keywords(self):
        cases = ["que donner à manger", "ration journalière", "fourrage aujourd'hui",
                 "ماذا أطعم الأبقار"]
        for q in cases:
            with self._mock_llm("FEED"):
                self.assertEqual(router.decide(q, [], "fr"), "feed_agent", msg=q)

    def test_normal_fallback(self):
        with self._mock_llm("NORMAL"):
            self.assertEqual(router.decide("bonjour", [], "fr"), "answer")

    def test_llm_failure_defaults_to_answer(self):
        with self._mock_llm(None):   # simulate Groq down
            self.assertEqual(router.decide("anything", [], "fr"), "answer")

    def test_noisy_llm_output_still_matches(self):
        """Model sometimes returns 'VET_AGENT' or 'VET.' — should still match."""
        with self._mock_llm("VET_AGENT"):
            self.assertEqual(router.decide("vet?", [], "fr"), "vet_agent")