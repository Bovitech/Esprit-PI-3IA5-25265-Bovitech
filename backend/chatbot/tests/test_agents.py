"""
Tests for individual agents.
All external calls (Groq, OpenStreetMap, Open-Meteo) are mocked.
Run with: python manage.py test chatbot.tests.test_agents
"""
import json
from unittest.mock import MagicMock, patch

from django.test import TestCase

from chatbot.services.agents.feed import FeedAgent
from chatbot.services.agents.meteo import MeteoAgent
from chatbot.services.agents.vet import VetAgent


# ------------------------------------------------------------------ #
# VetAgent
# ------------------------------------------------------------------ #
class VetAgentTest(TestCase):

    def test_no_location_returns_message(self):
        agent = VetAgent()
        result = agent.run(None, None, "fr")
        self.assertIn("localisation", result)

    def test_no_location_arabic(self):
        agent = VetAgent()
        result = agent.run(None, None, "ar")
        self.assertIn("الموقع", result)

    def test_no_vets_found(self):
        agent = VetAgent()
        mock_response = MagicMock()
        mock_response.json.return_value = {"elements": []}
        with patch("chatbot.services.agents.vet.requests.post", return_value=mock_response):
            result = agent.run(36.8, 10.1, "fr")
        self.assertIn("❌", result)

    def test_vet_found_renders_card(self):
        agent = VetAgent()
        mock_response = MagicMock()
        mock_response.json.return_value = {"elements": [{
            "lat": 36.81, "lon": 10.11,
            "tags": {"name": "Clinique Test", "phone": "+21612345678"}
        }]}
        with patch("chatbot.services.agents.vet.requests.post", return_value=mock_response):
            result = agent.run(36.8, 10.1, "fr")
        self.assertIn("Clinique Test", result)
        self.assertIn("map-btn", result)
        self.assertIn("+21612345678", result)


# ------------------------------------------------------------------ #
# MeteoAgent
# ------------------------------------------------------------------ #
class MeteoAgentTest(TestCase):

    def _mock_weather(self, temp=22, rain=0, wind=10, is_day=1, next_3h=0):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": temp,
                "precipitation": rain,
                "windspeed_10m": wind,
                "weathercode": 0,
                "is_day": is_day,
            },
            "hourly": {"precipitation_probability": [next_3h, next_3h, next_3h]},
        }
        return mock_resp

    def test_no_location(self):
        result = MeteoAgent().run(None, None, "fr")
        self.assertIn("localisation", result)

    def test_good_weather_shows_out(self):
        mock_llm = json.dumps({"decision": "out", "reason": "Beau temps", "tip": "Profitez-en"})
        with patch("chatbot.services.agents.meteo.requests.get", return_value=self._mock_weather()):
            with patch("chatbot.services.agents.meteo.llm_client.call_text", return_value=mock_llm):
                result = MeteoAgent().run(36.8, 10.1, "fr")
        self.assertIn("2d6a4f", result)   # green color = "out"
        self.assertIn("peuvent sortir", result)

    def test_night_override_forces_in(self):
        """Even if LLM says 'out', night must always force 'in'."""
        mock_llm = json.dumps({"decision": "out", "reason": "ok", "tip": ""})
        with patch("chatbot.services.agents.meteo.requests.get", return_value=self._mock_weather(is_day=0)):
            with patch("chatbot.services.agents.meteo.llm_client.call_text", return_value=mock_llm):
                result = MeteoAgent().run(36.8, 10.1, "fr")
        self.assertIn("c0392b", result)   # red = "in"
        self.assertIn("nuit", result.lower())

    def test_rain_warning_shown(self):
        mock_llm = json.dumps({"decision": "in", "reason": "Pluie", "tip": "Rester"})
        with patch("chatbot.services.agents.meteo.requests.get", return_value=self._mock_weather(next_3h=80)):
            with patch("chatbot.services.agents.meteo.llm_client.call_text", return_value=mock_llm):
                result = MeteoAgent().run(36.8, 10.1, "fr")
        self.assertIn("meteo-rain-warn", result)


# ------------------------------------------------------------------ #
# FeedAgent
# ------------------------------------------------------------------ #
class FeedAgentTest(TestCase):

    def _mock_weather(self, temp=20, rain=0, is_day=1):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "current": {"temperature_2m": temp, "precipitation": rain, "is_day": is_day}
        }
        return mock_resp

    def _mock_llm_rec(self):
        return json.dumps({
            "main_feed": "Foin", "supplement": "Sel",
            "water": "Eau fraîche", "warning": "", "tip": "Surveiller"
        })

    def test_works_without_gps(self):
        """Feed agent must work even without GPS coordinates."""
        with patch("chatbot.services.agents.feed.llm_client.call_text", return_value=self._mock_llm_rec()):
            result = FeedAgent().run(None, None, "fr")
        self.assertIn("feed-card", result)

    def test_renders_warning_when_present(self):
        rec_with_warning = json.dumps({
            "main_feed": "Foin", "supplement": "Sel",
            "water": "Eau", "warning": "Attention météorisation", "tip": ""
        })
        with patch("chatbot.services.agents.feed.requests.get", return_value=self._mock_weather()):
            with patch("chatbot.services.agents.feed.llm_client.call_text", return_value=rec_with_warning):
                result = FeedAgent().run(36.8, 10.1, "fr")
        self.assertIn("feed-warning", result)
        self.assertIn("météorisation", result)

    def test_arabic_labels_used(self):
        with patch("chatbot.services.agents.feed.llm_client.call_text", return_value=self._mock_llm_rec()):
            result = FeedAgent().run(None, None, "ar")
        self.assertIn("العلف", result)