"""
Tests for individual agents.
Agents now return DICTS — not HTML strings.
All external calls (Groq, OpenStreetMap, Open-Meteo) are mocked.
Run: python manage.py test chatbot.tests.test_agents
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

    def test_no_location_returns_dict_with_warning(self):
        result = VetAgent().run(None, None, "fr")
        self.assertIsInstance(result, dict)
        self.assertFalse(result["found"])
        self.assertIsNone(result["best"])
        self.assertIn("localisation", result["warning"])

    def test_no_location_arabic(self):
        result = VetAgent().run(None, None, "ar")
        self.assertIsInstance(result, dict)
        self.assertFalse(result["found"])
        self.assertIn("الموقع", result["warning"])

    def test_no_vets_found_returns_not_found(self):
        mock_response = MagicMock()
        mock_response.text = '{"elements": []}'
        mock_response.json.return_value = {"elements": []}
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot.services.agents.vet.requests.post", return_value=mock_response):
            result = VetAgent().run(36.8, 10.1, "fr")
        self.assertIsInstance(result, dict)
        self.assertFalse(result["found"])
        self.assertIsNone(result["best"])

    def test_vet_found_returns_correct_shape(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({"elements": [{
            "lat": 36.81, "lon": 10.11,
            "tags": {"name": "Clinique Test", "phone": "+21612345678"}
        }]})
        mock_response.json.return_value = {"elements": [{
            "lat": 36.81, "lon": 10.11,
            "tags": {"name": "Clinique Test", "phone": "+21612345678"}
        }]}
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot.services.agents.vet.requests.post", return_value=mock_response):
            result = VetAgent().run(36.8, 10.1, "fr")
        self.assertTrue(result["found"])
        self.assertIsNotNone(result["best"])
        self.assertEqual(result["best"]["name"], "Clinique Test")
        self.assertEqual(result["best"]["phone"], "+21612345678")
        self.assertIn("map_url", result["best"])
        self.assertIn("distance_km", result["best"])
        self.assertIsInstance(result["others"], list)

    def test_vet_found_no_phone(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({"elements": [{
            "lat": 36.81, "lon": 10.11,
            "tags": {"name": "Clinique Sans Tel"}
        }]})
        mock_response.json.return_value = {"elements": [{
            "lat": 36.81, "lon": 10.11,
            "tags": {"name": "Clinique Sans Tel"}
        }]}
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot.services.agents.vet.requests.post", return_value=mock_response):
            result = VetAgent().run(36.8, 10.1, "fr")
        self.assertIsNone(result["best"]["phone"])

    def test_far_vet_adds_warning(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({"elements": [{
            "lat": 37.5, "lon": 10.1,    # ~77km away
            "tags": {"name": "Clinique Lointaine"}
        }]})
        mock_response.json.return_value = {"elements": [{
            "lat": 37.5, "lon": 10.1,
            "tags": {"name": "Clinique Lointaine"}
        }]}
        mock_response.raise_for_status = MagicMock()
        with patch("chatbot.services.agents.vet.requests.post", return_value=mock_response):
            result = VetAgent().run(36.8, 10.1, "fr")
        self.assertIsNotNone(result["warning"])


# ------------------------------------------------------------------ #
# MeteoAgent
# ------------------------------------------------------------------ #
class MeteoAgentTest(TestCase):

    def _mock_weather_response(self, temp=22, rain=0, wind=10, is_day=1, next_3h=0):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": temp,
                "precipitation":  rain,
                "windspeed_10m":  wind,
                "weathercode":    0,
                "is_day":         is_day,
            },
            "hourly": {"precipitation_probability": [next_3h, next_3h, next_3h]},
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_no_location_returns_none(self):
        """MeteoAgent returns None when no location — views.py handles the message."""
        result = MeteoAgent().run(None, None, "fr")
        self.assertIsNone(result)

    def test_good_weather_returns_out_decision(self):
        mock_llm = json.dumps({"decision": "out", "reason": "Beau temps", "tip": "Profitez-en"})
        with patch("chatbot.services.agents.meteo.requests.get",
                   return_value=self._mock_weather_response()):
            with patch("chatbot.services.agents.meteo.llm_client.call_text",
                       return_value=mock_llm):
                result = MeteoAgent().run(36.8, 10.1, "fr")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["decision"], "out")
        self.assertEqual(result["temp"], 22)
        self.assertTrue(result["is_day"])

    def test_night_override_forces_in_regardless_of_llm(self):
        """Night must always produce 'in' even if LLM returns 'out'."""
        mock_llm = json.dumps({"decision": "out", "reason": "ok", "tip": ""})
        with patch("chatbot.services.agents.meteo.requests.get",
                   return_value=self._mock_weather_response(is_day=0)):
            with patch("chatbot.services.agents.meteo.llm_client.call_text",
                       return_value=mock_llm):
                result = MeteoAgent().run(36.8, 10.1, "fr")
        self.assertEqual(result["decision"], "in")
        self.assertFalse(result["is_day"])
        self.assertIn("nuit", result["reason"].lower())

    def test_rain_warning_in_data(self):
        mock_llm = json.dumps({"decision": "in", "reason": "Pluie", "tip": "Rester"})
        with patch("chatbot.services.agents.meteo.requests.get",
                   return_value=self._mock_weather_response(next_3h=80)):
            with patch("chatbot.services.agents.meteo.llm_client.call_text",
                       return_value=mock_llm):
                result = MeteoAgent().run(36.8, 10.1, "fr")
        self.assertEqual(result["next_3h_rain_pct"], 80)

    def test_result_contains_all_required_fields(self):
        mock_llm = json.dumps({"decision": "out", "reason": "ok", "tip": "go"})
        with patch("chatbot.services.agents.meteo.requests.get",
                   return_value=self._mock_weather_response()):
            with patch("chatbot.services.agents.meteo.llm_client.call_text",
                       return_value=mock_llm):
                result = MeteoAgent().run(36.8, 10.1, "fr")
        required = {"decision", "temp", "rain", "wind", "is_day", "next_3h_rain_pct", "reason", "tip"}
        self.assertEqual(required, required & result.keys())

    def test_weather_fetch_failure_returns_none(self):
        with patch("chatbot.services.agents.meteo.requests.get",
                   side_effect=Exception("network down")):
            result = MeteoAgent().run(36.8, 10.1, "fr")
        self.assertIsNone(result)


# ------------------------------------------------------------------ #
# FeedAgent
# ------------------------------------------------------------------ #
class FeedAgentTest(TestCase):

    def _mock_weather(self, temp=20, rain=0, is_day=1):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "current": {
                "temperature_2m": temp,
                "precipitation":  rain,
                "is_day":         is_day,
            }
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def _mock_llm_rec(self, warning=""):
        return json.dumps({
            "main_feed":  "Foin",
            "supplement": "Sel",
            "water":      "Eau fraîche",
            "warning":    warning,
            "tip":        "Surveiller",
        })

    def test_works_without_gps(self):
        """Feed agent must return a valid dict even without GPS."""
        with patch("chatbot.services.agents.feed.llm_client.call_text",
                   return_value=self._mock_llm_rec()):
            result = FeedAgent().run(None, None, "fr")
        self.assertIsInstance(result, dict)
        self.assertIn("season", result)
        self.assertIn("main_feed", result)
        self.assertIn("supplement", result)
        self.assertIn("water", result)

    def test_warning_field_present_when_set(self):
        with patch("chatbot.services.agents.feed.requests.get",
                   return_value=self._mock_weather()):
            with patch("chatbot.services.agents.feed.llm_client.call_text",
                       return_value=self._mock_llm_rec(warning="Attention météorisation")):
                result = FeedAgent().run(36.8, 10.1, "fr")
        self.assertEqual(result["warning"], "Attention météorisation")

    def test_warning_empty_when_not_set(self):
        with patch("chatbot.services.agents.feed.llm_client.call_text",
                   return_value=self._mock_llm_rec(warning="")):
            result = FeedAgent().run(None, None, "fr")
        self.assertEqual(result["warning"], "")

    def test_contains_all_required_fields(self):
        with patch("chatbot.services.agents.feed.llm_client.call_text",
                   return_value=self._mock_llm_rec()):
            result = FeedAgent().run(None, None, "fr")
        required = {"season", "temp", "is_day", "main_feed", "supplement", "water", "warning", "tip"}
        self.assertEqual(required, required & result.keys())

    def test_season_is_valid(self):
        with patch("chatbot.services.agents.feed.llm_client.call_text",
                   return_value=self._mock_llm_rec()):
            result = FeedAgent().run(None, None, "fr")
        self.assertIn(result["season"], ["hiver", "printemps", "été", "automne"])

    def test_temp_reflects_weather(self):
        with patch("chatbot.services.agents.feed.requests.get",
                   return_value=self._mock_weather(temp=35)):
            with patch("chatbot.services.agents.feed.llm_client.call_text",
                       return_value=self._mock_llm_rec()):
                result = FeedAgent().run(36.8, 10.1, "fr")
        self.assertEqual(result["temp"], 35)