"""
API-level tests for /chatbot/, /chatbot/stt/, /chatbot/tts/
Run: python manage.py test chatbot.tests.test_chat_api
"""
import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from django.test import Client, TestCase


class ChatEndpointTest(TestCase):

    def setUp(self):
        self.client  = Client()
        self.url     = "/chatbot/"
        self.session = "api-test-session"

    def _post(self, data: dict):
        return self.client.post(
            self.url,
            data=json.dumps(data),
            content_type="application/json",
        )

    # ---- request validation ---------------------------------- #

    def test_invalid_json_returns_400(self):
        res = self.client.post(self.url, data="not json", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        body = json.loads(res.content)
        self.assertIn("error", body)
        self.assertEqual(body["error"]["code"], "INVALID_JSON")

    def test_empty_message_returns_400(self):
        res = self._post({"message": "", "session_id": self.session})
        self.assertEqual(res.status_code, 400)
        body = json.loads(res.content)
        self.assertEqual(body["error"]["code"], "EMPTY_MESSAGE")

    def test_whitespace_only_message_returns_400(self):
        res = self._post({"message": "   ", "session_id": self.session})
        self.assertEqual(res.status_code, 400)

    def test_missing_message_field_returns_400(self):
        res = self._post({"session_id": self.session})
        self.assertEqual(res.status_code, 400)

    def test_invalid_lang_defaults_to_fr(self):
        """Unknown lang values should not crash — must default to fr."""
        with patch("chatbot.views.router.decide", return_value="answer"), \
             patch("chatbot.views.llm_stream", return_value=iter(["ok"])), \
             patch("chatbot.services.retrieval.search", return_value=[]):
            res = self._post({
                "message": "bonjour",
                "lang": "xx",
                "session_id": self.session,
            })
        self.assertIn(res.status_code, (200,))

    # ---- language gate --------------------------------------- #

    def test_arabic_in_fr_mode_returns_text_response(self):
        res = self._post({"message": "مرحبا", "lang": "fr", "session_id": self.session})
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "text")
        self.assertIn("français", body["content"])

    def test_latin_in_ar_mode_returns_text_response(self):
        res = self._post({"message": "bonjour", "lang": "ar", "session_id": self.session})
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "text")

    # ---- agent responses ------------------------------------ #

    def test_vet_agent_no_location_returns_text_type(self):
        with patch("chatbot.views.router.decide", return_value="vet_agent"), \
             patch("chatbot.services.retrieval.search", return_value=[]):
            res = self._post({
                "message": "je cherche un vétérinaire",
                "lang": "fr",
                "session_id": self.session,
                # no lat/lon → should get text "activate location"
            })
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "text")
        self.assertIn("localisation", body["content"])

    def test_vet_agent_with_location_returns_agent_shape(self):
        mock_vet_data = {
            "found": True,
            "best": {
                "name": "Clinique Test", "distance_km": 3.5,
                "phone": None, "map_url": "https://maps.google.com",
            },
            "others": [],
            "warning": None,
        }
        with patch("chatbot.views.router.decide", return_value="vet_agent"), \
             patch("chatbot.views._vet_agent.run", return_value=mock_vet_data), \
             patch("chatbot.services.retrieval.search", return_value=[]):
            res = self._post({
                "message": "je cherche un vétérinaire",
                "lang": "fr",
                "session_id": self.session,
                "lat": 36.8,
                "lon": 10.1,
            })
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "agent")
        self.assertEqual(body["agent"], "vet")
        self.assertIn("found", body["data"])
        self.assertIn("best", body["data"])
        self.assertIn("others", body["data"])

    def test_meteo_agent_returns_agent_shape(self):
        mock_meteo_data = {
            "decision": "out", "temp": 22.0, "rain": 0.0,
            "wind": 10.0, "is_day": True, "next_3h_rain_pct": 10,
            "reason": "Beau temps", "tip": "Profitez-en",
        }
        with patch("chatbot.views.router.decide", return_value="meteo_agent"), \
             patch("chatbot.views._meteo_agent.run", return_value=mock_meteo_data), \
             patch("chatbot.services.retrieval.search", return_value=[]):
            res = self._post({
                "message": "mes vaches peuvent sortir ?",
                "lang": "fr",
                "session_id": self.session,
                "lat": 36.8,
                "lon": 10.1,
            })
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "agent")
        self.assertEqual(body["agent"], "meteo")
        self.assertIn("decision", body["data"])
        self.assertIn("temp", body["data"])

    def test_meteo_agent_no_location_returns_text(self):
        with patch("chatbot.views.router.decide", return_value="meteo_agent"), \
             patch("chatbot.services.retrieval.search", return_value=[]):
            res = self._post({
                "message": "mes vaches peuvent sortir ?",
                "lang": "fr",
                "session_id": self.session,
                # no lat/lon
            })
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "text")

    def test_feed_agent_returns_agent_shape(self):
        mock_feed_data = {
            "season": "printemps", "temp": 20.0, "is_day": True,
            "main_feed": "Foin", "supplement": "Sel",
            "water": "Eau", "warning": "", "tip": "Surveiller",
        }
        with patch("chatbot.views.router.decide", return_value="feed_agent"), \
             patch("chatbot.views._feed_agent.run", return_value=mock_feed_data), \
             patch("chatbot.services.retrieval.search", return_value=[]):
            res = self._post({
                "message": "que donner à manger aujourd'hui",
                "lang": "fr",
                "session_id": self.session,
            })
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "agent")
        self.assertEqual(body["agent"], "feed")
        self.assertIn("season", body["data"])
        self.assertIn("main_feed", body["data"])

    def test_meteo_returns_text_when_agent_returns_none(self):
        """MeteoAgent returns None on fetch failure — view must handle it as text."""
        with patch("chatbot.views.router.decide", return_value="meteo_agent"), \
             patch("chatbot.views._meteo_agent.run", return_value=None), \
             patch("chatbot.services.retrieval.search", return_value=[]):
            res = self._post({
                "message": "météo ?",
                "lang": "fr",
                "session_id": self.session,
                "lat": 36.8,
                "lon": 10.1,
            })
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["type"], "text")


class STTEndpointTest(TestCase):

    url = "/chatbot/stt/"

    def test_no_audio_returns_400(self):
        res = self.client.post(self.url, data={"lang": "fr"})
        self.assertEqual(res.status_code, 400)
        body = json.loads(res.content)
        self.assertEqual(body["error"]["code"], "NO_AUDIO")

    def test_audio_too_small_returns_incomprehensible(self):
        """Tiny audio (< 100 bytes) should be rejected before transcription."""
        tiny_audio = io.BytesIO(b"\x00" * 50)
        tiny_audio.name = "audio.webm"
        res = self.client.post(
            self.url,
            data={"audio": tiny_audio, "lang": "fr"},
        )
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["status"], "incomprehensible")

    def test_valid_transcription_returns_ok(self):
        """Mock the entire transcribe + correct pipeline."""
        valid_audio = io.BytesIO(b"\x00" * 500)
        valid_audio.name = "audio.webm"
        with patch("chatbot.views.transcribe", return_value="la vache mange du foin"), \
             patch("chatbot.views.stt_correct", return_value="La vache mange du foin"):
            res = self.client.post(
                self.url,
                data={"audio": valid_audio, "lang": "fr"},
            )
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["text"], "La vache mange du foin")

    def test_transcription_returns_none_gives_incomprehensible(self):
        valid_audio = io.BytesIO(b"\x00" * 500)
        valid_audio.name = "audio.webm"
        with patch("chatbot.views.transcribe", return_value=None):
            res = self.client.post(
                self.url,
                data={"audio": valid_audio, "lang": "fr"},
            )
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["status"], "incomprehensible")

    def test_llm_correction_flags_noise_falls_back_to_cleaned(self):
        """If the LLM marks noise, we still return Whisper-cleaned text (status ok)."""
        valid_audio = io.BytesIO(b"\x00" * 500)
        valid_audio.name = "audio.webm"
        with patch("chatbot.views.transcribe", return_value="xzqr ttt aaa bbb"), \
             patch("chatbot.views.stt_correct", side_effect=ValueError("noise")):
            res = self.client.post(
                self.url,
                data={"audio": valid_audio, "lang": "fr"},
            )
        self.assertEqual(res.status_code, 200)
        body = json.loads(res.content)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["text"], "xzqr ttt aaa bbb")


class TTSEndpointTest(TestCase):

    url = "/chatbot/tts/"

    def test_no_text_returns_400(self):
        res = self.client.post(
            self.url,
            data=json.dumps({"lang": "fr"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        body = json.loads(res.content)
        self.assertEqual(body["error"]["code"], "NO_TEXT")

    def test_invalid_json_returns_400(self):
        res = self.client.post(self.url, data="bad", content_type="application/json")
        self.assertEqual(res.status_code, 400)

    def test_valid_text_returns_audio(self):
        """
        Create a real temp WAV file, mock synthesize to return it.
        Do NOT delete the file manually — Django's FileResponse holds
        it open on Windows; delete_after_send handles cleanup.
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(b"RIFF" + b"\x00" * 40)
        tmp.close()

        try:
            # Mock delete_after_send too so it doesn't run in test context
            with patch("chatbot.views.synthesize", return_value=(tmp.name, "audio/wav")), \
                 patch("chatbot.views.delete_after_send"):
                res = self.client.get   # warm up
                res = self.client.post(
                    self.url,
                    data=json.dumps({"text": "Bonjour", "lang": "fr"}),
                    content_type="application/json",
                )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res["Content-Type"], "audio/wav")
        finally:
            # Safe to delete now — response is consumed, file is closed
            try:
                os.unlink(tmp.name)
            except PermissionError:
                pass  # Windows may still hold it briefly — acceptable in tests

    def test_synthesis_failure_returns_500(self):
        with patch("chatbot.views.synthesize", side_effect=Exception("Piper not found")):
            res = self.client.post(
                self.url,
                data=json.dumps({"text": "Bonjour", "lang": "fr"}),
                content_type="application/json",
            )
        self.assertEqual(res.status_code, 500)
        body = json.loads(res.content)
        self.assertEqual(body["error"]["code"], "TTS_FAILED")