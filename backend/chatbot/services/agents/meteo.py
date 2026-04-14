import json
import logging
import re

import requests

from chatbot.services import llm_client
from .base import BaseAgent

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class MeteoAgent(BaseAgent):

    def run(self, lat: float | None, lon: float | None, lang: str) -> dict | None:
        """Returns None if location is missing (caller handles the message)."""
        if lat is None or lon is None:
            return None

        weather = self._fetch_weather(lat, lon)
        if weather is None:
            return None

        decision, reason, tip = self._llm_decision(weather, lang)

        # Hard safety override at night — regardless of LLM
        if not weather["is_day"]:
            decision = "in"
            reason   = "الوقت ليلاً — الأبقار يجب أن تكون في الداخل" if lang == "ar" else "Il fait nuit — les vaches doivent rester à l'intérieur"
            tip      = "أعد التحقق صباحاً" if lang == "ar" else "Revérifiez le matin"

        logger.info(
            "action=meteo decision=%s temp=%.1f rain=%.1f wind=%.1f is_day=%s",
            decision, weather["temp"], weather["rain"],
            weather["wind"], weather["is_day"]
        )

        return {
            "decision":          decision,
            "temp":              weather["temp"],
            "rain":              weather["rain"],
            "wind":              weather["wind"],
            "is_day":            bool(weather["is_day"]),
            "next_3h_rain_pct":  weather["next_3h_rain"],
            "reason":            reason,
            "tip":               tip,
        }

    def _fetch_weather(self, lat: float, lon: float) -> dict | None:
        try:
            params = (
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,precipitation,windspeed_10m,weathercode,is_day"
                f"&hourly=precipitation_probability"
                f"&forecast_days=1&windspeed_unit=kmh&timezone=auto"
            )
            res  = requests.get(OPEN_METEO_URL + params, timeout=10)
            res.raise_for_status()
            data = res.json()
            cur  = data["current"]
            prob = data.get("hourly", {}).get("precipitation_probability", [])
            return {
                "temp":         cur["temperature_2m"],
                "rain":         cur["precipitation"],
                "wind":         cur["windspeed_10m"],
                "wcode":        cur["weathercode"],
                "is_day":       cur.get("is_day", 1),
                "next_3h_rain": max(prob[:3]) if prob else 0,
            }
        except Exception as exc:
            logger.error("Open-Meteo fetch failed: %s", exc)
            return None

    def _llm_decision(self, w: dict, lang: str) -> tuple[str, str, str]:
        time_ctx = "nuit" if not w["is_day"] else "jour"
        ctx = (
            f"Heure : {time_ctx}\nTempérature : {w['temp']}°C\n"
            f"Précipitations : {w['rain']} mm\nProbabilité pluie 3h : {w['next_3h_rain']}%\n"
            f"Vent : {w['wind']} km/h"
        )

        if lang == "ar":
            prompt = (
                f"أنت خبير في تربية الأبقار. قرر إذا كان يمكن إخراج الأبقار.\n\n{ctx}\n\n"
                f"القواعد: ليل→in، حرارة>32→limited، حرارة<0→in، مطر>2mm أو احتمال>60%→in، رياح>50→in، وإلا→out\n"
                f'أجب فقط بـ JSON: {{"decision":"out|in|limited","reason":"...","tip":"..."}}'
            )
        else:
            prompt = (
                f"Tu es un expert en élevage bovin. Décide si les vaches peuvent sortir.\n\n{ctx}\n\n"
                f"Règles: nuit→in, temp>32→limited, temp<0→in, pluie>2mm ou prob>60%→in, vent>50→in, sinon→out\n"
                f'Réponds UNIQUEMENT en JSON: {{"decision":"out|in|limited","reason":"...","tip":"..."}}'
            )

        raw = llm_client.call_text(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        if raw:
            try:
                parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
                return (
                    parsed.get("decision", "out"),
                    parsed.get("reason", ""),
                    parsed.get("tip", ""),
                )
            except Exception:
                pass
        return "out", "Données insuffisantes", "Surveiller les conditions"