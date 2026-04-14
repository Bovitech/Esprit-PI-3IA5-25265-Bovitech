import json
import logging
import re
from datetime import datetime

import requests

from chatbot.services import llm_client
from .base import BaseAgent

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SEASON_MAP = {
    12: "hiver", 1: "hiver",  2: "hiver",
    3: "printemps", 4: "printemps", 5: "printemps",
    6: "été",  7: "été",  8: "été",
    9: "automne", 10: "automne", 11: "automne",
}


class FeedAgent(BaseAgent):

    def run(self, lat: float | None, lon: float | None, lang: str) -> dict:
        season  = SEASON_MAP[datetime.now().month]
        weather = self._fetch_weather(lat, lon)
        rec     = self._llm_recommendation(season, weather, lang)

        logger.info(
            "action=feed season=%s temp=%.1f is_day=%s",
            season, weather["temp"], weather["is_day"]
        )

        return {
            "season":     season,
            "temp":       weather["temp"],
            "is_day":     bool(weather["is_day"]),
            "main_feed":  rec.get("main_feed",  "Foin à volonté"),
            "supplement": rec.get("supplement", "Minéraux standards"),
            "water":      rec.get("water",      "Eau fraîche en permanence"),
            "warning":    rec.get("warning",    ""),
            "tip":        rec.get("tip",        ""),
        }

    def _fetch_weather(self, lat: float | None, lon: float | None) -> dict:
        defaults = {"temp": 20.0, "rain": 0.0, "is_day": 1}
        if lat is None or lon is None:
            return defaults
        try:
            params = (
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,precipitation,is_day"
                f"&timezone=auto"
            )
            res  = requests.get(OPEN_METEO_URL + params, timeout=10)
            res.raise_for_status()
            data = res.json()
            cur  = data["current"]
            return {
                "temp":   cur["temperature_2m"],
                "rain":   cur["precipitation"],
                "is_day": cur.get("is_day", 1),
            }
        except Exception as exc:
            logger.warning("Feed agent weather fetch failed: %s", exc)
            return defaults

    def _llm_recommendation(self, season: str, w: dict, lang: str) -> dict:
        time_of_day = "matin" if w["is_day"] else "soir/nuit"

        if lang == "ar":
            prompt = (
                f"أنت خبير تغذية أبقار. أعطِ توصيات تغذية اليوم.\n\n"
                f"الفصل: {season} | الوقت: {time_of_day} | الحرارة: {w['temp']}°C | الأمطار: {w['rain']}mm\n\n"
                f"القواعد: صيف/حرارة>28→زد الماء وقلل الحبوب وأضف أملاح، شتاء/برد<5→زد الطاقة وأضف دهون، "
                f"ربيع→انتبه لانتفاخ المرعى، خريف→احتياطيات قبل الشتاء\n\n"
                f'أجب فقط بـ JSON: {{"main_feed":"...","supplement":"...","water":"...","warning":"...","tip":"..."}}'
            )
        else:
            prompt = (
                f"Tu es un expert en nutrition bovine. Recommande l'alimentation d'aujourd'hui.\n\n"
                f"Saison: {season} | Moment: {time_of_day} | Température: {w['temp']}°C | Pluie: {w['rain']}mm\n\n"
                f"Règles: été/chaleur>28→augmenter eau réduire concentrés sel minéral, "
                f"hiver/froid<5→augmenter énergie lipides, printemps→attention météorisation, "
                f"automne→constituer réserves\n\n"
                f'Réponds UNIQUEMENT en JSON: {{"main_feed":"...","supplement":"...","water":"...","warning":"...","tip":"..."}}'
            )

        raw = llm_client.call_text(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        if raw:
            try:
                return json.loads(re.sub(r"```json|```", "", raw).strip())
            except Exception:
                pass
        return {
            "main_feed":  "Foin à volonté",
            "supplement": "Minéraux standards",
            "water":      "Eau fraîche en permanence",
            "warning":    "",
            "tip":        "",
        }