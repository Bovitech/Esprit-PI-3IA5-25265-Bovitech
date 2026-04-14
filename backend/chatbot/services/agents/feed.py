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
    12: "hiver", 1: "hiver", 2: "hiver",
    3: "printemps", 4: "printemps", 5: "printemps",
    6: "été", 7: "été", 8: "été",
    9: "automne", 10: "automne", 11: "automne",
}

SEASON_EMOJI = {"hiver": "❄️", "printemps": "🌱", "été": "☀️", "automne": "🍂"}


class FeedAgent(BaseAgent):

    def run(self, lat: float | None, lon: float | None, lang: str) -> str:
        season  = SEASON_MAP[datetime.now().month]
        weather = self._fetch_weather(lat, lon)   # may return defaults
        rec     = self._llm_recommendation(season, weather, lang)
        return self._build_card(season, weather, rec, lang)

    # ------------------------------------------------------------------ #

    def _fetch_weather(self, lat: float | None, lon: float | None) -> dict:
        defaults = {"temp": 20, "rain": 0, "is_day": 1}
        if lat is None or lon is None:
            return defaults
        try:
            params = (
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,precipitation,is_day"
                f"&timezone=auto"
            )
            res    = requests.get(OPEN_METEO_URL + params, timeout=10)
            data   = res.json()
            cur    = data["current"]
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
            prompt = f"""أنت خبير تغذية أبقار. أعطِ توصيات تغذية اليوم.

الفصل: {season} | الوقت: {time_of_day} | درجة الحرارة: {w['temp']}°C | الأمطار: {w['rain']}mm

القواعد:
- صيف/حرارة>28 → زد الماء، قلل الحبوب، أضف أملاح
- شتاء/برد<5 → زد الطاقة (شعير/ذرة)، أضف دهون
- ربيع → انتبه لانتفاخ المرعى
- خريف → احتياطيات قبل الشتاء

أجب فقط بـ JSON: {{"main_feed":"...","supplement":"...","water":"...","warning":"...","tip":"..."}}"""
        else:
            prompt = f"""Tu es un expert en nutrition bovine. Recommande l'alimentation d'aujourd'hui.

Saison: {season} | Moment: {time_of_day} | Température: {w['temp']}°C | Pluie: {w['rain']}mm

Règles:
- Été/chaleur>28 → augmenter eau, réduire concentrés, sel minéral
- Hiver/froid<5 → augmenter énergie (orge/maïs), lipides
- Printemps → attention météorisation au pâturage
- Automne → constituer réserves avant hiver

Réponds UNIQUEMENT en JSON: {{"main_feed":"...","supplement":"...","water":"...","warning":"...","tip":"..."}}"""

        raw = llm_client.call_text([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=200)
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

    def _build_card(self, season: str, w: dict, rec: dict, lang: str) -> str:
        ar           = lang == "ar"
        emoji        = SEASON_EMOJI.get(season, "🌿")
        time_of_day  = "matin" if w["is_day"] else "soir/nuit"
        warning_html = f'<p class="feed-warning">⚠️ {rec["warning"]}</p>' if rec.get("warning") else ""

        if ar:
            labels = {"title": f"{emoji} توصيات التغذية اليوم", "main": "🌾 العلف الأساسي",
                      "supp": "💊 المكملات", "water": "💧 الماء", "tip": "👉 نصيحة"}
        else:
            labels = {"title": f"{emoji} Alimentation recommandée aujourd'hui", "main": "🌾 Fourrage principal",
                      "supp": "💊 Compléments", "water": "💧 Eau", "tip": "👉 Conseil"}

        return f"""<div class="feed-card">
  <p class="feed-title">{labels['title']}</p>
  <p class="feed-stats">🌡️ {w['temp']}°C &nbsp;|&nbsp; {emoji} {season.capitalize()} &nbsp;|&nbsp; 🕐 {time_of_day}</p>
  <div class="feed-row"><span class="feed-label">{labels['main']}</span><span>{rec['main_feed']}</span></div>
  <div class="feed-row"><span class="feed-label">{labels['supp']}</span><span>{rec['supplement']}</span></div>
  <div class="feed-row"><span class="feed-label">{labels['water']}</span><span>{rec['water']}</span></div>
  {warning_html}
  <p class="feed-tip">{labels['tip']} {rec.get('tip','')}</p>
</div>"""