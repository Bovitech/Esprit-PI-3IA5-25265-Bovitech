import json
import logging
import re

import requests

from chatbot.services import llm_client
from .base import BaseAgent

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class MeteoAgent(BaseAgent):

    def run(self, lat: float | None, lon: float | None, lang: str) -> str:
        if lat is None or lon is None:
            return (
                "📍 يرجى تفعيل الموقع الجغرافي للحصول على توقعات الطقس"
                if lang == "ar"
                else "📍 Veuillez activer la localisation pour obtenir la météo"
            )

        weather = self._fetch_weather(lat, lon)
        if weather is None:
            return (
                "تعذّر الحصول على بيانات الطقس." if lang == "ar"
                else "Impossible de récupérer la météo."
            )

        decision, reason, tip = self._llm_decision(weather, lang)

        # Hard safety override at night
        if not weather["is_day"]:
            decision = "in"
            reason   = "الوقت ليلاً — الأبقار يجب أن تكون في الداخل" if lang == "ar" else "Il fait nuit — les vaches doivent rester à l'intérieur"
            tip      = "أعد التحقق صباحاً" if lang == "ar" else "Revérifiez le matin"

        return self._build_card(weather, decision, reason, tip, lang)

    # ------------------------------------------------------------------ #

    def _fetch_weather(self, lat: float, lon: float) -> dict | None:
        try:
            params = (
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,precipitation,windspeed_10m,weathercode,is_day"
                f"&hourly=precipitation_probability"
                f"&forecast_days=1&windspeed_unit=kmh&timezone=auto"
            )
            res  = requests.get(OPEN_METEO_URL + params, timeout=10)
            data = res.json()
            cur  = data["current"]
            prob = data.get("hourly", {}).get("precipitation_probability", [])
            return {
                "temp":          cur["temperature_2m"],
                "rain":          cur["precipitation"],
                "wind":          cur["windspeed_10m"],
                "wcode":         cur["weathercode"],
                "is_day":        cur.get("is_day", 1),
                "next_3h_rain":  max(prob[:3]) if prob else 0,
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
            prompt = f"""أنت خبير في تربية الأبقار. قرر إذا كان يمكن إخراج الأبقار.

{ctx}

القواعد: ليل→داخل، حرارة>32→محدود، حرارة<0→داخل، مطر>2mm أو احتمال>60%→داخل، رياح>50→داخل، وإلا→خارج
أجب فقط بـ JSON: {{"decision":"out|in|limited","reason":"...","tip":"..."}}"""
        else:
            prompt = f"""Tu es un expert en élevage bovin. Décide si les vaches peuvent sortir.

{ctx}

Règles: nuit→in, temp>32→limited, temp<0→in, pluie>2mm ou prob>60%→in, vent>50→in, sinon→out
Réponds UNIQUEMENT en JSON: {{"decision":"out|in|limited","reason":"...","tip":"..."}}"""

        raw = llm_client.call_text([{"role": "user", "content": prompt}], temperature=0, max_tokens=150)
        if raw:
            try:
                parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
                return parsed.get("decision", "out"), parsed.get("reason", ""), parsed.get("tip", "")
            except Exception:
                pass
        return "out", "Données insuffisantes", "Surveiller les conditions"

    def _build_card(self, w: dict, decision: str, reason: str, tip: str, lang: str) -> str:
        ar = lang == "ar"

        if decision == "out":
            header  = "🌤️ الطقس مناسب"         if ar else "🌤️ Conditions favorables"
            verdict = "✅ يمكن إخراج الأبقار"   if ar else "✅ Les vaches peuvent sortir"
            color   = "#2d6a4f"
        elif decision == "in":
            if not w["is_day"]:
                header  = "🌙 وقت الليل"              if ar else "🌙 C'est la nuit"
                verdict = "❌ الأبقار تبقى في الداخل" if ar else "❌ Les vaches restent à l'intérieur"
            else:
                header  = "🌧️ الطقس غير مناسب"        if ar else "🌧️ Conditions défavorables"
                verdict = "❌ أبقِ الأبقار في الداخل" if ar else "❌ Gardez les vaches à l'intérieur"
            color = "#c0392b"
        else:
            header  = "⚠️ خروج محدود"                  if ar else "⚠️ Sortie limitée"
            verdict = "⚠️ الخروج صباحاً أو مساءً فقط" if ar else "⚠️ Sortir tôt matin / soir uniquement"
            color   = "#e67e22"

        time_label = "🌙 Nuit" if not w["is_day"] else "☀️ Jour"
        stats      = f"🌡️ {w['temp']}°C &nbsp;|&nbsp; 🌧️ {w['rain']}mm &nbsp;|&nbsp; 💨 {w['wind']} km/h &nbsp;|&nbsp; {time_label}"

        rain_warn = ""
        if w["next_3h_rain"] > 60:
            label   = "احتمال مطر خلال 3 ساعات" if ar else "Pluie probable dans 3h"
            rain_warn = f"<p class='meteo-rain-warn'>🌂 {label} ({w['next_3h_rain']}%)</p>"

        return f"""<div class="meteo-card" style="border-color:{color}">
  <p class="meteo-header">{header}</p>
  <p class="meteo-verdict" style="color:{color}">{verdict}</p>
  <p class="meteo-stats">{stats}</p>
  {rain_warn}
  <p class="meteo-reason">• {reason}</p>
  <p class="meteo-tip">👉 {tip}</p>
</div>"""