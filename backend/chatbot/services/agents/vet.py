import logging
import requests
from chatbot.utils.geo import haversine
from .base import BaseAgent

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADII = [5000, 10000, 20000]


class VetAgent(BaseAgent):

    def run(self, lat: float | None, lon: float | None, lang: str) -> str:
        if lat is None or lon is None:
            return (
                "📍 يرجى تفعيل الموقع الجغرافي للعثور على طبيب بيطري قريب"
                if lang == "ar"
                else "📍 Veuillez activer la localisation pour trouver un vétérinaire proche"
            )
        return self._find_vets(lat, lon, lang)

    def _find_vets(self, lat: float, lon: float, lang: str) -> str:
        vets = []

        for radius in SEARCH_RADII:
            query = f"""
            [out:json][timeout:25];
            (
            node["amenity"="veterinary"](around:{radius},{lat},{lon});
            way["amenity"="veterinary"](around:{radius},{lat},{lon});
            );
            out center;
            """
            try:
                res = requests.post(
                    OVERPASS_URL,
                    data={"data": query},  # Correct Overpass POST format
                    timeout=30,
                    headers={"Accept": "application/json"},
                )
                res.raise_for_status()
                
                # Guard against empty or HTML response
                content = res.text.strip()
                if not content or content.startswith("<"):
                    logger.warning("Overpass returned non-JSON (radius=%d), retrying...", radius)
                    continue
                    
                data = res.json()
            except requests.exceptions.Timeout:
                logger.warning("Overpass timeout (radius=%d)", radius)
                continue
            except Exception as exc:
                logger.warning("Overpass request failed (radius=%d): %s", radius, exc)
                continue

            for el in data.get("elements", []):
                tags = el.get("tags", {})
                vlat = el.get("lat")
                vlon = el.get("lon")
                if vlat is None or vlon is None:
                    continue
                vets.append({
                    "name":     tags.get("name", "Vétérinaire"),
                    "phone":    tags.get("phone"),
                    "lat":      vlat,
                    "lon":      vlon,
                    "distance": round(haversine(lat, lon, vlat, vlon), 2),
                })

            if vets:
                break

        if not vets:
            return """<div class="vet-card" style="border-color:#e67e22">
        <p>⚠️ <strong>Aucun vétérinaire trouvé via OpenStreetMap</strong></p>
        <p>Les données peuvent être incomplètes dans votre région.</p>
        <p>👉 Essayez <a href="https://www.google.com/maps/search/vétérinaire" target="_blank" class="map-btn">Google Maps</a></p>
        </div>"""

        vets.sort(key=lambda v: v["distance"])
        best   = vets[0]
        others = vets[1:3]

        def map_link(v: dict) -> str:
            return f"https://www.google.com/maps/search/?api=1&query={v['lat']},{v['lon']}"

        warning    = "<p>⚠️ Vétérinaire le plus proche (>10 km) :</p>" if best["distance"] > 10 else ""
        phone_html = f"<p>📞 {best['phone']}</p>" if best["phone"] else "<p>📞 Téléphone non disponible</p>"
        others_html = ""
        if others:
            items       = "".join(f"<li>{v['name']} — {v['distance']} km</li>" for v in others)
            others_html = f"<p><strong>🔹 Autres options :</strong></p><ul>{items}</ul>"

        return f"""{warning}
<div class="vet-card">
  <p>🐄 <strong>Vétérinaire recommandé</strong></p>
  <p>📍 <strong>{best['name']}</strong> — {best['distance']} km</p>
  {phone_html}
  <a href="{map_link(best)}" target="_blank" class="map-btn">📍 Voir sur Google Maps</a>
</div>
{others_html}"""