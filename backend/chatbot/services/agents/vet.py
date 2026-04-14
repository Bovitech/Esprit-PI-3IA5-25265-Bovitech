import logging

import requests

from chatbot.utils.geo import haversine
from .base import BaseAgent

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADII = [5000, 10000, 20000]


class VetAgent(BaseAgent):

    def run(self, lat: float | None, lon: float | None, lang: str) -> dict:
        if lat is None or lon is None:
            return {
                "found": False,
                "best": None,
                "others": [],
                "warning": (
                    "يرجى تفعيل الموقع الجغرافي للعثور على طبيب بيطري قريب"
                    if lang == "ar"
                    else "Veuillez activer la localisation pour trouver un vétérinaire proche"
                ),
            }
        return self._find_vets(lat, lon)

    def _find_vets(self, lat: float, lon: float) -> dict:
        vets = []

        for radius in SEARCH_RADII:
            query = f"""
            [out:json][timeout:25];
            node["amenity"="veterinary"](around:{radius},{lat},{lon});
            out;
            """
            try:
                res = requests.post(
                    OVERPASS_URL,
                    data={"data": query},
                    timeout=30,
                    headers={"Accept": "application/json"},
                )
                res.raise_for_status()

                content = res.text.strip()
                if not content or content.startswith("<"):
                    logger.warning(
                        "Overpass non-JSON response (radius=%d)", radius
                    )
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
                    "name":        tags.get("name", "Vétérinaire"),
                    "phone":       tags.get("phone"),
                    "lat":         vlat,
                    "lon":         vlon,
                    "distance_km": round(haversine(lat, lon, vlat, vlon), 2),
                })

            if vets:
                break

        if not vets:
            return {"found": False, "best": None, "others": [], "warning": None}

        vets.sort(key=lambda v: v["distance_km"])
        best   = vets[0]
        others = vets[1:3]

        def map_url(v: dict) -> str:
            return f"https://www.google.com/maps/search/?api=1&query={v['lat']},{v['lon']}"

        logger.info(
            "action=vet_search found=%d best=%s dist=%.1fkm",
            len(vets), best["name"], best["distance_km"]
        )

        return {
            "found": True,
            "best": {
                "name":        best["name"],
                "distance_km": best["distance_km"],
                "phone":       best.get("phone"),
                "map_url":     map_url(best),
            },
            "others": [
                {"name": v["name"], "distance_km": v["distance_km"], "map_url": map_url(v)}
                for v in others
            ],
            "warning": "Vétérinaire éloigné (>10 km)" if best["distance_km"] > 10 else None,
        }