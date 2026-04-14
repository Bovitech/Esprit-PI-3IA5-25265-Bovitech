"""
contracts.py — All response shapes in one place.

Rules:
  - views.py imports from here — never builds dicts inline
  - _err() helper in views.py is REMOVED — use make_error() here instead
  - Every response has a `type` field so the frontend can branch cleanly
"""
from django.http import JsonResponse


# ------------------------------------------------------------------ #
# Success shapes
# ------------------------------------------------------------------ #

def text_response(content: str) -> dict:
    """Normal LLM text answer or language-gate message."""
    return {"type": "text", "content": content}


def agent_response(agent: str, data: dict) -> dict:
    """
    Structured agent result.
    agent: "vet" | "meteo" | "feed"
    data:  agent-specific dict (see agent files for shape)
    """
    return {"type": "agent", "agent": agent, "data": data}


# ------------------------------------------------------------------ #
# Error shape — single entry point, used everywhere
# ------------------------------------------------------------------ #

def make_error(code: str, message: str, status: int = 400) -> JsonResponse:
    """
    Return a JsonResponse with a uniform error shape.

    Usage in views:
        return make_error("EMPTY_MESSAGE", "Message must not be empty.")
        return make_error("TTS_FAILED", "Audio synthesis failed.", status=500)

    Shape:
        {"error": {"code": "...", "message": "..."}}
    """
    return JsonResponse(
        {"error": {"code": code, "message": message}},
        status=status,
    )


# ------------------------------------------------------------------ #
# Agent data shapes (documented, not enforced — Python is not TS)
# ------------------------------------------------------------------ #

# vet_data = {
#     "found": bool,
#     "best": {
#         "name": str,
#         "distance_km": float,
#         "phone": str | None,
#         "map_url": str,
#     } | None,
#     "others": [{"name": str, "distance_km": float}],
#     "warning": str | None,
# }

# meteo_data = {
#     "decision": "out" | "in" | "limited",
#     "temp": float,
#     "rain": float,
#     "wind": float,
#     "is_day": bool,
#     "next_3h_rain_pct": int,
#     "reason": str,
#     "tip": str,
# }

# feed_data = {
#     "season": str,
#     "temp": float,
#     "is_day": bool,
#     "main_feed": str,
#     "supplement": str,
#     "water": str,
#     "warning": str,
#     "tip": str,
# }