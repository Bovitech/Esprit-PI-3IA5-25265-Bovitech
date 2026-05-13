"""
BaseAgent — every agent must implement run().
"""
from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def run(self, lat: float | None, lon: float | None, lang: str) -> dict | None:
        """
        Execute the agent and return a data dict for agent_response(),
        or None if the agent cannot complete (e.g. weather fetch failed).

        lat/lon may be None if the user has not shared their location.
        views.py handles the None case by returning a text_response instead.
        """
        ...