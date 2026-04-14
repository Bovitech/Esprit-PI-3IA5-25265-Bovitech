"""
BaseAgent — every agent must implement run().
"""
from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def run(self, lat: float | None, lon: float | None, lang: str) -> str:
        """
        Execute the agent and return an HTML string ready for the frontend.
        lat/lon may be None if the user has not shared their location.
        """
        ...