# Generic MC model

from abc import ABC, abstractmethod


# Define Base Class for State Object for MC Models
# Todo add the abstract methods and what else is expected from this class.
class MCBase(ABC):
    """Class to maintain the state of a single asset MC process."""

    stats: dict = {}

    def set_stat(self, key: str, val):
        self.stats[key] = val

    def get_value(self, unit):
        """Return the value of the asset at the current time,
        if this asset is handled by the model, otherwise return None."""
        return None

    @abstractmethod
    def advance(self, new_time: float): ...
