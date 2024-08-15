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

    def get_df(self):
        """Return the discount factor at the current time."""
        ...

    @abstractmethod
    def advance(self, new_time: float):
        """Advance the model to a new time. It might require multiple time steps."""
        ...


class MCFixedStep(ABC):
    def advance(self, new_time):
        while new_time > self.cur_time + self.dt:
            self.advance_step(self.cur_time + self.dt)
        if new_time > self.cur_time + 1e-10:
            self.advance_step(new_time)

    @abstractmethod
    def advance_step(self, new_time: float):
        """Advance the model with a single time step."""
        ...
