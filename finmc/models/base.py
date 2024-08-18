# Generic MC model

from abc import ABC, abstractmethod


# Define Base Class for State Object for MC Models
# Todo add the abstract methods and what else is expected from this class.
class MCBase(ABC):
    """Class to maintain the state of a single asset MC process."""

    stats: dict = {}

    def __init__(self, dataset: dict) -> None:
        self.reset(dataset)

    @abstractmethod
    def reset(self, dataset: dict):
        """Reset the state of the model."""
        ...

    def set_stat(self, key: str, val):
        self.stats[key] = val

    def get_value(self, unit):
        """Return the value of the asset at the current time,
        if this asset is handled by the model, otherwise return None.
        The return value is none, float, or an np array of floats."""
        return None

    def get_df(self):
        """Return the discount factor at the current time.
        The return value is a float, or an np array of floats."""
        ...

    @abstractmethod
    def advance(self, new_time: float):
        """Advance the model to a new time. It might require multiple time steps."""
        ...


class MCFixedStep(MCBase):
    def advance(self, new_time):
        while new_time > self.cur_time + self.timestep:
            self.advance_step(self.cur_time + self.timestep)
        if new_time > self.cur_time + 1e-10:
            self.advance_step(new_time)

    @abstractmethod
    def advance_step(self, new_time: float):
        """Advance the model with a single time step."""
        ...
