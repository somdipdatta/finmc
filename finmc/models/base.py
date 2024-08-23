# Generic MC model

from abc import ABC, abstractmethod


class MCBase(ABC):
    """Base class for a Monte-Carlo process."""

    stats: dict = {}

    def __init__(self, dataset: dict) -> None:
        self.reset(dataset)

    @abstractmethod
    def reset(self, dataset: dict):
        """The derived class must implement this method to reset the state of the model
        to time zero."""
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
        """The derived class must implement this method to advance the state of the model
        to a new time. The model may do so in multiple time steps."""
        ...


class MCFixedStep(MCBase):
    """A Monte-Carlo process which breaks down the 'advance' step into fixed time steps
    specified by the TIMESTEP parameter."""

    def advance(self, new_time):
        while new_time > self.cur_time + self.timestep:
            self.advance_step(self.cur_time + self.timestep)
        if new_time > self.cur_time + 1e-10:
            self.advance_step(new_time)

    @abstractmethod
    def advance_step(self, new_time: float):
        """The derived class must implement this method, which advances the model by a timestep equal to or
        less than the TIMESTEP parameter."""
        ...
