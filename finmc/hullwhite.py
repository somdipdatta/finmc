# Custom Monte Carlo Pricer for a HullWhite Model

from math import sqrt

import numpy as np
from numpy.random import SFC64, Generator

from finmc.base.mc import MCBase
from finmc.base.utils import Discounter


# Define a class for the state of a single asset HullWhite MC process
class HullWhiteMC(MCBase):
    def __init__(self, dataset):
        self.shape = dataset["MC"]["PATHS"]
        self.dt = dataset["MC"]["TIMESTEP"]

        # create a random number generator
        self.rng = Generator(SFC64(dataset["MC"]["SEED"]))

        self.asset = dataset["HW"]["ASSET"]
        self.asset_fwd = Discounter(dataset["ASSETS"][self.asset])

        self.meanrev = dataset["HW"]["MEANREV"]
        self.vol = dataset["HW"]["VOL"]

        # Initialize the arrays
        self.x_vec = np.zeros(self.shape)  # processes x
        self.r_vec = np.zeros(self.shape)  # processes r (short rate)

        # We will reduce time spent in memory allocation by creating a tmp arrays
        self.tmp_vec = np.empty(self.shape, dtype=np.float64)

        self.cur_time = 0

    def advance(self, new_time):
        while new_time > self.cur_time + self.dt:
            self._advance(self.cur_time + self.dt)
        if new_time > self.cur_time + 1e-10:
            self._advance(new_time)

    def _advance(self, new_time):
        """Update x_vec, v_vec in place when we move simulation by time dt."""
        dt = new_time - self.cur_time
        if dt < 1e-10:
            return 1

        sqrtdt = sqrt(dt)

        # update the current value of x
        # first term: x -= a * x * dt
        np.multiply(self.x_vec, self.meanrev * dt, out=self.tmp_vec)
        np.subtract(self.x_vec, self.tmp_vec, out=self.x_vec)

        # second term: x += vol * dW
        self.rng.standard_normal(self.shape, out=self.tmp_vec)
        np.multiply(sqrtdt * self.vol, self.tmp_vec, out=self.tmp_vec)
        np.add(self.x_vec, self.tmp_vec, out=self.x_vec)

        # r = x + alpha
        fwd_rate = self.asset_fwd.rate(new_time, self.cur_time)
        temp = (self.vol / self.meanrev) * (
            1 - np.exp(-self.meanrev * new_time)
        )
        alpha = fwd_rate + temp * temp / 2
        np.add(self.x_vec, alpha, out=self.r_vec)

        self.cur_time = new_time

        return np.exp(-self.r_vec * dt)  # return the discount factor
