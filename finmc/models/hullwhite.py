# Custom Monte Carlo Pricer for a HullWhite Model

from math import sqrt

import numpy as np
from numpy.random import SFC64, Generator

from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter


# Define a class for the state of a single asset HullWhite MC process
class HullWhiteMC(MCFixedStep):
    def __init__(self, dataset):
        self.shape = dataset["MC"]["PATHS"]
        self.dt = dataset["MC"]["TIMESTEP"]

        # create a random number generator
        self.rng = Generator(SFC64(dataset["MC"]["SEED"]))

        self.asset = dataset["HW"]["ASSET"]
        self.asset_fwd = Discounter(dataset["ASSETS"][self.asset])

        self.meanrev = dataset["HW"]["MEANREV"]
        self.vol = dataset["HW"]["VOL"]

        # Initialize the arrays for the processes
        self.x_vec = np.zeros(self.shape, dtype=np.float64)  # x
        self.r_vec = np.zeros(self.shape, dtype=np.float64)  # r (short rate)
        self.df_vec = np.ones(self.shape, dtype=np.float64)  # discount factors

        # We will reduce time spent in memory allocation by creating a tmp arrays
        self.tmp_vec = np.empty(self.shape, dtype=np.float64)

        self.cur_time = 0

    def advance_step(self, new_time):
        """Update x_vec, v_vec in place when we move simulation by time dt."""
        dt = new_time - self.cur_time

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

        np.multiply(self.df_vec, np.exp(-self.r_vec * dt), out=self.df_vec)
        # np.exp(-self.r_vec * dt)  # return the discount factor

    def get_df(self):
        """Return the discount factor at the current time."""
        return self.df_vec  # discount factor
