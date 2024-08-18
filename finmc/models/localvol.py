from math import sqrt

import numpy as np
from numpy.random import SFC64, Generator

from finmc.models.base import MCFixedStep
from finmc.utils.assets import Discounter, Forwards


# Define a class for the state of a single asset BS Local Vol MC process
class LVMC(MCFixedStep):
    def reset(self, dataset):
        # fetch the model parameters from the dataset
        self.n = dataset["MC"]["PATHS"]
        self.asset = dataset["LV"]["ASSET"]
        self.asset_fwd = Forwards(dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)
        self.vol = dataset["LV"]["VOL"]
        self.logspot = np.log(self.spot)
        self.discounter = Discounter(dataset["ASSETS"][dataset["BASE"]])

        # Initialize rng and any arrays
        self.rng = Generator(SFC64(dataset["MC"].get("SEED")))
        self.x_vec = np.zeros(self.n)  # process x (log stock)
        self.dz_vec = np.empty(self.n, dtype=np.float64)
        self.tmp = np.empty(self.n, dtype=np.float64)

        self.timestep = dataset["MC"]["TIMESTEP"]

        self.cur_time = 0

    def advance_step(self, new_time):
        """Update x_vec in place when we move simulation by time dt."""

        dt = new_time - self.cur_time

        fwd_rate = self.asset_fwd.rate(new_time, self.cur_time)
        fwd = self.asset_fwd.forward(self.cur_time)
        logfwd_shift = np.log(fwd) - self.logspot
        if callable(self.vol):
            vol = self.vol((self.cur_time, self.x_vec - logfwd_shift))
        else:
            vol = self.vol

        # # generate the random numbers and advance the log stock process
        self.rng.standard_normal(self.n, out=self.dz_vec)
        self.dz_vec *= sqrt(dt)
        self.dz_vec *= vol

        # add drift to x_vec: (fwd_rate - vol * vol / 2.0) * dt
        np.multiply(vol, vol, out=self.tmp)
        self.tmp *= -0.5 * dt
        self.tmp += fwd_rate * dt
        self.x_vec += self.tmp
        # add the random part to x_vec
        self.x_vec += self.dz_vec

        self.cur_time = new_time

    def get_value(self, unit):
        """Return the value of the modeled asset at the current time."""
        if unit == self.asset:
            return self.spot * np.exp(self.x_vec)

    def get_df(self):
        return self.discounter.discount(self.cur_time)
