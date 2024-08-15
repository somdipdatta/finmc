# Utility classes and functions for models.

import numpy as np
from scipy import interpolate


# Define a class for discount factors and rates.
class Discounter:
    """A class for discount factors and rates."""

    def __init__(self, discount_data):
        data_type, data = discount_data
        if data[0, 0] > 0.0:
            data = np.insert(data, [0], [0.0, 0.0], axis=0)
        if data_type == "LOG_DISCOUNTS":
            # the columns are times and log discounts
            times = data[:, 0]
            log_discounts = data[:, 1]
        elif data_type == "ZERO_RATES":
            # the columns are times and zero rates
            times = data[:, 0]
            zero_rates = data[:, 1]
            log_discounts = -zero_rates * times

        self.log_discount_fn = interpolate.interp1d(times, log_discounts)

    def rate(self, end, start=0):
        ld_end, ld_start = self.log_discount_fn([end, start])

        return (ld_start - ld_end) / (end - start)

    def discount(self, t):
        return np.exp(self.log_discount_fn(t))


# Define a class for forwards of an asset.
class Forwards:
    """A class for forwards and forward rates."""

    def __init__(self, forwards_data):
        _, data = forwards_data

        times = data[:, 0]
        fwds = data[:, 1]

        self.log_forward_fn = interpolate.interp1d(times, np.log(fwds))

    def rate(self, end, start=0):
        try:
            ld_end, ld_start = self.log_forward_fn([end, start])
        except ValueError as e:
            raise ValueError(
                f"Error calculating forward rate from time {start} to {end}, caused by \n {e}. \n Perhaps the time range in forwards/discount data is too short."
            )
        return (ld_end - ld_start) / (end - start)

    def forward(self, t):
        return np.exp(self.log_forward_fn(t))


def flat_discount(rate, max_time):
    """Create a flat discount curve."""
    return ("ZERO_RATES", np.array([[max_time, rate]]))


def flat_fwds(spot, rate, div_rate, max_time):
    """Create a flat forward curve."""
    times = np.array([0.0, max_time])
    fwds = spot * np.exp((rate - div_rate) * times)
    return ("FORWARDS", np.column_stack((times, fwds)))
