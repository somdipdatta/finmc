# Description: Tests for a custom MC Hull White model with ZCB.

import numpy as np

from finmc.models.hullwhite import HullWhiteMC


def data_hwmc():
    """Data for the MC HW model."""

    # define dataset
    # first define the discount curve
    times = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    term_rates = np.array([0.04, 0.04, 0.045, 0.05, 0.05])
    discount_data = ("ZERO_RATES", np.column_stack((times, term_rates)))

    dataset = {
        "MC": {
            "PATHS": 100_000,
            "TIMESTEP": 1 / 250,
            "SEED": 1,
        },
        "BASE": "USD",
        "ASSETS": {"USD": discount_data},
        "HW": {
            "ASSET": "USD",
            "MEANREV": 0.1,
            "VOL": 0.03,
        },
    }

    return HullWhiteMC, dataset, {}
