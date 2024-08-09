# Description: datasets for the Heston model.


from datetime import datetime

from finmc.base.utils import flat_discount, flat_fwds
from finmc.heston import HestonMC


def data_heston_kruse():
    spot = 100  # spot doesn't matter, payoff is on returns

    # define dataset
    asset_name = "EQ"
    spot = 100
    rate = 0.0
    pricing_dt = datetime(2023, 12, 31)

    dataset = {
        "MC": {
            "PATHS": 100_000,
            "TIMESTEP": 1 / 250,
            "SEED": 1,
        },
        "BASE": "USD",
        "ASSETS": {
            "USD": flat_discount(rate, 3.0),
            asset_name: flat_fwds(spot, rate, 0.0, 3.0),
        },
        "HESTON": {
            "ASSET": asset_name,
            "INITIAL_VAR": 0.09,
            "LONG_VAR": 0.06,
            "VOL_OF_VAR": 0.65,
            "MEANREV": 4.0,
            "CORRELATION": -0.90,
        },
    }

    return (
        HestonMC,
        dataset,
        {"spot": spot, "rate": rate, "pricing_dt": pricing_dt},
    )
