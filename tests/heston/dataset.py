# Description: datasets for the Heston model.

from finmc.models.heston import HestonMC
from finmc.utils.assets import flat_discount, flat_fwds


def data_heston_kruse():
    """Define the dataset from Kruse."""
    asset_name = "EQ"
    spot = 100
    rate = 0.0

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
            "VOL_OF_VOL": 0.65,
            "MEANREV": 4.0,
            "CORRELATION": -0.90,
        },
    }

    return (
        HestonMC,
        dataset,
        {"spot": spot, "rate": rate},
    )
