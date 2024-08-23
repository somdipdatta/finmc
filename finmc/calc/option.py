"""
Utility to calculate prices of european contracts from a MC Simulation model.
"""

import numpy as np

from finmc.models.base import MCBase


def opt_price_mc(
    strike: float,
    maturity: float,
    option_type: str,
    asset_name: str,
    model: MCBase,
) -> float:
    """Calculate the price of a Vanilla European Option using MC Simulation.

    Args:
        strike: The strike price of the option.
        maturity: The time to maturity of the option in years.
        option_type: The type of the option. Either "Call" or "Put".
        asset_name: The name of the asset.
        model: The model used to simulate the asset price.

    Returns:
        The price of the option.

    Examples:
        price = opt_price_mc(K, T, "Call", "SPX", model)
    """

    model.advance(maturity)
    expiration_spots = model.get_value(asset_name)
    df = model.get_df()

    if option_type == "Call":
        price = np.maximum(expiration_spots - strike, 0).mean() * df
    else:
        price = np.maximum(strike - expiration_spots, 0).mean() * df
    return price


if __name__ == "__main__":
    from finmc.models.localvol import LVMC

    dataset = {
        "MC": {
            "PATHS": 100_000,
            "TIMESTEP": 1 / 10,
            "SEED": 1,
        },
        "BASE": "USD",
        "ASSETS": {
            "USD": ("ZERO_RATES", np.array([[1.0, 0.04]])),
            "SPX": ("FORWARDS", np.array([[0.0, 2900], [1.0, 3000]])),
        },
        "LV": {
            "ASSET": "SPX",
            "VOL": 0.3,
        },
    }

    model = LVMC(dataset)
    price = opt_price_mc(
        strike=2900,
        maturity=1 / 12,
        option_type="Call",
        asset_name="SPX",
        model=model,
    )
    print(f"price = {price}")
