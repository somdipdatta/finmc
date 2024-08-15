"""
Utility to calculate implied volatility surface from a MC Simulation model.
"""

import numpy as np

from finmc.calc.bs import find_vol


def iv_surface_sim(
    strikes,
    expirations,  # in years, increasing order
    asset_name: str,
    model,
) -> np.ndarray:
    iv_mat = np.zeros((len(expirations), len(strikes)))
    for i, exp in enumerate(expirations):
        model.advance(exp)
        expiration_spots = model.get_value(asset_name)
        fwd = expiration_spots.mean()

        # Use a call option for strikes above forward, a put option otherwise
        is_call = strikes > fwd
        is_call_c = is_call[..., None]  # Turn into a column vector

        # calculate prices (value as of expiration date)
        strikes_c = strikes[..., None]  # Turn into a column vector
        pay = np.where(
            is_call_c,
            expiration_spots - strikes_c,
            strikes_c - expiration_spots,
        )
        prices = np.maximum(pay, 0).mean(axis=1)

        # calculate implied vols
        iv_mat[i, :] = [
            find_vol(p, fwd, k, exp, ic)
            for p, k, ic in zip(prices, strikes, is_call)
        ]

    return iv_mat


if __name__ == "__main__":
    import pandas as pd

    from finmc.models.localvol import LVMC

    # create the dataset
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

    # create the model and calculate the implied volatility surface
    model = LVMC(dataset)
    strikes = np.linspace(2900, 3100, 3)
    expirations = [1 / 12, 1 / 6, 1 / 4, 1 / 2, 1]
    surface = iv_surface_sim(
        strikes,
        expirations,
        asset_name="SPX",
        model=model,
    )
    # print the surface as a DataFrame
    df = pd.DataFrame(surface, columns=strikes, index=expirations)
    print(f"surface:\n{df}")
