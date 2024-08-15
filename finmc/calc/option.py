import numpy as np


def opt_price_sim(
    strike,
    maturity,  # in years
    option_type,  # "Call" or "Put"
    asset_name,
    model,
):
    """Calculate the price of a Vanilla European Option using MC Simulation."""

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
    price = opt_price_sim(
        strike=2900,
        maturity=1 / 12,
        option_type="Call",
        asset_name="SPX",
        model=model,
    )
    print(f"price = {price}")
