# Helper functions for Black Scholes model, which includes calibration,
# and closed form pricing for special cases.

import numpy as np
from scipy.stats import norm

N = norm.cdf


def d1_d2(
    K,  # strike
    T,  # option maturity in years
    F,
    sigma,
):
    """Calculate d1, d2 from Black Scholes."""
    d1 = (np.log(F / K) + (sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def opt_price_bs(
    K,  # strike
    T,  # option maturity in years
    option_type,  # "Call" or "Put"
    F,  # forward
    df,  # discount factor
    sigma,  # volatility
):
    """Calculate the price of a Vanilla European Option using Closed from Black Scholes."""

    d1, d2 = d1_d2(K, T, F, sigma)

    if option_type == "Call":
        price = F * N(d1) - K * N(d2)
    else:
        price = K * N(-d2) - F * N(-d1)

    return price * df, {}


def digital_price_bs(
    K,  # strike
    T,  # option maturity in years
    option_type,  # "Call" or "Put"
    F,  # forward
    df,  # discount factor
    sigma,  # volatility
    apply_discounter=True,
):
    """Calculate the price of a Digital European Call Option using Closed from Black Scholes."""

    _, d2 = d1_d2(K, T, F, sigma)

    if option_type == "Call":
        price = N(d2)
    else:
        price = N(-d2)

    if apply_discounter:
        price = price * df
    return price, {}


def opt_price_sim(
    strike,
    maturity,  # in years
    option_type,  # "Call" or "Put"
    asset_name,
    dataset,
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
