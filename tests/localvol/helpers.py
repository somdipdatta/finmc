# Helper functions for Black Scholes model, which includes calibration,
# and closed form pricing for special cases.

import numpy as np
from scipy.stats import norm

from finmc.base.utils import Forwards, discounter_from_dataset

N = norm.cdf


def d1_d2(
    K,  # strike
    T,  # option maturity in years
    asset_name,
    dataset,
):
    """Calculate d1, d2 from Black Scholes."""

    asset_fwds = Forwards(dataset["ASSETS"][asset_name])
    sigma = dataset["BS"]["VOL"]

    if callable(sigma):
        spot = asset_fwds.forward(0)
        x = np.log(K / spot)
        sigma = float(sigma((T, x)))

    F = asset_fwds.forward(T)  # forward

    d1 = (np.log(F / K) + (sigma * sigma / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price_vanilla_opt(
    K,  # strike
    T,  # option maturity in years
    option_type,  # "Call" or "Put"
    asset_name,
    dataset,
):
    """Calculate the price of a Vanilla European Option using Closed from Black Scholes."""

    d1, d2 = d1_d2(K, T, asset_name, dataset)

    discounter = discounter_from_dataset(dataset)
    df = discounter.discount(T)

    asset_fwds = Forwards(dataset["ASSETS"][asset_name])
    F = asset_fwds.forward(T)

    if option_type == "Call":
        price = F * N(d1) - K * N(d2)
    else:
        price = K * N(-d2) - F * N(-d1)

    return price * df, {}


def price_digital_opt(
    K,  # strike
    T,  # option maturity in years
    option_type,  # "Call" or "Put"
    asset_name,
    dataset,
    apply_discounter=True,
):
    """Calculate the price of a Digital European Call Option using Closed from Black Scholes."""

    _, d2 = d1_d2(K, T, asset_name, dataset)

    discounter = discounter_from_dataset(dataset)
    df = discounter.discount(T)

    if option_type == "Call":
        price = N(d2)
    else:
        price = N(-d2)

    if apply_discounter:
        price = price * df
    return price, {}


def price_vanilla_option_sim(
    K,  # strike
    T,  # option maturity in years
    option_type,  # "Call" or "Put"
    asset_name,
    dataset,
    model,
):
    """Calculate the price of a Vanilla European Option using MC Simulation from Black Scholes."""

    # get price from timetable
    model.advance(T)
    vals = model.get_value(asset_name)
    discounter = discounter_from_dataset(dataset)
    df = discounter.discount(T)
    if option_type == "Call":
        price = np.maximum(vals - K, 0).mean() * df
    else:
        price = np.maximum(K - vals, 0).mean() * df
    return price
