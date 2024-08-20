"""
Utility to calculate implied volatility using closed form Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm

N = norm.cdf


def d1_d2(
    K,  # strike
    T,  # option maturity in years
    F,
    vol,
):
    """Calculate d1, d2 from Black Scholes."""
    d1 = (np.log(F / K) + (vol * vol / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return d1, d2


def impliedvol(target, F, K, T, is_call):
    """Find the implied volatility of a European option using the Black-Scholes model.

    Args:
        target (float): The target value of the option, undiscounted.
        F (float): The forward price of the underlying.
        K (float): The strike price of the option.
        T (float): The time to maturity of the option in years.
        is_call (bool): True if the option is a call, False if put.

    """
    if target < 1e-4 * F:
        return None
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    vol = 0.5
    for i in range(0, MAX_ITERATIONS):
        d1, d2 = d1_d2(K, T, F, vol)

        if is_call:
            price = F * N(d1) - K * N(d2)
        else:
            price = K * N(-d2) - F * N(-d1)
        vega = F * N(d1) * np.sqrt(T)
        diff = target - price  # our root
        if abs(diff) < PRECISION:
            return vol
        vol = vol + diff / vega  # f(x) / f'(x)
    return vol


def opt_price(
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


def digital_price(
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
