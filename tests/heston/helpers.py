# Helper functions for Heston model, which includes calibration,
# and closed form pricing for special cases.

# Based on https://github.com/cantaro86/Financial-Models-Numerical-Methods/

from functools import partial

import numpy as np
from scipy.integrate import quad

from finmc.base.utils import Forwards, discounter_from_dataset


def cf_heston(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta)
        / (sigma**2)
        * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2)
        * (xi - d)
        * (1 - np.exp(-d * t))
        / (1 - g2 * np.exp(-d * t))
    )
    return cf


def Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real(
            (np.exp(-u * k * 1j) / (u * 1j))
            * cf(u - 1j)
            / cf(-1.0000000000001j)
        )

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]


def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]


def price_vanilla_call(
    K,  # strike
    T,  # option maturity in years
    asset_name,
    dataset,
):
    """Calculate the price of a Vanilla European Option using Heston Model characteristic function."""
    # TODO feat: handle time dependent params.

    discounter = discounter_from_dataset(dataset)
    asset_fwds = Forwards(dataset["ASSETS"][asset_name])
    heston_data = dataset["HESTON"]

    r = discounter.rate(T)
    S0 = asset_fwds.forward(0)  # current Spot.
    mu = asset_fwds.rate(T)

    cf_reduced = partial(
        cf_heston,
        t=T,
        v0=heston_data["INITIAL_VAR"],
        mu=mu,
        theta=heston_data["LONG_VAR"],
        sigma=heston_data["VOL_OF_VAR"],
        kappa=heston_data["MEANREV"],
        rho=heston_data["CORRELATION"],
    )
    limit_max = 1000
    k = np.log(K / S0)  # log strike
    price = S0 * np.exp((mu - r) * T) * Q1(
        k, cf_reduced, limit_max
    ) - K * np.exp(-r * T) * Q2(k, cf_reduced, limit_max)
    return price, {}


# TODO feat. a forward starting option can be evaluated using cf as in
# 10.1007/s00780-004-0146-3
# It requires a double integral, and approximation of a Bessel function.
