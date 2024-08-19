# Description: datasets for the LV MC and BS FD model.

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from finmc.models.localvol import BSMC, LVMC


# Description: datasets for the LVMC model.
def _bs_vol(points):
    """A simple BS implied vol function with no skew."""
    (t, x_vec) = points
    var = 0.04 + 0.04 * (1 - np.exp(-0.5 * t))  # No skew
    return np.sqrt(var)


def _local_vol(points):
    """A local vol function derived from the above BS Vol function."""
    (t, x_vec) = points
    f = np.exp(-0.5 * t)
    var = 0.04 + 0.04 * (1 - f + 0.5 * t * f)
    return np.sqrt(var)


def _interp(volfn):
    """Return a RegularGridInterpolator for the vol function."""
    num_t = 10
    num_x = 20
    times = np.linspace(0.0, 3.0, num_t)
    strikes = np.linspace(-5.0, 5.0, num_x)
    vi = np.zeros((num_t, num_x))
    for i in range(num_t):
        vi[i, :] = volfn((times[i], strikes))
    return RegularGridInterpolator(
        (times, strikes), vi, fill_value=None, bounds_error=False
    )


def data_lvmc():
    """Data for the LVMC model."""

    # define dataset
    # first define the discount curve and forward curve
    times = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    term_rates = np.array([0.04, 0.04, 0.045, 0.05, 0.05])
    div_rate = 0.01
    discount_data = ("ZERO_RATES", np.column_stack((times, term_rates)))

    asset_name = "SPX"
    spot = 2900
    fwds = spot * np.exp((term_rates - div_rate) * times)
    fwd_data = ("FORWARDS", np.column_stack((times, fwds)))

    dataset = {
        "MC": {
            "PATHS": 100_000,
            "TIMESTEP": 1 / 10,
            "SEED": 1,
        },
        "BASE": "USD",
        "ASSETS": {"USD": discount_data, asset_name: fwd_data},
        "BS": {
            "ASSET": "SPX",
            "VOL": 0.3,
        },  # For closed form evaluation
        "LV": {
            "ASSET": "SPX",
            "VOL": 0.3,
        },
    }

    return LVMC, dataset, {"spot": spot}


def data_lvmc_grid():
    """Test when vol is a RegularGridInterpolator."""
    model_cls, dataset, other = data_lvmc()
    dataset["BS"]["VOL"] = _interp(_bs_vol)
    dataset["LV"]["VOL"] = _interp(_local_vol)

    return model_cls, dataset, other


def data_lvmc_fn():
    """Test when vol is a function."""
    model_cls, dataset, other = data_lvmc()
    dataset["BS"]["VOL"] = _bs_vol
    dataset["LV"]["VOL"] = _local_vol

    return model_cls, dataset, other


def data_bsmc():
    """Test when vol is a RegularGridInterpolator."""
    _, dataset, other = data_lvmc()

    return BSMC, dataset, other
