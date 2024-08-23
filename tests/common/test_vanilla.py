"""
Testing Vanilla Option for all Black Scholes models.
"""

import numpy as np
import pytest
from pytest import approx

from finmc.calc.option import opt_price_mc
from finmc.utils.assets import Discounter, Forwards
from finmc.utils.bs import opt_price
from tests.localvol.dataset import (
    data_bsmc,
    data_lvmc,
    data_lvmc_fn,
    data_lvmc_grid,
)


# datasets for all models that can price a vanilla option.
@pytest.fixture(
    params=[
        data_bsmc,
        data_lvmc,
        data_lvmc_fn,
        data_lvmc_grid,
    ]
)
def data(request):
    return request.param()


@pytest.mark.parametrize("maturity", [1, 12, 36])  # in months
@pytest.mark.parametrize("strike_x", [0.01, 0.95, 1.00, 1.05, 1.2, 10])
def test_call(data, maturity, strike_x):
    """Test the price of a vanilla call option."""

    model_cls, dataset, other = data
    spot = other["spot"]
    asset_name = dataset["BS"]["ASSET"]

    strike = strike_x * spot
    maturity_yrs = maturity / 12

    model = model_cls(dataset)
    price = opt_price_mc(
        strike,
        maturity_yrs,
        "Call",
        asset_name,
        model,
    )

    # get closed form price
    discounter = Discounter(dataset["ASSETS"][dataset["BASE"]])
    df = discounter.discount(maturity_yrs)

    asset_fwds = Forwards(dataset["ASSETS"][asset_name])
    F = asset_fwds.forward(maturity_yrs)

    sigma = dataset["BS"]["VOL"]
    if callable(sigma):
        spot = asset_fwds.forward(0)
        x = np.log(strike / spot)
        sigma = float(sigma((maturity_yrs, x)))

    expected_price, _ = opt_price(
        strike,
        maturity_yrs,
        "Call",
        F=F,
        df=df,
        sigma=sigma,
    )
    error = (price - expected_price) / spot
    # TODO feat: revisit the ATM Options at 0.10 maturity where error is high.
    # It did not go down much even with MAX_X at 0.5, where dx = 0.002
    assert error == approx(0.0, abs=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
