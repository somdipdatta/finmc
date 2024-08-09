"""
Testing Vanilla Option for all Black Scholes models.
"""

import pytest
from pytest import approx

from tests.localvol.dataset import (
    data_lvmc,
    data_lvmc_fn,
    data_lvmc_grid,
)
from tests.localvol.helpers import (
    price_vanilla_opt,
    price_vanilla_option_sim,
)


# datasets for all models that can price a vanilla option.
@pytest.fixture(
    params=[
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

    model = model_cls(dataset)
    asset_name = dataset["BS"]["ASSET"]

    maturity_yrs = maturity / 12

    price = price_vanilla_option_sim(
        strike,
        maturity_yrs,
        "Call",
        asset_name,
        dataset,
        model,
    )

    # get closed form price
    expected_price, _ = price_vanilla_opt(
        strike,
        maturity_yrs,
        "Call",
        asset_name,
        dataset,
    )
    error = (price - expected_price) / spot
    # TODO feat: revisit the ATM Options at 0.10 maturity where error is high.
    # It did not go down much even with MAX_X at 0.5, where dx = 0.002
    assert error == approx(0.0, abs=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
