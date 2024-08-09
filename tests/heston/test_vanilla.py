# Description: Tests for the Heston model for vanilla options.

import pytest
from pytest import approx

from tests.localvol.helpers import price_vanilla_option_sim
from tests.heston.dataset import data_heston_kruse
from tests.heston.helpers import price_vanilla_call


@pytest.fixture(scope="module")
def data():
    return data_heston_kruse()


@pytest.mark.parametrize("maturity", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("strike_x", [0.01, 0.95, 1.00, 1.05, 1.2])
def test_call(data, maturity, strike_x):
    """Test the price of a vanilla call option."""

    model_cls, dataset, other = data
    spot = other["spot"]
    asset_name = dataset["HESTON"]["ASSET"]

    strike = strike_x * spot

    model = model_cls(dataset)

    price = price_vanilla_option_sim(
        strike,
        maturity,
        "Call",
        asset_name,
        dataset,
        model,
    )

    # get closed form price
    expected_price, _ = price_vanilla_call(
        strike,
        maturity,
        asset_name,
        dataset,
    )
    error = (price - expected_price) / spot
    contract = f"Call {maturity:4.2f} {strike:6.0f}"
    assert error == approx(0.0, abs=1e-3)
    print(f"{contract}: {price:11.6f} {expected_price:11.6f} {error:9.6f}")


if __name__ == "__main__":
    pytest.main([__file__])
