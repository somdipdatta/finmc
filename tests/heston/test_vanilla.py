# Description: Tests for the Heston model for vanilla options.

import pytest
from pytest import approx

from finmc.calc.option import opt_price_mc
from finmc.utils.assets import Discounter, Forwards
from tests.heston.dataset import data_heston_kruse
from tests.heston.helpers import opt_price_heston


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
    ccy = dataset["BASE"]
    strike = strike_x * spot

    model = model_cls(dataset)

    price = opt_price_mc(
        strike,
        maturity,
        "Call",
        asset_name,
        model,
    )

    # get closed form price
    discounter = Discounter(dataset["ASSETS"][ccy])
    asset_fwds = Forwards(dataset["ASSETS"][asset_name])
    r = discounter.rate(maturity)
    S0 = asset_fwds.forward(0)
    mu = asset_fwds.rate(maturity)

    expected_price, _ = opt_price_heston(
        strike,
        maturity,
        S0=S0,
        r=r,
        mu=mu,
        heston_params=dataset["HESTON"],
    )
    error = (price - expected_price) / spot
    contract = f"Call {maturity:4.2f} {strike:6.0f}"
    assert error == approx(0.0, abs=1e-3)
    print(f"{contract}: {price:11.6f} {expected_price:11.6f} {error:9.6f}")


if __name__ == "__main__":
    pytest.main([__file__])
