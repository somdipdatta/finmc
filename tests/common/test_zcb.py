"""
Testing ZCB for all models.
"""

import pytest
from pytest import approx

from finmc.utils.assets import Discounter
from tests.hullwhite.dataset import data_hwmc
from tests.localvol.dataset import data_lvmc


@pytest.fixture(
    params=[
        data_lvmc,
        data_hwmc,
    ]
)
def data(request):
    return request.param()


def test_zcb(data):
    """Test the price of a zero coupon bond."""

    model_cls, dataset, _ = data

    for maturity in [0.1, 1.0, 3.0, 10.0]:  # years
        # Simulated Price
        model = model_cls(dataset)
        model.advance(maturity)
        price = model.get_df().mean()

        # Get closed form price
        discounter = Discounter(dataset["ASSETS"][dataset["BASE"]])
        expected_price = discounter.discount(maturity)

        error = price - expected_price
        contract = f"ZCB {maturity:4.2f}"
        assert error == approx(0.0, abs=1e-3)

        print(f"{contract}: {price:11.6f} {expected_price:11.6f} {error:9.6f}")
