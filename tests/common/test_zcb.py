"""
Testing ZCB for all models.
"""

import pytest
from pytest import approx

from finmc.base.utils import discounter_from_dataset
from tests.hullwhite.test_hw import data_hwmc
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

    model_cls, dataset, other = data

    for maturity in [0.1, 1.0, 3.0]:  # months
        model = model_cls(dataset)
        model.advance(maturity)

        # get closed form price
        discounter = discounter_from_dataset(dataset)
        expected_price = discounter.discount(maturity)

        price = expected_price  # trivial for now, modify later

        error = price - expected_price
        contract = f"ZCB {maturity:4.2f}"
        assert error == approx(0.0, abs=1e-3)

        print(f"{contract}: {price:11.6f} {expected_price:11.6f} {error:9.6f}")
