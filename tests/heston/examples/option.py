# Description: Run Heston MC model with a vanilla option contract.

from finmc.calc.option import opt_price_mc
from tests.heston.dataset import data_heston_kruse


def run_model():
    """Price a vanilla option with the Heston MC model."""

    model_cls, dataset, other = data_heston_kruse()

    # Create an option timetable with 1 month maturity

    model = model_cls(dataset)
    price = opt_price_mc(
        100,
        1 / 12,
        "Call",
        "EQ",
        model,
    )

    return price


if __name__ == "__main__":
    price = run_model()
    print(f"price = {price}")
