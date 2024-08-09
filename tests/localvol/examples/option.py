# Description: Run MSFD and LVMC models with a vanilla option contract.
from tests.blackscholes.dataset import data_lvmc_fn, data_lvmc_grid
from tests.blackscholes.helpers import price_vanilla_option_sim


def run_model(type="grid"):
    """Price a vanilla option with the LV MC model."""

    if type == "grid":
        model_cls, dataset, other = data_lvmc_grid()
    elif type == "fn":
        model_cls, dataset, other = data_lvmc_fn()

    spot = other["spot"]
    strike = spot

    asset_name = dataset["BS"]["ASSET"]

    # get price from timetable
    model = model_cls(dataset)
    price = price_vanilla_option_sim(
        strike,
        1 / 12,
        "Call",
        asset_name,
        dataset,
        model,
    )
    return price


if __name__ == "__main__":
    price = run_model()
    print(f"price = {price}")
