# Description: Run MSFD and LVMC models with a vanilla option contract.
from finmc.calc.option import opt_price_mc
from tests.localvol.dataset import data_lvmc_fn, data_lvmc_grid


def run_model(type="grid"):
    """Price a vanilla option with the LV MC model."""

    if type == "grid":
        model_cls, dataset, other = data_lvmc_grid()
    elif type == "fn":
        model_cls, dataset, other = data_lvmc_fn()

    spot = other["spot"]
    strike = spot

    asset_name = dataset["LV"]["ASSET"]

    # get price from timetable
    model = model_cls(dataset)
    price = opt_price_mc(
        strike,
        1 / 12,
        "Call",
        asset_name,
        model,
    )
    return price


if __name__ == "__main__":
    price = run_model()
    print(f"price = {price}")
