import matplotlib.pyplot as plt
import numpy as np


def plot_asset(
    model,
    asset_name,
    sample_idxs=np.arange(0, 3, 1),
    q_levels=np.linspace(0.05, 0.95, 30),
    times=np.linspace(0, 1, 101),
):
    num_levels = len(q_levels)
    num_steps = len(times)

    samples = np.zeros((len(sample_idxs), num_steps))
    quantiles = np.zeros((len(q_levels), num_steps))

    # enumerate over the time steps and calculate the spot price
    for i, t in enumerate(times):
        model.advance(t)
        spots = model.get_value(asset_name)
        quantiles[:, i] = np.quantile(spots, q_levels)
        samples[:, i] = spots[sample_idxs]

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(num_levels >> 1):
        ax.fill_between(
            times,
            quantiles[i, :],
            quantiles[num_levels - 1 - i, :],
            color="darkred",
            alpha=1.5 / num_levels,
            edgecolor="none",
        )
    for sample in samples:
        ax.plot(times, sample, label="Sample Path")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel(asset_name)
    plt.show()


if __name__ == "__main__":
    from finmc.models.hullwhite import HullWhiteMC

    discount_data = (
        "ZERO_RATES",
        np.array([[0.5, 0.05], [1.0, 0.04], [3, 0.04]]),
    )

    dataset = {
        "MC": {"PATHS": 100_000, "TIMESTEP": 1 / 250, "SEED": 1},
        "BASE": "USD",
        "ASSETS": {"USD": discount_data},
        "HW": {
            "ASSET": "USD",
            "MEANREV": 0.1,
            "VOL": 0.03,
        },
    }
    # create the model and plot the progression of short rate???
    model = HullWhiteMC(dataset)
    model.reset(dataset)
    plot_asset(model, "r")
