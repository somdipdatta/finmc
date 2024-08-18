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

    fig, ax = plt.subplots()

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
