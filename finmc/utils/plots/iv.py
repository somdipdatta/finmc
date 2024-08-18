import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_iv(strikes, expirations, surface, atm_vols, fwds):
    X, Y = np.meshgrid(strikes, expirations)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.plot_surface(
        X,
        Y,
        surface,
        cmap=cm.inferno,
        alpha=0.5,
    )
    # Add wireframes for each expiration
    ax.plot_wireframe(X, Y, surface, color="brown", rstride=1, cstride=0)

    ax.set_xlabel("Strike (K)")
    ax.set_ylabel("Maturity (years)")
    plt.show()

    # Add atm vol curve
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(
        expirations, atm_vols, color="brown", label="Forward Curve", marker="o"
    )

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("ATM Vol")
    plt.show()
