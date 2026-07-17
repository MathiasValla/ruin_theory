"""Orthogonal-polynomial ultimate-ruin approximation example."""

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import (
    CramerLundbergProcess,
    exponential,
    fit_ultimate_ruin_polynomial_expansion,
    plot_polynomial_expansion_error,
    plot_ruin_curve,
    ultimate_ruin_exponential,
)


def main() -> None:
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    surplus = np.linspace(0.0, 5.0, 80)
    exact = ultimate_ruin_exponential(model, surplus)

    approximations = {}
    for order in (0, 3, 8):
        expansion = fit_ultimate_ruin_polynomial_expansion(model, order=order)
        approximations[f"K={order}"] = expansion(surplus)

    _, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    plot_ruin_curve(surplus, exact, ax=axes[0], label="exact")
    for label, values in approximations.items():
        axes[0].plot(surplus, values, linewidth=1.4, linestyle="--", label=label)
    axes[0].legend()
    plot_polynomial_expansion_error(surplus, exact, approximations, ax=axes[1])
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
