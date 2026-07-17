"""Ordered risk and dual-risk formulas."""

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import (
    DualRiskProcess,
    PremiumBoundary,
    deterministic,
    dual_poisson_exponential_ruin_time_density,
    exponential,
    ordered_two_sided_survival,
    ordered_win_first_time_density,
    plot_dual_ruin_time_density,
    plot_ordered_two_sided_boundaries,
    poisson_ospp,
)


def main() -> None:
    ospp = poisson_ospp(rate=2.0)
    boundary = PremiumBoundary.linear(rate=1.0)

    survival = ordered_two_sided_survival(
        ospp,
        deterministic(0.1),
        boundary,
        initial_capital=0.5,
        upper_barrier=2.0,
        horizon=1.0,
        max_claims=8,
    )
    density_at_two = ordered_win_first_time_density(
        ospp,
        exponential(rate=1.0),
        boundary,
        initial_capital=0.5,
        upper_barrier=2.0,
        t=2.0,
        max_claims=10,
        n_conditional_samples=1024,
        seed=123,
    )
    print(f"Two-sided survival: {survival:.4f}")
    print(f"Win-first density at t=2: {density_at_two:.4f}")

    dual = DualRiskProcess(
        initial_capital=1.0,
        cost_rate=1.0,
        profit_arrival_rate=1.0,
        profit_distribution=exponential(rate=1.0),
    )
    times = np.linspace(0.0, 6.0, 200)
    density = dual_poisson_exponential_ruin_time_density(dual, times)
    print(f"Dual density max: {density.max():.4f}")

    _, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    plot_ordered_two_sided_boundaries(boundary, 0.5, 2.0, 3.0, ax=axes[0])
    plot_dual_ruin_time_density(dual, times, ax=axes[1])
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
