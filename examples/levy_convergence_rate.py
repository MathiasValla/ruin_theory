"""Finite-to-ultimate convergence rates for Levy-driven risk processes."""

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import (
    CompoundPoissonSubordinator,
    LevelDependentLevyRiskProcess,
    exponential_convergence_rate,
    gamma,
    plot_convergence_rate_bound,
    plot_exponential_convergence_rates,
)


def main() -> None:
    loadings = (0.1, 0.2, 0.3)
    volatilities = np.linspace(0.0, 5.0, 11)
    liability = CompoundPoissonSubordinator(rate=1.0, jump_distribution=gamma(shape=2, rate=1))
    mean_liability = liability.mean_rate

    curves = {}
    for loading in loadings:
        rates = []
        for sigma in volatilities:
            process = LevelDependentLevyRiskProcess(
                premium_rate=(1.0 + loading) * mean_liability,
                liability=liability,
                diffusion=float(sigma),
            )
            rates.append(exponential_convergence_rate(process).rate)
        curves[f"eta={loading:.1f}"] = np.array(rates)

    reference = LevelDependentLevyRiskProcess(
        premium_rate=1.2 * mean_liability,
        liability=liability,
        diffusion=0.0,
    )
    result = exponential_convergence_rate(reference)

    _, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    plot_exponential_convergence_rates(
        volatilities,
        curves,
        ax=axes[0],
        x_label="diffusion volatility",
    )
    plot_convergence_rate_bound(result, np.linspace(0.0, 50.0, 100), ax=axes[1])
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
