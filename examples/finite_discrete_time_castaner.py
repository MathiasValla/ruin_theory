"""Non-homogeneous discrete-time finite-horizon ruin diagnostics."""

from __future__ import annotations

from ruin_theory import (
    discounted_premiums,
    exchangeable_bernoulli_claim_scenarios,
    finite_time_dependent_discrete_time_ruin,
    finite_time_discrete_time_ruin,
    finite_time_lundberg_bounds,
    period_lundberg_roots_from_pmf,
    plot_discrete_time_deficit_cdf,
    plot_discrete_time_surplus_cdf,
    plot_finite_time_lundberg_bounds,
    ruin_deficit_quantile,
)


def main() -> None:
    premiums = discounted_premiums([1.1, 1.1, 1.1], [0.03, 0.03, 0.03])
    result = finite_time_discrete_time_ruin(
        [
            [0.55, 0.35, 0.10],
            [0.60, 0.30, 0.10],
            [0.65, 0.25, 0.10],
        ],
        premiums=premiums,
        initial_capital=0.0,
        return_result=True,
    )
    print("Discrete-time ruin probabilities:", result.ruin_probabilities)
    print("95% deficit quantile at t=1:", ruin_deficit_quantile(result, period=0, probability=0.95))

    scenarios, probabilities = exchangeable_bernoulli_claim_scenarios([0.25, 0.50, 0.25])
    dependent = finite_time_dependent_discrete_time_ruin(
        scenarios,
        probabilities,
        premiums=[0.0, 0.0],
        return_result=True,
    )
    print("Exchangeable Bernoulli ruin probability:", dependent.ruin_probability)

    roots = period_lundberg_roots_from_pmf(
        [[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]],
        premiums=[0.5, 0.5, 0.5],
    )
    bounds = finite_time_lundberg_bounds(roots, initial_capital=2.0)
    print("Period roots:", roots)
    print("Finite-time Lundberg bounds:", bounds.bounds)

    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    plot_discrete_time_surplus_cdf(result, period=2, ax=axes[0])
    plot_discrete_time_deficit_cdf(result, period=0, ax=axes[1])
    plot_finite_time_lundberg_bounds(bounds, ax=axes[2])
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
