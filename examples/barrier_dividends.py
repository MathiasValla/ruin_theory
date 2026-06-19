"""Generate horizontal dividend-barrier diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import (
    barrier_dividend_analytic_exponential_interest_force,
    deterministic,
    estimate_barrier_dividends,
    exponential,
    plot_barrier_comparison,
    plot_barrier_dividend_distribution,
    plot_barrier_dividend_path,
    plot_barrier_ruin_time_distribution,
    simulate_barrier_dividend_path,
)


OUTPUT_DIR = Path("output/figures")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = dict(
        initial_capital=1.0,
        premium_rate=1.2,
        claim_arrival_rate=0.7,
        claim_rate=1.4,
        interest_force=0.08,
    )
    barriers = np.linspace(1.0, 5.0, 9)
    expected = np.array(
        [
            barrier_dividend_analytic_exponential_interest_force(
                barrier=float(barrier),
                **params,
            ).expected_dividends
            for barrier in barriers
        ]
    )

    path = simulate_barrier_dividend_path(
        exponential(params["claim_rate"]),
        initial_capital=params["initial_capital"],
        premium_rate=params["premium_rate"],
        claim_arrival_rate=params["claim_arrival_rate"],
        barrier=3.0,
        interest_force=params["interest_force"],
        horizon=20.0,
        seed=2026,
    )
    estimate = estimate_barrier_dividends(
        deterministic(2.0),
        initial_capital=1.0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        barrier=1.0,
        n_simulations=3000,
        seed=2026,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plot_barrier_dividend_path(path, ax=axes[0, 0])
    plot_barrier_dividend_distribution(estimate.total_dividends, ax=axes[0, 1])
    plot_barrier_ruin_time_distribution(estimate.ruin_times, ax=axes[1, 0])
    plot_barrier_comparison(barriers, expected, ax=axes[1, 1])
    fig.savefig(OUTPUT_DIR / "fig_barrier_dividend_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
