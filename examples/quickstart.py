"""Minimal ruin-theory workflow."""

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import CramerLundbergProcess, exponential, ultimate_ruin_exponential
from ruin_theory.plotting import plot_path, plot_ruin_curve, plot_ruin_time_histogram
from ruin_theory.simulation import estimate_ruin_probability, simulate_path


def main() -> None:
    model = CramerLundbergProcess(
        initial_capital=2.0,
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=exponential(rate=5.0),
    )
    u = np.linspace(0.0, 8.0, 100)
    probabilities = ultimate_ruin_exponential(model, u)
    estimate = estimate_ruin_probability(model, horizon=10.0, n_simulations=2_000, seed=123)
    path = simulate_path(model, horizon=10.0, seed=123)
    print(f"Estimated P(ruin by 10): {estimate.probability:.3f}")
    print(f"One path: ruined={path.ruined}, terminal reserve={path.terminal_reserve:.3f}")

    _, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    plot_path(path, ax=axes[0])
    plot_ruin_curve(u, probabilities, ax=axes[1], label="ultimate")
    plot_ruin_time_histogram(estimate, ax=axes[2])
    plt.show()


if __name__ == "__main__":
    main()
