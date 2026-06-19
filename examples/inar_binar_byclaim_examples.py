"""Generate INAR/BINAR by-claim simulation examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from ruin_theory import (
    BINARByClaimModel,
    INARByClaimModel,
    deterministic,
    estimate_binar_byclaim_ruin_probability,
    estimate_inar_byclaim_ruin_probability,
    plot_integer_byclaim_counts,
    plot_integer_byclaim_path,
    simulate_binar_byclaim_path,
    simulate_inar_byclaim_path,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "output" / "figures"


def make_inar_example() -> Path:
    model = INARByClaimModel(
        initial_capital=0.0,
        premium_per_period=36.0,
        primary_count_mean=10.0,
        initial_byclaim_mean=10.0,
        reproduction=0.9,
        primary_distribution=deterministic(2.0),
        byclaim_distribution=deterministic(1.0),
    )
    path = simulate_inar_byclaim_path(model, periods=11, seed=123, stop_at_ruin=False)
    estimate = estimate_inar_byclaim_ruin_probability(
        model,
        periods=11,
        n_simulations=5000,
        seed=123,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    plot_integer_byclaim_path(path, ax=axes[0])
    plot_integer_byclaim_counts(path, ax=axes[1])
    fig.suptitle(f"INAR ruin estimate: {estimate.probability:.3f}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "fig_inar_byclaim_path.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def make_binar_example() -> Path:
    model = BINARByClaimModel(
        initial_capital=1000.0,
        premium_per_period=15000.0,
        primary_count_means=(5.0, 7.0),
        initial_byclaim_means=(1.0, 1.0),
        reproduction_matrix=((0.41, 0.10), (0.05, 0.30)),
        primary_distributions=(deterministic(10.0), deterministic(1.0)),
        byclaim_distributions=(deterministic(0.5), deterministic(0.5)),
    )
    path = simulate_binar_byclaim_path(model, periods=10, seed=123, stop_at_ruin=False)
    estimate = estimate_binar_byclaim_ruin_probability(
        model,
        periods=10,
        n_simulations=5000,
        seed=123,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    plot_integer_byclaim_path(path, ax=axes[0])
    plot_integer_byclaim_counts(path, ax=axes[1])
    fig.suptitle(f"BINAR ruin estimate: {estimate.probability:.3f}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "fig_binar_byclaim_counts.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    outputs = [make_inar_example(), make_binar_example()]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
