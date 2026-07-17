"""Reproduce Loisel-Privault finite-time ruin sensitivity diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import (
    CramerLundbergProcess,
    deterministic,
    estimate_finite_time_ruin_sensitivity_ibp,
    exponential,
    finite_time_ruin_exponential,
    pareto,
    plot_finite_time_ruin_sensitivity,
)


def _exponential_finite_difference_density(
    model: CramerLundbergProcess,
    surplus: np.ndarray,
    horizon: float,
    *,
    step: float = 1e-4,
) -> np.ndarray:
    derivative = np.array(
        [
            (
                finite_time_ruin_exponential(model, u=float(value + step), horizon=horizon)
                - finite_time_ruin_exponential(model, u=float(max(value - step, 0.0)), horizon=horizon)
            )
            / (value + step - max(value - step, 0.0))
            for value in surplus
        ]
    )
    return -derivative


def main() -> None:
    surplus = np.linspace(0.05, 8.0, 80)
    horizon = 10.0
    n_simulations = 20_000

    unit_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        claim_distribution=deterministic(1.0),
    )
    exponential_model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=0.5,
        claim_distribution=exponential(rate=1.0),
    )
    pareto_model = CramerLundbergProcess(
        premium_rate=2.0,
        claim_arrival_rate=0.4,
        claim_distribution=pareto(shape=1.5, scale=1.0),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, model, title in (
        (axes[0, 0], unit_model, "Unit claims"),
        (axes[0, 1], exponential_model, "Exponential claims"),
        (axes[1, 0], pareto_model, "Pareto claims"),
    ):
        estimate = estimate_finite_time_ruin_sensitivity_ibp(
            model,
            surplus,
            horizon=horizon,
            n_simulations=n_simulations,
            seed=123,
        )
        plot_finite_time_ruin_sensitivity(
            estimate,
            ax=axis,
            x_axis="infimum",
            label="IBP",
            show_ci=False,
        )
        if model is exponential_model:
            finite_difference = _exponential_finite_difference_density(model, surplus, horizon)
            axis.plot(-surplus, finite_difference, color="#b00020", linestyle="--", label="finite diff.")
            axis.legend()
        axis.set_title(title)

    sample_sizes = np.array([500, 1_000, 2_000, 5_000, 10_000, 20_000])
    target_surplus = np.array([0.5])
    convergence = np.array(
        [
            estimate_finite_time_ruin_sensitivity_ibp(
                exponential_model,
                target_surplus,
                horizon=horizon,
                n_simulations=int(size),
                seed=321,
            ).density.item()
            for size in sample_sizes
        ]
    )
    exact_density = _exponential_finite_difference_density(
        exponential_model,
        target_surplus,
        horizon,
    ).item()
    axes[1, 1].plot(sample_sizes, convergence, marker="o", color="#1f77b4", label="IBP")
    axes[1, 1].axhline(exact_density, color="#b00020", linestyle="--", label="finite diff.")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("simulations")
    axes[1, 1].set_ylabel("density at -0.5")
    axes[1, 1].set_title("IBP convergence")
    axes[1, 1].legend()

    plt.show()


if __name__ == "__main__":
    main()
