"""Exact finite-time ruin probabilities for integer-valued claims."""

from __future__ import annotations

import numpy as np

from ruin_theory import (
    finite_time_ruin_discrete_appell,
    finite_time_ruin_discrete_boundary_function,
    plot_finite_time_appell_coefficients,
    plot_finite_time_discrete_boundary,
    finite_time_ruin_discrete,
    plot_finite_time_discrete_computation_set,
    plot_finite_time_discrete_survival,
    plot_ruin_curve,
)


def main() -> None:
    deterministic_unit_claim = [0.0, 1.0]
    reserves = np.array([0, 5, 10, 15, 20])
    ruin = np.array(
        [
            finite_time_ruin_discrete(
                deterministic_unit_claim,
                initial_capital=int(u),
                premium_rate=1.25,
                claim_arrival_rate=1.0,
                horizon=10.0,
                method="seal",
            )
            for u in reserves
        ]
    )
    print("De Vylder/Picard-Lefevre deterministic-claim check")
    for u, value in zip(reserves, ruin):
        print(f"u={u:2d}  psi(u, 10)={value:.12g}")

    result = finite_time_ruin_discrete(
        [0.0, 0.25, 0.50, 0.25],
        initial_capital=4,
        premium_rate=1.3,
        claim_arrival_rate=0.8,
        horizon=4.2,
        method="inventory",
        return_result=True,
    )
    print("\nInventory dates:", np.round(result.inventory_times, 4))
    print("Survival at inventory dates:", np.round(result.survival_probabilities, 6))

    boundary_result = finite_time_ruin_discrete_boundary_function(
        [0.0, 0.25, 0.50, 0.25],
        boundary=lambda time: 4.0 + 1.3 * time,
        horizon=4.2,
        claim_arrival_rate=0.8,
        convention="negative",
        return_result=True,
    )
    print("Boundary recursion psi(4, 4.2)=", f"{boundary_result.ruin_probability:.12g}")

    appell_result = finite_time_ruin_discrete_appell(
        [0.0, 0.25, 0.50, 0.25],
        boundary=lambda time: 4.0 + 1.3 * time,
        horizon=4.2,
        claim_arrival_rate=0.8,
        return_result=True,
    )
    print("Appell formula psi(4, 4.2)=", f"{appell_result.ruin_probability:.12g}")

    non_integer = finite_time_ruin_discrete(
        deterministic_unit_claim,
        initial_capital=0.5,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        horizon=1.0,
        method="seal",
    )
    print(f"Non-integer reserve check psi(0.5, 1)={non_integer:.12g}")

    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    plot_ruin_curve(reserves, ruin, ax=axes[0, 0], label="Seal/Takacs")
    axes[0, 0].set_title("Exact finite-time lattice ruin")
    plot_finite_time_discrete_survival(result, ax=axes[0, 1], label="inventory")
    plot_finite_time_discrete_boundary(boundary_result, ax=axes[0, 2], label="h(t)")
    plot_finite_time_discrete_computation_set(
        initial_capital=5,
        premium_units=10,
        method="picard-lefevre",
        ax=axes[1, 0],
    )
    plot_finite_time_discrete_computation_set(
        initial_capital=5,
        premium_units=10,
        method="seal",
        ax=axes[1, 1],
    )
    plot_finite_time_appell_coefficients(appell_result, ax=axes[1, 2])
    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
