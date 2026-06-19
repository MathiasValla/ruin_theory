"""Exact finite-time ruin probabilities for integer-valued claims."""

from __future__ import annotations

import numpy as np

from ruin_theory import finite_time_ruin_discrete, plot_ruin_curve


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

    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return
    _, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    plot_ruin_curve(reserves, ruin, ax=ax, label="Seal/Takacs")
    ax.set_title("Exact finite-time lattice ruin")
    plt.show()


if __name__ == "__main__":
    main()

