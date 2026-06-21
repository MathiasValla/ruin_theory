"""Matrix-analytic and advanced-prevention examples."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import (
    CramerLundbergProcess,
    exponential,
    gerber_shiu_exponential_closed_form,
    matrix_exponential,
    optimize_dynamic_prevention_calendar,
    optimize_two_claim_prevention,
    phase_type,
    phase_type_renewal_count_pmf,
    plot_dynamic_prevention_policy,
    plot_gerber_shiu_closed_form,
    plot_phase_type_renewal_count,
    plot_two_claim_prevention_summary,
    sparre_andersen_phase_type_ruin_probability_by_count,
    ultimate_ruin_matrix_exponential,
)


def main() -> None:
    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    claims = matrix_exponential([1.0], [[-5.0]], [5.0])
    model = CramerLundbergProcess(
        premium_rate=1.0,
        claim_arrival_rate=3.0,
        claim_distribution=claims,
    )
    surplus = np.linspace(0.0, 4.0, 80)
    ruin = ultimate_ruin_matrix_exponential(model, surplus)
    print(f"ME ultimate ruin at u=0: {ruin[0]:.6f}")
    print(f"ME ultimate ruin at u=4: {ruin[-1]:.6f}")

    closed_gs = gerber_shiu_exponential_closed_form(
        CramerLundbergProcess(
            premium_rate=1.0,
            claim_arrival_rate=3.0,
            claim_distribution=exponential(rate=5.0),
        ),
        surplus,
        deficit_moment_order=1,
        return_result=True,
    )

    wait = phase_type([1.0], [[-2.0]])
    count_law = phase_type_renewal_count_pmf(wait, horizon=1.5, max_count=8)
    ruin_by_count = np.linspace(0.0, 0.8, count_law.probabilities.size)
    mixed_ruin = sparre_andersen_phase_type_ruin_probability_by_count(
        ruin_by_count,
        wait,
        horizon=1.5,
    )
    print(f"PH renewal count total mass: {count_law.total_mass:.6f}")
    print(f"Sparre-Andersen count-mixture ruin proxy: {mixed_ruin:.6f}")

    dynamic = optimize_dynamic_prevention_calendar(
        [1.0, 5.0, 1.0],
        initial_budget=0.4,
        max_prevention=1.0,
        prevention_response=lambda amount: 1.0 / (1.0 + 2.0 * amount),
        n_cycles=2,
    )
    print(f"Dynamic prevention reduction: {dynamic.pressure_reduction:.6f}")

    two_claim = optimize_two_claim_prevention(
        exponential(rate=1.0),
        exponential(rate=0.2),
        premium_rate=12.0,
        small_claim_arrival_rate=0.1,
        large_claim_frequency_function=lambda amount: 2.0 * math.exp(-3.0 * amount),
        max_prevention=2.0,
    )
    print(f"Two-claim prevention amount: {two_claim.amount:.6f}")
    print(f"Two-claim loss ratio: {two_claim.loss_ratio:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    plot_gerber_shiu_closed_form(closed_gs, ax=axes[0, 0], label="deficit mean")
    plot_phase_type_renewal_count(count_law, ax=axes[0, 1])
    plot_dynamic_prevention_policy(dynamic, ax=axes[1, 0])
    plot_two_claim_prevention_summary(two_claim, ax=axes[1, 1])
    figure_path = output_dir / "fig_matrix_analytic_prevention.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Figure written to {figure_path}")


if __name__ == "__main__":
    main()
