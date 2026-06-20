"""Generate time-in-red and reserve-allocation diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ruin_theory import (
    RedTimeCurveResult,
    evaluate_reserve_allocation_grid,
    expected_negative_area_exponential,
    expected_time_in_red_exponential,
    optimize_reserve_allocation,
    plot_red_time_allocation,
    plot_red_time_curve,
    plot_simplex_allocation_surface,
    plot_two_line_allocation_curve,
    simplex_reserve_grid,
)


OUTPUT_DIR = Path("output/figures")


def _red_function(params):
    return lambda reserve: expected_time_in_red_exponential(reserve, **params)


def _area_function(params):
    return lambda reserve: expected_negative_area_exponential(reserve, **params)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    branch_params = [
        dict(premium_rate=1.4, claim_arrival_rate=0.7, claim_rate=1.0),
        dict(premium_rate=1.2, claim_arrival_rate=0.9, claim_rate=1.1),
        dict(premium_rate=1.6, claim_arrival_rate=0.6, claim_rate=0.9),
    ]
    red_functions = tuple(_red_function(params) for params in branch_params)
    area_functions = tuple(_area_function(params) for params in branch_params)

    capital = np.linspace(0.0, 8.0, 41)
    curve = RedTimeCurveResult(
        initial_capitals=capital,
        expected_time_in_red=expected_time_in_red_exponential(capital, **branch_params[0]),
        expected_negative_area=expected_negative_area_exponential(capital, **branch_params[0]),
        time_in_red_standard_error=np.zeros_like(capital),
        negative_area_standard_error=np.zeros_like(capital),
        n_simulations=0,
        horizon=np.inf,
    )

    allocation = optimize_reserve_allocation(
        total_reserve=6.0,
        red_time_functions=red_functions,
        negative_area_functions=area_functions,
    )
    two_line = evaluate_reserve_allocation_grid(
        simplex_reserve_grid(total_reserve=6.0, n_lines=2, subdivisions=60),
        area_functions[:2],
        red_time_functions=red_functions[:2],
    )
    simplex = evaluate_reserve_allocation_grid(
        simplex_reserve_grid(total_reserve=6.0, n_lines=3, subdivisions=24),
        area_functions,
        red_time_functions=red_functions,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_red_time_curve(curve, ax=axes[0, 0])
    plot_red_time_allocation(allocation, ax=axes[0, 1])
    plot_two_line_allocation_curve(two_line, ax=axes[1, 0])
    plot_simplex_allocation_surface(simplex, ax=axes[1, 1])
    fig.savefig(OUTPUT_DIR / "fig_red_time_allocation_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
