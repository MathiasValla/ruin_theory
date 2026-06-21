"""Generate infinite-mean regularly varying ruin diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ruin_theory import (
    InfiniteMeanRuinModel,
    PolynomialPremiumGrowth,
    RegularlyVaryingTail,
    calibrate_polynomial_premium_coefficient,
    infinite_mean_ruin_curve,
    plot_infinite_mean_ruin_curve,
    plot_premium_power_calibration,
    plot_regular_variation_tail_diagnostic,
    premium_power_calibration_grid,
    premium_power_condition,
    regular_variation_tail_diagnostic,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tail = RegularlyVaryingTail(tail_index=0.8, scale=1.0)
    condition = premium_power_condition(tail_index=0.8, premium_power=1.6)
    capital_constraints = np.array([50.0, 100.0, 200.0, 500.0])
    calibration = calibrate_polynomial_premium_coefficient(
        tail,
        capital_constraints,
        target_probability=0.02,
        claim_arrival_rate=1.0,
        premium_power=1.6,
    )
    model = InfiniteMeanRuinModel(
        claim_arrival_rate=1.0,
        tail=tail,
        premium=PolynomialPremiumGrowth(
            coefficient=calibration.required_coefficient,
            power=1.6,
        ),
    )

    capital_grid = np.geomspace(30.0, 1000.0, 18)
    asymptotic = infinite_mean_ruin_curve(model, capital_grid, method="asymptotic")
    integral = infinite_mean_ruin_curve(model, capital_grid[:8], method="integral")
    diagnostic = regular_variation_tail_diagnostic(
        tail,
        thresholds=np.logspace(1.0, 6.0, 24),
        multipliers=[2.0, 5.0, 10.0],
    )
    power_grid = premium_power_calibration_grid(
        tail,
        capital_constraints,
        np.linspace(1.05, 2.5, 24),
        target_probability=0.02,
        claim_arrival_rate=1.0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    plot_infinite_mean_ruin_curve(asymptotic, ax=axes[0], label="KLR equivalent")
    plot_infinite_mean_ruin_curve(integral, ax=axes[0], label="tail integral")
    plot_regular_variation_tail_diagnostic(diagnostic, ax=axes[1])
    plot_premium_power_calibration(power_grid, ax=axes[2])

    output_path = OUTPUT_DIR / "fig_infinite_mean_regular_variation.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Condition beta > 1 / alpha: {condition.holds}")
    print(f"Threshold 1 / alpha: {condition.threshold:.3f}")
    print(f"Required premium coefficient: {calibration.required_coefficient:.6f}")
    print(f"Binding initial capital: {calibration.binding_initial_capital:.3f}")
    print(f"Achieved asymptotic: {calibration.achieved_asymptotic:.6f}")
    print(f"Tail diagnostic max relative error: {diagnostic.max_relative_error:.6f}")
    print(f"Figure written to {output_path}")


if __name__ == "__main__":
    main()
