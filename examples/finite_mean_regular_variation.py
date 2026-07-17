"""Generate finite-mean regular-variation ruin diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ruin_theory import (
    RegularlyVaryingTail,
    finite_mean_equilibrium_tail,
    finite_mean_equilibrium_tail_asymptotic,
    finite_mean_regular_variation_curve,
    finite_mean_regularly_varying_ruin_asymptotic,
    plot_finite_mean_regular_variation_curve,
    regular_variation_tail_diagnostic,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tail = RegularlyVaryingTail(tail_index=2.5, scale=4.0)
    rho = 0.4
    capital_grid = np.geomspace(10.0, 10_000.0, 24)

    exact_tail = finite_mean_equilibrium_tail(tail, capital_grid)
    karamata_tail = finite_mean_equilibrium_tail_asymptotic(tail, capital_grid)
    ruin_asymptotic = finite_mean_regularly_varying_ruin_asymptotic(
        tail,
        capital_grid,
        rho=rho,
    )
    curve = finite_mean_regular_variation_curve(tail, capital_grid, rho=rho)
    diagnostic = regular_variation_tail_diagnostic(
        tail,
        thresholds=np.logspace(2.0, 6.0, 20),
        multipliers=[2.0, 5.0, 10.0],
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    plot_finite_mean_regular_variation_curve(curve, ax=ax)
    output_path = OUTPUT_DIR / "fig_finite_mean_regular_variation.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Tail mean: {tail.mean():.6f}")
    print(f"Largest-grid integrated-tail ratio: {karamata_tail[-1] / exact_tail[-1]:.6f}")
    print(f"Largest-grid ruin asymptotic: {ruin_asymptotic[-1]:.8f}")
    print(f"Tail diagnostic max relative error: {diagnostic.max_relative_error:.6f}")
    print(f"Figure written to {output_path}")


if __name__ == "__main__":
    main()
