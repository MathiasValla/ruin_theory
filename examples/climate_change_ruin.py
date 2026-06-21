"""Generate KLR worsening-risk climate-change ruin diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ruin_theory import (
    WorseningParetoModel,
    climate_change_ruin_table,
    plot_climate_change_ruin_table,
    plot_uninsurability_times,
    plot_worsening_pareto_path,
    simulate_worsening_pareto_path,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = WorseningParetoModel(
        initial_capital=500.0,
        claim_arrival_rate=1.0,
        pareto_scale=1.0,
        initial_shape=1.5,
        worsening_speed=0.1,
        safety_loading=1.0,
        mode="shape",
    )
    path = simulate_worsening_pareto_path(
        model,
        horizon=1000.0,
        seed=123,
        stop_at_ruin=False,
    )
    table = climate_change_ruin_table(
        np.array([0.01, 0.02, 0.05, 0.1, 0.2]),
        initial_capital=500.0,
        claim_arrival_rate=1.0,
        pareto_scale=1.0,
        initial_shape=1.5,
        safety_loading=1.0,
        n_simulations=300,
        seed=123,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_worsening_pareto_path(path, ax=axes[0, 0])
    plot_worsening_pareto_path(path, model=model, rescale=True, ax=axes[0, 1])
    plot_climate_change_ruin_table(table, kind="asymptotic", ax=axes[1, 0])
    plot_uninsurability_times(table, ax=axes[1, 1])

    output_path = OUTPUT_DIR / "fig_climate_change_ruin.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Premium ceiling used in the table: {table.premium_rate_max:.3f}")
    print(f"Uninsurability horizons: {table.horizons}")
    print(f"KLR shape asymptotics: {table.shape_asymptotic}")
    print(f"KLR scale asymptotics: {table.scale_asymptotic}")
    print(f"Finite-time shape estimates: {table.shape_finite_ruin}")
    print(f"Finite-time scale estimates: {table.scale_finite_ruin}")
    print(f"Figure written to {output_path}")


if __name__ == "__main__":
    main()
