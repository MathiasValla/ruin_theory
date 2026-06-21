"""Generate multirisk dividend and insolvency-penalty diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt

from ruin_theory import (
    CommonShock,
    estimate_multirisk_dividend_penalties_ctmc,
    linear_status_premium_function,
    multirisk_dividend_convergence,
    plot_multirisk_dividend_convergence,
    plot_multirisk_dividend_penalty_bars,
    plot_multirisk_ruin_state_distribution,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"


def _units(amount: float, step: float) -> int:
    return int(round(amount / step))


def solve(step: float):
    premium = linear_status_premium_function(
        [1.0, 0.45],
        interaction_matrix=[[0.0, 0.35], [0.0, 0.0]],
    )
    main_claim = _units(2.5, step)
    secondary_claim = _units(1.5, step)
    shock = CommonShock(
        intensities=[0.55],
        claim_pmfs={
            (main_claim, 0): 0.65,
            (0, secondary_claim): 0.25,
            (main_claim, secondary_claim): 0.10,
        },
    )
    return estimate_multirisk_dividend_penalties_ctmc(
        initial_reserves=[2.0, 0.0],
        barriers=[3.0, 1.0],
        lower_bounds=[0.0, -1.0],
        grid_step=step,
        environment_generator=[[0.0]],
        environment_initial=[1.0],
        shocks=[shock],
        premium_rate_function=premium,
        ruin_lines=[0],
        max_states=50000,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [solve(step) for step in (1.0, 0.5, 0.25)]
    convergence = multirisk_dividend_convergence(results)
    finest = results[-1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_multirisk_dividend_penalty_bars(finest, ax=axes[0, 0])
    plot_multirisk_ruin_state_distribution(finest, ax=axes[0, 1])
    plot_multirisk_dividend_convergence(convergence, metric="time", ax=axes[1, 0])
    plot_multirisk_dividend_convergence(
        convergence,
        metric="penalties",
        line=0,
        ax=axes[1, 1],
    )

    path = OUTPUT_DIR / "fig_multirisk_dividend_penalties.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"Expected time to ruin: {finest.expected_time_to_ruin:.4f}")
    print(f"Ruin probability: {finest.ruin_probability:.4f}")
    print(f"Expected dividends: {finest.expected_dividends}")
    print(f"Expected insolvency penalties: {finest.expected_penalties}")
    print(f"Last time-convergence change: {convergence.last_time_change:.4f}")
    print(f"Figure written to {path}")


if __name__ == "__main__":
    main()
