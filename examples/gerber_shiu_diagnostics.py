"""Generate Gerber-Shiu deficit and surplus diagnostics."""

from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt

from ruin_theory import (
    CramerLundbergProcess,
    exponential,
    estimate_gerber_shiu,
    plot_deficit_at_ruin,
    plot_gerber_shiu_scatter,
    plot_surplus_before_ruin,
)


OUTPUT_DIR = Path("output/figures")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = CramerLundbergProcess(
        initial_capital=2.0,
        premium_rate=1.25,
        claim_arrival_rate=0.8,
        claim_distribution=exponential(1.0),
    )
    result = estimate_gerber_shiu(
        model,
        horizon=10.0,
        n_simulations=5000,
        penalty=lambda surplus, deficit: deficit,
        discount_rate=0.03,
        seed=2026,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    plot_deficit_at_ruin(result, ax=axes[0], bins=28)
    plot_surplus_before_ruin(result, ax=axes[1], bins=28)
    plot_gerber_shiu_scatter(result, ax=axes[2], alpha=0.55)
    fig.suptitle(f"Gerber-Shiu estimate: {result.estimate:.4f}")
    fig.savefig(OUTPUT_DIR / "fig_gerber_shiu_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
