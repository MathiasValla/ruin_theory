"""Generate Markov-modulated multirisk common-shock diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt

from ruin_theory import (
    CommonShock,
    MarkovEnvironment,
    common_shock_increment_pmfs,
    dependence_impact,
    finite_time_markov_modulated_ruin,
    independent_common_shock_pmf,
    plot_dependence_impact,
    plot_environment_state_survival,
    plot_markov_modulated_ruin_curves,
    plot_solvency_region_2d,
    transition_matrix_from_generator,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    environment = MarkovEnvironment(
        initial_distribution=[0.7, 0.3],
        transition_matrix=transition_matrix_from_generator(
            [[-0.35, 0.35], [0.2, -0.2]],
            step=1.0,
        ),
    )
    weather_shock = CommonShock(
        intensities=[0.25, 0.65],
        claim_pmfs=(
            independent_common_shock_pmf(
                [0.35, 0.2],
                [{1: 0.6, 2: 0.4}, {1: 1.0}],
            ),
            {(0, 0): 0.35, (2, 1): 0.25, (1, 2): 0.2, (3, 3): 0.2},
        ),
        name="weather",
    )
    motor_shock = CommonShock(
        intensities=[0.5, 0.7],
        claim_pmfs={(1, 0): 0.8, (2, 0): 0.2},
        name="motor",
    )
    property_shock = CommonShock(
        intensities=[0.4, 0.6],
        claim_pmfs={(0, 1): 0.9, (0, 2): 0.1},
        name="property",
    )
    increments = common_shock_increment_pmfs(
        [weather_shock, motor_shock, property_shock],
        max_count=8,
    )

    common_kwargs = dict(
        increment_pmfs=increments.increment_pmfs,
        environment=environment,
        initial_capitals=[4.0, 4.0],
        premiums=[0.9, 0.8],
        horizon=8,
        truncation_error_bounds=increments.truncation_error_bounds,
    )
    any_line = finite_time_markov_modulated_ruin(**common_kwargs, region="any_line")
    total = finite_time_markov_modulated_ruin(**common_kwargs, region="total")
    hybrid = finite_time_markov_modulated_ruin(
        **common_kwargs,
        region="hybrid",
        severity_limit=[1.0, 1.0],
    )

    one_state = MarkovEnvironment([1.0], [[1.0]])
    independent = finite_time_markov_modulated_ruin(
        [{(0, 0): 0.25, (2, 0): 0.25, (0, 2): 0.25, (2, 2): 0.25}],
        one_state,
        initial_capitals=[1.0, 1.0],
        premiums=[0.0, 0.0],
        horizon=1,
        region="any_line",
    )
    positive_dependence = finite_time_markov_modulated_ruin(
        [{(0, 0): 0.5, (2, 2): 0.5}],
        one_state,
        initial_capitals=[1.0, 1.0],
        premiums=[0.0, 0.0],
        horizon=1,
        region="any_line",
    )
    impact = dependence_impact(
        independent,
        positive_dependence,
        reference_label="independent",
        comparison_label="positive dependence",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_markov_modulated_ruin_curves(
        [any_line, total, hybrid],
        ax=axes[0, 0],
        labels=["any line", "total wealth", "hybrid"],
    )
    plot_environment_state_survival(any_line, ax=axes[0, 1], normalize=True)
    plot_dependence_impact(impact, ax=axes[1, 0])
    plot_solvency_region_2d(
        [4.0, 4.0],
        [0.9, 0.8],
        period=4,
        region="hybrid",
        severity_limit=[1.0, 1.0],
        ax=axes[1, 1],
    )

    path = OUTPUT_DIR / "fig_markov_modulated_common_shocks.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"Any-line ruin at horizon 8: {any_line.ruin_probabilities[-1]:.4f}")
    print(f"Total-wealth ruin at horizon 8: {total.ruin_probabilities[-1]:.4f}")
    print(f"Hybrid-region ruin at horizon 8: {hybrid.ruin_probabilities[-1]:.4f}")
    print(f"Compound-Poisson truncation bound: {any_line.truncation_error_bound:.2e}")
    print(f"Figure written to {path}")


if __name__ == "__main__":
    main()
