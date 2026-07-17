"""Create lightweight validation notebooks for the software-paper plan."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "validation"


def markdown(source: str) -> dict:
    cleaned = textwrap.dedent(source).strip()
    return {"cell_type": "markdown", "metadata": {}, "source": cleaned.splitlines(True)}


def code(source: str) -> dict:
    cleaned = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cleaned.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "markov_modulated_multirisk_reproduction.ipynb": notebook(
        [
            markdown(
                """
                # Markov-Modulated Multirisk Common-Shock Reproduction

                Source family: Picard, Lefevre and Coulibaly (2003), Loisel
                (2004, 2005), Cossette/Landriault/Marceau and related
                common-shock multirisk models.

                Paper-specific target: reproduce the one-period manual checks
                used for the package validation:

                - any-line ruin probability: `0.75`;
                - total-wealth ruin probability: `0.25`;
                - hybrid-region ruin probability: `0.25`;
                - positive-dependence impact on any-line ruin: `-0.25`.

                Reviewer rubric: please separate comments into mathematical
                correctness, API/usability, and teaching value.
                """
            ),
            code(
                """
                import numpy as np

                from ruin_theory import (
                    MarkovEnvironment,
                    dependence_impact,
                    finite_time_markov_modulated_ruin,
                )

                environment = MarkovEnvironment([1.0], [[1.0]])
                increment = [{(0, 0): 0.25, (2, 0): 0.25, (0, 2): 0.25, (2, 2): 0.25}]

                common = dict(
                    increment_pmfs=increment,
                    environment=environment,
                    initial_capitals=[1.0, 1.0],
                    premiums=[0.0, 0.0],
                    horizon=1,
                )

                any_line = finite_time_markov_modulated_ruin(**common, region="any_line")
                total = finite_time_markov_modulated_ruin(**common, region="total")
                hybrid = finite_time_markov_modulated_ruin(
                    **common,
                    region="hybrid",
                    severity_limit=1.0,
                )

                results = {
                    "any_line": any_line.ruin_probabilities[-1],
                    "total": total.ruin_probabilities[-1],
                    "hybrid": hybrid.ruin_probabilities[-1],
                }
                results
                """
            ),
            code(
                """
                np.testing.assert_allclose(
                    [results["any_line"], results["total"], results["hybrid"]],
                    [0.75, 0.25, 0.25],
                )

                independent = any_line
                positive_dependence = finite_time_markov_modulated_ruin(
                    [{(0, 0): 0.5, (2, 2): 0.5}],
                    environment,
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

                assert impact.final_difference == -0.25
                impact.final_difference
                """
            ),
        ]
    ),
    "multirisk_ctmc_dividends_reproduction.ipynb": notebook(
        [
            markdown(
                """
                # Multirisk CTMC Dividends And Insolvency Penalties

                Source family: Asmussen and Kella (2000), Frostig (2004), and
                Loisel's multirisk dividend/insolvency-penalty working paper.

                Paper-specific target: reproduce a one-line CTMC barrier case
                with an exponential ruin clock:

                - expected time to ruin: `0.5`;
                - ruin probability: `1.0`;
                - expected dividends on the line: `1.5`;
                - expected deficit at ruin: `1.0`.

                Reviewer rubric: please separate comments into mathematical
                correctness, API/usability, and teaching value.
                """
            ),
            code(
                """
                import numpy as np

                from ruin_theory import (
                    CommonShock,
                    estimate_multirisk_dividend_penalties_ctmc,
                )

                result = estimate_multirisk_dividend_penalties_ctmc(
                    initial_reserves=[1.0],
                    barriers=[1.0],
                    lower_bounds=[0.0],
                    grid_step=1.0,
                    environment_generator=[[0.0]],
                    environment_initial=[1.0],
                    shocks=[CommonShock(intensities=[2.0], claim_pmfs={(2,): 1.0})],
                    base_premium_rates=[3.0],
                    ruin_lines=[0],
                )

                summary = {
                    "state_count": result.state_count,
                    "expected_time_to_ruin": result.expected_time_to_ruin,
                    "ruin_probability": result.ruin_probability,
                    "expected_dividends": result.expected_dividends.tolist(),
                    "expected_deficit_at_ruin": result.expected_deficit_at_ruin.tolist(),
                }
                summary
                """
            ),
            code(
                """
                assert result.state_count == 2
                assert result.expected_time_to_ruin == 0.5
                assert result.ruin_probability == 1.0
                np.testing.assert_allclose(result.expected_dividends, [1.5])
                np.testing.assert_allclose(result.expected_deficit_at_ruin, [1.0])
                assert result.ruin_state_probabilities[(-1.0,)] == 1.0
                """
            ),
        ]
    ),
    "loisel_privault_sensitivity_reproduction.ipynb": notebook(
        [
            markdown(
                """
                # Loisel-Privault Finite-Time Sensitivity Reproduction

                Source family: Loisel and Privault integration-by-parts
                representation for infimum-density / finite-time ruin
                sensitivity.

                Paper-specific target: compare the IBP estimate of
                `d psi(u,T) / du` with a finite-difference derivative of the
                exact exponential finite-time ruin formula at `u = 0.5, 1.0,
                1.5`.

                Reviewer rubric: please separate comments into mathematical
                correctness, API/usability, and teaching value.
                """
            ),
            code(
                """
                import numpy as np

                from ruin_theory import (
                    CramerLundbergProcess,
                    estimate_finite_time_ruin_sensitivity_ibp,
                    exponential,
                    finite_time_ruin_exponential,
                )

                model = CramerLundbergProcess(
                    premium_rate=1.0,
                    claim_arrival_rate=0.5,
                    claim_distribution=exponential(rate=1.0),
                )
                surplus = np.array([0.5, 1.0, 1.5])

                estimate = estimate_finite_time_ruin_sensitivity_ibp(
                    model,
                    surplus,
                    horizon=2.0,
                    n_simulations=50_000,
                    seed=123,
                )

                step = 1e-4
                finite_difference = np.array(
                    [
                        (
                            finite_time_ruin_exponential(model, u=float(value + step), horizon=2.0)
                            - finite_time_ruin_exponential(model, u=float(value - step), horizon=2.0)
                        )
                        / (2.0 * step)
                        for value in surplus
                    ]
                )

                comparison = np.column_stack(
                    [
                        surplus,
                        estimate.ruin_probability_derivative,
                        finite_difference,
                        estimate.standard_error,
                    ]
                )
                comparison
                """
            ),
            code(
                """
                np.testing.assert_allclose(
                    estimate.ruin_probability_derivative,
                    finite_difference,
                    atol=4.0 * np.max(estimate.standard_error),
                )
                np.testing.assert_allclose(
                    estimate.density,
                    -estimate.ruin_probability_derivative,
                )
                """
            ),
        ]
    ),
    "inar_binar_byclaim_reproduction.ipynb": notebook(
        [
            markdown(
                """
                # INAR/BINAR By-Claim Reproduction

                Source family: McKenzie (1985), Al-Osh and Alzaid (1987), Du
                and Li (1991), Pedeli and Karlis (2011), plus the local
                INAR/BINAR by-claim simulation scripts.

                Paper-specific target: reproduce the deterministic expected
                by-claim count recursions and terminal-reserve means used to
                validate the package.

                Reviewer rubric: please separate comments into mathematical
                correctness, API/usability, and teaching value.
                """
            ),
            code(
                """
                import numpy as np

                from ruin_theory import (
                    BINARByClaimModel,
                    INARByClaimModel,
                    deterministic,
                    simulate_binar_byclaim_terminal_reserves,
                    simulate_inar_byclaim_terminal_reserves,
                )

                inar = INARByClaimModel(
                    initial_capital=0.0,
                    premium_per_period=37.0,
                    primary_count_mean=10.0,
                    initial_byclaim_mean=10.0,
                    reproduction=0.1,
                    primary_distribution=deterministic(2.0),
                    byclaim_distribution=deterministic(3.0),
                )

                inar_counts = inar.expected_byclaim_counts(4)
                inar_expected_terminal = inar.expected_terminal_reserve(4)
                inar_counts, inar_expected_terminal
                """
            ),
            code(
                """
                np.testing.assert_allclose(inar_counts, [11.0, 11.1, 11.11, 11.111])
                manual_terminal = 4 * 37.0 - 4 * 10.0 * 2.0 - np.sum(inar_counts) * 3.0
                assert inar_expected_terminal == manual_terminal

                reserves = simulate_inar_byclaim_terminal_reserves(
                    inar,
                    periods=8,
                    n_simulations=60_000,
                    seed=123,
                )
                assert abs(reserves.mean() - inar.expected_terminal_reserve(8)) < 0.45
                """
            ),
            code(
                """
                binar = BINARByClaimModel(
                    initial_capital=1000.0,
                    premium_per_period=15000.0,
                    primary_count_means=(5.0, 7.0),
                    initial_byclaim_means=(1.0, 1.0),
                    reproduction_matrix=((0.41, 0.1), (0.05, 0.3)),
                    primary_distributions=(deterministic(10.0), deterministic(1.0)),
                    byclaim_distributions=(deterministic(0.5), deterministic(0.5)),
                )

                matrix = np.array([[0.41, 0.1], [0.05, 0.3]])
                expected = np.empty((3, 2))
                previous = matrix @ np.ones(2) + np.array([5.0, 7.0])
                expected[0] = previous
                for index in range(1, 3):
                    previous = matrix @ previous + np.array([5.0, 7.0])
                    expected[index] = previous

                np.testing.assert_allclose(binar.expected_byclaim_counts(3), expected)
                expected
                """
            ),
        ]
    ),
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
