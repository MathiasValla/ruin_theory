"""Reproduce storm policy objectives from the authors' private standardized CSVs.

No loss samples are included in the package. Supply the numerical_v3/outputs
directory from the seasonal manuscript pipeline. Results use billion EUR,
monthly annualized rates, the retained cap already applied upstream, and
independent package simulations (not identical random seeds to the manuscript).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ruin_theory import (
    PreventionResponseCurve,
    compare_seasonal_prevention,
    empirical,
    optimize_seasonal_lundberg,
    optimize_seasonal_prevention,
    seasonal_lundberg_coefficient,
)


def read_calibrations(directory):
    paths = [
        Path(directory) / name
        for name in (
            "monthly_frequency_severity_inputs.csv",
            "monthly_retained_severity_samples.csv",
        )
    ]
    tables = []
    for path in paths:
        with path.open(newline="") as handle:
            tables.append(list(csv.DictReader(handle)))
    monthly, samples = tables
    calibrations = {}
    for run in sorted({row["run_id"] for row in monthly}):
        rows = sorted(
            (row for row in monthly if row["run_id"] == run), key=lambda row: int(row["month"])
        )
        if [int(row["month"]) for row in rows] != list(range(1, 13)):
            raise ValueError(f"incomplete monthly calibration: {run}")
        rates = np.array([float(row["annualized_intensity"]) for row in rows])
        laws = [
            empirical(
                [
                    float(row["retained_loss_bn"])
                    for row in samples
                    if row["run_id"] == run and int(row["month"]) == month
                ]
            )
            for month in range(1, 13)
        ]
        weights = rates * np.array([law.mean() for law in laws]) / 12
        reported = [float(row["kkt_expected_loss_weight_bn_per_year"]) for row in rows]
        np.testing.assert_allclose(weights, reported, rtol=1e-12, atol=1e-14)
        calibrations[run] = (rates, laws, weights)
    fingerprints = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return calibrations, fingerprints


def reproduce(
    directory,
    *,
    n_simulations=3000,
    bootstrap_replicates=0,
    premium_rule="fixed_control",
    historical_paths=0,
):
    calibrations, fingerprints = read_calibrations(directory)
    rates, laws, weights = calibrations["era5_historical_1976_2005"]
    curve = PreventionResponseCurve(shape=2)
    constant = np.full(12, 0.08)
    loss = optimize_seasonal_prevention(weights, response=curve, max_prevention=0.25, budget=0.08)
    frequency = optimize_seasonal_prevention(
        rates / 12, response=curve, max_prevention=0.25, budget=0.08
    )
    premium = 0.08 + 1.3 * float(weights @ curve(constant))
    lundberg = optimize_seasonal_lundberg(
        rates, laws, response=curve, premium_rate=premium, max_prevention=0.25, budget=0.08
    )
    policies = {
        "constant": constant,
        "frequency_KKT": frequency.amounts,
        "expected_loss_KKT": loss.amounts,
        "Lundberg_weighted": lundberg.allocation.amounts,
    }
    metrics = []
    for label, policy in policies.items():
        spending, annual_loss = float(policy.mean()), float(weights @ curve(policy))
        metrics.append(
            {
                "policy": label,
                "budget": spending,
                "loss": annual_loss,
                "outflow": spending + annual_loss,
                "net_profit": premium - spending - annual_loss,
                "root": seasonal_lundberg_coefficient(
                    rates, laws, prevention=policy, response=curve, premium_rate=premium
                ),
                "calendar": policy.tolist(),
            }
        )
    # Rounded values in Table 1 of the supplied October manuscript.
    reference = np.array([[1.061, 0.4970], [0.867, 0.8260], [0.841, 0.8693], [0.843, 0.8723]])
    computed = np.array([[row["loss"], row["root"]] for row in metrics])
    np.testing.assert_allclose(computed[:, 0], reference[:, 0], rtol=0, atol=0.0005)
    np.testing.assert_allclose(computed[:, 1], reference[:, 1], rtol=0, atol=0.00005)
    simulations = []
    seeds = np.random.SeedSequence(20261009).spawn(len(calibrations) + 3)
    for index, (run, (run_rates, run_laws, run_weights)) in enumerate(calibrations.items()):
        if n_simulations == 0:
            continue
        if premium_rule == "fixed_control" and run != "era5_historical_1976_2005":
            model = run.split("_rcp")[0].split("_historical")[0]
            premium_weights = calibrations[f"{model}_historical_control_1976_2005"][2]
        elif premium_rule == "scenario_repriced" or run == "era5_historical_1976_2005":
            premium_weights = run_weights
        else:
            raise ValueError("premium_rule must be fixed_control or scenario_repriced")
        run_premium = 0.08 + 1.3 * float(premium_weights @ curve(constant))
        result = compare_seasonal_prevention(
            run_rates,
            run_laws,
            {"constant": constant, "transferred_historical": loss.amounts},
            response=curve,
            premium_rate=run_premium,
            initial_capital=2,
            horizon=2,
            n_simulations=n_simulations,
            seed=int(seeds[index].generate_state(1)[0]),
        )
        simulations.append(
            {
                "run": run,
                "premium_rule": premium_rule,
                "premium": run_premium,
                "probabilities": [estimate.probability for estimate in result.estimates],
                "absolute_gain": float(result.absolute_gains[1]),
                "paired_gain_se": float(result.gain_standard_errors[1]),
            }
        )
    historical = []
    if historical_paths:
        for index, (capital, paper_probabilities) in enumerate(
            [(1, [0.2648, 0.1764]), (2, [0.0951, 0.0524]), (4, [0.0099, 0.0037])]
        ):
            paired = compare_seasonal_prevention(
                rates,
                laws,
                {"constant": constant, "expected_loss_KKT": loss.amounts},
                response=curve,
                premium_rate=premium,
                initial_capital=capital,
                horizon=2,
                n_simulations=historical_paths,
                seed=20261011 + index,
            )
            estimates = np.array([item.probability for item in paired.estimates])
            se = np.array([item.standard_error for item in paired.estimates])
            paper_values = np.array(paper_probabilities)
            paper_se = np.sqrt(paper_values * (1 - paper_values) / 50000)
            discrepancy = abs(estimates - paper_values) / np.hypot(se, paper_se)
            historical.append(
                {
                    "capital": capital,
                    "paths": historical_paths,
                    "probabilities": estimates.tolist(),
                    "mc_se": se.tolist(),
                    "absolute_gain": float(paired.absolute_gains[1]),
                    "paired_gain_se": float(paired.gain_standard_errors[1]),
                    "paper_probabilities": paper_probabilities,
                    "standardized_differences": discrepancy.tolist(),
                }
            )
    bootstrap = None
    if bootstrap_replicates:
        # Resample complete years, preserving within-year seasonal dependence.
        with (Path(directory) / "monthly_retained_severity_samples.csv").open(newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["run_id"] == "era5_historical_1976_2005"
            ]
        years = sorted({int(row["date"][:4]) for row in rows})
        totals = np.zeros((len(years), 12))
        counts = np.zeros(len(years))
        for row in rows:
            totals[years.index(int(row["date"][:4])), int(row["month"]) - 1] += float(
                row["retained_loss_bn"]
            )
            counts[years.index(int(row["date"][:4]))] += 1
        rng = np.random.default_rng(20261010)
        samples = []
        for _ in range(bootstrap_replicates):
            indices = rng.integers(0, len(years), len(years))
            fitted_weights = totals[indices].mean(axis=0)
            allocation = optimize_seasonal_prevention(
                fitted_weights, response=curve, max_prevention=0.25, budget=0.08
            )
            baseline_loss = float(fitted_weights @ curve(constant))
            samples.append(
                [
                    float(counts[indices].mean()),
                    float(fitted_weights.sum()),
                    100 * (1 - allocation.controlled_loss / baseline_loss),
                ]
            )
        bootstrap = {
            "replicates": bootstrap_replicates,
            "columns": ["positive_days_per_year", "annual_retained_loss", "timing_gain_percent"],
            "mean": np.mean(samples, axis=0).tolist(),
            "percentile_95": np.quantile(samples, [0.025, 0.975], axis=0).tolist(),
            "scope": "expected-loss calibration uncertainty; not ruin-probability intervals",
        }
    return {
        "input_sha256": fingerprints,
        "raw_data_distributed": False,
        "units": "billion EUR",
        "premium": premium,
        "paths_per_scenario": n_simulations,
        "paper_table_1_passed": True,
        "policy_metrics": metrics,
        "scenario_simulations": simulations,
        "historical_simulations": historical,
        "year_block_bootstrap": bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory")
    parser.add_argument("--paths", type=int, default=3000)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--historical-paths", type=int, default=0)
    parser.add_argument(
        "--premium-rule", choices=["fixed_control", "scenario_repriced"], default="fixed_control"
    )
    parser.add_argument("--output", default="output/seasonal_storm_replication.json")
    args = parser.parse_args()
    result = reproduce(
        args.input_directory,
        n_simulations=args.paths,
        bootstrap_replicates=args.bootstrap,
        premium_rule=args.premium_rule,
        historical_paths=args.historical_paths,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(target.resolve())


if __name__ == "__main__":
    main()
