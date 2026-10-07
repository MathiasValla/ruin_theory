"""Run the two 2026 prevention papers' model experiments with explicit MC error.

Default smoke run is intentionally small; --full uses the heavy-tail paper's
grids and sample sizes. Storm tables require the authors' daily data; no private
or copyrighted inputs are distributed. --storm-data accepts year,month,loss CSV.
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from ruin_theory import (
    AnnualHeavyTailModel,
    AnnualRuinTimeEstimate,
    PreventionResponseCurve,
    calibrate_storm_loss_days,
    compare_seasonal_prevention,
    exponential,
    finite_mean_iso_loss_ruin,
    infinite_mean_prevention_multiplier,
    integrate_seasonal_pressure,
    optimize_dynamic_prevention_calendar,
    optimize_seasonal_prevention,
    optimize_seasonal_lundberg,
    premium_for_infinite_mean_scale,
    seasonal_exponential_profile_root,
    seasonal_lundberg_coefficient,
    simulate_annual_heavy_tail_ruin_times,
    stable_ruin_time_laplace,
)


def seasonal_cases(storm_data=None, n_simulations=300):
    response = PreventionResponseCurve(shape=2)
    t = (np.arange(12) + 0.5) / 12
    weights = 1.24 * (1 + 0.8 * np.cos(2 * np.pi * t)) / 12
    rates, laws = 12 * weights / (1.24 / 39.7), [exponential(39.7 / 1.24)] * 12
    calibration = "synthetic seasonal illustration, not ERA5 table replication"
    if storm_data is not None:
        with Path(storm_data).open() as handle:
            rows = list(csv.DictReader(handle))
        fitted = calibrate_storm_loss_days(
            [float(row["loss"]) for row in rows],
            [int(row["month"]) for row in rows],
            [int(row["year"]) for row in rows],
            retained_loss_cap=2e9,
        )
        weights, rates, laws = fitted.loss_weights, fitted.frequencies, list(fitted.distributions)
        calibration = "user-supplied daily storm data, retained cap EUR 2bn"
    allocations = {}
    for family in ("exponential", "quadratic", "reciprocal"):
        curve = PreventionResponseCurve(family, shape=2, offset=1)
        for convention in ("rate", "amount"):
            for budget in (None, 0.08):
                result = optimize_seasonal_prevention(
                    weights,
                    response=curve,
                    max_prevention=0.25,
                    budget=budget,
                    budget_convention=convention,
                )
                key = f"{family}_{convention}_{'free' if budget is None else 'fixed'}"
                allocations[key] = {
                    "calendar": result.amounts.tolist(),
                    "budget": result.budget_spent,
                    "loss": result.controlled_loss,
                    "outflow": result.total_outflow,
                }
    fixed = optimize_seasonal_prevention(
        weights, response=response, max_prevention=0.25, budget=0.08
    )
    numeric = optimize_seasonal_prevention(
        weights, response=lambda p: np.exp(-2 * p), max_prevention=0.25, budget=0.08
    )
    delayed = optimize_seasonal_prevention(
        weights, response=response, max_prevention=0.25, budget=0.08, lag_steps=2
    )
    dynamic = optimize_dynamic_prevention_calendar(
        weights,
        initial_budget=0.16,
        max_prevention=0.25,
        effectiveness=2,
        n_cycles=2,
        budget_grid_size=31,
    )
    paired = compare_seasonal_prevention(
        rates,
        laws,
        {"constant": np.full(12, 0.08), "periodic": fixed.amounts},
        response=response,
        premium_rate=1.46,
        initial_capital=2,
        horizon=2,
        n_simulations=n_simulations,
        seed=20261007,
    )
    profiles = {}
    for profile in ("cosine", "triangular", "affine"):
        exact = seasonal_exponential_profile_root(
            base_rate=2, variation=0.4, frequency=1, premium_rate=1, profile=profile
        )
        approximations = []
        for m in (12, 48, 192):
            points = (np.arange(m) + 0.5) / m
            eta = 2 + 0.4 * (
                np.cos(2 * np.pi * points)
                if profile == "cosine"
                else (points if profile == "affine" else np.minimum(points, 1 - points))
            )
            root = seasonal_lundberg_coefficient(
                np.ones(m),
                [exponential(x) for x in eta],
                prevention=np.zeros(m),
                response=lambda p: 1,
                premium_rate=1,
            )
            approximations.append(
                {"intervals": m, "root": root, "absolute_error": abs(root - exact)}
            )
        profiles[profile] = {"closed_root": exact, "mesh": approximations}
    continuous = integrate_seasonal_pressure(
        lambda x: 1 + 0.8 * np.cos(2 * np.pi * x), breaks=np.linspace(0, 1, 13)
    )
    root_optimum = optimize_seasonal_lundberg(
        rates, laws, response=response, premium_rate=1.46, max_prevention=0.25, budget=0.08
    )
    return {
        "calibration": calibration,
        "allocations": allocations,
        "generic_solver_calendar_error": float(np.max(abs(fixed.amounts - numeric.amounts))),
        "delayed_loss": delayed.controlled_loss,
        "dynamic_result": {"controlled_pressure": float(dynamic.controlled_pressure)},
        "continuous_pressure_total": float(continuous.sum()),
        "profile_roots": profiles,
        "root_optimal_calendar": root_optimum.allocation.amounts.tolist(),
        "optimal_root": root_optimum.coefficient,
        "paired_ruin": {
            "probabilities": [x.probability for x in paired.estimates],
            "individual_mc_se": [x.standard_error for x in paired.estimates],
            "absolute_gain": float(paired.absolute_gains[1]),
            "paired_gain_mc_se": float(paired.gain_standard_errors[1]),
        },
    }


def heavy_tail_cases(full=False, cache_directory=None, progress=False):
    seeds = np.random.SeedSequence(20261007).spawn(150)
    source = Path(__file__).resolve().parents[1] / "src/ruin_theory/heavy_tail_prevention.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    index = 0

    def run(model, premium, n):
        nonlocal index
        seed = int(seeds[index].generate_state(1)[0])
        index += 1
        limit = 10_000_000 if model.light_mean * model.light_severity_factor == 0 else 2_000_000
        cache = None
        if cache_directory is not None:
            settings = json.dumps(
                [asdict(model), float(premium), n, seed, limit, source_hash], sort_keys=True
            )
            key = hashlib.sha256(settings.encode()).hexdigest()
            cache = Path(cache_directory) / (key + ".npz")
            cache.parent.mkdir(parents=True, exist_ok=True)
        if cache is not None and cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                times = saved["ruin_times"]
            mean, scale = float(times.mean()), model.scale(premium)
            result = AnnualRuinTimeEstimate(
                times, mean, float(times.std(ddof=1) / np.sqrt(n)), scale, mean / scale
            )
        else:
            result = simulate_annual_heavy_tail_ruin_times(
                model, premium=premium, n_simulations=n, seed=seed, max_steps=limit
            )
            if cache is not None:
                np.savez_compressed(cache, ruin_times=result.ruin_times)
        if progress:
            print(
                f"completed case {index}: alpha={model.alpha}, premium={premium:.6g}, N={n}",
                flush=True,
            )
        return result

    base = AnnualHeavyTailModel(light_family="deterministic")
    mean_rows = []
    for premium in (
        np.unique(np.r_[np.linspace(25, 1500, 31), [25, 50, 100, 200, 400, 800, 1200, 1500]])
        if full
        else [25, 100]
    ):
        result = run(base, premium, 100000 if full else 150)
        mean_rows.append(
            {
                "premium": float(premium),
                "mean": result.mean,
                "mc_se": result.standard_error,
                "asymptotic": result.asymptotic_mean,
                "ratio": result.mean_ratio,
            }
        )
    s = np.linspace(0.1, 3, 30) if full else np.array([0.1, 0.5, 1, 2])
    laplace = []
    for premium in [50, 200, 800] if full else [50, 200]:
        result = run(base, premium, 30000 if full else 150)
        draws = np.exp(-s[:, None] * result.ruin_times / result.asymptotic_mean)
        laplace.append(
            {
                "premium": premium,
                "empirical": draws.mean(axis=1).tolist(),
                "mc_se": (draws.std(axis=1, ddof=1) / np.sqrt(draws.shape[1])).tolist(),
            }
        )
    robust = []
    for alpha in (0.3, 0.5, 0.7):
        for scale in [200, 400, 800, 1600, 3200, 8000, 20000] if full else [200]:
            premium = premium_for_infinite_mean_scale(scale, alpha, 0.2)
            result = run(replace(base, alpha=alpha), premium, 12000 if full else 150)
            robust.append(
                {
                    "alpha": alpha,
                    "scale": scale,
                    "premium": premium,
                    "ratio": result.mean_ratio,
                    "ratio_mc_se": result.standard_error / scale,
                }
            )
    two = replace(base, light_family="exponential", light_mean=1)
    reference = run(two, 400, 12000 if full else 150)
    prevention = []
    for strength in np.arange(0.10, 0.401, 0.05) if full else [0.2]:
        for mechanism, parameter in [
            ("frequency", "frequency_factor"),
            ("heavy_severity", "heavy_severity_factor"),
            ("light_severity", "light_severity_factor"),
        ]:
            result = run(replace(two, **{parameter: 1 - strength}), 400, 12000 if full else 150)
            ratio = result.mean / reference.mean
            error = ratio * np.hypot(
                result.standard_error / result.mean, reference.standard_error / reference.mean
            )
            prevention.append(
                {
                    "strength": float(strength),
                    "mechanism": mechanism,
                    "ratio": ratio,
                    "mc_se": error,
                    "theory": float(
                        infinite_mean_prevention_multiplier(strength, 0.5, mechanism=mechanism)
                    ),
                }
            )
    finite = finite_mean_iso_loss_ruin(
        np.linspace(0, 20, 41), n_simulations=600000 if full else 10000, seed=20261008
    )
    return {
        "mode": "paper-sized" if full else "smoke, not paper-sized",
        "mean_curve": mean_rows,
        "laplace_s": s.tolist(),
        "laplace": laplace,
        "laplace_theory": stable_ruin_time_laplace(s, 0.5).tolist(),
        "tail_indices": robust,
        "prevention": prevention,
        "finite_mean": {
            "capital": finite.capital.tolist(),
            "labels": finite.labels,
            "probabilities": finite.probabilities.tolist(),
            "mc_se": finite.standard_errors.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--storm-data")
    parser.add_argument("--output", default="output/prevention_paper_reproductions.json")
    args = parser.parse_args()
    result = {
        "seed": 20261007,
        "seasonal": seasonal_cases(args.storm_data, 10000 if args.full else 300),
        "heavy_tail": heavy_tail_cases(
            args.full,
            cache_directory=Path(args.output).parent / "prevention_cache",
            progress=True,
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(path.resolve())


if __name__ == "__main__":
    main()
