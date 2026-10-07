"""Standalone replication of all numerical tables and figures in the JSS draft.

Run: python examples/jss_replication.py. No external data or downloads.
Elapsed times are metadata, not exactly reproducible quantities.
"""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import scipy

from ruin_theory import (
    AnnualHeavyTailModel,
    CramerLundbergProcess,
    INARByClaimModel,
    PreventionResponseCurve,
    adjustment_coefficient,
    compare_seasonal_prevention,
    deterministic,
    estimate_ruin_probability,
    exponential,
    finite_mean_iso_loss_ruin,
    finite_time_ruin_discrete,
    finite_time_ruin_exponential,
    optimize_seasonal_prevention,
    pollaczek_khinchine_monte_carlo,
    simulate_annual_heavy_tail_ruin_times,
    simulate_inar_byclaim_terminal_reserves,
    stable_ruin_time_laplace,
    ultimate_ruin_exponential,
)
from prevention_paper_reproductions import seasonal_cases

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "jss"


def table(name, header, rows):
    lines = [
        "\\begin{tabular}{" + "r" * len(header) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUTPUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "package_version": version("ruin-theory"),
        "seed": 20261007,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/ruin_theory").glob("*.py"))
        },
    }
    start = time.perf_counter()
    model = CramerLundbergProcess(
        premium_rate=1, claim_arrival_rate=3, claim_distribution=exponential(5)
    )
    reserves = [0, 1, 2]
    exact = ultimate_ruin_exponential(model, u=np.array(reserves))
    pk = pollaczek_khinchine_monte_carlo(model, reserves, n_simulations=30000, seed=20261007)
    classical, rows = [], []
    for i, u in enumerate(reserves):
        finite = finite_time_ruin_exponential(model, u, 10)
        process = CramerLundbergProcess(
            initial_capital=u,
            premium_rate=1,
            claim_arrival_rate=3,
            claim_distribution=exponential(5),
        )
        mc = estimate_ruin_probability(process, horizon=10, n_simulations=20000, seed=20261007 + i)
        pkse = float(np.sqrt(pk[i] * (1 - pk[i]) / 30000))
        rows.append(
            [
                str(u),
                f"{exact[i]:.7f}",
                f"{finite:.7f}",
                f"{mc.probability:.7f}",
                f"{mc.standard_error:.7f}",
                f"{pk[i]:.7f}",
                f"{pkse:.7f}",
            ]
        )
        classical.append(
            {
                "u": u,
                "ultimate_exact": float(exact[i]),
                "finite_exact": finite,
                "finite_mc": mc.probability,
                "finite_se": mc.standard_error,
                "pk_mc": float(pk[i]),
                "pk_se": pkse,
            }
        )
    assert abs(adjustment_coefficient(model) - 2) < 1e-8
    table(
        "classical_table",
        ["$u$", "$\\psi$", "$\\psi_{10}$", "Path MC", "MC SE", "PK MC", "MC SE"],
        rows,
    )
    lattice, timings, rows = [], {}, []
    for u in [0, 5, 10, 15, 20]:
        results = {}
        for method in ["seal", "picard-lefevre", "inventory"]:
            elapsed = []
            for _ in range(3):
                begin = time.perf_counter()
                value = finite_time_ruin_discrete(
                    [0, 1],
                    initial_capital=u,
                    premium_rate=1.25,
                    claim_arrival_rate=1,
                    horizon=10,
                    method=method,
                )
                elapsed.append(time.perf_counter() - begin)
            results[method] = value
            timings[f"{method}_u{u}"] = float(np.median(elapsed))
        error = max(abs(value - results["seal"]) for value in results.values())
        assert error < 1e-10
        rows.append([str(u), f"{results['seal']:.12g}", f"{error:.2g}"])
        lattice.append({"u": u, "results": results, "maximum_route_difference": error})
    table("lattice_table", ["$u$", "$\\psi_{10}(u)$", "Maximum difference"], rows)
    seasonal = seasonal_cases(n_simulations=300)
    weights = 1.24 * (1 + 0.8 * np.cos(2 * np.pi * (np.arange(12) + 0.5) / 12)) / 12
    curve = PreventionResponseCurve(shape=2)
    fixed = optimize_seasonal_prevention(weights, response=curve, max_prevention=0.25, budget=0.08)
    free = optimize_seasonal_prevention(weights, response=curve, max_prevention=0.25)
    policies = {
        "Constant": np.full(12, 0.08),
        "Fixed-budget": fixed.amounts,
        "Free-budget": free.amounts,
    }
    compared = compare_seasonal_prevention(
        12 * weights / (1.24 / 39.7),
        [exponential(39.7 / 1.24)] * 12,
        policies,
        response=curve,
        premium_rate=1.46,
        initial_capital=0.2,
        horizon=2,
        n_simulations=3000,
        seed=20261007,
        record_paths=1,
    )
    rows = []
    for i, (label, policy) in enumerate(policies.items()):
        loss, spent = float(weights @ curve(policy)), float(np.mean(policy))
        rows.append(
            [
                label,
                f"{spent:.6f}",
                f"{loss:.6f}",
                f"{spent + loss:.6f}",
                f"{compared.estimates[i].probability:.5f}",
                f"{compared.estimates[i].standard_error:.5f}",
                f"{compared.absolute_gains[i]:.5f}",
                f"{compared.gain_standard_errors[i]:.5f}",
            ]
        )
    table(
        "seasonal_table",
        ["Policy", "$B$", "$S$", "$B+S$", "$\\hat\\psi_2$", "MC SE", "Gain", "Paired SE"],
        rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    months = np.arange(1, 13)
    axes[0].plot(months, weights, color="0.4", marker=".", label="Loss weight")
    for label, policy in policies.items():
        axes[0].plot(months, policy, marker=".", label=label)
        times, values = compared.trajectories[label][0]
        axes[1].plot(times, values, label=label)
    axes[0].set(xlabel="Month", ylabel="Loss weight / prevention rate")
    axes[1].set(xlabel="Time (years)", ylabel="Reserve")
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.savefig(OUTPUT / "seasonal.png", dpi=200)
    plt.close(fig)
    rootrows = []
    for profile, result in seasonal["profile_roots"].items():
        for mesh in result["mesh"]:
            rootrows.append(
                [
                    profile,
                    str(mesh["intervals"]),
                    f"{result['closed_root']:.9f}",
                    f"{mesh['root']:.9f}",
                    f"{mesh['absolute_error']:.3g}",
                ]
            )
    table("root_table", ["Profile", "$m$", "Closed form", "Mesh root", "Error"], rootrows)
    annual = simulate_annual_heavy_tail_ruin_times(
        AnnualHeavyTailModel(), premium=50, n_simulations=2000, seed=20261007
    )
    s = np.array([0, 0.1, 0.5, 1, 2, 4])
    transformed = np.exp(-s[:, None] * annual.ruin_times / annual.asymptotic_mean)
    finite = finite_mean_iso_loss_ruin(np.arange(0, 21), n_simulations=60000, seed=20261007)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    axes[0].plot(s, stable_ruin_time_laplace(s, 0.5), label="Stable limit")
    axes[0].errorbar(
        s,
        transformed.mean(axis=1),
        yerr=1.96 * transformed.std(axis=1, ddof=1) / np.sqrt(2000),
        fmt="o",
        label="Annual MC",
    )
    axes[0].set(xlabel="Laplace argument", ylabel="Transform")
    axes[0].legend(fontsize=8)
    for label, p, se in zip(finite.labels, finite.probabilities, finite.standard_errors):
        axes[1].plot(finite.capital, p, label=label.replace("_", " "))
        axes[1].fill_between(finite.capital, p - 1.96 * se, p + 1.96 * se, alpha=0.15)
    axes[1].set(xlabel="Initial reserve", ylabel="Ultimate ruin probability")
    axes[1].legend(fontsize=8)
    fig.savefig(OUTPUT / "heavy_tail.png", dpi=200)
    plt.close(fig)
    table(
        "heavy_table",
        ["$c$", "$b_c$", "Mean time", "MC SE", "Mean/$b_c$"],
        [
            [
                "50",
                f"{annual.asymptotic_mean:.4f}",
                f"{annual.mean:.4f}",
                f"{annual.standard_error:.4f}",
                f"{annual.mean_ratio:.5f}",
            ]
        ],
    )
    inar = INARByClaimModel(
        initial_capital=0,
        premium_per_period=37,
        primary_count_mean=10,
        initial_byclaim_mean=10,
        reproduction=0.1,
        primary_distribution=deterministic(2),
        byclaim_distribution=deterministic(3),
    )
    terminal = simulate_inar_byclaim_terminal_reserves(
        inar, periods=4, n_simulations=20000, seed=20261007
    )
    inar_result = {
        "counts": inar.expected_byclaim_counts(4).tolist(),
        "terminal_exact": inar.expected_terminal_reserve(4),
        "terminal_mc": float(terminal.mean()),
        "mc_se": float(terminal.std(ddof=1) / np.sqrt(terminal.size)),
    }
    table(
        "inar_table",
        ["Periods", "Exact reserve mean", "MC mean", "MC SE"],
        [
            [
                "4",
                f"{inar_result['terminal_exact']:.6f}",
                f"{inar_result['terminal_mc']:.6f}",
                f"{inar_result['mc_se']:.6f}",
            ]
        ],
    )
    output = {
        "metadata": metadata,
        "classical": classical,
        "lattice": lattice,
        "seasonal": seasonal,
        "paired_probabilities": [r.probability for r in compared.estimates],
        "annual": {
            "mean": annual.mean,
            "se": annual.standard_error,
            "scale": annual.asymptotic_mean,
        },
        "inar": inar_result,
        "median_runtime_seconds": timings,
        "total_runtime_seconds": time.perf_counter() - start,
    }
    (OUTPUT / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Replicated all draft tables and figures in {OUTPUT}")


if __name__ == "__main__":
    main()
