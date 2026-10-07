"""Compare independent package runs with the heavy-tail manuscript's CSVs.

Inputs are private author files; only aggregate discrepancies are written.
A five-combined-standard-error screen checks agreement between two Monte
Carlo estimates, not agreement with a first-order asymptotic approximation.
It is a diagnostic, not a formal simultaneous hypothesis test.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(result_path, paper_directory):
    result = json.loads(Path(result_path).read_text())["heavy_tail"]
    if result["mode"] != "paper-sized":
        raise ValueError("use --full before comparing with the paper-sized CSVs")
    directory = Path(paper_directory)
    checks, fingerprints = [], {}

    def read(name):
        path = directory / name
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline="") as handle:
            return [
                {key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)
            ]

    def add(family, point, estimate, se, reference, reference_se):
        combined = np.hypot(se, reference_se)
        z = float(abs(estimate - reference) / combined)
        checks.append(
            {
                "family": family,
                "point": point,
                "standardized_difference": z,
                "within_five_combined_se": z <= 5,
            }
        )

    for row in read("fig1_fig2_baseline_values.csv"):
        current = next(x for x in result["mean_curve"] if np.isclose(x["premium"], row["c"]))
        add("annual_mean", row["c"], current["mean"], current["mc_se"], row["mean"], row["se"])
    for row in read("fig_stable_laplace_values.csv"):
        j = int(np.argmin(abs(np.array(result["laplace_s"]) - row["s"])))
        for current in result["laplace"]:
            key = f"c{int(current['premium'])}"
            add(
                "laplace",
                [current["premium"], row["s"]],
                current["empirical"][j],
                current["mc_se"][j],
                row[key],
                row[key + "_se"],
            )
    for row in read("fig_alpha_robustness_values.csv"):
        current = next(
            x
            for x in result["tail_indices"]
            if np.isclose(x["alpha"], row["alpha"]) and np.isclose(x["scale"], row["b_c"])
        )
        add(
            "tail_index",
            [row["alpha"], row["b_c"]],
            current["ratio"],
            current["ratio_mc_se"],
            row["ratio"],
            row["ratio_se"],
        )
    keys = {
        "frequency": "heavy_frequency",
        "heavy_severity": "heavy_severity",
        "light_severity": "light_component",
    }
    for row in read("fig3_prevention_values.csv"):
        for mechanism, key in keys.items():
            current = next(
                x
                for x in result["prevention"]
                if x["mechanism"] == mechanism
                and np.isclose(100 * x["strength"], row["prevention_percent"])
            )
            add(
                "prevention",
                [mechanism, row["prevention_percent"]],
                current["ratio"],
                current["mc_se"],
                row[key + "_ratio"],
                row[key + "_se"],
            )
    for row in read("fig4_finite_mean_values.csv"):
        j = int(np.argmin(abs(np.array(result["finite_mean"]["capital"]) - row["u"])))
        for i, label in enumerate(result["finite_mean"]["labels"]):
            p = row[label]
            add(
                "finite_mean",
                [label, row["u"]],
                result["finite_mean"]["probabilities"][i][j],
                result["finite_mean"]["mc_se"][i][j],
                p,
                np.sqrt(p * (1 - p) / 600000),
            )
    return {
        "source_sha256": fingerprints,
        "number_of_checks": len(checks),
        "all_within_five_combined_se": all(x["within_five_combined_se"] for x in checks),
        "maximum_standardized_difference": max(x["standardized_difference"] for x in checks),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json")
    parser.add_argument("paper_figures_directory")
    parser.add_argument("--output", default="output/heavy_tail_paper_comparison.json")
    args = parser.parse_args()
    result = compare(args.result_json, args.paper_figures_directory)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"{result['number_of_checks']} comparisons; maximum standardized difference "
        f"{result['maximum_standardized_difference']:.3f}; "
        f"screen passed: {result['all_within_five_combined_se']}"
    )


if __name__ == "__main__":
    main()
