"""Reproducible IBP timing and peak-allocation benchmark (no speed assertion)."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy

from ruin_theory import (
    CramerLundbergProcess,
    estimate_finite_time_ruin_sensitivity_ibp,
    exponential,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.simulations, args.grid_size, args.repeats) <= 0:
        parser.error("simulations, grid-size and repeats must be positive")

    model = CramerLundbergProcess(
        premium_rate=6.0, claim_arrival_rate=5.0, claim_distribution=exponential(1.0),
    )
    surplus = np.linspace(0.0, 10.0, args.grid_size)

    def run():
        return estimate_finite_time_ruin_sensitivity_ibp(
            model, surplus, horizon=4.0, n_simulations=args.simulations, seed=2026,
        )

    run()
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run()
        durations.append(time.perf_counter() - start)
    tracemalloc.start()
    result = run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "simulations": args.simulations,
        "grid_size": args.grid_size,
        "seed": 2026,
        "horizon": 4.0,
        "median_seconds": statistics.median(durations),
        "all_seconds": durations,
        "peak_traced_bytes": peak,
        "density_sum": float(result.density.sum()),
        "standard_error_sum": float(result.standard_error.sum()),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        np.savez(args.output.with_suffix(".npz"), density=result.density,
                 standard_error=result.standard_error)
    print(rendered, end="")


if __name__ == "__main__":
    main()
