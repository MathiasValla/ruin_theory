"""Plotting diagnostics tests."""

# ruff: noqa: E402

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from ruin_theory.plotting import (
    plot_path,
    plot_paths,
    plot_ruin_curve,
    plot_ruin_time_histogram,
)
from ruin_theory.results import RuinEstimate, SimulationPath


def _path(*, horizon: float = 3.0, ruin_time: float | None = 1.0) -> SimulationPath:
    return SimulationPath(
        times=np.array([0.0, 1.0, 1.0, 2.0]),
        reserves=np.array([2.0, 3.0, -0.5, 0.5]),
        claim_times=np.array([1.0]),
        claim_sizes=np.array([3.5]),
        ruin_time=ruin_time,
        horizon=horizon,
        initial_capital=2.0,
        premium_rate=1.0,
    )


def _estimate(ruin_times: np.ndarray, *, horizon: float = 3.0) -> RuinEstimate:
    finite = np.isfinite(ruin_times)
    probability = float(np.mean(finite))
    return RuinEstimate(
        probability=probability,
        standard_error=0.0,
        ci_low=probability,
        ci_high=probability,
        n_simulations=ruin_times.size,
        horizon=horizon,
        ruin_times=ruin_times,
    )


def test_plot_path_marks_reserve_and_ruin_time():
    fig, ax = plt.subplots()
    try:
        result = plot_path(_path(), ax=ax)

        assert result is ax
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "reserve"
        assert ax.get_title() == "Reserve trajectory"
        assert ax.get_xlim() == pytest.approx((0.0, 3.0))
        assert len(ax.lines) == 3
    finally:
        plt.close(fig)


def test_plot_paths_overlays_paths_and_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        plot_paths([])

    fig, ax = plt.subplots()
    try:
        result = plot_paths([_path(horizon=2.0), _path(horizon=4.0, ruin_time=None)], ax=ax)

        assert result is ax
        assert ax.get_title() == "Simulated reserve trajectories"
        assert ax.get_xlim() == pytest.approx((0.0, 4.0))
        assert len(ax.lines) == 3
    finally:
        plt.close(fig)


def test_plot_ruin_curve_validates_and_labels_probability_curve():
    fig, ax = plt.subplots()
    try:
        result = plot_ruin_curve([0.0, 1.0, 2.0], [0.8, 0.5, 0.2], ax=ax, label="ultimate")

        assert result is ax
        assert ax.get_xlabel() == "initial surplus"
        assert ax.get_ylabel() == "ruin probability"
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))
        assert ax.get_legend() is not None
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.8, 0.5, 0.2])
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="matching shapes"):
        plot_ruin_curve([0.0, 1.0], [0.5])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        plot_ruin_curve([0.0], [1.2])


def test_plot_ruin_time_histogram_handles_ruined_and_unruined_samples():
    fig, ax = plt.subplots()
    try:
        result = plot_ruin_time_histogram(_estimate(np.array([0.5, 1.0, np.inf, 2.0])), ax=ax)

        assert result is ax
        assert ax.get_xlabel() == "time to ruin"
        assert ax.get_ylabel() == "frequency"
        assert ax.get_xlim() == pytest.approx((0.0, 3.0))
        assert len(ax.patches) > 0
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_ruin_time_histogram(_estimate(np.array([np.inf, np.inf])), ax=ax)

        assert len(ax.patches) == 0
        assert [text.get_text() for text in ax.texts] == ["no ruin observed"]
        assert list(ax.get_yticks()) == []
    finally:
        plt.close(fig)
