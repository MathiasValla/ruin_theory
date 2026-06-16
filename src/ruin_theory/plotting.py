"""Matplotlib diagnostics for ruin models and simulations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .results import RuinEstimate, SimulationPath


def _axis(ax: Axes | None) -> Axes:
    return plt.gca() if ax is None else ax


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _plot_reserve_path(
    axis: Axes,
    path: SimulationPath,
    *,
    alpha: float = 1.0,
    linewidth: float = 1.8,
) -> np.ndarray:
    times = _as_1d_float(path.times, "path.times")
    reserves = _as_1d_float(path.reserves, "path.reserves")
    if times.shape != reserves.shape:
        raise ValueError("path.times and path.reserves must have matching shapes")
    axis.step(times, reserves, where="post", color="#1f77b4", alpha=alpha, linewidth=linewidth)
    return times


def _path_xlim(paths: Iterable[SimulationPath]) -> tuple[float, float] | None:
    upper = 0.0
    for path in paths:
        times = np.asarray(path.times, dtype=float)
        candidates = [float(path.horizon)]
        if times.size:
            candidates.append(float(np.nanmax(times)))
        finite = [value for value in candidates if np.isfinite(value) and value > 0.0]
        if finite:
            upper = max(upper, max(finite))
    if upper <= 0.0:
        return None
    return (0.0, upper)


def plot_path(path: SimulationPath, *, ax: Axes | None = None, show_ruin: bool = True) -> Axes:
    """Plot a simulated reserve path."""

    axis = _axis(ax)
    _plot_reserve_path(axis, path)
    axis.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    if show_ruin and path.ruin_time is not None:
        axis.axvline(path.ruin_time, color="#b00020", linewidth=1.2, linestyle=":")
    xlim = _path_xlim((path,))
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.set_xlabel("time")
    axis.set_ylabel("reserve")
    axis.set_title("Reserve trajectory")
    return axis


def plot_paths(
    paths: Iterable[SimulationPath],
    *,
    ax: Axes | None = None,
    alpha: float = 0.25,
) -> Axes:
    """Overlay several reserve trajectories."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    path_list = tuple(paths)
    if not path_list:
        raise ValueError("paths must contain at least one SimulationPath")

    axis = _axis(ax)
    for path in path_list:
        _plot_reserve_path(axis, path, alpha=alpha, linewidth=1.2)
    axis.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    xlim = _path_xlim(path_list)
    if xlim is not None:
        axis.set_xlim(*xlim)
    axis.set_xlabel("time")
    axis.set_ylabel("reserve")
    axis.set_title("Simulated reserve trajectories")
    return axis


def plot_ruin_curve(
    u: ArrayLike,
    probabilities: ArrayLike,
    *,
    ax: Axes | None = None,
    label: str | None = None,
) -> Axes:
    """Plot ruin probability against initial surplus."""

    surplus = _as_1d_float(u, "u")
    ruin_probabilities = _as_1d_float(probabilities, "probabilities")
    if surplus.shape != ruin_probabilities.shape:
        raise ValueError("u and probabilities must have matching shapes")
    if np.any((ruin_probabilities < 0.0) | (ruin_probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    axis = _axis(ax)
    axis.plot(surplus, ruin_probabilities, color="#0b6e4f", linewidth=2.0, label=label)
    axis.set_xlabel("initial surplus")
    axis.set_ylabel("ruin probability")
    axis.set_ylim(0.0, 1.0)
    if label:
        axis.legend()
    return axis


def plot_ruin_time_histogram(
    estimate: RuinEstimate,
    *,
    ax: Axes | None = None,
    bins: int = 30,
) -> Axes:
    """Plot conditional ruin-time histogram from a Monte Carlo estimate."""

    if bins <= 0:
        raise ValueError("bins must be positive")

    axis = _axis(ax)
    ruin_times = np.asarray(estimate.ruin_times, dtype=float)
    finite = ruin_times[np.isfinite(ruin_times)]
    if np.any(finite < 0.0):
        raise ValueError("ruin times must be non-negative")
    if finite.size:
        axis.hist(finite, bins=bins, color="#4c78a8", alpha=0.85)
    else:
        axis.text(
            0.5,
            0.5,
            "no ruin observed",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_yticks([])
    if estimate.horizon is not None and np.isfinite(estimate.horizon) and estimate.horizon > 0:
        axis.set_xlim(0.0, float(estimate.horizon))
    axis.set_xlabel("time to ruin")
    axis.set_ylabel("frequency")
    axis.set_title("Conditional time to ruin")
    return axis
