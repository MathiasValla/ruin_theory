"""Gerber-Shiu discounted penalty diagnostics tests."""

import math

import numpy as np
import pytest

from ruin_theory import (
    CramerLundbergProcess,
    GerberShiuResult,
    deterministic,
    estimate_gerber_shiu,
    gerber_shiu_from_paths,
)
from ruin_theory.results import SimulationPath


def _ruined_path() -> SimulationPath:
    return SimulationPath(
        times=np.array([0.0, 1.0, 1.0]),
        reserves=np.array([2.0, 3.0, -1.0]),
        claim_times=np.array([1.0]),
        claim_sizes=np.array([4.0]),
        ruin_time=1.0,
        horizon=2.0,
        initial_capital=2.0,
        premium_rate=1.0,
    )


def _safe_path() -> SimulationPath:
    return SimulationPath(
        times=np.array([0.0, 2.0]),
        reserves=np.array([2.0, 4.0]),
        claim_times=np.empty(0),
        claim_sizes=np.empty(0),
        ruin_time=None,
        horizon=2.0,
        initial_capital=2.0,
        premium_rate=1.0,
    )


def test_simulation_path_exposes_surplus_deficit_and_ruin_claim():
    path = _ruined_path()

    assert path.surplus_before_ruin == pytest.approx(3.0)
    assert path.deficit_at_ruin == pytest.approx(1.0)
    assert path.claim_causing_ruin == pytest.approx(4.0)
    assert _safe_path().surplus_before_ruin is None
    assert _safe_path().deficit_at_ruin is None
    assert _safe_path().claim_causing_ruin is None


def test_gerber_shiu_from_paths_matches_discounted_penalty_definition():
    result = gerber_shiu_from_paths(
        [_ruined_path(), _safe_path()],
        penalty=lambda surplus, deficit: surplus + 2.0 * deficit,
        discount_rate=math.log(2.0),
    )

    assert isinstance(result, GerberShiuResult)
    assert result.n_simulations == 2
    assert result.ruin_probability == pytest.approx(0.5)
    np.testing.assert_allclose(result.ruin_times, [1.0, np.inf])
    np.testing.assert_allclose(result.surplus_before_ruin[:1], [3.0])
    np.testing.assert_allclose(result.deficits_at_ruin[:1], [1.0])
    np.testing.assert_allclose(result.penalty_values, [5.0, 0.0])
    np.testing.assert_allclose(result.discounted_penalties, [2.5, 0.0])
    assert result.estimate == pytest.approx(1.25)
    assert result.mean_surplus_before_ruin == pytest.approx(3.0)
    assert result.mean_deficit_at_ruin == pytest.approx(1.0)


def test_gerber_shiu_default_penalty_is_ruin_probability():
    result = gerber_shiu_from_paths([_ruined_path(), _safe_path()])

    assert result.estimate == pytest.approx(0.5)
    assert result.ci_method == "normal"


def test_gerber_shiu_validates_inputs():
    with pytest.raises(ValueError, match="paths"):
        gerber_shiu_from_paths([])
    with pytest.raises(ValueError, match="discount_rate"):
        gerber_shiu_from_paths([_safe_path()], discount_rate=-1.0)
    with pytest.raises(TypeError, match="penalty"):
        gerber_shiu_from_paths([_safe_path()], penalty=1.0)
    with pytest.raises(ValueError, match="penalty"):
        gerber_shiu_from_paths([_ruined_path()], penalty=lambda surplus, deficit: -1.0)


def test_estimate_gerber_shiu_validates_simulation_arguments():
    model = CramerLundbergProcess(
        initial_capital=1.0,
        premium_rate=1.0,
        claim_arrival_rate=1.0,
        claim_distribution=deterministic(1.0),
    )

    with pytest.raises(TypeError, match="n_simulations"):
        estimate_gerber_shiu(model, horizon=1.0, n_simulations=2.5)
    with pytest.raises(ValueError, match="max_events"):
        estimate_gerber_shiu(model, horizon=1.0, n_simulations=1, max_events=0)


def test_gerber_shiu_reproduces_poisson_first_claim_probability():
    rate = 1.5
    horizon = 2.0
    model = CramerLundbergProcess(
        initial_capital=0.0,
        premium_rate=0.0,
        claim_arrival_rate=rate,
        claim_distribution=deterministic(1.0),
    )

    result = estimate_gerber_shiu(
        model,
        horizon=horizon,
        n_simulations=8000,
        seed=2026,
    )

    assert result.estimate == pytest.approx(1.0 - math.exp(-rate * horizon), abs=0.015)
    assert result.mean_surplus_before_ruin == pytest.approx(0.0)
    assert result.mean_deficit_at_ruin == pytest.approx(1.0)


def test_estimate_gerber_shiu_runs_against_risk_process():
    model = CramerLundbergProcess(
        initial_capital=0.0,
        premium_rate=0.0,
        claim_arrival_rate=2.0,
        claim_distribution=deterministic(1.0),
    )

    result, paths = estimate_gerber_shiu(
        model,
        horizon=3.0,
        n_simulations=200,
        penalty=lambda surplus, deficit: deficit,
        seed=123,
        return_paths=True,
    )

    assert isinstance(result, GerberShiuResult)
    assert len(paths) == 200
    assert result.ruin_probability > 0.95
    assert result.estimate == pytest.approx(result.ruin_probability)
    assert np.nanmean(result.deficits_at_ruin) == pytest.approx(1.0)
