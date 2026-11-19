"""Tests for baseline, greedy, simulated annealing, and genetic algorithm schedulers."""

from pathlib import Path
import matplotlib.figure
import numpy as np
import pytest

from constraints import validate
from cost import hard_violation_count, total_cost
from models import Problem, Schedule
from scheduler import (
    plot_convergence,
    schedule_annealing,
    schedule_baseline,
    schedule_genetic,
    schedule_greedy,
)


def test_baseline_places_all_events_basic_rules(default_problem: Problem) -> None:
    """Verify baseline schedule places all events without basic room conflicts."""
    sched = schedule_baseline(default_problem)
    assert not np.any(sched.room_idx == -1)

    viol = validate(default_problem, sched)
    basic_rules = {"ROOM_DOUBLE_BOOKED", "ROOM_TOO_SMALL", "WRONG_ROOM_TYPE", "OUT_OF_DAY"}
    viol_basic = viol[viol["rule"].isin(basic_rules)]
    assert viol_basic.empty, f"Baseline violated basic rules: {viol_basic['rule'].tolist()}"


def test_greedy_zero_hard_violations(default_problem: Problem) -> None:
    """Verify greedy schedule achieves zero hard violations on default_problem."""
    sched = schedule_greedy(default_problem)
    assert hard_violation_count(default_problem, sched, partial=False) == 0


def test_cost_ordering(default_problem: Problem) -> None:
    """Verify on default_problem seed 42: total_cost(SA) <= total_cost(greedy) <= total_cost(baseline)."""
    base_sched = schedule_baseline(default_problem)
    cost_base = total_cost(default_problem, base_sched)

    greedy_sched = schedule_greedy(default_problem)
    cost_greedy = total_cost(default_problem, greedy_sched)

    sa_sched, _ = schedule_annealing(default_problem, init=greedy_sched, n_iter=2000, seed=42)
    cost_sa = total_cost(default_problem, sa_sched)

    assert cost_greedy <= cost_base, f"Greedy ({cost_greedy}) > Baseline ({cost_base})"
    assert cost_sa <= cost_greedy, f"SA ({cost_sa}) > Greedy ({cost_greedy})"


def test_sa_history_properties(default_problem: Problem) -> None:
    """Verify SA history is non-increasing, has length n_iter + 1, and final value equals total_cost(best)."""
    n_iter = 1000
    sa_sched, hist = schedule_annealing(default_problem, n_iter=n_iter, seed=42)

    assert len(hist) == n_iter + 1
    # Check non-increasing
    for i in range(len(hist) - 1):
        assert hist[i] >= hist[i + 1] - 1e-9

    assert pytest.approx(hist[-1], 1e-6) == total_cost(default_problem, sa_sched)


def test_sa_determinism(default_problem: Problem) -> None:
    """Verify SA is deterministic given the same seed."""
    s1, h1 = schedule_annealing(default_problem, n_iter=500, seed=123)
    s2, h2 = schedule_annealing(default_problem, n_iter=500, seed=123)

    np.testing.assert_array_equal(s1.room_idx, s2.room_idx)
    np.testing.assert_array_equal(s1.day, s2.day)
    np.testing.assert_array_equal(s1.start, s2.start)
    assert h1 == h2


def test_schedulers_respect_unavailable_rooms(default_problem: Problem) -> None:
    """Verify no scheduler places events in unavailable rooms."""
    r01_idx = default_problem.room_ids.index("R01")
    mod_prob = default_problem.with_unavailable_rooms(["R01"])

    base_s = schedule_baseline(mod_prob)
    assert not np.any(base_s.room_idx == r01_idx)

    greedy_s = schedule_greedy(mod_prob)
    assert not np.any(greedy_s.room_idx == r01_idx)

    sa_s, _ = schedule_annealing(mod_prob, init=greedy_s, n_iter=500, seed=42)
    assert not np.any(sa_s.room_idx == r01_idx)


def test_sa_does_not_mutate_init(default_problem: Problem) -> None:
    """Verify SA does not mutate the initial Schedule passed into it."""
    greedy_s = schedule_greedy(default_problem)
    orig_room = greedy_s.room_idx.copy()
    orig_day = greedy_s.day.copy()
    orig_start = greedy_s.start.copy()

    _, _ = schedule_annealing(default_problem, init=greedy_s, n_iter=500, seed=42)

    np.testing.assert_array_equal(greedy_s.room_idx, orig_room)
    np.testing.assert_array_equal(greedy_s.day, orig_day)
    np.testing.assert_array_equal(greedy_s.start, orig_start)


def test_genetic_algorithm(default_problem: Problem) -> None:
    """Verify GA returns a fully placed schedule with zero hard violations and non-increasing history."""
    pop_size = 20
    generations = 30
    ga_sched, hist = schedule_genetic(
        default_problem, pop_size=pop_size, generations=generations, seed=42
    )

    assert not np.any(ga_sched.room_idx == -1)
    assert hard_violation_count(default_problem, ga_sched, partial=False) == 0
    assert len(hist) == generations + 1
    for i in range(len(hist) - 1):
        assert hist[i] >= hist[i + 1] - 1e-9


def test_plot_convergence(tmp_path: Path) -> None:
    """Verify plot_convergence creates a matplotlib Figure and saves to disk."""
    histories = {
        "Annealing": [100.0, 90.0, 80.0, 70.0],
        "Genetic": [120.0, 110.0, 95.0, 85.0],
    }
    img_path = tmp_path / "conv.png"
    fig = plot_convergence(histories, path=str(img_path))

    assert isinstance(fig, matplotlib.figure.Figure)
    assert img_path.exists()
    assert img_path.stat().st_size > 0
