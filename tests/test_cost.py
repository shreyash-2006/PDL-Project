"""Tests for cost calculations, clash matrices, and performance metrics."""

import time
import numpy as np
import pandas as pd
import pytest

import config
from constraints import HARD_RULES, validate
from cost import (
    clash_matrix,
    cost_breakdown,
    hard_violation_count,
    overlap_matrix,
    total_cost,
)
from models import Problem, Schedule
from tests.helpers import serial_schedule


def test_clash_matrix_manual_double_loop(small_problem: Problem) -> None:
    """Verify clash_matrix matches a manual double-loop computation on small_problem."""
    C = clash_matrix(small_problem)
    n_events = len(small_problem.event_ids)
    R = small_problem.R

    manual_C = np.zeros((n_events, n_events), dtype=int)
    for i in range(n_events):
        for j in range(n_events):
            manual_C[i, j] = int(np.sum(R[:, i] * R[:, j]))

    np.testing.assert_array_equal(C, manual_C)


def test_clash_matrix_diagonal(default_problem: Problem) -> None:
    """Verify clash matrix diagonal equals total student registration counts per event."""
    C = clash_matrix(default_problem)
    n_events = len(default_problem.event_ids)
    for i in range(n_events):
        expected_reg = int(np.sum(default_problem.R[:, i]))
        assert C[i, i] == expected_reg


def test_hand_built_three_event_clashes(small_problem: Problem) -> None:
    """Verify known hand-built clash values: overlapping sharing 2 students, non-overlapping, and back-to-back."""
    # Artificially set registrations for 3 events: E0, E1, E2
    prob = small_problem
    # R shape: (S, E)
    custom_R = np.zeros((10, len(prob.event_ids)), dtype=np.uint8)
    # 2 students registered for both E0 and E1
    custom_R[0, 0] = 1
    custom_R[0, 1] = 1
    custom_R[1, 0] = 1
    custom_R[1, 1] = 1
    prob.R = custom_R
    prob._C = None  # reset cache

    # Case 1: E0 and E1 overlap on day 0
    sched = Schedule.empty(len(prob.event_ids))
    sched.room_idx[0] = 0
    sched.day[0] = 0
    sched.start[0] = 0

    sched.room_idx[1] = 1
    sched.day[1] = 0
    sched.start[1] = 1  # overlaps with E0

    bd1 = cost_breakdown(prob, sched, partial=True)
    assert bd1["raw_clash"] == 2.0

    # Case 2: Non-overlapping (different days)
    sched.day[1] = 1
    bd2 = cost_breakdown(prob, sched, partial=True)
    assert bd2["raw_clash"] == 0.0

    # Case 3: Back-to-back on same day
    sched.day[1] = 0
    sched.start[1] = int(prob.dur[0])  # starts exactly when E0 ends
    bd3 = cost_breakdown(prob, sched, partial=True)
    assert bd3["raw_clash"] == 0.0


def test_hand_computed_terms(small_problem: Problem) -> None:
    """Verify room, waste, gap, and late terms on a hand-constructed schedule."""
    prob = small_problem
    sched = Schedule.empty(len(prob.event_ids))

    # Place event 0 in room 0, day 0, start 2, duration = prob.dur[0]
    dur0 = int(prob.dur[0])
    sched.room_idx[0] = 0
    sched.day[0] = 0
    sched.start[0] = 2

    # Place event 1 in room 0, day 0, start 2 + dur0 + 3 (gap of 3 slots)
    dur1 = int(prob.dur[1])
    sched.room_idx[1] = 0
    sched.day[1] = 0
    sched.start[1] = 2 + dur0 + 3

    # Place event 2 in room 1, day 1, start 19 (late event, >= 18)
    dur2 = int(prob.dur[2])
    sched.room_idx[2] = 1
    sched.day[2] = 1
    sched.start[2] = 19

    bd = cost_breakdown(prob, sched, partial=True)

    expected_room = (
        prob.room_cost[0] * (dur0 / 2.0)
        + prob.room_cost[0] * (dur1 / 2.0)
        + prob.room_cost[1] * (dur2 / 2.0)
    )
    assert pytest.approx(bd["raw_room"], 1e-6) == expected_room

    expected_waste = (
        max(0, prob.room_cap[0] - prob.attendance[0])
        + max(0, prob.room_cap[0] - prob.attendance[1])
        + max(0, prob.room_cap[1] - prob.attendance[2])
    )
    assert bd["raw_waste"] == expected_waste
    assert bd["raw_gap"] == 3.0
    assert bd["raw_late"] == 1.0


def test_hard_count_consistency_with_validate(default_problem: Problem) -> None:
    """Verify hard_violation_count equals the number of HARD_RULES in validate output across multiple schedules."""
    rng = np.random.default_rng(999)
    n_events = len(default_problem.event_ids)
    n_rooms = len(default_problem.room_ids)

    for _ in range(5):
        # Generate random schedule (some unplaced, some overlapping, etc.)
        room_idx = rng.integers(-1, n_rooms, size=n_events)
        day = rng.integers(0, config.N_DAYS, size=n_events)
        start = rng.integers(0, config.SLOTS_PER_DAY, size=n_events)
        sched = Schedule(room_idx=room_idx, day=day, start=start)

        expected_count = hard_violation_count(default_problem, sched, partial=False)
        viol_df = validate(default_problem, sched)
        actual_count = len(viol_df[viol_df["rule"].isin(HARD_RULES)])

        assert expected_count == actual_count, f"Mismatch: {expected_count} vs {actual_count}"


def test_partial_ignores_unplaced(small_problem: Problem) -> None:
    """Verify partial=True ignores unplaced events and does not penalize unplaced violations."""
    sched = Schedule.empty(len(small_problem.event_ids))
    # All events unplaced
    bd_partial = cost_breakdown(small_problem, sched, partial=True)
    assert bd_partial["raw_hard"] == 0.0

    bd_full = cost_breakdown(small_problem, sched, partial=False)
    assert bd_full["raw_hard"] == float(len(small_problem.event_ids))


def test_cost_evaluation_speed(default_problem: Problem) -> None:
    """Verify cost evaluation meets speed target: 2,000 calls in under 2 seconds."""
    sched = serial_schedule(default_problem)
    # Warm-up clash cache
    _ = total_cost(default_problem, sched)

    n_calls = 2000
    t0 = time.perf_counter()
    for _ in range(n_calls):
        _ = total_cost(default_problem, sched)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"2,000 cost calls took {elapsed:.3f} s (expected < 2.0 s)"
