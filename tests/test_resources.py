"""Tests for volunteer assignments, equipment allocation, and forecasting."""

import numpy as np
import pandas as pd
import pytest

import config
from constraints import validate
from cost import KPI_DIRECTIONS, comparison_table, compute_kpis
from forecast import forecast_attendance
from models import Problem, Schedule
from resources import allocate_equipment, assign_volunteers, volunteer_workload
from tests.helpers import serial_schedule


def test_volunteer_assignment_validity(default_problem: Problem) -> None:
    """Verify volunteer assignments satisfy double-booking, skill match, and hour limit constraints."""
    sched = serial_schedule(default_problem)
    assignments, info = assign_volunteers(default_problem, sched)

    viol = validate(default_problem, sched, assignments=assignments)
    vdb = viol[viol["rule"] == "VOLUNTEER_DOUBLE_BOOKED"]
    vsm = viol[viol["rule"] == "VOLUNTEER_SKILL_MISMATCH"]
    voh = viol[viol["rule"] == "VOLUNTEER_OVER_HOURS"]

    assert vdb.empty, "Found VOLUNTEER_DOUBLE_BOOKED"
    assert vsm.empty, "Found VOLUNTEER_SKILL_MISMATCH"
    assert voh.empty, "Found VOLUNTEER_OVER_HOURS"


def test_volunteer_workload_load_balancing(default_problem: Problem) -> None:
    """Verify load-balanced assignments have workload std <= naive first-available assignment."""
    sched = serial_schedule(default_problem)
    assignments, _ = assign_volunteers(default_problem, sched)
    balanced_wl = volunteer_workload(default_problem, assignments)
    std_balanced = float(balanced_wl["hours"].std())

    # Naive assignment: assign first available volunteer always
    naive_assignments = []
    vol_skills = {}
    vol_max_h = {}
    for _, row in default_problem.volunteers.iterrows():
        vid = str(row["volunteer_id"])
        vol_skills[vid] = set(s.strip() for s in str(row["skills"]).split(";"))
        vol_max_h[vid] = float(row["max_hours"])

    naive_hours = {vid: 0.0 for vid in default_problem.volunteer_ids}
    naive_events = {vid: [] for vid in default_problem.volunteer_ids}

    for e in range(len(default_problem.event_ids)):
        eid = default_problem.event_ids[e]
        req_s = default_problem.skill_required[e]
        needed = int(default_problem.vol_needed[e])
        dur_h = float(default_problem.dur[e] / 2.0)
        d_e = sched.day[e]
        s_e = sched.start[e]
        end_e = s_e + default_problem.dur[e]

        chosen = 0
        for vid in default_problem.volunteer_ids:
            if req_s not in vol_skills[vid]:
                continue
            if naive_hours[vid] + dur_h > vol_max_h[vid]:
                continue
            overlap = False
            for prev_e in naive_events[vid]:
                if sched.day[prev_e] == d_e:
                    prev_s = sched.start[prev_e]
                    prev_end = prev_s + default_problem.dur[prev_e]
                    if s_e < prev_end and prev_s < end_e:
                        overlap = True
                        break
            if not overlap:
                naive_events[vid].append(e)
                naive_hours[vid] += dur_h
                naive_assignments.append({"event_id": eid, "volunteer_id": vid})
                chosen += 1
                if chosen >= needed:
                    break

    naive_df = pd.DataFrame(naive_assignments)
    naive_wl = volunteer_workload(default_problem, naive_df)
    std_naive = float(naive_wl["hours"].std())

    assert std_balanced <= std_naive + 1e-6


def test_volunteer_scarcity_shortfall(small_problem: Problem) -> None:
    """Verify shortfall is correctly reported when volunteers are scarce."""
    # Modify problem to have only 1 volunteer with 0 hours
    prob = small_problem
    prob.volunteer_ids = [prob.volunteer_ids[0]]
    prob.volunteers = prob.volunteers.iloc[:1].copy()
    prob.volunteers.loc[0, "max_hours"] = 0.0  # Cannot work at all

    sched = serial_schedule(prob)
    assignments, info = assign_volunteers(prob, sched)
    total_needed = int(np.sum(prob.vol_needed))
    assert info["unfilled"] == total_needed


def test_equipment_allocation_no_over_stock(default_problem: Problem) -> None:
    """Verify equipment allocation never allocates beyond stock in any slot even under contention."""
    sched = serial_schedule(default_problem)
    # Move all events to start simultaneously at day 0 slot 0 in different rooms
    for i in range(len(default_problem.event_ids)):
        sched.room_idx[i] = i % len(default_problem.room_ids)
        sched.day[i] = 0
        sched.start[i] = 0

    alloc_df = allocate_equipment(default_problem, sched)
    # Verify no stock exceeded
    ok_allocs = alloc_df[alloc_df["status"] == "OK"]
    for q_idx, eqid in enumerate(default_problem.equip_ids):
        stock = default_problem.equip_stock[q_idx]
        allocated_qty = ok_allocs[ok_allocs["equipment_id"] == eqid]["qty"].sum()
        assert allocated_qty <= stock


def test_forecast_attendance_linear_recovery() -> None:
    """Verify attendance forecasting recovers perfectly linear series, clamps at 0, and handles < 2 points."""
    # Perfectly linear: y = 20 * (year - 2021) + 100 -> slope = 20, 2026 forecast = 200
    past_linear = pd.DataFrame(
        [
            {"category": "Tech", "year": y, "attendance": 100 + 20 * (y - 2021)}
            for y in config.PAST_YEARS
        ]
    )
    fc = forecast_attendance(past_linear, target_year=2026, degree=1)
    row = fc[fc["category"] == "Tech"].iloc[0]
    assert pytest.approx(row["slope"], 1e-6) == 20.0
    assert pytest.approx(row["forecast"], 1e-6) == 200.0
    assert pytest.approx(row["r2"], 1e-6) == 1.0

    # Test negative trend clamped at 0
    past_neg = pd.DataFrame(
        [
            {"category": "Dying", "year": 2021, "attendance": 10},
            {"category": "Dying", "year": 2022, "attendance": 2},
            {"category": "Dying", "year": 2023, "attendance": 0},
        ]
    )
    fc_neg = forecast_attendance(past_neg, target_year=2026, degree=1)
    assert fc_neg.iloc[0]["forecast"] == 0

    # Test single point
    past_single = pd.DataFrame([{"category": "NewCat", "year": 2025, "attendance": 50}])
    fc_single = forecast_attendance(past_single, target_year=2026, degree=1)
    assert fc_single.iloc[0]["forecast"] == 50
    assert fc_single.iloc[0]["slope"] == 0.0


def test_kpi_keys_and_comparison_table(default_problem: Problem) -> None:
    """Verify compute_kpis contains exactly the 14 defined keys and comparison_table computes directions correctly."""
    sched = serial_schedule(default_problem)
    assignments, _ = assign_volunteers(default_problem, sched)

    kpis = compute_kpis(default_problem, sched, assignments=assignments)
    expected_keys = set(KPI_DIRECTIONS.keys())
    assert set(kpis.keys()) == expected_keys

    # Create dummy before/after dicts to test comparison_table
    before = {k: 100.0 for k in expected_keys}
    after = {k: 100.0 for k in expected_keys}
    after["total_cost"] = 80.0  # lower is better -> Yes
    after["room_utilization_pct"] = 120.0  # higher is better -> Yes
    after["student_clashes"] = 150.0  # lower is better -> No
    after["venue_utilization_pct"] = 120.0  # n/a -> -

    comp = comparison_table(before, after)
    assert comp.loc["total_cost", "Better"] == "Yes"
    assert comp.loc["total_cost", "Change"] == -20.0
    assert comp.loc["total_cost", "Change %"] == -20.0
    assert comp.loc["room_utilization_pct", "Better"] == "Yes"
    assert comp.loc["student_clashes", "Better"] == "No"
    assert comp.loc["venue_utilization_pct", "Better"] == "-"
