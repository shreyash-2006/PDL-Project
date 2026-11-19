"""Tests for constraint validation, cleaning, and dataset generator."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import config
from constraints import HARD_RULES, RULES, summarize, validate
from data_io import DataError, clean_tables, load_problem
from generator import generate_all
from models import Problem, Schedule
from tests.helpers import serial_schedule


def test_valid_serial_schedule(small_problem: Problem) -> None:
    """Ensure a valid serial schedule has zero constraint violations."""
    sched = serial_schedule(small_problem)
    viol = validate(small_problem, sched)
    assert viol.empty


def test_room_double_booked(small_problem: Problem) -> None:
    """Ensure two overlapping events in the same room trigger ROOM_DOUBLE_BOOKED."""
    sched = serial_schedule(small_problem)
    sched.room_idx[1] = sched.room_idx[0]
    sched.day[1] = sched.day[0]
    sched.start[1] = sched.start[0]

    viol = validate(small_problem, sched)
    db = viol[viol["rule"] == "ROOM_DOUBLE_BOOKED"]
    assert len(db) == 1
    row = db.iloc[0]
    assert {row["event_id"], row["other_event_id"]} == {
        small_problem.event_ids[0],
        small_problem.event_ids[1],
    }


def test_room_too_small(default_problem: Problem) -> None:
    """Ensure placing an event in an undersized room triggers ROOM_TOO_SMALL."""
    sched = serial_schedule(default_problem)
    # Find an event with attendance larger than the smallest room
    min_cap_idx = int(np.argmin(default_problem.room_cap))
    min_cap = default_problem.room_cap[min_cap_idx]

    candidate_events = [
        i for i in range(len(default_problem.event_ids)) if default_problem.attendance[i] > min_cap
    ]
    assert candidate_events, "No event with attendance > min_cap"
    e_idx = candidate_events[0]

    sched.room_idx[e_idx] = min_cap_idx
    viol = validate(default_problem, sched)
    ts = viol[
        (viol["rule"] == "ROOM_TOO_SMALL")
        & (viol["event_id"] == default_problem.event_ids[e_idx])
    ]
    assert not ts.empty


def test_wrong_room_type(small_problem: Problem) -> None:
    """Ensure placing an event in an incompatible room type triggers WRONG_ROOM_TYPE."""
    sched = serial_schedule(small_problem)
    req_type = small_problem.events.iloc[0]["required_room_type"]
    diff_rooms = [
        r for r, rt in enumerate(small_problem.rooms["room_type"]) if rt != req_type
    ]
    assert diff_rooms, "No room with different room type found"
    sched.room_idx[0] = diff_rooms[0]

    viol = validate(small_problem, sched)
    wt = viol[
        (viol["rule"] == "WRONG_ROOM_TYPE")
        & (viol["event_id"] == small_problem.event_ids[0])
    ]
    assert not wt.empty


def test_equipment_over_stock(tmp_path: Path) -> None:
    """Ensure overlapping events exceeding equipment stock trigger EQUIPMENT_OVER_STOCK."""
    tables = generate_all(
        seed=12,
        out_dir=str(tmp_path),
        n_events=4,
        n_rooms=4,
        n_students=20,
        n_volunteers=8,
    )
    # Ensure equipment demand exceeds stock for an item
    ee = tables["event_equipment"].copy()
    # Let E01 and E02 both demand stock of EQ01
    stk = int(tables["equipment"].loc[tables["equipment"]["equipment_id"] == "EQ01", "stock"].iloc[0])
    new_ee = pd.DataFrame(
        [
            {"event_id": "E01", "equipment_id": "EQ01", "qty": stk},
            {"event_id": "E02", "equipment_id": "EQ01", "qty": 1},
        ]
    )
    tables["event_equipment"] = new_ee
    for name, df in tables.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)

    prob = load_problem(str(tmp_path))
    # Place E01 and E02 at same time in different rooms
    sched = Schedule.empty(len(prob.event_ids))
    sched.room_idx[0] = 0
    sched.day[0] = 0
    sched.start[0] = 2

    sched.room_idx[1] = 1
    sched.day[1] = 0
    sched.start[1] = 2

    # Put remaining events elsewhere
    for i in range(2, len(prob.event_ids)):
        sched.room_idx[i] = 2
        sched.day[i] = 1
        sched.start[i] = i * 4

    viol = validate(prob, sched)
    eq_viol = viol[viol["rule"] == "EQUIPMENT_OVER_STOCK"]
    assert not eq_viol.empty
    assert eq_viol.iloc[0]["resource_id"] == "EQ01"


def test_out_of_day(small_problem: Problem) -> None:
    """Ensure scheduling an event beyond daily operating slots triggers OUT_OF_DAY."""
    sched = serial_schedule(small_problem)
    dur = small_problem.dur[0]
    sched.start[0] = config.SLOTS_PER_DAY - dur + 1

    viol = validate(small_problem, sched)
    ood = viol[viol["rule"] == "OUT_OF_DAY"]
    assert len(ood) == 1
    assert ood.iloc[0]["event_id"] == small_problem.event_ids[0]


def test_unplaced(small_problem: Problem) -> None:
    """Ensure an unplaced event triggers UNPLACED."""
    sched = serial_schedule(small_problem)
    sched.room_idx[0] = -1

    viol = validate(small_problem, sched)
    unpl = viol[viol["rule"] == "UNPLACED"]
    assert len(unpl) == 1
    assert unpl.iloc[0]["event_id"] == small_problem.event_ids[0]


def test_room_unavailable(small_problem: Problem) -> None:
    """Ensure placing an event in an unavailable room triggers ROOM_UNAVAILABLE."""
    sched = serial_schedule(small_problem)
    r_idx = sched.room_idx[0]
    rid = small_problem.room_ids[r_idx]
    prob = small_problem.with_unavailable_rooms([rid])

    viol = validate(prob, sched)
    ru = viol[viol["rule"] == "ROOM_UNAVAILABLE"]
    assert not ru.empty
    assert ru.iloc[0]["room_id"] == rid


def test_volunteer_double_booked(small_problem: Problem) -> None:
    """Ensure assigning a volunteer to overlapping events triggers VOLUNTEER_DOUBLE_BOOKED."""
    # Create two overlapping events in different rooms
    sched = Schedule.empty(len(small_problem.event_ids))
    sched.room_idx[0] = 0
    sched.day[0] = 0
    sched.start[0] = 0

    sched.room_idx[1] = 1
    sched.day[1] = 0
    sched.start[1] = 0

    vid = small_problem.volunteer_ids[0]
    assignments = pd.DataFrame(
        [
            {"event_id": small_problem.event_ids[0], "volunteer_id": vid},
            {"event_id": small_problem.event_ids[1], "volunteer_id": vid},
        ]
    )

    viol = validate(small_problem, sched, assignments=assignments)
    vdb = viol[viol["rule"] == "VOLUNTEER_DOUBLE_BOOKED"]
    assert not vdb.empty
    assert vdb.iloc[0]["resource_id"] == vid


def test_volunteer_skill_mismatch(small_problem: Problem) -> None:
    """Ensure assigning a volunteer lacking the required skill triggers VOLUNTEER_SKILL_MISMATCH."""
    sched = serial_schedule(small_problem)
    # Find an event's required skill and a volunteer who lacks it
    e_idx = 0
    req_skill = small_problem.skill_required[e_idx]
    eid = small_problem.event_ids[e_idx]

    vol_skills_map = dict(
        zip(small_problem.volunteers["volunteer_id"], small_problem.volunteers["skills"])
    )
    unskilled_vols = [
        vid
        for vid, s_str in vol_skills_map.items()
        if req_skill not in [s.strip() for s in s_str.split(";")]
    ]
    assert unskilled_vols, "All volunteers have this skill"
    chosen_vid = unskilled_vols[0]

    assignments = pd.DataFrame([{"event_id": eid, "volunteer_id": chosen_vid}])
    viol = validate(small_problem, sched, assignments=assignments)
    vsm = viol[viol["rule"] == "VOLUNTEER_SKILL_MISMATCH"]
    assert not vsm.empty
    assert vsm.iloc[0]["resource_id"] == chosen_vid


def test_volunteer_over_hours(small_problem: Problem) -> None:
    """Ensure exceeding volunteer maximum allowable hours triggers VOLUNTEER_OVER_HOURS."""
    sched = serial_schedule(small_problem)
    vid = small_problem.volunteer_ids[0]
    # Assign vid to all events to exceed max_hours
    assignments = pd.DataFrame(
        [{"event_id": eid, "volunteer_id": vid} for eid in small_problem.event_ids]
    )
    viol = validate(small_problem, sched, assignments=assignments)
    voh = viol[viol["rule"] == "VOLUNTEER_OVER_HOURS"]
    assert not voh.empty
    assert voh.iloc[0]["resource_id"] == vid


def test_volunteer_understaffed(small_problem: Problem) -> None:
    """Ensure understaffed events trigger VOLUNTEER_UNDERSTAFFED."""
    sched = serial_schedule(small_problem)
    empty_assignments = pd.DataFrame(columns=["event_id", "volunteer_id"])
    viol = validate(small_problem, sched, assignments=empty_assignments)
    # Since events need >= 1 volunteer, understaffed rows must be present
    vus = viol[viol["rule"] == "VOLUNTEER_UNDERSTAFFED"]
    assert len(vus) == len(small_problem.event_ids)


def test_back_to_back_not_flagged(small_problem: Problem) -> None:
    """Ensure back-to-back events in the same room are NOT treated as overlapping."""
    sched = Schedule.empty(len(small_problem.event_ids))
    dur0 = int(small_problem.dur[0])
    sched.room_idx[0] = 0
    sched.day[0] = 0
    sched.start[0] = 0

    sched.room_idx[1] = 0
    sched.day[1] = 0
    sched.start[1] = dur0  # starts exactly when event 0 ends

    viol = validate(small_problem, sched)
    db = viol[viol["rule"] == "ROOM_DOUBLE_BOOKED"]
    assert db.empty


def test_summarize_has_all_keys(small_problem: Problem) -> None:
    """Ensure summarize dictionary contains every defined rule code with zero counts for valid schedule."""
    sched = serial_schedule(small_problem)
    viol = validate(small_problem, sched)
    summary = summarize(viol)
    assert set(summary.keys()) == set(RULES)
    assert all(count == 0 for count in summary.values())


def test_load_problem_cleaning(tmp_path: Path) -> None:
    """Verify that load_problem cleans whitespace, upper-cases IDs, and drops invalid/orphan rows."""
    base_tables = generate_all(
        seed=101,
        out_dir=str(tmp_path),
        n_events=5,
        n_rooms=3,
        n_students=15,
        n_volunteers=5,
    )
    # Add messy duplicates and lowercase strings
    ev = base_tables["events"].copy()
    ev.loc[0, "name"] = "  Messy Technical Event  "
    ev.loc[len(ev)] = ev.iloc[0].copy()  # exact duplicate row

    # Add orphan registration pointing to nonexistent event
    reg = base_tables["registrations"].copy()
    reg.loc[len(reg)] = {"student_id": "s999", "event_id": "E999"}
    # Add row with invalid numeric
    rm = base_tables["rooms"].copy()
    rm.loc[len(rm)] = {
        "room_id": "r99",
        "name": "Bad Room",
        "room_type": "classroom",
        "capacity": "INVALID_CAP",
        "cost_per_hour": 100,
    }

    base_tables["events"] = ev
    base_tables["registrations"] = reg
    base_tables["rooms"] = rm

    for name, df in base_tables.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)

    cleaned_problem = load_problem(str(tmp_path))
    # Orphan event should not be in problem
    assert "E999" not in cleaned_problem.event_ids
    # Bad room should be dropped
    assert "R99" not in cleaned_problem.room_ids
    # Duplicate event should be deduplicated
    assert len(cleaned_problem.event_ids) == 5


def test_generator_determinism_and_feasibility() -> None:
    """Verify generator determinism and feasibility invariants."""
    t1 = generate_all(seed=42, out_dir="")
    t2 = generate_all(seed=42, out_dir="")

    for key in t1:
        pd.testing.assert_frame_equal(t1[key], t2[key])

    # Feasibility invariants
    events = t1["events"]
    rooms = t1["rooms"]
    volunteers = t1["volunteers"]
    equip = t1["equipment"]
    ee = t1["event_equipment"]

    # Every event has at least one compatible room
    for _, ev in events.iterrows():
        comp = rooms[
            (rooms["room_type"] == ev["required_room_type"])
            & (rooms["capacity"] >= ev["expected_attendance"])
        ]
        assert not comp.empty, f"Event {ev['event_id']} has no compatible room"

    # Equipment qty <= stock
    stock_dict = dict(zip(equip["equipment_id"], equip["stock"]))
    for _, row in ee.iterrows():
        assert row["qty"] <= stock_dict[row["equipment_id"]]

    # Each skill has >= 6 volunteers
    skill_counts = {s: 0 for s in config.SKILLS}
    for _, row in volunteers.iterrows():
        for s in str(row["skills"]).split(";"):
            s_clean = s.strip()
            if s_clean in skill_counts:
                skill_counts[s_clean] += 1
    for s, count in skill_counts.items():
        assert count >= 6, f"Skill '{s}' has only {count} volunteers (< 6)"


def test_default_generated_sizes(default_problem: Problem) -> None:
    """Ensure default dataset matches expected dimensions."""
    assert len(default_problem.event_ids) == 25
    assert len(default_problem.room_ids) == 10
    assert len(default_problem.student_ids) == 300
    assert len(default_problem.volunteer_ids) == 40
