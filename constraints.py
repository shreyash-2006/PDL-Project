"""Constraint verification and schedule validation according to domain rules."""

from typing import Optional
import numpy as np
import pandas as pd

import config
from models import Problem, Schedule

HARD_RULES = (
    "UNPLACED",
    "ROOM_DOUBLE_BOOKED",
    "ROOM_TOO_SMALL",
    "WRONG_ROOM_TYPE",
    "EQUIPMENT_OVER_STOCK",
    "OUT_OF_DAY",
    "ROOM_UNAVAILABLE",
)

RULES = HARD_RULES + (
    "VOLUNTEER_DOUBLE_BOOKED",
    "VOLUNTEER_SKILL_MISMATCH",
    "VOLUNTEER_OVER_HOURS",
    "VOLUNTEER_UNDERSTAFFED",
)


def validate(
    problem: Problem,
    schedule: Schedule,
    assignments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Validate a schedule against all hard and resource constraints, returning violation records."""
    violations: list[dict] = []
    n_events = len(problem.event_ids)
    room_types = list(problem.rooms["room_type"])
    req_room_types = list(problem.events["required_room_type"])

    # 1. UNPLACED
    for e_idx in range(n_events):
        if schedule.room_idx[e_idx] == -1:
            eid = problem.event_ids[e_idx]
            violations.append(
                {
                    "rule": "UNPLACED",
                    "event_id": eid,
                    "other_event_id": "",
                    "room_id": "",
                    "resource_id": "",
                    "day": "",
                    "slot": "",
                    "detail": f"Event {eid} is unplaced.",
                }
            )

    # 2. ROOM_DOUBLE_BOOKED
    for i in range(n_events):
        r_i = schedule.room_idx[i]
        if r_i < 0:
            continue
        d_i = schedule.day[i]
        s_i = schedule.start[i]
        dur_i = problem.dur[i]
        end_i = s_i + dur_i

        for j in range(i + 1, n_events):
            r_j = schedule.room_idx[j]
            if r_j != r_i:
                continue
            d_j = schedule.day[j]
            if d_j != d_i:
                continue
            s_j = schedule.start[j]
            dur_j = problem.dur[j]
            end_j = s_j + dur_j

            # Overlap test: s_i < end_j and s_j < end_i
            if s_i < end_j and s_j < end_i:
                eid_i = problem.event_ids[i]
                eid_j = problem.event_ids[j]
                rid = problem.room_ids[r_i]
                overlap_slot = max(s_i, s_j)
                violations.append(
                    {
                        "rule": "ROOM_DOUBLE_BOOKED",
                        "event_id": eid_i,
                        "other_event_id": eid_j,
                        "room_id": rid,
                        "resource_id": "",
                        "day": int(d_i),
                        "slot": int(overlap_slot),
                        "detail": f"Room {rid} double booked by {eid_i} and {eid_j}.",
                    }
                )

    # 3. ROOM_TOO_SMALL
    for e_idx in range(n_events):
        r_idx = schedule.room_idx[e_idx]
        if r_idx >= 0:
            cap = problem.room_cap[r_idx]
            att = problem.attendance[e_idx]
            if cap < att:
                eid = problem.event_ids[e_idx]
                rid = problem.room_ids[r_idx]
                violations.append(
                    {
                        "rule": "ROOM_TOO_SMALL",
                        "event_id": eid,
                        "other_event_id": "",
                        "room_id": rid,
                        "resource_id": "",
                        "day": int(schedule.day[e_idx]),
                        "slot": int(schedule.start[e_idx]),
                        "detail": f"Room {rid} (capacity {cap}) too small for {eid} (attendance {att}).",
                    }
                )

    # 4. WRONG_ROOM_TYPE
    for e_idx in range(n_events):
        r_idx = schedule.room_idx[e_idx]
        if r_idx >= 0:
            actual_type = room_types[r_idx]
            req_type = req_room_types[e_idx]
            if actual_type != req_type:
                eid = problem.event_ids[e_idx]
                rid = problem.room_ids[r_idx]
                violations.append(
                    {
                        "rule": "WRONG_ROOM_TYPE",
                        "event_id": eid,
                        "other_event_id": "",
                        "room_id": rid,
                        "resource_id": "",
                        "day": int(schedule.day[e_idx]),
                        "slot": int(schedule.start[e_idx]),
                        "detail": f"Room {rid} has type '{actual_type}' but {eid} requires '{req_type}'.",
                    }
                )

    # 5. EQUIPMENT_OVER_STOCK: (equipment, day, slot)
    n_equip = len(problem.equip_ids)
    equip_usage = np.zeros((n_equip, config.N_DAYS, config.SLOTS_PER_DAY), dtype=int)
    equip_events: dict[tuple[int, int, int], list[str]] = {}

    for e_idx in range(n_events):
        r_idx = schedule.room_idx[e_idx]
        if r_idx < 0:
            continue
        d = schedule.day[e_idx]
        s = schedule.start[e_idx]
        dur = problem.dur[e_idx]
        if not (0 <= d < config.N_DAYS):
            continue

        eid = problem.event_ids[e_idx]
        # Sum equipment demand across valid slots
        for slot in range(max(0, s), min(config.SLOTS_PER_DAY, s + dur)):
            for q_idx in range(n_equip):
                qty = problem.equip_need[e_idx, q_idx]
                if qty > 0:
                    equip_usage[q_idx, d, slot] += qty
                    key = (q_idx, int(d), int(slot))
                    equip_events.setdefault(key, []).append(eid)

    for q_idx in range(n_equip):
        stk = problem.equip_stock[q_idx]
        eqid = problem.equip_ids[q_idx]
        for d in range(config.N_DAYS):
            for slot in range(config.SLOTS_PER_DAY):
                usage = equip_usage[q_idx, d, slot]
                if usage > stk:
                    active_eids = equip_events.get((q_idx, d, slot), [])
                    joined_eids = ",".join(sorted(active_eids))
                    violations.append(
                        {
                            "rule": "EQUIPMENT_OVER_STOCK",
                            "event_id": joined_eids,
                            "other_event_id": "",
                            "room_id": "",
                            "resource_id": eqid,
                            "day": int(d),
                            "slot": int(slot),
                            "detail": f"Equipment {eqid} usage ({usage}) exceeds stock ({stk}) on day {d} slot {slot}.",
                        }
                    )

    # 6. OUT_OF_DAY
    for e_idx in range(n_events):
        r_idx = schedule.room_idx[e_idx]
        if r_idx >= 0:
            d = schedule.day[e_idx]
            s = schedule.start[e_idx]
            dur = problem.dur[e_idx]
            if d < 0 or d >= config.N_DAYS or s < 0 or s + dur > config.SLOTS_PER_DAY:
                eid = problem.event_ids[e_idx]
                violations.append(
                    {
                        "rule": "OUT_OF_DAY",
                        "event_id": eid,
                        "other_event_id": "",
                        "room_id": "",
                        "resource_id": "",
                        "day": int(d) if 0 <= d < config.N_DAYS else d,
                        "slot": int(s),
                        "detail": f"Event {eid} scheduled out of bounds (day={d}, start={s}, dur={dur}).",
                    }
                )

    # 7. ROOM_UNAVAILABLE
    for e_idx in range(n_events):
        r_idx = schedule.room_idx[e_idx]
        if r_idx >= 0:
            if not problem.room_available[r_idx]:
                eid = problem.event_ids[e_idx]
                rid = problem.room_ids[r_idx]
                violations.append(
                    {
                        "rule": "ROOM_UNAVAILABLE",
                        "event_id": eid,
                        "other_event_id": "",
                        "room_id": rid,
                        "resource_id": "",
                        "day": int(schedule.day[e_idx]),
                        "slot": int(schedule.start[e_idx]),
                        "detail": f"Event {eid} assigned to unavailable room {rid}.",
                    }
                )

    # Volunteer rules if assignments provided
    if assignments is not None:
        # Pre-process assignments
        event_idx_map = {eid: idx for idx, eid in enumerate(problem.event_ids)}
        vol_skills_map = dict(zip(problem.volunteers["volunteer_id"], problem.volunteers["skills"]))
        vol_max_hours = dict(zip(problem.volunteers["volunteer_id"], problem.volunteers["max_hours"]))

        vol_assigned_events: dict[str, list[str]] = {}
        event_assigned_vols: dict[str, list[str]] = {eid: [] for eid in problem.event_ids}

        for _, row in assignments.iterrows():
            vid = str(row["volunteer_id"])
            eid = str(row["event_id"])
            vol_assigned_events.setdefault(vid, []).append(eid)
            if eid in event_assigned_vols:
                event_assigned_vols[eid].append(vid)

        # 8. VOLUNTEER_DOUBLE_BOOKED: (volunteer, unordered event pair)
        for vid, e_list in vol_assigned_events.items():
            placed_e = [e for e in e_list if e in event_idx_map and schedule.room_idx[event_idx_map[e]] >= 0]
            for i in range(len(placed_e)):
                eid_i = placed_e[i]
                idx_i = event_idx_map[eid_i]
                d_i = schedule.day[idx_i]
                s_i = schedule.start[idx_i]
                end_i = s_i + problem.dur[idx_i]

                for j in range(i + 1, len(placed_e)):
                    eid_j = placed_e[j]
                    idx_j = event_idx_map[eid_j]
                    d_j = schedule.day[idx_j]
                    if d_j != d_i:
                        continue
                    s_j = schedule.start[idx_j]
                    end_j = s_j + problem.dur[idx_j]

                    if s_i < end_j and s_j < end_i:
                        violations.append(
                            {
                                "rule": "VOLUNTEER_DOUBLE_BOOKED",
                                "event_id": eid_i,
                                "other_event_id": eid_j,
                                "room_id": "",
                                "resource_id": vid,
                                "day": int(d_i),
                                "slot": int(max(s_i, s_j)),
                                "detail": f"Volunteer {vid} double booked on {eid_i} and {eid_j}.",
                            }
                        )

        # 9. VOLUNTEER_SKILL_MISMATCH: (volunteer, event)
        for _, row in assignments.iterrows():
            vid = str(row["volunteer_id"])
            eid = str(row["event_id"])
            if eid in event_idx_map:
                idx = event_idx_map[eid]
                req_skill = problem.skill_required[idx]
                skills_str = vol_skills_map.get(vid, "")
                skills_list = [s.strip() for s in skills_str.split(";") if s.strip()]
                if req_skill not in skills_list:
                    violations.append(
                        {
                            "rule": "VOLUNTEER_SKILL_MISMATCH",
                            "event_id": eid,
                            "other_event_id": "",
                            "room_id": "",
                            "resource_id": vid,
                            "day": "",
                            "slot": "",
                            "detail": f"Volunteer {vid} lacks required skill '{req_skill}' for {eid}.",
                        }
                    )

        # 10. VOLUNTEER_OVER_HOURS: volunteer
        for vid in problem.volunteer_ids:
            assigned_eids = vol_assigned_events.get(vid, [])
            total_slots = sum(
                problem.dur[event_idx_map[e]] for e in assigned_eids if e in event_idx_map
            )
            hours = total_slots * (config.SLOT_MINUTES / 60.0)
            max_h = vol_max_hours.get(vid, 0.0)
            if hours > max_h:
                violations.append(
                    {
                        "rule": "VOLUNTEER_OVER_HOURS",
                        "event_id": "",
                        "other_event_id": "",
                        "room_id": "",
                        "resource_id": vid,
                        "day": "",
                        "slot": "",
                        "detail": f"Volunteer {vid} assigned {hours:.1f} hours exceeding max {max_h:.1f}.",
                    }
                )

        # 11. VOLUNTEER_UNDERSTAFFED: event
        for e_idx in range(n_events):
            eid = problem.event_ids[e_idx]
            needed = int(problem.vol_needed[e_idx])
            assigned_cnt = len(event_assigned_vols.get(eid, []))
            if assigned_cnt < needed:
                violations.append(
                    {
                        "rule": "VOLUNTEER_UNDERSTAFFED",
                        "event_id": eid,
                        "other_event_id": "",
                        "room_id": "",
                        "resource_id": "",
                        "day": "",
                        "slot": "",
                        "detail": f"Event {eid} has {assigned_cnt} volunteers, needs {needed}.",
                    }
                )

    columns = [
        "rule",
        "event_id",
        "other_event_id",
        "room_id",
        "resource_id",
        "day",
        "slot",
        "detail",
    ]
    if not violations:
        return pd.DataFrame(columns=columns)

    df_viol = pd.DataFrame(violations, columns=columns)

    # Sort order of the output: by rule (order in RULES), then event_id
    rule_order_map = {r: i for i, r in enumerate(RULES)}
    df_viol["_rule_order"] = df_viol["rule"].map(rule_order_map)
    df_viol = df_viol.sort_values(by=["_rule_order", "event_id"]).drop(columns=["_rule_order"])
    return df_viol.reset_index(drop=True)


def summarize(violations: pd.DataFrame) -> dict[str, int]:
    """Summarize violation counts grouped by every standard rule code."""
    summary = {r: 0 for r in RULES}
    if violations is not None and not violations.empty and "rule" in violations.columns:
        counts = violations["rule"].value_counts().to_dict()
        for r, cnt in counts.items():
            if r in summary:
                summary[r] = int(cnt)
    return summary
