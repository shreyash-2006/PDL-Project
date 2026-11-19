"""Resource allocation and workload balancing for volunteers and equipment."""

import numpy as np
import pandas as pd

import config
from models import Problem, Schedule


def assign_volunteers(
    problem: Problem,
    schedule: Schedule,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Greedily assign volunteers to scheduled events with load balancing and skill matching."""
    n_events = len(problem.event_ids)
    event_idx_map = {eid: idx for idx, eid in enumerate(problem.event_ids)}

    vol_skills = {}
    vol_max_hours = {}
    for _, row in problem.volunteers.iterrows():
        vid = str(row["volunteer_id"])
        skills_str = str(row["skills"])
        vol_skills[vid] = set(s.strip() for s in skills_str.split(";") if s.strip())
        vol_max_hours[vid] = float(row["max_hours"])

    vol_assigned_events: dict[str, list[int]] = {vid: [] for vid in problem.volunteer_ids}
    vol_hours: dict[str, float] = {vid: 0.0 for vid in problem.volunteer_ids}

    # Order placed events by (day, start, event_id)
    placed_events = [e for e in range(n_events) if schedule.room_idx[e] >= 0]
    placed_events.sort(key=lambda e: (schedule.day[e], schedule.start[e], problem.event_ids[e]))

    assignments: list[dict[str, str]] = []
    unfilled = 0

    for e in placed_events:
        eid = problem.event_ids[e]
        req_skill = problem.skill_required[e]
        needed = int(problem.vol_needed[e])
        event_hours = float(problem.dur[e] / 2.0)
        d_e = schedule.day[e]
        s_e = schedule.start[e]
        end_e = s_e + problem.dur[e]

        candidates = []
        for vid in problem.volunteer_ids:
            # 1. Skill check
            if req_skill not in vol_skills.get(vid, set()):
                continue

            # 2. Max hours check
            if vol_hours[vid] + event_hours > vol_max_hours.get(vid, 0.0) + 1e-9:
                continue

            # 3. Overlap check with already assigned events
            overlap = False
            for prev_e in vol_assigned_events[vid]:
                if schedule.day[prev_e] == d_e:
                    prev_s = schedule.start[prev_e]
                    prev_end = prev_s + problem.dur[prev_e]
                    if s_e < prev_end and prev_s < end_e:
                        overlap = True
                        break
            if not overlap:
                candidates.append(vid)

        # Sort candidates by (current_hours, volunteer_id)
        candidates.sort(key=lambda v: (vol_hours[v], v))

        chosen = candidates[:needed]
        for vid in chosen:
            vol_assigned_events[vid].append(e)
            vol_hours[vid] += event_hours
            assignments.append({"event_id": eid, "volunteer_id": vid})

        if len(chosen) < needed:
            unfilled += needed - len(chosen)

    df_assign = pd.DataFrame(assignments, columns=["event_id", "volunteer_id"])
    return df_assign, {"unfilled": unfilled}


def allocate_equipment(problem: Problem, schedule: Schedule) -> pd.DataFrame:
    """Allocate equipment stock across events by priority and temporal availability."""
    n_events = len(problem.event_ids)
    n_equip = len(problem.equip_ids)

    # Remaining stock table: (Q, N_DAYS, SLOTS_PER_DAY)
    remaining_stock = np.zeros((n_equip, config.N_DAYS, config.SLOTS_PER_DAY), dtype=int)
    for q in range(n_equip):
        remaining_stock[q, :, :] = problem.equip_stock[q]

    # Process events by (priority desc, day, start, event_id)
    placed_events = [e for e in range(n_events) if schedule.room_idx[e] >= 0]
    placed_events.sort(
        key=lambda e: (-problem.priority[e], schedule.day[e], schedule.start[e], problem.event_ids[e])
    )

    rows: list[dict] = []
    for e in placed_events:
        eid = problem.event_ids[e]
        d = int(schedule.day[e])
        s = int(schedule.start[e])
        dur = int(problem.dur[e])

        if not (0 <= d < config.N_DAYS):
            continue

        s_start = max(0, s)
        s_end = min(config.SLOTS_PER_DAY, s + dur)

        for q in range(n_equip):
            qty = int(problem.equip_need[e, q])
            if qty > 0:
                eqid = problem.equip_ids[q]
                # Check availability across all slots
                available_in_all = True
                for slot in range(s_start, s_end):
                    if remaining_stock[q, d, slot] < qty:
                        available_in_all = False
                        break

                if available_in_all:
                    status = "OK"
                    remaining_stock[q, d, s_start:s_end] -= qty
                else:
                    status = "SHORTAGE"

                rows.append(
                    {
                        "event_id": eid,
                        "equipment_id": eqid,
                        "qty": qty,
                        "status": status,
                    }
                )

    return pd.DataFrame(rows, columns=["event_id", "equipment_id", "qty", "status"])


def volunteer_workload(problem: Problem, assignments: pd.DataFrame) -> pd.DataFrame:
    """Calculate the event count and total assigned hours for every volunteer."""
    event_idx_map = {eid: idx for idx, eid in enumerate(problem.event_ids)}
    vol_counts = {vid: 0 for vid in problem.volunteer_ids}
    vol_hours = {vid: 0.0 for vid in problem.volunteer_ids}

    if assignments is not None and not assignments.empty:
        for _, row in assignments.iterrows():
            vid = str(row["volunteer_id"])
            eid = str(row["event_id"])
            if vid in vol_counts and eid in event_idx_map:
                e_idx = event_idx_map[eid]
                vol_counts[vid] += 1
                vol_hours[vid] += float(problem.dur[e_idx] / 2.0)

    rows = [
        {"volunteer_id": vid, "n_events": vol_counts[vid], "hours": vol_hours[vid]}
        for vid in problem.volunteer_ids
    ]
    return pd.DataFrame(rows, columns=["volunteer_id", "n_events", "hours"])
