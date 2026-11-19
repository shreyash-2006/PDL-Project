"""Clash matrix, objective cost evaluation, and KPI analytics."""

from typing import Optional
import numpy as np
import pandas as pd

import config
from models import Problem, Schedule

KPI_DIRECTIONS = {
    "total_cost": "lower",
    "student_clashes": "lower",
    "clashing_pairs": "lower",
    "hard_violations": "lower",
    "room_cost_total": "lower",
    "wasted_seats": "lower",
    "idle_gap_hours": "lower",
    "late_events": "lower",
    "room_utilization_pct": "higher",
    "seat_utilization_pct": "higher",
    "venue_utilization_pct": "n/a",
    "rooms_used": "n/a",
    "volunteer_hours_std": "lower",
    "unfilled_volunteer_slots": "lower",
}


def clash_matrix(problem: Problem) -> np.ndarray:
    """Compute and cache the symmetric student registration clash matrix."""
    if problem._C is not None:
        return problem._C
    R_int = problem.R.astype(np.int32)
    C = R_int.T @ R_int
    problem._C = C
    return C


def overlap_matrix(problem: Problem, schedule: Schedule) -> np.ndarray:
    """Compute the symmetric boolean pairwise event overlap matrix."""
    n_events = len(problem.event_ids)
    placed = schedule.room_idx >= 0
    placed_2d = placed[:, None] & placed[None, :]

    same_day = schedule.day[:, None] == schedule.day[None, :]
    ends = schedule.start + problem.dur
    time_overlap = (schedule.start[:, None] < ends[None, :]) & (schedule.start[None, :] < ends[:, None])

    overlap = placed_2d & same_day & time_overlap
    np.fill_diagonal(overlap, False)
    return overlap


def hard_violation_count(problem: Problem, schedule: Schedule, partial: bool = False) -> int:
    """Count the total number of hard constraint violations."""
    n_events = len(problem.event_ids)
    placed = schedule.room_idx >= 0
    hard_count = 0

    # 1. UNPLACED
    if not partial:
        hard_count += int(np.sum(~placed))

    if not np.any(placed):
        return hard_count

    placed_indices = np.where(placed)[0]
    placed_rooms = schedule.room_idx[placed_indices]
    placed_days = schedule.day[placed_indices]
    placed_starts = schedule.start[placed_indices]
    placed_durs = problem.dur[placed_indices]
    placed_ends = placed_starts + placed_durs

    # 2. ROOM_DOUBLE_BOOKED
    same_room = placed_rooms[:, None] == placed_rooms[None, :]
    same_day = placed_days[:, None] == placed_days[None, :]
    time_overlap = (placed_starts[:, None] < placed_ends[None, :]) & (
        placed_starts[None, :] < placed_ends[:, None]
    )
    room_overlap = same_room & same_day & time_overlap
    np.fill_diagonal(room_overlap, False)
    hard_count += int(np.sum(room_overlap) // 2)

    # 3. ROOM_TOO_SMALL
    caps = problem.room_cap[placed_rooms]
    atts = problem.attendance[placed_indices]
    hard_count += int(np.sum(caps < atts))

    # 4. WRONG_ROOM_TYPE
    # Check compatibility for type match
    type_matches = problem.rooms["room_type"].to_numpy()[placed_rooms] == problem.events["required_room_type"].to_numpy()[placed_indices]
    hard_count += int(np.sum(~type_matches))

    # 5. EQUIPMENT_OVER_STOCK: (equipment, day, slot)
    n_equip = len(problem.equip_ids)
    equip_usage = np.zeros((n_equip, config.N_DAYS, config.SLOTS_PER_DAY), dtype=int)
    for idx, e_idx in enumerate(placed_indices):
        d = placed_days[idx]
        s = placed_starts[idx]
        dur = placed_durs[idx]
        if 0 <= d < config.N_DAYS:
            s_start = max(0, s)
            s_end = min(config.SLOTS_PER_DAY, s + dur)
            if s_end > s_start:
                needs = problem.equip_need[e_idx]  # (Q,)
                for q in range(n_equip):
                    if needs[q] > 0:
                        equip_usage[q, d, s_start:s_end] += needs[q]

    over_stock = equip_usage > problem.equip_stock[:, None, None]
    hard_count += int(np.sum(over_stock))

    # 6. OUT_OF_DAY
    out_of_day = (
        (placed_days < 0)
        | (placed_days >= config.N_DAYS)
        | (placed_starts < 0)
        | (placed_ends > config.SLOTS_PER_DAY)
    )
    hard_count += int(np.sum(out_of_day))

    # 7. ROOM_UNAVAILABLE
    avail = problem.room_available[placed_rooms]
    hard_count += int(np.sum(~avail))

    return hard_count


def cost_breakdown(
    problem: Problem,
    schedule: Schedule,
    partial: bool = False,
) -> dict[str, float]:
    """Compute detailed unweighted and weighted cost term components."""
    placed = schedule.room_idx >= 0
    placed_indices = np.where(placed)[0]

    # Clash
    C = clash_matrix(problem)
    overlap = overlap_matrix(problem, schedule)
    raw_clash = float(np.sum(C * overlap) // 2)

    # Placed events data
    if len(placed_indices) > 0:
        placed_rooms = schedule.room_idx[placed_indices]
        placed_starts = schedule.start[placed_indices]
        placed_durs = problem.dur[placed_indices]
        placed_days = schedule.day[placed_indices]

        # Room cost = cost_per_hour * dur / 2
        raw_room = float(np.sum(problem.room_cost[placed_rooms] * (placed_durs / 2.0)))

        # Waste = max(0, capacity - attendance)
        raw_waste = float(
            np.sum(
                np.maximum(
                    0,
                    problem.room_cap[placed_rooms] - problem.attendance[placed_indices],
                )
            )
        )

        # Gap = idle gap slots between events in same room-day
        raw_gap = 0.0
        # Group by (room, day)
        room_day_keys = set(zip(placed_rooms, placed_days))
        for r, d in room_day_keys:
            mask = (placed_rooms == r) & (placed_days == d)
            if np.sum(mask) >= 2:
                s_arr = placed_starts[mask]
                d_arr = placed_durs[mask]
                order = np.argsort(s_arr)
                s_sorted = s_arr[order]
                e_sorted = s_sorted + d_arr[order]
                gaps = np.maximum(0, s_sorted[1:] - e_sorted[:-1])
                raw_gap += float(np.sum(gaps))

        # Late = start >= LATE_START_SLOT
        raw_late = float(np.sum(placed_starts >= config.LATE_START_SLOT))
    else:
        raw_room = 0.0
        raw_waste = 0.0
        raw_gap = 0.0
        raw_late = 0.0

    raw_hard = float(hard_violation_count(problem, schedule, partial=partial))

    weighted_clash = config.W_CLASH * raw_clash
    weighted_room = config.W_ROOM * raw_room
    weighted_waste = config.W_WASTE * raw_waste
    weighted_gap = config.W_GAP * raw_gap
    weighted_late = config.W_LATE * raw_late
    weighted_hard = config.W_HARD * raw_hard

    total = (
        weighted_clash
        + weighted_room
        + weighted_waste
        + weighted_gap
        + weighted_late
        + weighted_hard
    )

    return {
        "raw_clash": raw_clash,
        "raw_room": raw_room,
        "raw_waste": raw_waste,
        "raw_gap": raw_gap,
        "raw_late": raw_late,
        "raw_hard": raw_hard,
        "clash": weighted_clash,
        "room": weighted_room,
        "waste": weighted_waste,
        "gap": weighted_gap,
        "late": weighted_late,
        "hard": weighted_hard,
        "total": total,
    }


def total_cost(problem: Problem, schedule: Schedule, partial: bool = False) -> float:
    """Calculate the scalar weighted objective cost of a schedule."""
    bd = cost_breakdown(problem, schedule, partial=partial)
    return bd["total"]


def compute_kpis(
    problem: Problem,
    schedule: Schedule,
    assignments: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Compute the 14 standard performance indicators for a schedule."""
    bd = cost_breakdown(problem, schedule, partial=False)
    placed = schedule.room_idx >= 0
    placed_indices = np.where(placed)[0]

    C = clash_matrix(problem)
    overlap = overlap_matrix(problem, schedule)
    clashing_pairs = float(np.sum((C > 0) & overlap) // 2)

    # Room utilization pct
    total_occupied_slots = 0.0
    total_span_slots = 0.0
    if len(placed_indices) > 0:
        placed_rooms = schedule.room_idx[placed_indices]
        placed_days = schedule.day[placed_indices]
        placed_starts = schedule.start[placed_indices]
        placed_durs = problem.dur[placed_indices]
        placed_ends = placed_starts + placed_durs

        room_day_keys = set(zip(placed_rooms, placed_days))
        for r, d in room_day_keys:
            mask = (placed_rooms == r) & (placed_days == d)
            occ = float(np.sum(placed_durs[mask]))
            span = float(np.max(placed_ends[mask]) - np.min(placed_starts[mask]))
            total_occupied_slots += occ
            total_span_slots += span

    room_util_pct = (
        (total_occupied_slots / total_span_slots * 100.0) if total_span_slots > 0 else 0.0
    )

    # Seat utilization pct
    if len(placed_indices) > 0:
        att_slots = float(np.sum(problem.attendance[placed_indices] * problem.dur[placed_indices]))
        cap_slots = float(np.sum(problem.room_cap[schedule.room_idx[placed_indices]] * problem.dur[placed_indices]))
        seat_util_pct = (att_slots / cap_slots * 100.0) if cap_slots > 0 else 0.0
    else:
        seat_util_pct = 0.0

    # Venue utilization pct
    avail_rooms_cnt = float(np.sum(problem.room_available))
    total_venue_slots = avail_rooms_cnt * config.N_DAYS * config.SLOTS_PER_DAY
    venue_util_pct = (
        (np.sum(problem.dur[placed]) / total_venue_slots * 100.0)
        if total_venue_slots > 0
        else 0.0
    )

    # Rooms used
    rooms_used = float(len(set(schedule.room_idx[placed]))) if len(placed_indices) > 0 else 0.0

    # Volunteer metrics
    if assignments is not None:
        # Volunteer workload std
        event_idx_map = {eid: idx for idx, eid in enumerate(problem.event_ids)}
        vol_assigned_hours = {vid: 0.0 for vid in problem.volunteer_ids}
        event_vol_counts = {eid: 0 for eid in problem.event_ids}

        for _, row in assignments.iterrows():
            vid = str(row["volunteer_id"])
            eid = str(row["event_id"])
            if vid in vol_assigned_hours and eid in event_idx_map:
                e_idx = event_idx_map[eid]
                vol_assigned_hours[vid] += float(problem.dur[e_idx] / 2.0)
            if eid in event_vol_counts:
                event_vol_counts[eid] += 1

        hours_array = np.array(list(vol_assigned_hours.values()))
        vol_hours_std = float(np.std(hours_array, ddof=1)) if len(hours_array) > 1 else 0.0

        # Unfilled slots
        unfilled = 0.0
        for e_idx, eid in enumerate(problem.event_ids):
            needed = problem.vol_needed[e_idx]
            assigned_cnt = event_vol_counts.get(eid, 0)
            unfilled += max(0.0, float(needed - assigned_cnt))
    else:
        vol_hours_std = float("nan")
        unfilled = float("nan")

    return {
        "total_cost": bd["total"],
        "student_clashes": bd["raw_clash"],
        "clashing_pairs": clashing_pairs,
        "hard_violations": bd["raw_hard"],
        "room_cost_total": bd["raw_room"],
        "wasted_seats": bd["raw_waste"],
        "idle_gap_hours": bd["raw_gap"] / 2.0,
        "late_events": bd["raw_late"],
        "room_utilization_pct": room_util_pct,
        "seat_utilization_pct": seat_util_pct,
        "venue_utilization_pct": venue_util_pct,
        "rooms_used": rooms_used,
        "volunteer_hours_std": vol_hours_std,
        "unfilled_volunteer_slots": unfilled,
    }


def comparison_table(kpis_before: dict[str, float], kpis_after: dict[str, float]) -> pd.DataFrame:
    """Generate a structured Before vs After KPI comparison table."""
    keys = list(KPI_DIRECTIONS.keys())
    rows = []

    for k in keys:
        before_val = kpis_before.get(k, float("nan"))
        after_val = kpis_after.get(k, float("nan"))
        change = after_val - before_val

        if not np.isnan(before_val) and before_val != 0:
            change_pct = (change / before_val) * 100.0
        else:
            change_pct = float("nan")

        direction = KPI_DIRECTIONS.get(k, "n/a")
        if direction == "lower":
            if after_val < before_val:
                better = "Yes"
            elif after_val > before_val:
                better = "No"
            else:
                better = "-"
        elif direction == "higher":
            if after_val > before_val:
                better = "Yes"
            elif after_val < before_val:
                better = "No"
            else:
                better = "-"
        else:
            better = "-"

        rows.append(
            {
                "KPI": k,
                "Before": before_val,
                "After": after_val,
                "Change": change,
                "Change %": change_pct,
                "Better": better,
            }
        )

    df = pd.DataFrame(rows).set_index("KPI")
    return df
