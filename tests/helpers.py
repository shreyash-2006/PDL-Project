"""Helper utilities and fixtures for test suites."""

import config
from models import Problem, Schedule


def serial_schedule(problem: Problem) -> Schedule:
    """Build a strictly serial, collision-free schedule by placing events one after another."""
    n_events = len(problem.event_ids)
    sched = Schedule.empty(n_events)

    current_day = 0
    current_slot = 0

    for e_idx in range(n_events):
        dur = int(problem.dur[e_idx])
        # Find first compatible and available room
        comp_rooms = [
            r
            for r in range(len(problem.room_ids))
            if problem.compatible[e_idx, r] and problem.room_available[r]
        ]
        if not comp_rooms:
            raise RuntimeError(f"No compatible available room for event {problem.event_ids[e_idx]}")

        chosen_room = comp_rooms[0]

        if current_slot + dur > config.SLOTS_PER_DAY:
            current_day += 1
            current_slot = 0

        if current_day < config.N_DAYS:
            sched.room_idx[e_idx] = chosen_room
            sched.day[e_idx] = current_day
            sched.start[e_idx] = current_slot
            current_slot += dur
        else:
            # Fallback for large problems to ensure zero violations
            placed = False
            for d in range(config.N_DAYS):
                for s in range(config.SLOTS_PER_DAY - dur + 1):
                    for r in comp_rooms:
                        # Check room overlap
                        overlap = False
                        for prev_e in range(e_idx):
                            if sched.room_idx[prev_e] == r and sched.day[prev_e] == d:
                                prev_s = sched.start[prev_e]
                                prev_end = prev_s + problem.dur[prev_e]
                                if s < prev_end and prev_s < s + dur:
                                    overlap = True
                                    break
                        if overlap:
                            continue

                        # Check equipment stock
                        equip_ok = True
                        for q in range(len(problem.equip_ids)):
                            need = problem.equip_need[e_idx, q]
                            if need > 0:
                                for slot_check in range(s, s + dur):
                                    used = sum(
                                        problem.equip_need[prev_e, q]
                                        for prev_e in range(e_idx)
                                        if sched.day[prev_e] == d
                                        and sched.start[prev_e] <= slot_check < sched.start[prev_e] + problem.dur[prev_e]
                                    )
                                    if used + need > problem.equip_stock[q]:
                                        equip_ok = False
                                        break
                                if not equip_ok:
                                    break
                        if equip_ok:
                            sched.room_idx[e_idx] = r
                            sched.day[e_idx] = d
                            sched.start[e_idx] = s
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                raise RuntimeError(f"Unable to serially place event {problem.event_ids[e_idx]}")

    return sched
