"""Synthetic test dataset generator for EventOptima fest scheduling."""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

import config


ROOM_TEMPLATE = [
    {"room_id": "R01", "name": "Main Auditorium", "room_type": "auditorium", "capacity": 300, "cost_per_hour": 800},
    {"room_id": "R02", "name": "Mini Auditorium", "room_type": "auditorium", "capacity": 150, "cost_per_hour": 500},
    {"room_id": "R03", "name": "Open Air Theatre", "room_type": "open_air", "capacity": 250, "cost_per_hour": 300},
    {"room_id": "R04", "name": "Classroom A", "room_type": "classroom", "capacity": 80, "cost_per_hour": 150},
    {"room_id": "R05", "name": "Classroom B", "room_type": "classroom", "capacity": 60, "cost_per_hour": 120},
    {"room_id": "R06", "name": "Classroom C", "room_type": "classroom", "capacity": 60, "cost_per_hour": 120},
    {"room_id": "R07", "name": "Classroom D", "room_type": "classroom", "capacity": 40, "cost_per_hour": 100},
    {"room_id": "R08", "name": "Computer Lab 1", "room_type": "lab", "capacity": 50, "cost_per_hour": 250},
    {"room_id": "R09", "name": "Computer Lab 2", "room_type": "lab", "capacity": 30, "cost_per_hour": 200},
    {"room_id": "R10", "name": "Seminar Hall", "room_type": "seminar", "capacity": 100, "cost_per_hour": 250},
]

EQUIPMENT_TEMPLATE = [
    {"equipment_id": "EQ01", "name": "Projector", "stock": 5},
    {"equipment_id": "EQ02", "name": "Microphone", "stock": 6},
    {"equipment_id": "EQ03", "name": "Sound System", "stock": 2},
    {"equipment_id": "EQ04", "name": "Laptop", "stock": 8},
    {"equipment_id": "EQ05", "name": "Camera", "stock": 2},
]

ROOM_TYPE_PROBS = {
    "Technical": [("lab", 0.5), ("classroom", 0.5)],
    "Cultural": [("auditorium", 0.5), ("open_air", 0.5)],
    "Workshop": [("lab", 0.5), ("seminar", 0.5)],
    "Talk": [("auditorium", 0.4), ("seminar", 0.3), ("classroom", 0.3)],
    "Competition": [("classroom", 0.5), ("lab", 0.2), ("open_air", 0.3)],
}


def _build_rooms(n_rooms: int) -> pd.DataFrame:
    """Generate the rooms table according to the fixed room specifications."""
    rows = []
    base_count = len(ROOM_TEMPLATE)
    for i in range(n_rooms):
        src = ROOM_TEMPLATE[i % base_count]
        rid = f"R{i+1:02d}"
        name = src["name"] if i < base_count else f"{src['name']} {i // base_count + 1}"
        rows.append(
            {
                "room_id": rid,
                "name": name,
                "room_type": src["room_type"],
                "capacity": int(src["capacity"]),
                "cost_per_hour": int(src["cost_per_hour"]),
            }
        )
    return pd.DataFrame(rows)


def _build_equipment() -> pd.DataFrame:
    """Generate the fixed equipment stock table."""
    return pd.DataFrame(EQUIPMENT_TEMPLATE)


def generate_all(
    seed: int = config.SEED,
    out_dir: str = "data",
    n_events: int = 25,
    n_rooms: int = 10,
    n_students: int = 300,
    n_volunteers: int = 40,
) -> dict[str, pd.DataFrame]:
    """Generate all seven dataset CSVs deterministically using the provided seed."""
    rng = np.random.default_rng(seed)

    # 1. Rooms
    rooms_df = _build_rooms(n_rooms)

    # 2. Equipment
    equip_df = _build_equipment()

    # Pre-calculate max capacity per room type to guarantee event feasibility
    available_room_types = set(rooms_df["room_type"])
    max_cap_by_type = {}
    for rtype in config.ROOM_TYPES:
        matching = rooms_df[rooms_df["room_type"] == rtype]["capacity"]
        max_cap_by_type[rtype] = int(matching.max()) if not matching.empty else 0

    # 3. Events initial generation (without final expected attendance)
    events_raw = []
    category_counts = {cat: 0 for cat in config.CATEGORIES}

    for i in range(n_events):
        event_id = f"E{i+1:02d}"
        category = rng.choice(config.CATEGORIES)
        category_counts[category] += 1
        name = f"{category} {category_counts[category]}"

        probs_info = ROOM_TYPE_PROBS[category]
        valid_pairs = [(rt, p) for rt, p in probs_info if rt in available_room_types]
        if valid_pairs:
            types, probs = zip(*valid_pairs)
            norm_probs = np.array(probs, dtype=float) / sum(probs)
            required_room_type = rng.choice(types, p=norm_probs)
        else:
            required_room_type = rng.choice(sorted(list(available_room_types)))

        duration_slots = int(rng.integers(2, 7))
        priority = int(rng.integers(1, 6))
        volunteers_needed = int(rng.integers(1, 5))
        required_skill = rng.choice(config.SKILLS)

        events_raw.append(
            {
                "event_id": event_id,
                "name": name,
                "category": category,
                "required_room_type": required_room_type,
                "duration_slots": duration_slots,
                "priority": priority,
                "volunteers_needed": volunteers_needed,
                "required_skill": required_skill,
            }
        )

    # 4. Registrations and Expected Attendance
    # Student registrations: Dirichlet weights over events
    pop_weights = rng.dirichlet(np.ones(n_events))
    student_ids = [f"S{i+1:03d}" for i in range(n_students)]
    student_registrations: dict[str, list[int]] = {sid: [] for sid in student_ids}
    event_reg_students: list[list[str]] = [[] for _ in range(n_events)]

    for sid in student_ids:
        k = int(rng.integers(2, 5))  # 2 to 4 distinct events
        chosen_events = rng.choice(n_events, size=k, replace=False, p=pop_weights)
        for e_idx in chosen_events:
            student_registrations[sid].append(e_idx)
            event_reg_students[e_idx].append(sid)

    # Capacity check: registered_count * 1.2 <= max capacity
    # If exceeded, randomly drop registrations for that event
    for e_idx, ev in enumerate(events_raw):
        rtype = ev["required_room_type"]
        max_cap = max_cap_by_type.get(rtype, 0)
        current_students = event_reg_students[e_idx]
        while math.ceil(len(current_students) * 1.2) > max_cap and current_students:
            drop_idx = rng.integers(0, len(current_students))
            dropped_student = current_students.pop(drop_idx)
            student_registrations[dropped_student].remove(e_idx)

    # Now compute final expected attendance per event
    expected_attendances = []
    for e_idx, ev in enumerate(events_raw):
        registered_count = len(event_reg_students[e_idx])
        factor = rng.uniform(1.0, 1.2)
        att = max(10, math.ceil(registered_count * factor))
        # Ensure it does not exceed max capacity
        rtype = ev["required_room_type"]
        max_cap = max_cap_by_type.get(rtype, 0)
        if att > max_cap and max_cap > 0:
            att = max_cap
        expected_attendances.append(att)

    # Finalize events DataFrame
    events_rows = []
    for ev, att in zip(events_raw, expected_attendances):
        events_rows.append(
            {
                "event_id": ev["event_id"],
                "name": ev["name"],
                "category": ev["category"],
                "required_room_type": ev["required_room_type"],
                "duration_slots": ev["duration_slots"],
                "expected_attendance": att,
                "priority": ev["priority"],
                "volunteers_needed": ev["volunteers_needed"],
                "required_skill": ev["required_skill"],
            }
        )
    events_df = pd.DataFrame(events_rows)

    # Finalize registrations DataFrame
    reg_rows = []
    for sid in student_ids:
        for e_idx in student_registrations[sid]:
            reg_rows.append({"student_id": sid, "event_id": events_raw[e_idx]["event_id"]})
    reg_df = pd.DataFrame(reg_rows).drop_duplicates()

    # 5. Event Equipment
    stock_dict = dict(zip(equip_df["equipment_id"], equip_df["stock"]))
    equip_ids = list(equip_df["equipment_id"])
    event_equip_rows = []
    qty_probs = [0.3, 0.3, 0.25, 0.15]

    for ev in events_raw:
        eid = ev["event_id"]
        n_items = int(rng.choice([0, 1, 2, 3], p=qty_probs))
        if n_items > 0:
            chosen_equip = rng.choice(equip_ids, size=n_items, replace=False)
            for eqid in chosen_equip:
                max_stk = stock_dict[eqid]
                qty = int(rng.integers(1, 3))
                qty = min(qty, max_stk)
                event_equip_rows.append({"event_id": eid, "equipment_id": eqid, "qty": qty})
    event_equip_df = pd.DataFrame(event_equip_rows)

    # 6. Volunteers
    volunteers_rows = []
    vol_skills: list[list[str]] = []
    skill_counts = {s: 0 for s in config.SKILLS}

    for i in range(n_volunteers):
        vid = f"V{i+1:03d}"
        name = f"Volunteer {vid}"
        n_s = int(rng.integers(1, 4))
        chosen_skills = list(rng.choice(config.SKILLS, size=n_s, replace=False))
        for s in chosen_skills:
            skill_counts[s] += 1
        vol_skills.append(chosen_skills)
        max_h = float(rng.choice([6.0, 7.0, 8.0, 9.0, 10.0]))
        volunteers_rows.append({"volunteer_id": vid, "name": name, "max_hours": max_h})

    # Repair step: guarantee >= 6 volunteers for every skill
    for s in config.SKILLS:
        while skill_counts[s] < 6:
            candidates = [idx for idx, sk_list in enumerate(vol_skills) if s not in sk_list]
            if not candidates:
                break
            chosen_idx = rng.choice(candidates)
            vol_skills[chosen_idx].append(s)
            skill_counts[s] += 1

    for i, r in enumerate(volunteers_rows):
        r["skills"] = ";".join(vol_skills[i])
    vol_df = pd.DataFrame(volunteers_rows)
    # Ensure correct column ordering
    vol_df = vol_df[["volunteer_id", "name", "skills", "max_hours"]]

    # 7. Past Attendance
    past_rows = []
    for cat in config.CATEGORIES:
        base = rng.uniform(150, 400)
        g = rng.uniform(0.03, 0.12)
        for year in config.PAST_YEARS:
            noise = rng.normal(0, 0.05 * base)
            val = base * (1.0 + g * (year - 2021)) + noise
            attendance = max(0, int(round(val)))
            past_rows.append({"category": cat, "year": year, "attendance": attendance})
    past_df = pd.DataFrame(past_rows)

    tables = {
        "events": events_df,
        "rooms": rooms_df,
        "equipment": equip_df,
        "event_equipment": event_equip_df,
        "volunteers": vol_df,
        "registrations": reg_df,
        "past_attendance": past_df,
    }

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            csv_path = out_path / f"{name}.csv"
            df.to_csv(csv_path, index=False)

    return tables


def main() -> None:
    """CLI entry point to generate synthetic dataset files."""
    parser = argparse.ArgumentParser(description="Generate EventOptima dataset CSVs.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    parser.add_argument("--events", type=int, default=25, help="Number of events")
    parser.add_argument("--rooms", type=int, default=10, help="Number of rooms")
    parser.add_argument("--students", type=int, default=300, help="Number of students")
    parser.add_argument("--volunteers", type=int, default=40, help="Number of volunteers")
    args = parser.parse_args()

    generate_all(
        seed=args.seed,
        out_dir=args.out,
        n_events=args.events,
        n_rooms=args.rooms,
        n_students=args.students,
        n_volunteers=args.volunteers,
    )
    print(f"Generated datasets in '{args.out}' with seed {args.seed}.")


if __name__ == "__main__":
    main()
