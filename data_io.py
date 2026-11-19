"""Data loading, cleaning, validation, and domain problem construction."""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from models import Problem

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "events": [
        "event_id",
        "name",
        "category",
        "required_room_type",
        "duration_slots",
        "expected_attendance",
        "priority",
        "volunteers_needed",
        "required_skill",
    ],
    "rooms": ["room_id", "name", "room_type", "capacity", "cost_per_hour"],
    "equipment": ["equipment_id", "name", "stock"],
    "event_equipment": ["event_id", "equipment_id", "qty"],
    "volunteers": ["volunteer_id", "name", "skills", "max_hours"],
    "registrations": ["student_id", "event_id"],
    "past_attendance": ["category", "year", "attendance"],
}

NUMERIC_COLUMNS = {
    "events": ["duration_slots", "expected_attendance", "priority", "volunteers_needed"],
    "rooms": ["capacity", "cost_per_hour"],
    "equipment": ["stock"],
    "event_equipment": ["qty"],
    "volunteers": ["max_hours"],
    "past_attendance": ["year", "attendance"],
}

ID_COLUMNS = ["event_id", "room_id", "equipment_id", "volunteer_id", "student_id"]


class DataError(Exception):
    """Raised when dataset files or data structures fail schema or feasibility requirements."""

    pass


def clean_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean and standardize input tables by stripping whitespace, uppercase IDs, and removing invalid rows."""
    cleaned: dict[str, pd.DataFrame] = {}

    for name, cols in REQUIRED_COLUMNS.items():
        if name not in tables:
            raise DataError(f"Missing required table: '{name}'")
        df = tables[name].copy()

        # Check required columns
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise DataError(f"Table '{name}' is missing required columns: {missing_cols}")

        # Strip whitespace for string/object columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()

        # Upper-case ID columns
        for col in ID_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper()

        # Coerce numerics and drop rows with invalid numeric values
        num_cols = NUMERIC_COLUMNS.get(name, [])
        initial_len = len(df)
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=num_cols)
        dropped_num = initial_len - len(df)
        if dropped_num > 0:
            logger.warning("Dropped %d rows in '%s' due to invalid numeric values.", dropped_num, name)

        # Drop exact duplicate rows
        pre_dup = len(df)
        if name == "registrations":
            df = df.drop_duplicates(subset=["student_id", "event_id"])
        else:
            df = df.drop_duplicates()
        dropped_dup = pre_dup - len(df)
        if dropped_dup > 0:
            logger.info("Dropped %d duplicate rows in '%s'.", dropped_dup, name)

        cleaned[name] = df

    # Drop link table rows with nonexistent foreign keys
    events_df = cleaned["events"]
    rooms_df = cleaned["rooms"]
    equip_df = cleaned["equipment"]

    valid_event_ids = set(events_df["event_id"])
    valid_equip_ids = set(equip_df["equipment_id"])

    # Clean event_equipment
    ee = cleaned["event_equipment"]
    ee_initial = len(ee)
    ee = ee[ee["event_id"].isin(valid_event_ids) & ee["equipment_id"].isin(valid_equip_ids)]
    dropped_ee = ee_initial - len(ee)
    if dropped_ee > 0:
        logger.warning("Dropped %d rows in 'event_equipment' with missing foreign keys.", dropped_ee)
    cleaned["event_equipment"] = ee

    # Clean registrations
    reg = cleaned["registrations"]
    reg_initial = len(reg)
    reg = reg[reg["event_id"].isin(valid_event_ids)]
    dropped_reg = reg_initial - len(reg)
    if dropped_reg > 0:
        logger.warning("Dropped %d rows in 'registrations' with missing event foreign keys.", dropped_reg)
    cleaned["registrations"] = reg

    return cleaned


def load_problem(data_dir: str = "data") -> Problem:
    """Load, clean, and validate all CSV datasets from a directory into a Problem instance."""
    path = Path(data_dir)
    if not path.exists() or not path.is_dir():
        raise DataError(f"Data directory '{data_dir}' does not exist or is not a directory.")

    raw_tables: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_COLUMNS:
        csv_file = path / f"{name}.csv"
        if not csv_file.exists():
            raise DataError(f"Required dataset file '{name}.csv' not found in '{data_dir}'.")
        raw_tables[name] = pd.read_csv(csv_file)

    cleaned = clean_tables(raw_tables)

    # Sort each table by primary identifier
    events_df = cleaned["events"].sort_values("event_id").reset_index(drop=True)
    rooms_df = cleaned["rooms"].sort_values("room_id").reset_index(drop=True)
    equip_df = cleaned["equipment"].sort_values("equipment_id").reset_index(drop=True)
    vol_df = cleaned["volunteers"].sort_values("volunteer_id").reset_index(drop=True)
    reg_df = cleaned["registrations"].sort_values(["student_id", "event_id"]).reset_index(drop=True)
    ee_df = cleaned["event_equipment"].sort_values(["event_id", "equipment_id"]).reset_index(drop=True)
    past_df = cleaned["past_attendance"].sort_values(["category", "year"]).reset_index(drop=True)

    event_ids = list(events_df["event_id"])
    room_ids = list(rooms_df["room_id"])
    equip_ids = list(equip_df["equipment_id"])
    volunteer_ids = list(vol_df["volunteer_id"])
    student_ids = sorted(list(reg_df["student_id"].unique()))

    n_events = len(event_ids)
    n_rooms = len(room_ids)
    n_equip = len(equip_ids)
    n_students = len(student_ids)

    dur = events_df["duration_slots"].to_numpy(dtype=int)
    attendance = events_df["expected_attendance"].to_numpy(dtype=int)
    priority = events_df["priority"].to_numpy(dtype=int)
    vol_needed = events_df["volunteers_needed"].to_numpy(dtype=int)
    skill_required = list(events_df["required_skill"])

    room_cap = rooms_df["capacity"].to_numpy(dtype=int)
    room_cost = rooms_df["cost_per_hour"].to_numpy(dtype=float)
    room_available = np.ones(n_rooms, dtype=bool)

    # Build compatibility matrix: (E, R)
    req_types = events_df["required_room_type"].to_numpy()[:, None]
    room_types = rooms_df["room_type"].to_numpy()[None, :]
    type_match = req_types == room_types
    cap_match = room_cap[None, :] >= attendance[:, None]
    compatible = type_match & cap_match

    # Ensure every event has at least one compatible room
    incompatible_events = [event_ids[i] for i in range(n_events) if not compatible[i].any()]
    if incompatible_events:
        raise DataError(f"The following events have no compatible room: {incompatible_events}")

    # Build registration matrix R: (S, E)
    student_idx_map = {sid: i for i, sid in enumerate(student_ids)}
    event_idx_map = {eid: i for i, eid in enumerate(event_ids)}

    R = np.zeros((n_students, n_events), dtype=np.uint8)
    for _, row in reg_df.iterrows():
        sid = row["student_id"]
        eid = row["event_id"]
        if sid in student_idx_map and eid in event_idx_map:
            R[student_idx_map[sid], event_idx_map[eid]] = 1

    # Build equip_need: (E, Q) and equip_stock: (Q,)
    equip_idx_map = {eqid: i for i, eqid in enumerate(equip_ids)}
    equip_need = np.zeros((n_events, n_equip), dtype=int)
    for _, row in ee_df.iterrows():
        eid = row["event_id"]
        eqid = row["equipment_id"]
        qty = int(row["qty"])
        if eid in event_idx_map and eqid in equip_idx_map:
            equip_need[event_idx_map[eid], equip_idx_map[eqid]] = qty

    equip_stock = equip_df["stock"].to_numpy(dtype=int)

    return Problem(
        events=events_df,
        rooms=rooms_df,
        equipment=equip_df,
        event_equipment=ee_df,
        volunteers=vol_df,
        registrations=reg_df,
        past_attendance=past_df,
        event_ids=event_ids,
        room_ids=room_ids,
        equip_ids=equip_ids,
        volunteer_ids=volunteer_ids,
        student_ids=student_ids,
        dur=dur,
        attendance=attendance,
        priority=priority,
        room_cap=room_cap,
        room_cost=room_cost,
        room_available=room_available,
        compatible=compatible,
        R=R,
        equip_need=equip_need,
        equip_stock=equip_stock,
        vol_needed=vol_needed,
        skill_required=skill_required,
    )
