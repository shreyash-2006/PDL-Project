"""Data models for EventOptima problem and schedule representations."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class Problem:
    """Problem instance container holding all domain entities, metadata, and numerical arrays."""

    events: pd.DataFrame
    rooms: pd.DataFrame
    equipment: pd.DataFrame
    event_equipment: pd.DataFrame
    volunteers: pd.DataFrame
    registrations: pd.DataFrame
    past_attendance: pd.DataFrame
    event_ids: list[str]
    room_ids: list[str]
    equip_ids: list[str]
    volunteer_ids: list[str]
    student_ids: list[str]
    dur: np.ndarray  # (E,) int, slots
    attendance: np.ndarray  # (E,) int
    priority: np.ndarray  # (E,) int
    room_cap: np.ndarray  # (R,) int
    room_cost: np.ndarray  # (R,) float, per hour
    room_available: np.ndarray  # (R,) bool, default all True
    compatible: np.ndarray  # (E,R) bool: type matches AND capacity >= attendance (ignores availability)
    R: np.ndarray  # (S,E) uint8, 1 if student s registered for event e
    equip_need: np.ndarray  # (E,Q) int
    equip_stock: np.ndarray  # (Q,) int
    vol_needed: np.ndarray  # (E,) int
    skill_required: list[str]  # length E
    _C: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def with_unavailable_rooms(self, room_ids: list[str]) -> "Problem":
        """Return a copy of the problem with the specified room IDs marked unavailable."""
        new_avail = self.room_available.copy()
        for rid in room_ids:
            if rid in self.room_ids:
                idx = self.room_ids.index(rid)
                new_avail[idx] = False

        return Problem(
            events=self.events,
            rooms=self.rooms,
            equipment=self.equipment,
            event_equipment=self.event_equipment,
            volunteers=self.volunteers,
            registrations=self.registrations,
            past_attendance=self.past_attendance,
            event_ids=self.event_ids,
            room_ids=self.room_ids,
            equip_ids=self.equip_ids,
            volunteer_ids=self.volunteer_ids,
            student_ids=self.student_ids,
            dur=self.dur,
            attendance=self.attendance,
            priority=self.priority,
            room_cap=self.room_cap,
            room_cost=self.room_cost,
            room_available=new_avail,
            compatible=self.compatible,
            R=self.R,
            equip_need=self.equip_need,
            equip_stock=self.equip_stock,
            vol_needed=self.vol_needed,
            skill_required=self.skill_required,
            _C=self._C,
        )


@dataclass
class Schedule:
    """Schedule assignment vector specifying room, day, and start slot for each event."""

    room_idx: np.ndarray  # (E,) int, -1 = unplaced
    day: np.ndarray  # (E,) int
    start: np.ndarray  # (E,) int, slot index

    def copy(self) -> "Schedule":
        """Return an independent copy of this schedule."""
        return Schedule(
            room_idx=self.room_idx.copy(),
            day=self.day.copy(),
            start=self.start.copy(),
        )

    def to_dataframe(self, problem: Problem) -> pd.DataFrame:
        """Convert placed events in this schedule into a DataFrame."""
        rows = []
        for e_idx, (r_idx, d, s) in enumerate(zip(self.room_idx, self.day, self.start)):
            if r_idx >= 0:
                event_id = problem.event_ids[e_idx]
                room_id = problem.room_ids[r_idx]
                dur = int(problem.dur[e_idx])
                rows.append(
                    {
                        "event_id": event_id,
                        "room_id": room_id,
                        "day": int(d),
                        "start_slot": int(s),
                        "end_slot": int(s + dur),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=["event_id", "room_id", "day", "start_slot", "end_slot"],
        )

    @staticmethod
    def from_dataframe(problem: Problem, df: pd.DataFrame) -> "Schedule":
        """Reconstruct a Schedule from a DataFrame of placed event assignments."""
        n_events = len(problem.event_ids)
        room_idx = np.full(n_events, -1, dtype=int)
        day = np.zeros(n_events, dtype=int)
        start = np.zeros(n_events, dtype=int)

        event_map = {eid: idx for idx, eid in enumerate(problem.event_ids)}
        room_map = {rid: idx for idx, rid in enumerate(problem.room_ids)}

        for _, row in df.iterrows():
            eid = str(row["event_id"])
            if eid in event_map:
                e_idx = event_map[eid]
                rid = str(row["room_id"])
                if rid in room_map:
                    room_idx[e_idx] = room_map[rid]
                    day[e_idx] = int(row["day"])
                    start[e_idx] = int(row["start_slot"])

        return Schedule(room_idx=room_idx, day=day, start=start)

    @staticmethod
    def empty(n_events: int) -> "Schedule":
        """Create an empty schedule where all events are unplaced."""
        return Schedule(
            room_idx=np.full(n_events, -1, dtype=int),
            day=np.zeros(n_events, dtype=int),
            start=np.zeros(n_events, dtype=int),
        )
