# EventOptima: Two-Day Fest Timetable Optimization Platform

**EventOptima** is an intelligent college fest timetable optimization engine and analytics dashboard. It schedules multi-category student events across constrained venue rooms, allocates shared equipment, assigns and balances student volunteer shifts, and eliminates student registration timetable clashes.

The system features a **Baseline First-Fit** scheduler (the "Before"), a **Greedy Heuristic**, and a metaheuristic **Simulated Annealing (SA)** and **Genetic Algorithm (GA)** optimizer (the "After") presented in a **Streamlit** visual dashboard with what-if scenario simulations.

---

## 🏛️ System Architecture

```
                                  +-----------------------+
                                  |    Synthetic Data     |
                                  |     generator.py      |
                                  +-----------+-----------+
                                              |
                                              v
+-------------------------+       +-----------+-----------+       +-------------------------+
|      Configuration      | ----> |     Data Ingestion    | <---- |     Domain Models       |
|        config.py        |       |      data_io.py       |       |        models.py        |
+-------------------------+       +-----------+-----------+       +-------------------------+
                                              |
                                              v
                                  +-----------+-----------+
                                  |     Problem Instance  |
                                  +-----------+-----------+
                                              |
                      +-----------------------+-----------------------+
                      |                                               |
                      v                                               v
          +-----------+-----------+                       +-----------+-----------+
          | Constraints Engine    |                       | Objective & Cost      |
          |    constraints.py     |                       |        cost.py        |
          +-----------+-----------+                       +-----------+-----------+
                      |                                               |
                      +-----------------------+-----------------------+
                                              |
                                              v
                                  +-----------+-----------+
                                  |  Scheduling Engines   |
                                  |     scheduler.py      |
                                  | (Baseline, Greedy, SA)|
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------+-----------+
                                  | Resource Allocation   |
                                  |     resources.py      |
                                  | (Volunteers & Equip)  |
                                  +-----------+-----------+
                                              |
                      +-----------------------+-----------------------+
                      |                                               |
                      v                                               v
          +-----------+-----------+                       +-----------+-----------+
          | CLI Pipeline Pipeline |                       | Streamlit Dashboard   |
          |        main.py        |                       |        app.py         |
          +-----------------------+                       +-----------------------+
```

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.11+
- Virtual environment recommended (`venv` or `conda`)

### 1. Installation
Clone the repository and install required dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Dataset
Generate the 7 standard fest dataset CSVs into the `data/` directory with seed 42:
```bash
python generator.py --seed 42 --out data
```

### 3. Run Optimization Pipeline (CLI)
Execute the end-to-end optimization pipeline and generate all artifacts in `outputs/`:
```bash
python main.py --iterations 20000 --compare
```

### 4. Launch Interactive Web Dashboard
Launch the interactive Streamlit analytics platform:
```bash
streamlit run app.py
```

### 5. Run Test Suite
Execute the comprehensive pytest suite:
```bash
pytest -q
```

---

## 📊 Dataset Schemas (`data/`)

All datasets are comma-separated UTF-8 CSVs:

| File | Primary Key / Index | Key Columns | Purpose |
| :--- | :--- | :--- | :--- |
| `events.csv` | `event_id` | `name`, `category`, `required_room_type`, `duration_slots`, `expected_attendance`, `priority`, `volunteers_needed`, `required_skill` | Core event definitions |
| `rooms.csv` | `room_id` | `name`, `room_type`, `capacity`, `cost_per_hour` | Venue specifications |
| `equipment.csv` | `equipment_id` | `name`, `stock` | Global inventory limits |
| `event_equipment.csv` | `(event_id, equipment_id)` | `qty` | Equipment demand per event |
| `volunteers.csv` | `volunteer_id` | `name`, `skills` (semicolon-separated), `max_hours` | Volunteer roster |
| `registrations.csv` | `(student_id, event_id)` | — | Student event registrations |
| `past_attendance.csv` | `(category, year)` | `attendance` | Historical attendance (2021-2025) |

---

## ⏱️ Core Time Model & Definitions

1. **Discrete Slots:** The fest spans $N = 2$ days (Day 0 and Day 1), operating daily from 09:00 to 21:00 (12 hours). Time is discretized into $\text{SLOT\_MINUTES} = 30$, giving $\text{SLOTS\_PER\_DAY} = 24$ slots.
2. **Intervals:** An event with start $s$ and duration $d$ occupies half-open interval $[s, s + d)$ on its assigned day. Valid bounds require $0 \le s \le s + d \le 24$.
3. **Clashes & Overlaps:** Events $A$ and $B$ overlap iff on the same day and $s_A < e_B \land s_B < e_A$. Back-to-back ($e_A = s_B$) is **not** an overlap.
4. **Student Clashes:** Pairwise clashes are computed via registration matrix $R \in \{0, 1\}^{S \times E}$ as $C = R^T R$. Total student clashes = $\sum_{i < j} C_{ij} \times \text{overlap}_{ij}$.
5. **Idle Gap Slots:** For each (room, day) with $\ge 2$ events sorted by start time, gap = $\sum \max(0, \text{next.start} - \text{prev.end})$.
6. **Late Events:** Any event scheduled with start slot $s \ge 18$ (18:00 onwards).

---

## 👥 Engineering Team Roles

- **Role 1 (Data + Constraints):** Data schema, synthetic generation (`generator.py`), tabular cleaning and problem loader (`data_io.py`), and 11-rule constraint validator (`constraints.py`).
- **Role 2 (Scheduling Algorithms):** First-fit baseline (`schedule_baseline`), greedy cost-minimizer (`schedule_greedy`), simulated annealing (`schedule_annealing`), and genetic algorithm (`schedule_genetic`).
- **Role 3 (Cost + Resources + Analytics):** Vectorized matrix math (`cost.py`), volunteer load-balancer and equipment allocator (`resources.py`), and polynomial attendance forecasting (`forecast.py`).
- **Role 4 (Dashboard + Integration):** CLI pipeline (`main.py`), interactive Streamlit dashboard (`app.py`), automated CI/CD, documentation, and end-to-end testing.
