# EventOptima: Implementation Plan (Build Spec for AI Agents)

Version 1.0. Python 3.11. Every file name, column name, function signature and number in this document is binding. Hand each agent: Sections 1 to 3, its own role section (4 to 7), and Sections 8 to 9.

## 1. Project and rules of engagement

**Goal.** Take events, rooms, equipment, volunteers and student registrations; build a clash-free two-day fest timetable; optimize it for cost and space; show Before (baseline) vs After (optimized) in a Streamlit dashboard.

**Stack.** Python 3.11, NumPy, pandas, matplotlib, plotly, Streamlit, pytest, Git/GitHub. No other third-party packages (no scipy, sklearn, ortools, DEAP).

**Rules for every agent**

1. Use only the names in this spec. Do not rename, add required parameters, or change return types. Extra optional keyword arguments are allowed.
2. Randomness only through `np.random.default_rng(seed)` passed in or created from a `seed` argument. Default seed is 42. Same seed must give identical output.
3. Type hints and a one-line docstring on every public function. No print debugging in library code; use `logging`.
4. Constants live only in `config.py`. Never hard-code 24, 2, 30, weights, etc. elsewhere.
5. If the spec is silent or ambiguous, pick the simplest reading, add one line to `DECISIONS.md` (what and why), and keep going. Do not stop to ask.
6. Never edit a file owned by another role except to fix an import error; if you find a bug there, note it in `DECISIONS.md` under 'Cross-role bugs'.
7. Every public function must have at least one pytest test. All tests must pass with `pytest -q` from the repo root before a branch is merged.

## 2. Shared contracts (frozen in Phase 0)

### 2.1 Repo layout

```
eventoptima/
  config.py            # constants (Role 1 writes in Phase 0, then frozen)
  models.py            # Problem, Schedule dataclasses (Role 1, Phase 0, frozen)
  data_io.py           # load_problem, clean_tables (Role 1)
  generator.py         # fake data generator (Role 1)
  constraints.py       # validate() (Role 1)
  scheduler.py         # baseline, greedy, SA, GA, plot_convergence (Role 2)
  cost.py              # clash matrix, cost, KPIs, comparison table (Role 3)
  resources.py         # volunteers, equipment (Role 3)
  forecast.py          # attendance forecast (Role 3)
  app.py               # Streamlit dashboard (Role 4)
  main.py              # CLI pipeline (Role 4)
  data/                # generated CSVs (committed)
  outputs/             # pipeline outputs (git-ignored)
  tests/
    helpers.py         # serial_schedule() (Role 1)
    conftest.py        # fixtures small_problem, default_problem (Role 1)
    test_constraints.py  test_scheduler.py  test_resources.py  test_cost.py  test_end_to_end.py
  requirements.txt  README.md  DECISIONS.md  REPORT.md  DEMO_SCRIPT.md
  .github/workflows/ci.yml
```

`models.py`, `config.py`, `data_io.py` and `tests/helpers.py` are additions to the original file lists; they exist so all four roles share one data model.

### 2.2 Time model and definitions (no other interpretation is allowed)

- Fest has `N_DAYS = 2` days, indexed 0 and 1. Each day starts 09:00 and ends 21:00.
- Time is discrete: `SLOT_MINUTES = 30`, `SLOTS_PER_DAY = 24`. Slot 0 = 09:00, slot 23 = 20:30-21:00. Slot s starts at 09:00 + 30\*s minutes.
- An event with `start`, `dur` on `day` occupies slots `[start, start+dur)` of that day. It must satisfy `start + dur <= SLOTS_PER_DAY` (otherwise: OUT\_OF\_DAY).
- Two events **overlap** iff same day and `a.start < b.end and b.start < a.end`. Back-to-back (a.end == b.start) is NOT an overlap; no changeover buffer.
- **Student clash** between overlapping events i, j = number of students registered for both (`C[i,j]`). Total student clashes = sum over unordered overlapping pairs of `C[i,j]`. 'Clashing pairs' = count of overlapping pairs with `C[i,j] > 0`.
- **Late event**: `start >= LATE_START_SLOT` (18 = 18:00).
- **Idle gap slots**: for each (room, day) with 2+ events, sort by start; gap = sum of `max(0, next.start - prev.end)`. Total = sum over all rooms and days. Idle time before the first and after the last event is not counted.
- **Room cost** of an event = `cost_per_hour * dur / 2`.
- **Wasted seats** of an event = `max(0, capacity - expected_attendance)`.
- Index convention: `event_idx` is the row position in `problem.event_ids` (sorted by event\_id). Same for rooms, equipment, volunteers, students.

### 2.3 `config.py` (exact contents of constants)

```python
SEED = 42
N_DAYS = 2
SLOT_MINUTES = 30
SLOTS_PER_DAY = 24
DAY_START = '09:00'
LATE_START_SLOT = 18
BASE_DATE = '2026-03-01'          # used only to build datetimes for charts
W_CLASH = 20.0     # per student-clash
W_ROOM = 0.1       # per currency unit of room cost
W_WASTE = 0.5      # per wasted seat
W_GAP = 3.0        # per idle gap slot
W_LATE = 40.0      # per late event
W_HARD = 10000.0   # per hard violation
SKILLS = ['tech_support', 'stage_management', 'registration', 'hospitality', 'first_aid']
CATEGORIES = ['Technical', 'Cultural', 'Workshop', 'Talk', 'Competition']
ROOM_TYPES = ['auditorium', 'open_air', 'classroom', 'lab', 'seminar']
PAST_YEARS = [2021, 2022, 2023, 2024, 2025]
FORECAST_YEAR = 2026
SA_ITER = 20000; SA_T0 = 50.0; SA_T_END = 0.05
GA_POP = 60; GA_GENERATIONS = 300; GA_ELITE = 2; GA_TOURNAMENT_K = 3; GA_MUT_RATE = 0.05
```

### 2.4 CSV schemas (all in `data/`, UTF-8, header row, comma separated)

| File | Columns (type) | Notes |
| --- | --- | --- |
| events.csv | event\_id (E01..E25), name, category (in CATEGORIES), required\_room\_type (in ROOM\_TYPES), duration\_slots (int 2-6), expected\_attendance (int), priority (int 1-5, 5 = most important), volunteers\_needed (int 1-4), required\_skill (in SKILLS) | one row per event |
| rooms.csv | room\_id (R01..R10), name, room\_type, capacity (int), cost\_per\_hour (int) | fixed table in 4.1 |
| equipment.csv | equipment\_id (EQ01..EQ05), name, stock (int) |  |
| event\_equipment.csv | event\_id, equipment\_id, qty (int >= 1) | link table; events with no equipment have no rows |
| volunteers.csv | volunteer\_id (V001..V040), name, skills (semicolon-separated subset of SKILLS), max\_hours (float) |  |
| registrations.csv | student\_id (S001..S300), event\_id | one row per (student, event); no duplicates |
| past\_attendance.csv | category, year, attendance (int) | total attendance per category per year, 2021-2025 |

The link table `event_equipment.csv` and `past_attendance.csv` are required additions to the five tables named in the brief.

### 2.5 `models.py`

```python
@dataclass
class Problem:
    events: pd.DataFrame; rooms: pd.DataFrame; equipment: pd.DataFrame
    event_equipment: pd.DataFrame; volunteers: pd.DataFrame; registrations: pd.DataFrame
    past_attendance: pd.DataFrame
    event_ids: list[str]; room_ids: list[str]; equip_ids: list[str]
    volunteer_ids: list[str]; student_ids: list[str]
    dur: np.ndarray            # (E,) int, slots
    attendance: np.ndarray     # (E,) int
    priority: np.ndarray       # (E,) int
    room_cap: np.ndarray       # (R,) int
    room_cost: np.ndarray      # (R,) float, per hour
    room_available: np.ndarray # (R,) bool, default all True
    compatible: np.ndarray     # (E,R) bool: type matches AND capacity >= attendance (ignores availability)
    R: np.ndarray              # (S,E) uint8, 1 if student s registered for event e
    equip_need: np.ndarray     # (E,Q) int
    equip_stock: np.ndarray    # (Q,) int
    vol_needed: np.ndarray     # (E,) int
    skill_required: list[str]  # length E
    def with_unavailable_rooms(self, room_ids: list[str]) -> 'Problem': ...  # returns copy, those rooms False

@dataclass
class Schedule:
    room_idx: np.ndarray   # (E,) int, -1 = unplaced
    day: np.ndarray        # (E,) int
    start: np.ndarray      # (E,) int, slot index
    def copy(self) -> 'Schedule': ...
    def to_dataframe(self, problem) -> pd.DataFrame  # columns: event_id, room_id, day, start_slot, end_slot (placed events only)
    @staticmethod
    def from_dataframe(problem, df) -> 'Schedule'
    @staticmethod
    def empty(n_events) -> 'Schedule'   # all room_idx = -1, day = 0, start = 0
```

Volunteer assignments are a separate DataFrame with columns `event_id, volunteer_id` (one row per assigned volunteer).

### 2.6 Rule codes and violation row granularity

`validate()` returns a DataFrame with columns `rule, event_id, other_event_id, room_id, resource_id, day, slot, detail` (unused cells are empty string or NaN). One row per item below.

| Rule code | Hard? | One row per | Trigger |
| --- | --- | --- | --- |
| UNPLACED | hard | event | room\_idx == -1 |
| ROOM\_DOUBLE\_BOOKED | hard | unordered event pair | same room, overlap |
| ROOM\_TOO\_SMALL | hard | event | capacity < expected\_attendance |
| WRONG\_ROOM\_TYPE | hard | event | room\_type != required\_room\_type |
| EQUIPMENT\_OVER\_STOCK | hard | (equipment, day, slot) | summed qty of events active in that slot > stock; `event_id` holds comma-joined active events |
| OUT\_OF\_DAY | hard | event | start + dur > SLOTS\_PER\_DAY, or start < 0, or day not in 0..N\_DAYS-1 |
| ROOM\_UNAVAILABLE | hard | event | room\_available is False |
| VOLUNTEER\_DOUBLE\_BOOKED | volunteer | (volunteer, unordered event pair) | same volunteer, overlap |
| VOLUNTEER\_SKILL\_MISMATCH | volunteer | (volunteer, event) | volunteer lacks event required\_skill |
| VOLUNTEER\_OVER\_HOURS | volunteer | volunteer | assigned hours (sum dur/2) > max\_hours |
| VOLUNTEER\_UNDERSTAFFED | volunteer | event | assigned count < volunteers\_needed |

The first seven are 'hard' and are counted by the cost function. The four VOLUNTEER\_ rules are evaluated only when `assignments` is passed and are reported as KPIs, not in the schedule cost.

Sort order of the output: by rule (order in table), then event\_id.

### 2.7 Public API (all roles code against exactly these)

```python
# data_io.py (R1)
load_problem(data_dir: str = 'data') -> Problem
# generator.py (R1)
generate_all(seed=42, out_dir='data', n_events=25, n_rooms=10, n_students=300, n_volunteers=40) -> dict[str, pd.DataFrame]
# constraints.py (R1)
RULES: tuple[str, ...]; HARD_RULES: tuple[str, ...]
validate(problem, schedule, assignments=None) -> pd.DataFrame
summarize(violations: pd.DataFrame) -> dict[str, int]   # every rule code present, 0 if none
# cost.py (R3)
clash_matrix(problem) -> np.ndarray                      # (E,E) int, C = R.T @ R
overlap_matrix(problem, schedule) -> np.ndarray          # (E,E) bool, zero diagonal, unplaced events never overlap
hard_violation_count(problem, schedule) -> int
cost_breakdown(problem, schedule) -> dict[str, float]    # keys: clash, room, waste, gap, late, hard, total (all weighted) plus raw_ keys
total_cost(problem, schedule) -> float
compute_kpis(problem, schedule, assignments=None) -> dict[str, float]
comparison_table(kpis_before: dict, kpis_after: dict) -> pd.DataFrame
# resources.py (R3)
assign_volunteers(problem, schedule) -> tuple[pd.DataFrame, dict]   # (assignments, info{'unfilled': int})
allocate_equipment(problem, schedule) -> pd.DataFrame     # event_id, equipment_id, qty, status ('OK'|'SHORTAGE')
volunteer_workload(problem, assignments) -> pd.DataFrame  # volunteer_id, n_events, hours
# forecast.py (R3)
forecast_attendance(past: pd.DataFrame, target_year=2026, degree=1) -> pd.DataFrame
# scheduler.py (R2)
schedule_baseline(problem) -> Schedule
schedule_greedy(problem) -> Schedule
schedule_annealing(problem, init=None, n_iter=20000, t0=50.0, t_end=0.05, seed=42) -> tuple[Schedule, list[float]]
schedule_genetic(problem, pop_size=60, generations=300, seed=42) -> tuple[Schedule, list[float]]
plot_convergence(histories: dict[str, list[float]], path=None) -> matplotlib.figure.Figure
```

`cost.total_cost` must be fast (vectorized NumPy, no pandas inside the loop). Target: 20,000 calls in under 20 s on an ordinary laptop.

## 3. Build order

| Phase | What | Who | Gate to leave the phase |
| --- | --- | --- | --- |
| 0: Contracts | repo, `config.py`, `models.py`, stub files with the 2.7 signatures raising NotImplementedError, `requirements.txt`, CI | R4 repo; R1 writes config/models/helpers | everyone can `import` every module; `pytest` runs (tests skipped or xfail) |
| 1: Parallel build | each role implements own files against the contracts. R1 delivers generator and `load_problem` first (within Phase 1) because everyone needs `data/` | R1, R2, R3, R4 | each role's tests pass on its own branch |
| 2: Integration | merge in this order: R1, R3, R2, R4. Fix imports, run `main.py`, run all tests | R4 | end-to-end test passes |
| 3: Polish | README, REPORT.md, DEMO\_SCRIPT.md, tuning of SA parameters, screenshots | R4 with all | acceptance checklist in Section 9 fully ticked |

While waiting for others, R2 and R4 build against stubs and the committed `data/` folder (R1 commits `data/` generated with seed 42 on day 1 of Phase 1).

## 4. ROLE 1: Data + Constraints

**Files:** `generator.py`, `data_io.py`, `constraints.py`, `config.py`, `models.py`, `tests/helpers.py`, `tests/conftest.py`, `tests/test_constraints.py`.

### 4.1 `generator.py`

CLI: `python generator.py --seed 42 --out data` writes all seven CSVs. Uses `np.random.default_rng(seed)`.

**Rooms: fixed table (not random)**

| room\_id | name | room\_type | capacity | cost\_per\_hour |
| --- | --- | --- | --- | --- |
| R01 | Main Auditorium | auditorium | 300 | 800 |
| R02 | Mini Auditorium | auditorium | 150 | 500 |
| R03 | Open Air Theatre | open\_air | 250 | 300 |
| R04 | Classroom A | classroom | 80 | 150 |
| R05 | Classroom B | classroom | 60 | 120 |
| R06 | Classroom C | classroom | 60 | 120 |
| R07 | Classroom D | classroom | 40 | 100 |
| R08 | Computer Lab 1 | lab | 50 | 250 |
| R09 | Computer Lab 2 | lab | 30 | 200 |
| R10 | Seminar Hall | seminar | 100 | 250 |

If `n_rooms != 10`, take the first n rows of this table (cycling with new ids if n > 10).

**Equipment (fixed):** EQ01 Projector stock 5; EQ02 Microphone 6; EQ03 Sound System 2; EQ04 Laptop 8; EQ05 Camera 2.

**Events (random, seeded):**

- Category chosen uniformly from CATEGORIES. Room type by category with probabilities: Technical (lab 0.5, classroom 0.5); Cultural (auditorium 0.5, open\_air 0.5); Workshop (lab 0.5, seminar 0.5); Talk (auditorium 0.4, seminar 0.3, classroom 0.3); Competition (classroom 0.5, lab 0.2, open\_air 0.3).
- `duration_slots` uniform integer 2 to 6; `priority` uniform 1 to 5; `volunteers_needed` uniform 1 to 4; `required_skill` uniform from SKILLS; names like `'Hackathon 1'` built from category plus counter.

**Registrations (drives expected\_attendance):** each of 300 students registers for a random 2 to 4 distinct events, sampled with event popularity weights drawn from a Dirichlet(alpha=1.0) over events. Then set `expected_attendance = ceil(registered_count * U(1.0, 1.2))`, minimum 10. If an event's attendance would exceed the capacity of the largest room of its required type, randomly drop registrations of that event until `registered_count * 1.2 <= max capacity`, then recompute attendance. This guarantees every event has a compatible room.

**Event equipment:** each event gets 0 to 3 distinct equipment items (probabilities 0.3, 0.3, 0.25, 0.15), qty uniform 1 to 2. Ensure qty <= stock.

**Volunteers:** 40 volunteers, each 1 to 3 skills (random distinct), `max_hours` uniform in {6, 7, 8, 9, 10}. Repair step: if any skill has fewer than 6 volunteers, add that skill to random volunteers until it has 6.

**Past attendance:** for each category and each year in PAST\_YEARS: `base * (1 + g*(year-2021)) + noise`, with base in 150-400, growth `g` in 0.03-0.12, noise normal with sd 5% of base; integer, minimum 0.

**Determinism:** two runs with the same seed give byte-identical CSVs.

### 4.2 `data_io.py`: `load_problem(data_dir)`

Read the seven CSVs with pandas and clean:

1. Strip whitespace in all string columns; upper-case ids.
2. Drop exact duplicate rows (and duplicate (student\_id, event\_id) pairs); log how many.
3. Coerce numeric columns; rows with unparseable numerics are dropped and logged.
4. Drop rows in link tables whose foreign key does not exist; log how many.
5. Raise `DataError` (define in `data_io.py`) if: a required column is missing; an event has no compatible room; a required file is missing.
6. Sort each table by id, build all arrays of `Problem` per 2.5. `R` must be shape (n\_students, n\_events) with students in `student_ids` order.

### 4.3 `constraints.py`: `validate`

Implements every rule in 2.6 exactly, with the stated row granularity. Pure function; does not mutate its inputs. Must work with unavailable rooms (via `problem.room_available`), with unplaced events, and with `assignments=None`. `summarize` returns a dict with every code in `RULES` as key.

### 4.4 Tests (`test_constraints.py`, each test builds a valid schedule with `serial_schedule(problem)` then breaks exactly one thing)

`serial_schedule(problem)` in `tests/helpers.py`: places events one after another (no two events at the same time anywhere), each in its first compatible available room, moving to the next day if it would run past 21:00. This schedule must produce zero violations.

Required tests: (1) valid serial schedule gives empty DataFrame; (2) room double-booked: move two events to the same room and time, expect exactly one ROOM\_DOUBLE\_BOOKED row naming both; (3) too small: assign an event to a room with capacity below attendance; (4) wrong type; (5) equipment over stock: put events needing the same equipment at the same time (different rooms) so the sum exceeds stock, expect rows for exactly the overlapping slots; (6) out of day: start = SLOTS\_PER\_DAY - dur + 1; (7) unplaced; (8) room unavailable via `with_unavailable_rooms`; (9) volunteer in two places: hand-built assignments with one volunteer on two overlapping events; (10) skill mismatch; (11) over hours; (12) understaffed; (13) back-to-back events in the same room are NOT flagged; (14) `summarize` has all keys; (15) `load_problem` cleaning: write a messy tmp CSV set (duplicates, whitespace, lowercase ids, orphan registration) and assert the cleaned counts; (16) generator determinism and feasibility (every event has at least one compatible room, equipment qty <= stock, each skill has >= 6 volunteers); (17) default generated sizes: 25 events, 10 rooms, 300 students.

Fixtures in `conftest.py`: `small_problem` = `generate_all(seed=7, n_events=6, n_rooms=4, n_students=30, n_volunteers=8)` written to a tmp dir and loaded; `default_problem` = `load_problem('data')`.

**Done when:** all tests above pass; `data/` committed; `python generator.py` regenerates identical files.

## 5. ROLE 2: Scheduling Algorithms

**Files:** `scheduler.py`, `tests/test_scheduler.py`. Depends on `cost.total_cost` and `models`. Until R3 is merged, use a temporary local stub of `total_cost` in the test file only; never inside `scheduler.py`.

All schedulers return a `Schedule` in which every event is placed. All must respect `problem.room_available`.

### 5.1 Baseline: `schedule_baseline` (the 'Before')

First-fit. Iterate events in `event_id` order. For each, scan day 0 then day 1, start slot 0 upward, rooms in `room_id` order; take the first (day, start, room) where: room is available, `compatible[e, r]` is True, the event fits inside the day, and the room is free for the whole interval. It ignores student clashes, equipment, cost, priority and late slots (that is why it is the baseline). If no placement exists, raise `RuntimeError`.

### 5.2 Greedy: `schedule_greedy`

1. Order events by (priority desc, expected\_attendance desc, duration desc, event\_id asc).
2. For each event, enumerate every candidate (room, day, start) with: room available and compatible, fits inside day, room free, and equipment stock not exceeded given events already placed.
3. Evaluate each candidate by `total_cost` of the partial schedule, calling the cost function with `partial=True` (unplaced events have `room_idx == -1` and are ignored entirely, see 6.2). Pick the lowest; tie-break by lower day, lower start, lower room index.
4. If no candidate satisfies all hard rules, relax equipment only (choose the min-cost candidate ignoring equipment) and log a warning.

### 5.3 Simulated annealing: `schedule_annealing`

- Start from `init` or, if None, `schedule_greedy(problem)`.
- Cost = `cost.total_cost` (hard violations are penalized by W\_HARD, so infeasible moves are rejected naturally).
- Temperature: geometric, `T_k = t0 * (t_end/t0) ** (k/(n_iter-1))`.
- Each iteration pick one move uniformly at random:
  - **MOVE\_TIME** (prob 0.4): pick event, random day, random start in `[0, SLOTS_PER_DAY - dur]`.
  - **MOVE\_ROOM** (prob 0.3): pick event, random room among `compatible` and available rooms (if only one, fall back to MOVE\_TIME).
  - **SWAP** (prob 0.3): pick two events, swap their (room, day, start); skip if either room is not compatible with the other event or either would run past the day.
- Accept if `delta < 0` or `rng.random() < exp(-delta / T)`.
- Track best-so-far. Return `(best_schedule, history)` where `history` has length `n_iter + 1` and `history[k]` is the best cost after k iterations (monotone non-increasing; `history[0]` is the initial cost).
- Deterministic for a given seed. Must not mutate `init`.

### 5.3b Genetic algorithm (bonus): `schedule_genetic`

Individual = a Schedule. Initial population: one greedy schedule plus `pop_size - 1` random placements (random compatible room, random day/start). Fitness = `total_cost`. Tournament selection (k = GA\_TOURNAMENT\_K). Uniform crossover per event gene (room, day, start together). Mutation: each gene with probability GA\_MUT\_RATE gets MOVE\_TIME or MOVE\_ROOM. Elitism GA\_ELITE. `history[g]` = best cost after generation g (length generations + 1). For a fair comparison, `main.py` and the dashboard run SA with `n_iter = pop_size * generations` so both use a similar number of cost evaluations; the comparison is reported in the dashboard and report.

### 5.4 `plot_convergence(histories, path=None)`

matplotlib line chart, x = iteration (or generation), y = cost, one line per key (e.g. 'Annealing', 'Genetic'), legend, axis labels, title 'Convergence'. Saves to `path` if given. Returns the Figure.

### 5.5 Tests (`test_scheduler.py`)

(1) baseline places all events and has zero ROOM\_DOUBLE\_BOOKED, ROOM\_TOO\_SMALL, WRONG\_ROOM\_TYPE, OUT\_OF\_DAY (use `constraints.validate`); (2) greedy has zero hard violations on `default_problem`; (3) on `default_problem` seed 42: `total_cost(SA) <= total_cost(greedy) <= total_cost(baseline)`; (4) SA history is non-increasing, has length n\_iter + 1, and final value equals `total_cost(best)`; (5) SA is deterministic (two runs, same seed, identical arrays); (6) with `with_unavailable_rooms(['R01'])` no scheduler places anything in R01; (7) SA does not mutate `init`; (8) GA returns a fully placed schedule with zero hard violations and non-increasing history; (9) `plot_convergence` returns a Figure and writes a file when `path` is given. Use small `n_iter` (e.g. 2000) in tests.

**Done when:** tests pass and on the default data SA reaches student clashes <= 10% of baseline clashes (target: 0) and zero hard violations. If not met, tune move probabilities or t0 and note it in `DECISIONS.md`.

## 6. ROLE 3: Cost + Resources + Analytics

**Files:** `cost.py`, `resources.py`, `forecast.py`, `tests/test_resources.py`, `tests/test_cost.py`.

### 6.1 Clash matrix

`clash_matrix(problem)` returns `problem.R.T @ problem.R` as an (E,E) int array (cast R to int32 before multiplying to avoid uint8 overflow). Diagonal = event registration count; symmetric. `overlap_matrix(problem, schedule)` builds the (E,E) boolean overlap per 2.2 with vectorized NumPy broadcasting (same day, `start_i < end_j` and `start_j < end_i`, both placed, diagonal False). Compute `C` once and cache it on the problem (lazily, e.g. `problem._C`) so the cost function never recomputes it.

### 6.2 Cost function

```
total = W_CLASH * clash + W_ROOM * room + W_WASTE * waste + W_GAP * gap + W_LATE * late + W_HARD * hard
clash = sum over i<j of C[i,j] * overlap[i,j]
room  = sum over placed events of room_cost[r] * dur / 2
waste = sum over placed events of max(0, room_cap[r] - attendance)
gap   = idle gap slots per 2.2
late  = number of placed events with start >= LATE_START_SLOT
hard  = hard_violation_count(problem, schedule)
```

`cost_breakdown` returns the weighted terms plus `raw_clash, raw_room, raw_waste, raw_gap, raw_late, raw_hard` (unweighted) and `total`. `cost_breakdown` and `total_cost` accept `partial: bool = False`; when True, unplaced events are ignored entirely and are not counted in `hard` (used by greedy).

`hard_violation_count` counts exactly the seven hard rules of 2.6 with the same row granularity as `constraints.validate`. **Consistency test:** for random schedules, `hard_violation_count(p, s)` equals the number of rows in `validate(p, s)` whose rule is in `HARD_RULES`.

### 6.3 Volunteers: `assign_volunteers(problem, schedule)`

Greedy with load balancing. Process events sorted by (day, start). For each, candidates = volunteers who have `required_skill`, are not already assigned to an overlapping event, and whose assigned hours + event hours (`dur/2`) <= `max_hours`. Pick the `volunteers_needed` candidates with the lowest current hours (tie: lower volunteer\_id). If fewer candidates than needed, assign all candidates and add the shortfall to `info['unfilled']` (number of missing volunteer positions). Returns `(assignments_df, {'unfilled': int})`. The result must produce zero VOLUNTEER\_DOUBLE\_BOOKED, VOLUNTEER\_SKILL\_MISMATCH and VOLUNTEER\_OVER\_HOURS violations; VOLUNTEER\_UNDERSTAFFED rows correspond to events with a shortfall.

`volunteer_workload(problem, assignments)` returns one row per volunteer (including those with zero events) with `n_events` and `hours`. Workload evenness KPI = `volunteer_hours_std`, the standard deviation of `hours` across all volunteers.

### 6.4 Equipment: `allocate_equipment(problem, schedule)`

Process events by (priority desc, day, start). For each (event, equipment, qty) need, check remaining stock for every slot of the event; if enough in all slots, allocate with `status='OK'` and reserve the stock for those slots; else `status='SHORTAGE'` and reserve nothing. Never allocate beyond stock in any slot. For a schedule with zero hard violations there must be zero SHORTAGE rows.

### 6.5 Forecast: `forecast_attendance(past, target_year=2026, degree=1)`

For each category: `np.polyfit(years, attendance, degree)`, `np.polyval` at `target_year`, clamp at 0, round to int. Returns columns `category, forecast, slope, r2, last_actual` (r2 from residuals; if the series has zero variance, r2 = 1.0). If a category has fewer than 2 data points, return its last value with slope 0. The forecast is informational (dashboard and report); it does NOT feed the scheduler.

### 6.6 KPIs and comparison table

`compute_kpis(problem, schedule, assignments=None)` returns exactly these keys:

| Key | Definition | Better is |
| --- | --- | --- |
| total\_cost | `total_cost` | lower |
| student\_clashes | raw\_clash | lower |
| clashing\_pairs | overlapping pairs with C > 0 | lower |
| hard\_violations | raw\_hard | lower |
| room\_cost\_total | raw\_room (currency) | lower |
| wasted\_seats | raw\_waste | lower |
| idle\_gap\_hours | raw\_gap / 2 | lower |
| late\_events | raw\_late | lower |
| room\_utilization\_pct | occupied room-slots / (slots inside the active span of every room-day that has an event) \* 100; active span = first start to last end | higher |
| seat\_utilization\_pct | sum(attendance \* dur) / sum(room\_cap \* dur) \* 100 over placed events | higher |
| venue\_utilization\_pct | occupied room-slots / (available rooms \* N\_DAYS \* SLOTS\_PER\_DAY) \* 100 (informational) | n/a |
| rooms\_used | distinct rooms with at least one event | n/a |
| volunteer\_hours\_std | std of volunteer hours (NaN if no assignments given) | lower |
| unfilled\_volunteer\_slots | missing volunteer positions, computed from assignments (NaN if none given) | lower |

`comparison_table(before, after)` returns a DataFrame indexed by KPI key with columns `Before, After, Change, Change %, Better`. Change = After - Before; Change % = Change / Before \* 100 (NaN if Before == 0); Better is `'Yes'`, `'No'` or `'-'` (equal or n/a) using the direction column above.

### 6.7 Tests

`test_cost.py`: clash matrix equals a manual double loop on `small_problem`; diagonal equals registration counts; hand-built 3-event example with a known value (two overlapping events sharing exactly 2 students give raw\_clash == 2; non-overlapping gives 0; back-to-back gives 0); each of the room, waste, gap, late terms checked on a hand-computed schedule; hard-count consistency with `validate`; `partial=True` ignores unplaced events; speed test (2,000 calls under 2 s). `test_resources.py`: volunteers never double-booked, always skilled, never over hours; workload std lower than or equal to a naive first-available assignment; shortfall reported when volunteers are scarce (problem with one volunteer); equipment never over stock for any (equipment, slot) even when demand exceeds stock; forecast recovers a perfectly linear series (slope and prediction within 1e-6), clamps at 0, handles fewer than 2 points; KPI keys exactly as in 6.6; comparison table directions and Change %.

**Done when:** all tests pass and `total_cost` meets the speed target in 2.7.

## 7. ROLE 4: Dashboard + Integration

**Files:** `app.py`, `main.py`, `README.md`, `REPORT.md`, `DEMO_SCRIPT.md`, `requirements.txt`, `.gitignore`, `.github/workflows/ci.yml`, `tests/test_end_to_end.py`.

### 7.1 Repo and Git

Create the GitHub repo with `main` protected (PR + passing CI). Branches: `role1-data`, `role2-scheduler`, `role3-cost`, `role4-dashboard`. Conventional commit messages. `requirements.txt` minimums: numpy>=1.26, pandas>=2.1, matplotlib>=3.8, plotly>=5.18, streamlit>=1.30, pytest>=8.0. CI (GitHub Actions): Python 3.11, `pip install -r requirements.txt`, `pytest -q`. `.gitignore`: `outputs/`, `__pycache__/`, `.venv/`, `.pytest_cache/`.

### 7.2 `main.py` (CLI)

`python main.py [--seed 42] [--data data] [--algo sa|ga|greedy] [--iterations 20000] [--unavailable R03,R05] [--compare] [--out outputs]`

Steps: (1) if the `--data` folder is missing or empty, call `generate_all`; (2) `load_problem`; (3) apply `with_unavailable_rooms` if given; (4) baseline schedule = Before; (5) run the chosen algorithm (default `sa`, initialised from greedy) = After; (6) `assign_volunteers` and `allocate_equipment` for both schedules; (7) `validate` both (with assignments); (8) `compute_kpis` for both and `comparison_table`; (9) write to `outputs/`: `schedule_before.csv`, `schedule_after.csv`, `assignments_after.csv`, `equipment_after.csv`, `kpi_comparison.csv`, `violations_before.csv`, `violations_after.csv`, `convergence.png` (SA curve, plus GA curve if `--algo ga` or `--compare`); (10) print the comparison table and output paths. Exit code 1 if the After schedule has any hard violation, else 0.

Also expose `run_pipeline(seed=42, data_dir='data', algo='sa', iterations=20000, unavailable=(), compare=False) -> dict` that `app.py` and the end-to-end test call. It returns keys: `problem, before, after, assign_before, assign_after, equip_after, violations_before, violations_after, kpis_before, kpis_after, comparison, histories, forecast`.

### 7.3 `app.py` (Streamlit, run with `streamlit run app.py`)

Use `st.set_page_config(layout='wide')` and cache heavy calls with `st.cache_data` or `st.session_state`. Layout:

- **Sidebar:** seed (number input), algorithm (Greedy / Simulated Annealing / Genetic), iterations slider (1000 to 50000, default 20000), 'Compare SA vs GA' checkbox, 'Rooms unavailable (what-if)' multiselect of room ids, a primary **Optimize** button (runs with no unavailable rooms) and a **Re-optimize with what-if** button (runs with the selected unavailable rooms). Show `st.spinner` while running. Before anything has run, show 'Click Optimize to start' and nothing else.
- **KPI cards (top row, `st.metric` with delta):** Total cost, Student clashes, Room utilization %, Hard violations, Late events. Delta = After - Before; `delta_color='inverse'` for lower-is-better metrics.
- **Tabs:**
  1. *Timetable*: radio Before / After and a day selector (Day 1, Day 2); `plotly.express.timeline` Gantt with y = room, color = event category, hover showing event, attendance and times. Build datetimes from `BASE_DATE` + day + slot minutes.
  2. *Utilization heatmap*: `plotly.graph_objects.Heatmap`, rows = rooms, columns = 48 time steps (2 days x 24 slots) with time labels, value = 1 if occupied; Before and After side by side. Unavailable rooms get the label suffix ' (down)'.
  3. *Comparison*: styled `comparison_table` plus a bar chart of the cost breakdown (clash, room, waste, gap, late) Before vs After.
  4. *Convergence*: cost vs iteration; both SA and GA curves when GA is selected or Compare is ticked.
  5. *Violations*: Before and After validator tables (`st.dataframe`) with counts per rule at the top.
  6. *Resources*: volunteer workload bar chart (After), equipment allocation table, unfilled volunteer slots metric.
  7. *Forecast*: table from `forecast_attendance` and a chart of past vs forecast per category, next to this year's registered totals per category.
- **What-if behaviour:** the re-optimized result replaces the After views; a banner shows 'What-if: R03, R05 unavailable'. No event may appear in an unavailable room; the app checks this and shows `st.error` if violated.

### 7.4 End-to-end test (`tests/test_end_to_end.py`)

Generate data into a tmp dir with seed 42, call `run_pipeline` with 2000 iterations and assert: After has zero hard violations; After total cost < Before total cost; After student clashes <= Before; no VOLUNTEER\_DOUBLE\_BOOKED, SKILL\_MISMATCH or OVER\_HOURS rows; zero SHORTAGE rows in equipment; all output files exist; `python main.py --iterations 1000` exits with code 0 (via `subprocess`). Second case: `--unavailable R01,R02` leaves all events placed and none in R01 or R02.

### 7.5 Documents

- **README.md:** purpose, architecture (text diagram), setup (`python -m venv`, `pip install -r requirements.txt`), commands (`python generator.py`, `python main.py`, `streamlit run app.py`, `pytest -q`), data schema summary, the definitions in 2.2, team roles, screenshots.
- **REPORT.md:** problem statement; data design; constraints and cost function with weights; algorithms (baseline, greedy, SA, GA) with parameters; results (paste the real comparison table and convergence chart from `outputs/`); SA vs GA comparison; forecasting; limitations (30-minute slots, no travel time, fixed two days); future work.
- **DEMO\_SCRIPT.md:** a 5-minute script: (1) problem in 30 seconds; (2) Before timetable with its clashes and violations; (3) click Optimize and narrate the KPI cards; (4) After timetable and heatmap; (5) convergence tab; (6) what-if: make Main Auditorium (R01) unavailable and re-optimize; (7) resources and forecast; (8) close with the headline numbers.

**Done when:** a fresh clone runs `pip install -r requirements.txt && python main.py && streamlit run app.py` without edits, and CI is green.

## 8. Quality and consistency rules

- Code style: PEP 8, max line length 100, `black`-compatible formatting.
- Pandas only for I/O and display. Anything inside the optimization loop is NumPy.
- Never mutate a `Problem` or an input `Schedule` (use `.copy()`).
- Log with `logging` (INFO for summaries, WARNING for dropped or relaxed items).
- All file paths via `pathlib.Path`; no absolute paths.
- Charts must have titles, axis labels and legends; colors are consistent per category across all charts.
- Weights and constants change only in `config.py`; every change is recorded in `DECISIONS.md` with the reason and the resulting KPIs.

## 9. Final acceptance checklist

1. `pytest -q` passes with 0 failures and 0 errors on a fresh clone; CI is green.
2. `data/` contains 7 CSVs: 25 events, 10 rooms, 5 equipment types, 40 volunteers, 300 students; regeneration with seed 42 is identical.
3. `validate` detects all 11 rule codes, each proven by a test.
4. On default data (seed 42): baseline has student clashes > 0; greedy has 0 hard violations; `cost(SA) <= cost(greedy) <= cost(baseline)`; SA student clashes <= 10% of baseline clashes; SA has 0 hard violations.
5. After schedule: no volunteer double-booking, skill mismatch or over-hours; workload std reported; zero equipment SHORTAGE rows.
6. `python main.py` prints the comparison table, writes every file listed in 7.2 and exits 0.
7. Dashboard: Optimize button, Gantt timetable, heatmap, KPI cards, convergence, violations, resources and forecast tabs all render; the what-if re-optimization respects unavailable rooms.
8. Forecast uses `np.polyfit`; clash matrix uses `R.T @ R`.
9. README, REPORT.md, DEMO\_SCRIPT.md and DECISIONS.md exist and contain real results, not placeholders.
10. Bonus (optional): GA implemented and compared against SA in the dashboard and report.
