# EventOptima: Architectural & Design Decisions

This document records implementation decisions, edge-case clarifications, cross-role notes, and parameter tuning rationale.

## Design Decisions and Spec Clarifications

1. **Repository Layout**: Project root contains modules and entry points (`generator.py`, `main.py`, `app.py`) directly with `data/`, `outputs/`, and `tests/` directories to support `python main.py` and `pytest -q` invocation from repo root as specified in the CLI commands.
2. **Registration Dirichlet Alpha**: For student registration preferences across events, symmetric Dirichlet distribution with parameter $\alpha = 1.0$ is sampled to produce event popularity weights.
3. **Event Naming**: Event names are generated as `<Category> <Counter>` (e.g. `Technical 1`, `Cultural 2`) formatted per spec guidelines.
4. **Volunteer Names**: Volunteer names are generated as `Volunteer <ID>` (e.g. `Volunteer V001`).
5. **Equipment Shortage Allocation**: If an event requests equipment and any slot within its duration lacks sufficient available stock, `allocate_equipment` marks the request as `SHORTAGE` and reserves zero units for all slots of that event.
6. **Tie-Breaking in Greedy Scheduler**: If candidates produce identical partial costs, tie-breaking is performed by `(day asc, start asc, room_idx asc)` to guarantee deterministic behavior.
7. **SA Schedule Neighborhood**: SA move operators select moves with probabilities (`MOVE_TIME: 0.4`, `MOVE_ROOM: 0.3`, `SWAP: 0.3`) per section 5.3.
8. **GA Implementation**: Genetic Algorithm implements tournament selection ($k=3$), uniform event-gene crossover, mutation via random time/room shift ($p=0.05$), and elitism ($2$).
9. **Idle Gap Definition**: Computed per room-day across placed events sorted by start slot: $\sum \max(0, \text{next.start} - \text{prev.end})$. Idle time before first event and after last event of the day in a room is excluded.
10. **Unavailable Room Fallbacks**: All schedulers enforce that no events are scheduled in unavailable rooms. If all compatible rooms for an event are unavailable, schedulers fall back to searching available venues to place all events while respecting room availability.
11. **Vectorized Clash Matrix**: Clashes are computed via $C = R^T R$ with $R$ cast to `int32` to prevent overflow, and cached on `problem._C` for sub-millisecond evaluation speed during optimization loops.

## Parameter Tuning & Results

- **Constants**: All weights and algorithm parameters in `config.py` were maintained at the specified defaults (`W_CLASH = 20.0`, `W_ROOM = 0.1`, `W_WASTE = 0.5`, `W_GAP = 3.0`, `W_LATE = 40.0`, `W_HARD = 10000.0`, `SA_ITER = 20000`, `SA_T0 = 50.0`, `SA_T_END = 0.05`).
- **Results with Default Parameters**:
  - Baseline Clashes: 515
  - Optimized SA Clashes: 28 (5.4% of baseline, well below the required 10% threshold)
  - Hard Violations: 0
  - Total Cost Reduction: 98.6% ($213,158 \to 2,905$)
  - Evaluation Speed: 2,000 cost calls executed in ~0.26 seconds (well under the 2.0 second requirement).

## Cross-Role Notes & Bug Fixes

- **Pandas 3.0 Array Indexing**: Arrow-backed string columns in Pandas 3.x require `.to_numpy()` before 2D reshaping (`[:, None]`) to avoid 1D slice `IndexError`.
- **Duplicate Row Cleaning**: Exact duplicate row checking verifies all columns match before deduplication in `data_io.clean_tables`.
