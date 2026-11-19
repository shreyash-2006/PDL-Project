# EventOptima: Technical & Empirical Evaluation Report

## 1. Executive Summary & Problem Statement

College fests involve complex multi-facility scheduling across competing academic, cultural, and technical events. Scheduling problems are classically NP-hard due to intersecting discrete constraints:
- Room capacity and facility requirements (e.g. computer labs vs auditoriums).
- Limited shared technical inventory (projectors, cameras, sound consoles).
- Volunteer staffing capacities and skill qualifications.
- Student registration clashes: maximizing student attendance by minimizing timetable overlap between events that share common registrants.

**EventOptima** implements an end-to-end mathematical optimization pipeline that schedules 25 multi-category events across 10 campus venues over 2 fest days (48 half-hour slots). The system optimizes venue expenditures, seat utilization, timetable gaps, and eliminates student timetable clashes using metaheuristic Simulated Annealing and Genetic Algorithms.

---

## 2. Data Architecture & Design

The problem state is modeled across seven interconnected relational datasets:

1. **Events ($E=25$):** Defined with required room type (auditorium, classroom, lab, open_air, seminar), duration (2 to 6 slots, i.e. 1 to 3 hours), priority rating (1 to 5), expected attendance, volunteer demand (1 to 4), and required volunteer skill.
2. **Rooms ($R=10$):** Fixed campus venue inventory ranging from large 300-capacity auditoriums to specialized 30-capacity computer labs with hourly rental tariffs.
3. **Equipment Inventory ($Q=5$):** Fixed stock limits for Projectors (5), Microphones (6), Sound Systems (2), Laptops (8), and Cameras (2).
4. **Event Equipment Demands:** Link table specifying per-event equipment quantities.
5. **Volunteer Pool ($V=40$):** Volunteers with certified skill sets and maximum allowable working hours ($\in \{6, 7, 8, 9, 10\}$ hrs).
6. **Student Registrations ($S=300$):** 300 students registering for 2 to 4 events sampled using a Dirichlet distribution ($\alpha = 1.0$) over event popularities.
7. **Past Attendance:** Longitudinal records from 2021 to 2025 across all 5 event categories.

---

## 3. Constraints & Objective Formulation

### 3.1 Hard Constraints
Hard constraints are strictly enforced or heavily penalized with weight $W_{\text{HARD}} = 10,000$:
1. **UNPLACED:** Every event must be placed in a room.
2. **ROOM_DOUBLE_BOOKED:** No two events may occupy the same room at overlapping time intervals.
3. **ROOM_TOO_SMALL:** Venue capacity $\ge$ expected event attendance.
4. **WRONG_ROOM_TYPE:** Room category must match event requirement.
5. **EQUIPMENT_OVER_STOCK:** Aggregate equipment usage across concurrent events in any slot must not exceed total stock.
6. **OUT_OF_DAY:** Events must fit strictly within daily operating hours ($09:00 - 21:00$, slots $0 \le s \le s + d \le 24$).
7. **ROOM_UNAVAILABLE:** Events must not be scheduled in rooms marked unavailable.

### 3.2 Soft Objectives & Cost Function
The global scalar objective function is defined as:
$$\text{Cost} = W_{\text{CLASH}} \cdot \text{Clash} + W_{\text{ROOM}} \cdot \text{RoomCost} + W_{\text{WASTE}} \cdot \text{Waste} + W_{\text{GAP}} \cdot \text{Gap} + W_{\text{LATE}} \cdot \text{Late} + W_{\text{HARD}} \cdot \text{Hard}$$

| Term | Weight | Definition |
| :--- | :--- | :--- |
| **Clash** | $W_{\text{CLASH}} = 20.0$ | Sum of shared student registrations over overlapping event pairs: $\sum_{i < j} C_{ij} \cdot \text{overlap}_{ij}$ |
| **RoomCost** | $W_{\text{ROOM}} = 0.1$ | Total venue expenditure: $\sum_e \text{cost\_per\_hour} \times \frac{\text{dur}_e}{2}$ |
| **Waste** | $W_{\text{WASTE}} = 0.5$ | Wasted seat capacity: $\sum_e \max(0, \text{Capacity}_r - \text{Attendance}_e)$ |
| **Gap** | $W_{\text{GAP}} = 3.0$ | Idle room gap slots between consecutive events in the same room |
| **Late** | $W_{\text{LATE}} = 40.0$ | Number of events starting at or after slot 18 (18:00) |
| **Hard** | $W_{\text{HARD}} = 10000.0$ | Number of hard rule violations |

---

## 4. Optimization Algorithms & Methodology

### 4.1 Baseline First-Fit
Constructs an initial naive schedule by scanning days ($0 \to 1$), slots ($0 \to 24$), and rooms ($R01 \to R10$) sequentially. It satisfies basic room availability and room compatibility but ignores student clashes, equipment stock, priority, and venue costs. This provides the unoptimized "Before" benchmark.

### 4.2 Greedy Heuristic
Orders events by priority descending, expected attendance descending, duration descending, and event ID ascending. For each event, enumerates all feasible $(d, s, r)$ candidate placements and selects the one minimizing the partial schedule cost while verifying equipment capacity.

### 4.3 Simulated Annealing (SA)
- **Initialization:** Starts from the greedy schedule.
- **Cooling Schedule:** Geometric cooling $T_k = T_0 \cdot \left(\frac{T_{\text{end}}}{T_0}\right)^{\frac{k}{K-1}}$ with $T_0 = 50.0$, $T_{\text{end}} = 0.05$, and $K = 20,000$ iterations.
- **Neighborhood Operators:**
  1. `MOVE_TIME` ($p=0.4$): Shift an event to a new random day and start slot.
  2. `MOVE_ROOM` ($p=0.3$): Reassign an event to an alternative compatible available venue.
  3. `SWAP` ($p=0.3$): Swap the room, day, and slot of two compatible events.
- **Acceptance:** Metropolis acceptance rule $P = \exp(-\Delta / T)$ for uphill moves.

### 4.4 Genetic Algorithm (GA)
- **Population & Generations:** Population size $N = 60$, generations $G = 300$, tournament size $k = 3$, elitism $E = 2$.
- **Crossover & Mutation:** Uniform event-gene crossover with independent mutation probability $p_{\text{mut}} = 0.05$.

---

## 5. Empirical Results & Performance Comparison

The complete pipeline was evaluated on the benchmark fest dataset (`seed = 42`, 20,000 iterations):

### 5.1 Before vs After KPI Comparison Table

| Metric | Before (Baseline) | After (Optimized SA) | Absolute Change | Relative Change (%) | Improved? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Cost** | 213,158.0 | 2,905.0 | -210,253.0 | **-98.64%** | **Yes** |
| **Student Clashes** | 515.0 | 28.0 | -487.0 | **-94.56%** | **Yes** |
| **Clashing Pairs** | 95.0 | 10.0 | -85.0 | **-89.47%** | **Yes** |
| **Hard Violations** | 20.0 | 0.0 | -20.0 | **-100.00%** | **Yes** |
| **Room Rental Cost** | 16,915.0 | 14,525.0 | -2,390.0 | **-14.13%** | **Yes** |
| **Wasted Seats** | 2,333.0 | 1,633.0 | -700.0 | **-30.00%** | **Yes** |
| **Seat Utilization %** | 33.65% | 41.24% | +7.59% | **+22.56%** | **Yes** |
| **Venue Utilization %**| 21.67% | 21.67% | 0.00% | 0.00% | — |
| **Rooms Used** | 10 | 8 | -2 | -20.00% | — |
| **Volunteer Workload Std** | 0.95 hrs | 0.91 hrs | -0.05 hrs | **-5.06%** | **Yes** |
| **Unfilled Volunteer Slots** | 0 | 0 | 0 | 0.0% | — |
| **Equipment Shortages** | 20 (violations) | 0 | -20 | **-100.00%** | **Yes** |

### 5.2 Key Findings
1. **Clash Elimination:** Student registration clashes dropped by 94.6% (from 515 clashes down to 28), ensuring students can attend their registered events.
2. **Hard Violations:** 20 equipment stock over-allocations and room conflicts present in the baseline schedule were completely resolved to **0 hard violations**.
3. **Space & Financial Efficiency:** By consolidating events into 8 rooms instead of 10, room rental expenses dropped by 14.1% ($16,915 \to \$14,525$) and seat utilization increased from 33.6% to 41.2%.
4. **Volunteer Fairness:** Greedy load-balanced volunteer staffing achieved an even distribution across the 40 volunteers with standard deviation of only 0.91 hours and 0 unfilled positions.

---

## 6. SA vs GA Comparative Analysis

| Feature | Simulated Annealing (SA) | Genetic Algorithm (GA) |
| :--- | :--- | :--- |
| **Final Objective Cost** | **2,905.0** | 3,140.0 |
| **Convergence Speed** | Rapid descent in first 3,000 iterations | Slower, steady exploratory improvements |
| **Memory Footprint** | Extremely low (single state copy) | Moderate ($N=60$ individual states) |
| **Search Exploitation** | High fine-tuning via local swaps | High diversity, slower local convergence |
| **Deterministic Consistency** | High with seeded PRNG | High with seeded PRNG |

Simulated Annealing outperformed the Genetic Algorithm on fine-grained constraint tuning because point-based neighborhood operators preserve feasibility better than uniform crossover recombination.

---

## 7. Attendance Forecasting Analytics

Historical attendance (2021–2025) per category was regressed using 1st-degree polynomial trend fitting (`np.polyfit`) to project 2026 fest demands:
- **Technical:** Growth trend +8.4% YoY ($R^2 = 0.94$), 2026 projection: ~340 attendees.
- **Cultural:** Strong upward trajectory ($R^2 = 0.96$), 2026 projection: ~385 attendees.
- **Workshops & Talks:** Consistent linear participation, $R^2 \ge 0.91$.
All models achieved $R^2 > 0.90$, providing reliable intelligence for room sizing and catering quotas.

---

## 8. Limitations & Future Extensions

1. **Discrete Half-Hour Granularity:** Events are quantized into 30-minute blocks without micro-interval changeover buffers.
2. **Intra-Campus Transit Times:** Travel times between remote venues (e.g. Open Air Theatre vs Computer Lab 2) are not explicitly modeled in student clash penalties.
3. **Fixed Two-Day Horizon:** Extension to variable $N$-day festivals with multi-stage preliminary competitions.
4. **Dynamic Attendance Rescheduling:** Future work can incorporate real-time mobile check-in data to rebalance room assignments dynamically during live fest operations.
