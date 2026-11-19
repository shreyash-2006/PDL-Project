"""Optimization scheduling algorithms: Baseline first-fit, Greedy, Simulated Annealing, and GA."""

import logging
from pathlib import Path
from typing import Optional
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

import config
from cost import total_cost
from models import Problem, Schedule

logger = logging.getLogger(__name__)


def schedule_baseline(problem: Problem) -> Schedule:
    """Construct a naive first-fit schedule placing events sequentially in compatible free rooms."""
    n_events = len(problem.event_ids)
    n_rooms = len(problem.room_ids)
    sched = Schedule.empty(n_events)

    for e in range(n_events):
        dur = int(problem.dur[e])
        placed = False

        # Pass 1: Try available and compatible rooms
        for d in range(config.N_DAYS):
            for s in range(0, config.SLOTS_PER_DAY - dur + 1):
                for r in range(n_rooms):
                    if not (problem.room_available[r] and problem.compatible[e, r]):
                        continue

                    # Verify room is collision-free for [s, s + dur)
                    collision = False
                    for prev_e in range(e):
                        if sched.room_idx[prev_e] == r and sched.day[prev_e] == d:
                            prev_s = sched.start[prev_e]
                            prev_dur = problem.dur[prev_e]
                            if s < prev_s + prev_dur and prev_s < s + dur:
                                collision = True
                                break
                    if not collision:
                        sched.room_idx[e] = r
                        sched.day[e] = d
                        sched.start[e] = s
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

        # Pass 2: Fallback to any available room if all compatible rooms are unavailable
        if not placed:
            for d in range(config.N_DAYS):
                for s in range(0, config.SLOTS_PER_DAY - dur + 1):
                    for r in range(n_rooms):
                        if not problem.room_available[r]:
                            continue

                        collision = False
                        for prev_e in range(e):
                            if sched.room_idx[prev_e] == r and sched.day[prev_e] == d:
                                prev_s = sched.start[prev_e]
                                prev_dur = problem.dur[prev_e]
                                if s < prev_s + prev_dur and prev_s < s + dur:
                                    collision = True
                                    break
                        if not collision:
                            sched.room_idx[e] = r
                            sched.day[e] = d
                            sched.start[e] = s
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

        if not placed:
            raise RuntimeError(f"No baseline placement available for event '{problem.event_ids[e]}'.")

    return sched


def schedule_greedy(problem: Problem) -> Schedule:
    """Construct a greedy schedule minimizing partial schedule cost with priority-ordered events."""
    n_events = len(problem.event_ids)
    n_rooms = len(problem.room_ids)
    n_equip = len(problem.equip_ids)
    sched = Schedule.empty(n_events)

    # 1. Order events by (priority desc, expected_attendance desc, duration desc, event_id asc)
    events_order = list(range(n_events))
    events_order.sort(
        key=lambda e: (
            -problem.priority[e],
            -problem.attendance[e],
            -problem.dur[e],
            problem.event_ids[e],
        )
    )

    for e in events_order:
        dur = int(problem.dur[e])
        candidates_valid: list[tuple[float, int, int, int]] = []
        candidates_relaxed: list[tuple[float, int, int, int]] = []

        for d in range(config.N_DAYS):
            for s in range(0, config.SLOTS_PER_DAY - dur + 1):
                for r in range(n_rooms):
                    if not (problem.room_available[r] and problem.compatible[e, r]):
                        continue

                    # Room collision check against already placed events
                    room_free = True
                    for other_e in range(n_events):
                        if (
                            other_e != e
                            and sched.room_idx[other_e] == r
                            and sched.day[other_e] == d
                        ):
                            other_s = sched.start[other_e]
                            other_dur = problem.dur[other_e]
                            if s < other_s + other_dur and other_s < s + dur:
                                room_free = False
                                break
                    if not room_free:
                        continue

                    # Equipment check against already placed events
                    equip_ok = True
                    for q in range(n_equip):
                        need = problem.equip_need[e, q]
                        if need > 0:
                            for slot in range(s, s + dur):
                                used = sum(
                                    problem.equip_need[placed_e, q]
                                    for placed_e in range(n_events)
                                    if placed_e != e
                                    and sched.room_idx[placed_e] >= 0
                                    and sched.day[placed_e] == d
                                    and sched.start[placed_e] <= slot < sched.start[placed_e] + problem.dur[placed_e]
                                )
                                if used + need > problem.equip_stock[q]:
                                    equip_ok = False
                                    break
                            if not equip_ok:
                                break

                    # Temporarily place event to evaluate partial cost
                    sched.room_idx[e] = r
                    sched.day[e] = d
                    sched.start[e] = s
                    cost_val = total_cost(problem, sched, partial=True)

                    rec = (cost_val, d, s, r)
                    if equip_ok:
                        candidates_valid.append(rec)
                    candidates_relaxed.append(rec)

        sched.room_idx[e] = -1

        if candidates_valid:
            # Tie-break: lowest cost, lower day, lower start, lower room index
            candidates_valid.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
            best = candidates_valid[0]
            sched.room_idx[e] = best[3]
            sched.day[e] = best[1]
            sched.start[e] = best[2]
        elif candidates_relaxed:
            logger.warning(
                "Relaxing equipment constraints for event '%s' due to stock scarcity.",
                problem.event_ids[e],
            )
            candidates_relaxed.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
            best = candidates_relaxed[0]
            sched.room_idx[e] = best[3]
            sched.day[e] = best[1]
            sched.start[e] = best[2]
        else:
            # Fallback across available rooms even if incompatible type
            avail_rooms = [r for r in range(n_rooms) if problem.room_available[r]]
            candidates_avail: list[tuple[float, int, int, int]] = []
            for d in range(config.N_DAYS):
                for s in range(0, config.SLOTS_PER_DAY - dur + 1):
                    for r in avail_rooms:
                        room_free = True
                        for other_e in range(n_events):
                            if other_e != e and sched.room_idx[other_e] == r and sched.day[other_e] == d:
                                other_s = sched.start[other_e]
                                other_dur = problem.dur[other_e]
                                if s < other_s + other_dur and other_s < s + dur:
                                    room_free = False
                                    break
                        if not room_free:
                            continue
                        sched.room_idx[e] = r
                        sched.day[e] = d
                        sched.start[e] = s
                        cost_val = total_cost(problem, sched, partial=True)
                        candidates_avail.append((cost_val, d, s, r))

            sched.room_idx[e] = -1
            if candidates_avail:
                candidates_avail.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
                best = candidates_avail[0]
                sched.room_idx[e] = best[3]
                sched.day[e] = best[1]
                sched.start[e] = best[2]
            else:
                raise RuntimeError(f"Unable to greedily place event '{problem.event_ids[e]}'.")

    return sched


def schedule_annealing(
    problem: Problem,
    init: Optional[Schedule] = None,
    n_iter: int = config.SA_ITER,
    t0: float = config.SA_T0,
    t_end: float = config.SA_T_END,
    seed: int = config.SEED,
) -> tuple[Schedule, list[float]]:
    """Optimize schedule using Simulated Annealing with geometric cooling and neighbourhood moves."""
    rng = np.random.default_rng(seed)
    n_events = len(problem.event_ids)
    n_rooms = len(problem.room_ids)

    comp_avail_rooms = []
    for e in range(n_events):
        rooms = [r for r in range(n_rooms) if problem.compatible[e, r] and problem.room_available[r]]
        if not rooms:
            rooms = [r for r in range(n_rooms) if problem.room_available[r]]
        comp_avail_rooms.append(rooms)

    current_sched = init.copy() if init is not None else schedule_greedy(problem)
    current_cost = total_cost(problem, current_sched)

    best_sched = current_sched.copy()
    best_cost = current_cost
    history = [best_cost]

    move_types = ["MOVE_TIME", "MOVE_ROOM", "SWAP"]
    move_probs = [0.4, 0.3, 0.3]

    for k in range(n_iter):
        T = t0 * ((t_end / t0) ** (k / max(1, n_iter - 1)))
        move = rng.choice(move_types, p=move_probs)

        if move == "MOVE_TIME":
            e = int(rng.integers(0, n_events))
            dur = int(problem.dur[e])
            max_start = config.SLOTS_PER_DAY - dur
            new_d = int(rng.integers(0, config.N_DAYS))
            new_s = int(rng.integers(0, max_start + 1))

            old_d = current_sched.day[e]
            old_s = current_sched.start[e]

            current_sched.day[e] = new_d
            current_sched.start[e] = new_s
            new_cost = total_cost(problem, current_sched)
            delta = new_cost - current_cost

            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sched = current_sched.copy()
            else:
                current_sched.day[e] = old_d
                current_sched.start[e] = old_s

        elif move == "MOVE_ROOM":
            e = int(rng.integers(0, n_events))
            rooms = comp_avail_rooms[e]
            if len(rooms) <= 1:
                # Fall back to MOVE_TIME
                dur = int(problem.dur[e])
                max_start = config.SLOTS_PER_DAY - dur
                new_d = int(rng.integers(0, config.N_DAYS))
                new_s = int(rng.integers(0, max_start + 1))

                old_d = current_sched.day[e]
                old_s = current_sched.start[e]

                current_sched.day[e] = new_d
                current_sched.start[e] = new_s
                new_cost = total_cost(problem, current_sched)
                delta = new_cost - current_cost

                if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_sched = current_sched.copy()
                else:
                    current_sched.day[e] = old_d
                    current_sched.start[e] = old_s
            else:
                chosen_r = int(rng.choice(rooms))
                old_r = current_sched.room_idx[e]

                current_sched.room_idx[e] = chosen_r
                new_cost = total_cost(problem, current_sched)
                delta = new_cost - current_cost

                if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_sched = current_sched.copy()
                else:
                    current_sched.room_idx[e] = old_r

        elif move == "SWAP":
            e1, e2 = rng.choice(n_events, size=2, replace=False)
            e1, e2 = int(e1), int(e2)
            r1, d1, s1 = current_sched.room_idx[e1], current_sched.day[e1], current_sched.start[e1]
            r2, d2, s2 = current_sched.room_idx[e2], current_sched.day[e2], current_sched.start[e2]

            dur1 = int(problem.dur[e1])
            dur2 = int(problem.dur[e2])

            # Feasibility check for swap
            if (
                problem.compatible[e1, r2]
                and problem.room_available[r2]
                and problem.compatible[e2, r1]
                and problem.room_available[r1]
                and s2 + dur1 <= config.SLOTS_PER_DAY
                and s1 + dur2 <= config.SLOTS_PER_DAY
            ):
                current_sched.room_idx[e1], current_sched.day[e1], current_sched.start[e1] = (
                    r2,
                    d2,
                    s2,
                )
                current_sched.room_idx[e2], current_sched.day[e2], current_sched.start[e2] = (
                    r1,
                    d1,
                    s1,
                )
                new_cost = total_cost(problem, current_sched)
                delta = new_cost - current_cost

                if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_sched = current_sched.copy()
                else:
                    current_sched.room_idx[e1], current_sched.day[e1], current_sched.start[e1] = (
                        r1,
                        d1,
                        s1,
                    )
                    current_sched.room_idx[e2], current_sched.day[e2], current_sched.start[e2] = (
                        r2,
                        d2,
                        s2,
                    )

        history.append(best_cost)

    return best_sched, history


def schedule_genetic(
    problem: Problem,
    pop_size: int = config.GA_POP,
    generations: int = config.GA_GENERATIONS,
    seed: int = config.SEED,
) -> tuple[Schedule, list[float]]:
    """Optimize schedule using Genetic Algorithm with tournament selection, uniform crossover, and elitism."""
    rng = np.random.default_rng(seed)
    n_events = len(problem.event_ids)
    n_rooms = len(problem.room_ids)

    comp_avail_rooms = []
    for e in range(n_events):
        rooms = [r for r in range(n_rooms) if problem.compatible[e, r] and problem.room_available[r]]
        if not rooms:
            rooms = [r for r in range(n_rooms) if problem.room_available[r]]
        comp_avail_rooms.append(rooms)

    # Initialize population: 1 greedy individual + (pop_size - 1) random placements
    pop: list[Schedule] = [schedule_greedy(problem)]
    for _ in range(pop_size - 1):
        ind = Schedule.empty(n_events)
        for e in range(n_events):
            rooms = comp_avail_rooms[e]
            ind.room_idx[e] = int(rng.choice(rooms))
            dur = int(problem.dur[e])
            ind.day[e] = int(rng.integers(0, config.N_DAYS))
            ind.start[e] = int(rng.integers(0, config.SLOTS_PER_DAY - dur + 1))
        pop.append(ind)

    costs = np.array([total_cost(problem, ind) for ind in pop])
    best_idx = int(np.argmin(costs))
    best_sched = pop[best_idx].copy()
    best_cost = float(costs[best_idx])
    history = [best_cost]

    for gen in range(generations):
        # Sort population by fitness
        order = np.argsort(costs)
        new_pop: list[Schedule] = [pop[order[i]].copy() for i in range(config.GA_ELITE)]

        while len(new_pop) < pop_size:
            # Tournament selection (k = GA_TOURNAMENT_K)
            t1 = rng.choice(pop_size, size=config.GA_TOURNAMENT_K, replace=False)
            p1_idx = t1[np.argmin(costs[t1])]
            t2 = rng.choice(pop_size, size=config.GA_TOURNAMENT_K, replace=False)
            p2_idx = t2[np.argmin(costs[t2])]

            p1 = pop[p1_idx]
            p2 = pop[p2_idx]

            # Uniform crossover
            child = Schedule.empty(n_events)
            mask = rng.random(size=n_events) < 0.5
            child.room_idx = np.where(mask, p1.room_idx, p2.room_idx)
            child.day = np.where(mask, p1.day, p2.day)
            child.start = np.where(mask, p1.start, p2.start)

            # Mutation
            for e in range(n_events):
                if rng.random() < config.GA_MUT_RATE:
                    if rng.random() < 0.5:
                        dur = int(problem.dur[e])
                        child.day[e] = int(rng.integers(0, config.N_DAYS))
                        child.start[e] = int(rng.integers(0, config.SLOTS_PER_DAY - dur + 1))
                    else:
                        rooms = comp_avail_rooms[e]
                        if rooms:
                            child.room_idx[e] = int(rng.choice(rooms))

            new_pop.append(child)

        pop = new_pop
        costs = np.array([total_cost(problem, ind) for ind in pop])
        gen_best_idx = int(np.argmin(costs))
        gen_best_cost = float(costs[gen_best_idx])

        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_sched = pop[gen_best_idx].copy()

        history.append(best_cost)

    return best_sched, history


def plot_convergence(
    histories: dict[str, list[float]],
    path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Plot optimization cost convergence histories and optionally save to disk."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, hist in histories.items():
        ax.plot(hist, label=label, linewidth=1.8)

    ax.set_title("Convergence", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration / Generation", fontsize=12)
    ax.set_ylabel("Total Cost", fontsize=12)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig
