"""Command-line pipeline interface for EventOptima fest scheduling."""

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence
import pandas as pd

import config
from cost import comparison_table, compute_kpis, hard_violation_count, total_cost
from constraints import validate
from data_io import load_problem
from forecast import forecast_attendance
from generator import generate_all
from models import Problem, Schedule
from resources import allocate_equipment, assign_volunteers
from scheduler import (
    plot_convergence,
    schedule_annealing,
    schedule_baseline,
    schedule_genetic,
    schedule_greedy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    seed: int = config.SEED,
    data_dir: str = "data",
    algo: str = "sa",
    iterations: int = config.SA_ITER,
    unavailable: Sequence[str] = (),
    compare: bool = False,
) -> dict:
    """Execute end-to-end scheduling pipeline and return all artifacts, schedules, and metrics."""
    data_path = Path(data_dir)
    if not data_path.exists() or not (data_path / "events.csv").exists():
        logger.info("Data directory '%s' missing or empty. Generating synthetic data...", data_dir)
        generate_all(seed=seed, out_dir=data_dir)

    # 1. Load problem
    prob = load_problem(data_dir)

    # 2. Apply unavailable rooms
    if unavailable:
        unavail_list = [r.strip() for r in unavailable if r.strip()]
        if unavail_list:
            prob = prob.with_unavailable_rooms(unavail_list)

    # 3. Baseline schedule = Before
    logger.info("Constructing baseline (Before) schedule...")
    before = schedule_baseline(prob)

    # 4. Optimized schedule = After
    algo_key = algo.lower()
    histories: dict[str, list[float]] = {}
    logger.info("Executing optimization algorithm '%s'...", algo_key)

    if algo_key == "greedy":
        after = schedule_greedy(prob)
        histories["Greedy"] = [total_cost(prob, after)]
        if compare:
            _, sa_hist = schedule_annealing(prob, init=after, n_iter=iterations, seed=seed)
            histories["Simulated Annealing"] = sa_hist
            gens = max(1, iterations // config.GA_POP)
            _, ga_hist = schedule_genetic(prob, pop_size=config.GA_POP, generations=gens, seed=seed)
            histories["Genetic Algorithm"] = ga_hist

    elif algo_key == "ga":
        gens = max(1, iterations // config.GA_POP)
        after, ga_hist = schedule_genetic(
            prob, pop_size=config.GA_POP, generations=gens, seed=seed
        )
        histories["Genetic Algorithm"] = ga_hist
        if compare:
            greedy_init = schedule_greedy(prob)
            _, sa_hist = schedule_annealing(prob, init=greedy_init, n_iter=iterations, seed=seed)
            histories["Simulated Annealing"] = sa_hist

    else:  # default 'sa'
        greedy_init = schedule_greedy(prob)
        after, sa_hist = schedule_annealing(
            prob, init=greedy_init, n_iter=iterations, seed=seed
        )
        histories["Simulated Annealing"] = sa_hist
        if compare:
            gens = max(1, iterations // config.GA_POP)
            _, ga_hist = schedule_genetic(
                prob, pop_size=config.GA_POP, generations=gens, seed=seed
            )
            histories["Genetic Algorithm"] = ga_hist

    # 5. Volunteers and equipment
    assign_before, _ = assign_volunteers(prob, before)
    assign_after, _ = assign_volunteers(prob, after)
    equip_after = allocate_equipment(prob, after)

    # 6. Validate
    violations_before = validate(prob, before, assignments=assign_before)
    violations_after = validate(prob, after, assignments=assign_after)

    # 7. KPIs and comparison
    kpis_before = compute_kpis(prob, before, assignments=assign_before)
    kpis_after = compute_kpis(prob, after, assignments=assign_after)
    comparison = comparison_table(kpis_before, kpis_after)

    # 8. Forecast
    forecast = forecast_attendance(prob.past_attendance, target_year=config.FORECAST_YEAR)

    return {
        "problem": prob,
        "before": before,
        "after": after,
        "assign_before": assign_before,
        "assign_after": assign_after,
        "equip_after": equip_after,
        "violations_before": violations_before,
        "violations_after": violations_after,
        "kpis_before": kpis_before,
        "kpis_after": kpis_after,
        "comparison": comparison,
        "histories": histories,
        "forecast": forecast,
    }


def main() -> None:
    """CLI entry point for pipeline execution and output artifact generation."""
    parser = argparse.ArgumentParser(description="EventOptima fest timetable optimization.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    parser.add_argument("--data", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--algo",
        type=str,
        default="sa",
        choices=["sa", "ga", "greedy"],
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=config.SA_ITER,
        help="Number of iterations / budget",
    )
    parser.add_argument(
        "--unavailable",
        type=str,
        default="",
        help="Comma-separated unavailable room IDs (e.g. 'R03,R05')",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Simulated Annealing and Genetic Algorithm convergence",
    )
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    unavail = [r.strip() for r in args.unavailable.split(",") if r.strip()]

    results = run_pipeline(
        seed=args.seed,
        data_dir=args.data,
        algo=args.algo,
        iterations=args.iterations,
        unavailable=unavail,
        compare=args.compare,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prob: Problem = results["problem"]
    before: Schedule = results["before"]
    after: Schedule = results["after"]

    # Save CSV outputs
    before.to_dataframe(prob).to_csv(out_dir / "schedule_before.csv", index=False)
    after.to_dataframe(prob).to_csv(out_dir / "schedule_after.csv", index=False)
    results["assign_after"].to_csv(out_dir / "assignments_after.csv", index=False)
    results["equip_after"].to_csv(out_dir / "equipment_after.csv", index=False)
    results["comparison"].to_csv(out_dir / "kpi_comparison.csv")
    results["violations_before"].to_csv(out_dir / "violations_before.csv", index=False)
    results["violations_after"].to_csv(out_dir / "violations_after.csv", index=False)

    # Save convergence plot
    plot_convergence(results["histories"], path=str(out_dir / "convergence.png"))

    print("\n======================== KPI COMPARISON TABLE ========================")
    print(results["comparison"].to_string())
    print("======================================================================\n")
    print(f"Artifacts successfully written to '{out_dir}/':")
    for f in out_dir.iterdir():
        print(f"  - {f.name}")

    hard_violations = hard_violation_count(prob, after, partial=False)
    if hard_violations > 0:
        logger.error("After schedule contains %d hard constraint violation(s)!", hard_violations)
        sys.exit(1)
    else:
        logger.info("Optimization completed successfully with 0 hard violations.")
        sys.exit(0)


if __name__ == "__main__":
    main()
