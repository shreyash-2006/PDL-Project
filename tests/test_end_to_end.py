"""End-to-end integration tests for complete optimization pipeline and CLI execution."""

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from generator import generate_all
from main import run_pipeline


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    """Verify end-to-end pipeline run produces valid schedule, lower cost, and complete artifacts."""
    data_dir = tmp_path / "data"
    generate_all(seed=42, out_dir=str(data_dir))

    results = run_pipeline(seed=42, data_dir=str(data_dir), iterations=2000)

    prob = results["problem"]
    before = results["before"]
    after = results["after"]
    kpis_before = results["kpis_before"]
    kpis_after = results["kpis_after"]
    viol_after = results["violations_after"]
    equip_after = results["equip_after"]

    # 1. After has zero hard violations
    assert kpis_after["hard_violations"] == 0

    # 2. After total cost < Before total cost
    assert kpis_after["total_cost"] < kpis_before["total_cost"]

    # 3. After student clashes <= Before
    assert kpis_after["student_clashes"] <= kpis_before["student_clashes"]

    # 4. Volunteer constraints
    vol_rules = {"VOLUNTEER_DOUBLE_BOOKED", "VOLUNTEER_SKILL_MISMATCH", "VOLUNTEER_OVER_HOURS"}
    vol_viol = viol_after[viol_after["rule"].isin(vol_rules)]
    assert vol_viol.empty, f"Found volunteer violations: {vol_viol['rule'].tolist()}"

    # 5. Zero SHORTAGE rows in equipment
    shortage_rows = equip_after[equip_after["status"] == "SHORTAGE"]
    assert shortage_rows.empty, "Found equipment shortage rows"


def test_cli_execution_and_artifacts(tmp_path: Path) -> None:
    """Verify python main.py executes successfully and creates all specified output files."""
    out_dir = tmp_path / "outputs"
    cmd = [
        sys.executable,
        "main.py",
        "--iterations",
        "1000",
        "--out",
        str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI exited with error: {res.stderr}"

    expected_files = [
        "schedule_before.csv",
        "schedule_after.csv",
        "assignments_after.csv",
        "equipment_after.csv",
        "kpi_comparison.csv",
        "violations_before.csv",
        "violations_after.csv",
        "convergence.png",
    ]
    for fname in expected_files:
        p = out_dir / fname
        assert p.exists(), f"Expected artifact '{fname}' was not generated"


def test_pipeline_unavailable_rooms() -> None:
    """Verify pipeline with unavailable rooms leaves all events placed and none in unavailable rooms."""
    unavail = ["R01", "R02"]
    results = run_pipeline(seed=42, iterations=1000, unavailable=unavail)

    prob = results["problem"]
    after = results["after"]

    # All events placed
    assert not np.any(after.room_idx == -1)

    # No events in R01 or R02
    r01_idx = prob.room_ids.index("R01")
    r02_idx = prob.room_ids.index("R02")
    assert not np.any(after.room_idx == r01_idx)
    assert not np.any(after.room_idx == r02_idx)
