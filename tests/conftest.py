"""Pytest fixtures for EventOptima testing."""

import sys
from pathlib import Path
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_io import load_problem
from generator import generate_all
from models import Problem


@pytest.fixture(scope="session")
def small_problem(tmp_path_factory) -> Problem:
    """Fixture providing a small 6-event Problem instance."""
    tmp_dir = tmp_path_factory.mktemp("small_data")
    generate_all(
        seed=7,
        out_dir=str(tmp_dir),
        n_events=6,
        n_rooms=4,
        n_students=30,
        n_volunteers=8,
    )
    return load_problem(str(tmp_dir))


@pytest.fixture(scope="session")
def default_problem() -> Problem:
    """Fixture providing the default 25-event Problem loaded from 'data/'."""
    data_path = Path("data")
    if not (data_path / "events.csv").exists():
        generate_all(seed=42, out_dir="data")
    return load_problem("data")
