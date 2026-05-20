"""Shared fixtures for baseline tests."""

import pytest
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"


@pytest.fixture
def koga_available():
    """Skip test if Koga baseline is not cloned."""
    koga_path = BASELINES_DIR / "koga"
    if not koga_path.exists():
        pytest.skip("Koga baseline not cloned. Run `python setup_baselines.py`.")


@pytest.fixture
def grislain_available():
    """Skip test if Grislain baseline is not cloned."""
    grislain_path = BASELINES_DIR / "grislain"
    if not grislain_path.exists():
        pytest.skip("Grislain baseline not cloned. Run `python setup_baselines.py`.")
