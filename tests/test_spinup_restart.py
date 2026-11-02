"""Spin-up save/restore (model.spinup_state_out / model.spinup_state_in) on a small thermal-only run.

A 61-day spin-up plus 3 days without XBeach on a 7-node transect with a 40-node thermal grid. Synthetic forcing, see run_helpers.py.
Run from the repository root with:  python -m pytest tests -q
"""
import os
from pathlib import Path

import numpy as np
import pytest

from run_helpers import build_run, reset_logger, run_main

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_KEYS = ("enthalpy_matrix", "temp_matrix", "thaw_depth", "ghost_nodes_temperature", "heat_flux", "xgr", "zgr")


@pytest.fixture(scope="module")
def full_and_restart(tmp_path_factory):
    """(full run saving the state, restart from that state, state file)."""
    root = tmp_path_factory.mktemp("restart")
    state = root / "spinup_state.npz"
    full_dir = build_run(root / "full", spinup_state_out=str(state))
    restart_dir = build_run(root / "restart", spinup_state_in=str(state))
    full = run_main(full_dir, REPO_ROOT)
    restart = run_main(restart_dir, REPO_ROOT)
    return full, restart, state, root


def test_state_file_written(full_and_restart):
    _, _, state, _ = full_and_restart
    assert state.exists() and state.stat().st_size > 0
    assert not state.with_name(state.name + ".tmp").exists()


def test_restart_is_bit_identical(full_and_restart):
    full, restart, _, _ = full_and_restart
    for key in STATE_KEYS:
        assert np.array_equal(full[key], restart[key]), f"{key} differs after restart"
    # the active layer really thawed during the run (the test would be vacuous on a frozen column)
    assert full["thaw_depth"].max() > 0
    assert (full["temp_matrix"] > 273.15).any() and (full["temp_matrix"] < 273.15).any()


def test_restart_skips_the_spinup(full_and_restart):
    full, restart, _, root = full_and_restart
    assert full["diag"].spinup_wall_s > 0 and not full["diag"].spinup_restored
    assert restart["diag"].spinup_wall_s == 0 and restart["diag"].spinup_restored
    assert restart["diag"].thermal_wall_s < 0.2 * full["diag"].thermal_wall_s  # 3 of 64 days
    log = (root / "restart" / "run.log").read_text(encoding="utf-8")
    assert "Spin-up skipped: state restored" in log and "RUN DIAGNOSTICS SUMMARY" in log


def test_restart_refuses_changed_thermal_config(full_and_restart, tmp_path):
    _, _, state, _ = full_and_restart
    changed = build_run(tmp_path / "changed", thermal=dict(k_soil_frozen_min=0.95), spinup_state_in=str(state))
    with pytest.raises(ValueError, match="thermal config differs.*k_soil_frozen_min"):
        run_main(changed, REPO_ROOT)


def test_restart_refuses_changed_option_default(full_and_restart, tmp_path):
    """A solver option counts as part of the thermal config (here the legacy conduction form instead of flux)."""
    _, _, state, _ = full_and_restart
    changed = build_run(tmp_path / "changed_opt", thermal=dict(conduction_form="nodal"), spinup_state_in=str(state))
    with pytest.raises(ValueError, match="conduction_form"):
        run_main(changed, REPO_ROOT)


def test_restart_refuses_other_start_time_and_missing_file(full_and_restart, tmp_path):
    _, _, state, _ = full_and_restart
    shifted = build_run(tmp_path / "shifted", time_start="02-05-2010", time_end="05-07-2010", spinup_state_in=str(state))
    with pytest.raises(ValueError, match="run context differs"):
        run_main(shifted, REPO_ROOT)
    missing = build_run(tmp_path / "missing", spinup_state_in=str(tmp_path / "nope.npz"))
    with pytest.raises(FileNotFoundError):
        run_main(missing, REPO_ROOT)


def test_other_end_time_is_allowed(full_and_restart, tmp_path):
    """The state does not depend on how long the run continues afterwards."""
    _, _, state, _ = full_and_restart
    longer = build_run(tmp_path / "longer", time_end="06-07-2010", spinup_state_in=str(state))
    out = run_main(longer, REPO_ROOT)
    assert np.isfinite(out["enthalpy_matrix"]).all()
