"""xbeach.max_failed_calls: main.py stops with a nonzero exit once more XBeach calls have failed than allowed, after recording the
failed call in xbeach_calls.csv. Without the key a failed call is logged and the run carries on (the calibration driver scores it).

XBeach itself is stubbed: every timestep is a call, set-up and grid update do nothing and start_xbeach fails."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from run_helpers import build_run, run_main
from arctic_xbeach.model import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def failing_xbeach(monkeypatch):
    calls = []
    monkeypatch.setattr(Simulation, "check_xbeach", lambda self, i: 1.0)
    monkeypatch.setattr(Simulation, "write_ne_layer", lambda self: None)
    monkeypatch.setattr(Simulation, "xbeach_setup", lambda self, i: None)
    monkeypatch.setattr(Simulation, "start_xbeach", lambda self, *a, **k: calls.append(1) or False)
    return calls


def make_run(tmp_path, max_failed_calls=None):
    run_dir = build_run(tmp_path / "run", time_start="01-05-2010", time_end="02-05-2010", spin_up_days=0)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["xbeach"]["with_xbeach"] = True
    if max_failed_calls is not None:
        cfg["xbeach"]["max_failed_calls"] = max_failed_calls
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir


def calls_csv(run_dir):
    return pd.read_csv(Path(run_dir) / "results" / "xbeach_calls.csv")


def test_stops_with_nonzero_exit_once_failures_exceed_the_limit(tmp_path, failing_xbeach):
    run_dir = make_run(tmp_path, max_failed_calls=2)
    with pytest.raises(SystemExit) as exc:
        run_main(run_dir, REPO_ROOT)
    assert exc.value.code not in (0, None)
    assert len(failing_xbeach) == 3  # two failures allowed, the third one stops the run
    calls = calls_csv(run_dir)
    assert len(calls) == 3 and (calls["success"] == 0).all()  # the failed call that triggered the stop is recorded
    assert "exceed xbeach.max_failed_calls = 2" in (run_dir / "run.log").read_text()


def test_limit_zero_stops_at_the_first_failure(tmp_path, failing_xbeach):
    run_dir = make_run(tmp_path, max_failed_calls=0)
    with pytest.raises(SystemExit):
        run_main(run_dir, REPO_ROOT)
    assert len(failing_xbeach) == 1 and len(calls_csv(run_dir)) == 1


def test_without_the_key_failed_calls_are_logged_and_the_run_carries_on(tmp_path, failing_xbeach):
    run_dir = make_run(tmp_path)
    run_main(run_dir, REPO_ROOT)
    assert len(failing_xbeach) > 3
    assert (calls_csv(run_dir)["success"] == 0).all()
