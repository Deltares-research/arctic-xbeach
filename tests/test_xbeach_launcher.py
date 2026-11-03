"""start_xbeach runs the executable directly (no shell, no batch file) in the work folder, so it works on Linux and on any drive."""
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_helpers import build_run
from arctic_xbeach.model import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = "arctic_xbeach.model"


@pytest.fixture()
def sim_and_dir(tmp_path):
    """(Simulation, work folder of timestep 7). Simulation() changes the cwd, which is restored afterwards."""
    cwd = os.getcwd()
    sim = Simulation(str(build_run(tmp_path / "run")), proj_dir=str(REPO_ROOT))
    os.chdir(cwd)
    work_dir = Path(sim.result_dir) / "xb_files" / f"{7:010d}"
    work_dir.mkdir(parents=True)
    yield sim, work_dir


def record_run(monkeypatch, returncode):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=returncode, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_runs_the_executable_directly_in_the_work_folder(sim_and_dir, monkeypatch):
    sim, work_dir = sim_and_dir
    calls = record_run(monkeypatch, 0)
    assert sim.start_xbeach("/some/xbeach", "params.txt", timestep_id=7) is True
    (args, kwargs), = calls
    assert args[0] == ["/some/xbeach"]  # an argv list, not a command string
    assert not kwargs.get("shell")
    assert Path(kwargs["cwd"]) == work_dir
    assert not list(work_dir.glob("*.bat"))


def test_failed_run_logs_exit_code_and_xberror(sim_and_dir, monkeypatch, caplog):
    sim, work_dir = sim_and_dir
    (work_dir / "XBerror.txt").write_text("line 1\nboom\n")
    record_run(monkeypatch, 3)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert sim.start_xbeach("/some/xbeach", "params.txt", timestep_id=7) is False
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "3" in text and "boom" in text


def test_real_process_runs_without_shell_or_batch_file(sim_and_dir):
    """The interpreter stands in for xbeach.exe: started without arguments it reads the (empty) stdin and exits with 0."""
    sim, work_dir = sim_and_dir
    assert sim.start_xbeach(sys.executable, "params.txt", timestep_id=7) is True
    assert not list(work_dir.glob("*.bat"))


def test_missing_executable_is_a_configuration_error(sim_and_dir, caplog):
    sim, work_dir = sim_and_dir
    missing = str(work_dir / "no_such_xbeach.exe")
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        with pytest.raises(OSError, match="no_such_xbeach"):
            sim.start_xbeach(missing, "params.txt", timestep_id=7)
    assert any("no_such_xbeach" in r.getMessage() for r in caplog.records)
