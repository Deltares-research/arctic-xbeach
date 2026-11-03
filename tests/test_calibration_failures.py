"""The calibration drivers must not score a failed or incomplete run.

Both drivers are imported and their run_simulation is called with a stubbed child process (subprocess.run / launch) and a small
hand-made results.nc, so no model run is needed. The coupled driver reads the observation csv files at import; without them the
tests of that driver are skipped.
"""
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2010-01-01")
N_Z = 20


def load_driver(name):
    sys.path.insert(0, str(REPO_ROOT / "calibration"))
    try:
        return importlib.import_module(name)
    except (FileNotFoundError, ImportError) as e:  # observation csv of the coupled driver, optuna
        pytest.skip(f"cannot import calibration/{name}.py here: {e}")
    finally:
        sys.path.remove(str(REPO_ROOT / "calibration"))


def make_trial(tmp_path, n_records, time_end="05-01-2010", calls_success=(1, 1)):
    """Trial folder with a config.yaml, a results.nc of n_records hourly records labelled at the end of their step and an
    xbeach_calls.csv with one row per entry of calls_success (None = no file)."""
    sim_dir = tmp_path / "trial"
    (sim_dir / "results").mkdir(parents=True)
    if calls_success is not None:
        pd.DataFrame({"call_id": range(1, len(calls_success) + 1), "success": calls_success}).to_csv(sim_dir / "results" / "xbeach_calls.csv", index=False)
    cfg = dict(model=dict(time_start="01-01-2010", time_end=time_end, timestep=1), output=dict(output_res=1))
    (sim_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    with netCDF4.Dataset(sim_dir / "results" / "results.nc", "w") as nc:
        for dim, size in (("time", None), ("time_xbeach", None), ("xgr", 3), ("depth_id", N_Z)):
            nc.createDimension(dim, size)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = f"seconds since {T0}"
        t[:] = 3600.0 * np.arange(1, n_records + 1)
        tx = nc.createVariable("time_xbeach", "f8", ("time_xbeach",))
        tx.units = f"seconds since {T0}"
        tx[:] = 3600.0 * np.arange(1, 3)  # XBeach records stop early in every case, as they do between calls
        z = np.linspace(0.0, -4.0, N_Z)
        nc.createVariable("abs_zgr", "f4", ("time", "xgr", "depth_id"))[:] = np.tile(z, (n_records, 3, 1))
        nc.createVariable("ground_temperature_distribution", "f4", ("time", "xgr", "depth_id"))[:] = 270.0
        nc.createVariable("xgr", "f4", ("xgr",))[:] = [0.0, 10.0, 20.0]
        zb = np.tile([-2.0, 5.0, 15.0], (2, 1))
        nc.createVariable("zb_xbeach", "f4", ("time_xbeach", "xgr"))[:] = zb
    return sim_dir


def stub_child(monkeypatch, module, returncode):
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=returncode))
    monkeypatch.setattr(module, "launch", lambda sim_dir: returncode, raising=False)


@pytest.fixture()
def thermal(monkeypatch, tmp_path):
    m = load_driver("run_thermal_calibration")
    times = pd.date_range("2010-01-02", periods=24, freq="h")
    obs = pd.DataFrame({"date_time": times, **{f"dpth{d}cm": -1.0 for d in range(0, 426, 25) if d != 75}})
    obs.to_csv(tmp_path / "obs.csv", index=False)
    monkeypatch.setattr(m, "OBSERVATION_CSV", tmp_path / "obs.csv")
    monkeypatch.setattr(m, "OUTPUT_WINDOW", ("02-01-2010", "03-01-2010"))  # records labelled 02-01 00:00 to 02-01 23:00
    monkeypatch.setattr(m, "MAKE_FIGURE", False)
    return m


@pytest.fixture()
def coupled(monkeypatch, tmp_path):
    m = load_driver("auto_calibrate_coupled_model")
    obs = pd.DataFrame({"time": pd.to_datetime(["2010-01-01 12:00", "2010-01-02 12:00"]), "distance_from_ref": [4.0, 5.0]})
    for name in ("DF_BE", "DF_SL", "DF_BE_CALIB", "DF_SL_CALIB"):
        monkeypatch.setattr(m, name, obs)
    monkeypatch.setattr(m, "NORM_DATE", pd.Timestamp("2010-01-01 12:00"))
    monkeypatch.setattr(m, "MAKE_FIGURE", False)
    return m


# thermal driver: the scored period ends at the end of OUTPUT_WINDOW (record labelled 02-01 23:00 is the last one)
def test_thermal_failed_child_is_not_scored(thermal, monkeypatch, tmp_path):
    sim_dir = make_trial(tmp_path, n_records=72)  # complete-looking results
    stub_child(monkeypatch, thermal, returncode=1)
    assert thermal.run_simulation(sim_dir) == float("inf")
    assert not (sim_dir / "rmse_bias_layers.txt").exists()


def test_thermal_results_that_stop_early_are_not_scored(thermal, monkeypatch, tmp_path):
    sim_dir = make_trial(tmp_path, n_records=36)  # last record 02-01 12:00
    stub_child(monkeypatch, thermal, returncode=0)
    assert thermal.run_simulation(sim_dir) == float("inf")
    assert not (sim_dir / "rmse_bias_layers.txt").exists()


def test_thermal_complete_results_are_scored(thermal, monkeypatch, tmp_path):
    sim_dir = make_trial(tmp_path, n_records=48)  # last record 02-01 23:00 plus later records outside the window are not needed
    stub_child(monkeypatch, thermal, returncode=0)
    assert np.isfinite(thermal.run_simulation(sim_dir))
    assert (sim_dir / "rmse_bias_layers.txt").exists()


# coupled driver: the thermal time variable decides, time_xbeach is only written at XBeach calls
def test_coupled_failed_child_is_not_scored(coupled, monkeypatch, tmp_path):
    sim_dir = make_trial(tmp_path, n_records=96, time_end="05-01-2010")
    stub_child(monkeypatch, coupled, returncode=1)
    assert coupled.run_simulation(sim_dir)[0] == float("inf")
    assert not (sim_dir / "metrics.txt").exists()


def test_coupled_results_that_stop_early_are_not_scored(coupled, monkeypatch, tmp_path):
    sim_dir = make_trial(tmp_path, n_records=60, time_end="05-01-2010")
    stub_child(monkeypatch, coupled, returncode=0)
    assert coupled.run_simulation(sim_dir)[0] == float("inf")
    assert not (sim_dir / "metrics.txt").exists()


def test_coupled_complete_results_are_scored(coupled, monkeypatch, tmp_path):
    sim_dir = make_trial(tmp_path, n_records=96, time_end="05-01-2010")
    stub_child(monkeypatch, coupled, returncode=0)
    rmse, n_failed = coupled.run_simulation(sim_dir)
    assert np.isfinite(rmse) and n_failed == 0
    assert (sim_dir / "metrics.txt").exists()


# failed XBeach calls: main.py logs them and carries on, xbeach_calls.csv (success column) is the record
def test_coupled_failed_xbeach_calls_are_not_scored(coupled, monkeypatch, tmp_path, capsys):
    sim_dir = make_trial(tmp_path, n_records=96, calls_success=(1, 0, 1, 0))
    stub_child(monkeypatch, coupled, returncode=0)
    assert coupled.run_simulation(sim_dir) == (float("inf"), 2)
    assert "2 failed XBeach calls (allowed: 0)" in capsys.readouterr().out
    assert not (sim_dir / "metrics.txt").exists()


def test_coupled_failed_xbeach_calls_up_to_the_limit_are_scored(coupled, monkeypatch, tmp_path):
    sim_dir = make_trial(tmp_path, n_records=96, calls_success=(1, 0, 1))
    stub_child(monkeypatch, coupled, returncode=0)
    monkeypatch.setattr(coupled, "MAX_FAILED_XBEACH_CALLS", 1)
    rmse, n_failed = coupled.run_simulation(sim_dir)
    assert np.isfinite(rmse) and n_failed == 1


def test_coupled_missing_call_log_is_not_scored(coupled, monkeypatch, tmp_path, capsys):
    sim_dir = make_trial(tmp_path, n_records=96, calls_success=None)
    stub_child(monkeypatch, coupled, returncode=0)
    assert coupled.run_simulation(sim_dir) == (float("inf"), None)
    assert "xbeach_calls.csv not found" in capsys.readouterr().out
