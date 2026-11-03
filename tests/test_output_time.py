"""Output records are labelled at the END of their model step: the state written after step t is the state at timestamps[t] + dt."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from run_helpers import build_run, run_main

REPO_ROOT = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2010-05-01")
FLUXES = ["heat_flux", "lw_flux", "sw_flux", "latent_flux", "convective_flux"]


def short_run(run_dir, **output):
    """Hourly output for 2 days from 01-05-2010 without spin-up, with the ground temperature csv switched on."""
    build_run(run_dir, time_start="01-05-2010", time_end="03-05-2010", spin_up_days=0)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["output"].update(output_res=1, save_ground_temp_layers=[0, 0.5], heat_fluxes=FLUXES, **output)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    run_main(run_dir, REPO_ROOT)
    return run_dir / "results"


def test_records_are_labelled_at_the_end_of_the_step(tmp_path):
    results = short_run(tmp_path / "run")
    with xr.open_dataset(results / "results.nc") as ds:
        time, step = pd.to_datetime(ds.time.values), ds.timestep_id.values
        assert ds.attrs["time_label"] == "end_of_step"  # tells new files from older ones (labelled at the start of the step)
    assert len(time) == 48
    assert time[0] == T0 + pd.Timedelta(hours=1)  # the state after the first hour, not at the start
    assert np.all(np.diff(time) == pd.Timedelta(hours=1))
    assert np.array_equal(time, T0 + pd.to_timedelta(step + 1, unit="h"))

    csv = pd.read_csv(results / "ground_temperature_timeseries.csv", parse_dates=["time"])
    assert np.array_equal(csv["time"].values, time.values)


def test_output_window_refers_to_the_record_labels(tmp_path):
    results = short_run(tmp_path / "run", output_start="02-05-2010", output_end="02-05-2010 12:00")
    with xr.open_dataset(results / "results.nc") as ds:
        time, step = pd.to_datetime(ds.time.values), ds.timestep_id.values
    assert len(time) == 12
    assert time[0] == pd.Timestamp("2010-05-02 00:00") and time[-1] == pd.Timestamp("2010-05-02 11:00")
    assert step[0] == 23  # the state at the end of step 23 is labelled 2 May 00:00
