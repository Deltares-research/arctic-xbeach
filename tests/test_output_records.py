"""XBeach records in results.nc: one per successful call, at consecutive indices, whatever the thermal output steps are.

write_output is driven directly on a thermal-only sim (as in tests/test_diagnostics.py) with fake XBeach output files, at the
steps main.py calls it for: the thermal output steps (temp_output_ids) and every step with a successful call.
"""
import os
import time

import numpy as np
import pytest
import xarray as xr

from run_helpers import build_run
from test_diagnostics import make_sim

XB_VARS = ("H", "zb", "zs", "E", "Sxx", "Sxy", "Syy", "vmag", "urms")


def fake_xb_file(sim, step: int):
    """Minimal XBeach output for one call; every variable equals the step id so records can be told apart."""
    folder = os.path.join(sim.result_dir, "xb_files", f"{step:010d}")
    os.makedirs(folder, exist_ok=True)
    x = sim.xgr.copy()
    ds = xr.Dataset({v: (("globaltime", "x"), np.full((2, len(x)), float(step))) for v in XB_VARS},
                    coords={"x": x, "globaltime": [0.0, 3600.0]})
    ds.to_netcdf(os.path.join(folder, "xboutput.nc"))


def run_outputs(tmp_path, calls: dict, output_steps):
    """calls: {step: (xbeach_times value, success)}; output_steps are the thermal output steps. Returns the results.nc contents."""
    cwd = os.getcwd()
    try:
        sim = make_sim(build_run(tmp_path / "run"), with_diag=False)
        assert not sim.xbeach_success.any()  # nothing has been called yet
        for step, (value, success) in calls.items():
            sim.xbeach_times[step] = value
            sim.xbeach_success[step] = success
            if success:
                fake_xb_file(sim, step)
        sim.temp_output_ids = np.array(output_steps)
        for step in sorted(set(output_steps) | {s for s, (_, ok) in calls.items() if ok}):  # as main.py
            sim.write_output(step, time.time())
        sim.nc_writer.close()
        with xr.open_dataset(os.path.join(sim.result_dir, "results.nc"), decode_times=False) as ds:
            return ds.load()
    finally:
        os.chdir(cwd)


def test_success_flag_defaults_to_false(tmp_path):
    cwd = os.getcwd()
    try:
        sim = make_sim(build_run(tmp_path / "run"), with_diag=False)
        assert sim.xbeach_success.shape == sim.xbeach_times.shape and sim.xbeach_success.dtype == bool
    finally:
        os.chdir(cwd)


def test_call_on_storm_and_inter_step_is_written(tmp_path):
    ds = run_outputs(tmp_path, {3: (2.0, True)}, output_steps=[0, 3])
    assert ds.sizes["time_xbeach"] == 1
    assert ds.time_xbeach.values[0] == ds.time.values[1]  # label of the thermal record of the same step
    assert np.all(ds.zb_xbeach.values[0] == 3.0)


def test_calls_between_output_steps_are_written(tmp_path):
    calls = {1: (1.0, True), 3: (1.0, True), 6: (1.0, True)}
    ds = run_outputs(tmp_path, calls, output_steps=[0, 3, 6])
    assert ds.sizes["time"] == 3  # no thermal record for the call at step 1
    assert ds.sizes["time_xbeach"] == 3  # but an XBeach record
    assert np.array_equal(ds.time_xbeach.values[1:], ds.time.values[1:])
    assert ds.time_xbeach.values[0] < ds.time.values[1]
    assert np.array_equal(ds.zb_xbeach.values[:, 0], [1.0, 3.0, 6.0])  # no empty or fill records in between


def test_failed_call_has_no_record(tmp_path):
    ds = run_outputs(tmp_path, {3: (1.0, False), 6: (1.0, True)}, output_steps=[0, 3, 6])
    assert ds.sizes["time_xbeach"] == 1
    assert ds.time_xbeach.values[0] == ds.time.values[2]
    assert np.all(ds.zb_xbeach.values[0] == 6.0)
