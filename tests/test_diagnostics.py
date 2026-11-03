"""Run diagnostics (xbeach_calls.csv helpers, conservation log at a grid update). Diagnostics must not change results.

The grid update is driven with a fake XBeach output file (erosion and deposition on a 7-node transect), so no XBeach is needed.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from run_helpers import build_run
import arctic_xbeach.diagnostics as dg
from arctic_xbeach.diagnostics import RunDiagnostics
from arctic_xbeach.model import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
SEDERO = np.array([0.0, -0.30, -0.10, 0.20, 0.40, 0.0, 0.0])  # m, per node of the 7-node transect
TID = 120  # hours after the start


def make_sim(run_dir: Path, with_diag: bool) -> Simulation:
    """Initialise like main.main does (thermal only) and run TID hours of thermal updates."""
    cwd = os.getcwd()
    sim = Simulation(str(run_dir), proj_dir=str(REPO_ROOT))
    os.chdir(cwd)
    cfg = sim.config
    sim.set_temporal_params(cfg.model.time_start, cfg.model.time_end, cfg.model.timestep)
    sim.load_forcing(cfg.data.forcing_data_path)
    sim.initialize_hydro_forcing(cfg.data.storm_data_path)
    sim.timesteps_with_xbeach_active()
    sim.generate_initial_grid(nx=cfg.bathymetry.nx, bathy_path=str(run_dir / "bed.dep"), bathy_grid_path=str(run_dir / "x.grd"))
    sim.initialize_xbeach_module()
    sim.check_r2_criteria()
    sim.initialize_thermal_module()
    if with_diag:
        sim.diag = RunDiagnostics(run_dir / "results", dt_hours=1, sea_ice_threshold=0.85, spinup_seconds=0.0)
        sim.diag.begin_call()
    for tid in range(TID + 1):
        for sub in range(0, 3600, int(cfg.thermal.dt)):
            sim.thermal_update(tid, sub)
        sim.find_thaw_depth()
    return sim


def fake_xbeach_output(sim: Simulation, sedero: np.ndarray, tid: int = TID):
    folder = Path(sim.result_dir) / "xb_files" / f"{tid:010d}"
    folder.mkdir(parents=True, exist_ok=True)
    x = sim.xgr.copy()
    ds = xr.Dataset(
        {"zb": (("globaltime", "x"), np.vstack([sim.zgr, sim.zgr + sedero])),
         "sedero": (("globaltime", "x"), np.vstack([np.zeros_like(x), sedero]))},
        coords={"x": x, "globaltime": [0.0, 3600.0]})
    ds.to_netcdf(folder / "xboutput.nc")


@pytest.fixture()
def sims(tmp_path):
    cwd = os.getcwd()
    run_a, run_b = build_run(tmp_path / "with_diag"), build_run(tmp_path / "without_diag")
    out = []
    for run, flag in ((run_a, True), (run_b, False)):
        sim = make_sim(run, flag)
        out.append(sim)
    yield out
    os.chdir(cwd)


def test_diagnostics_do_not_change_results_and_split_is_exact(sims):
    a, b = sims
    for sim in (a, b):
        fake_xbeach_output(sim, SEDERO)
    before = dict(H=a.enthalpy_matrix.copy(), rho=a.soil_density_matrix.copy(), Cs=a.Cs_matrix.copy(), z=a.zgr.copy(), x=a.xgr.copy(),
                  abs_zgr=a.abs_zgr.copy())
    a.update_grid(TID, "xboutput.nc")
    b.update_grid(TID, "xboutput.nc")

    # 1. same model state with and without diagnostics
    for name in ("enthalpy_matrix", "nb_matrix", "soil_density_matrix", "zgr", "abs_xgr", "abs_zgr", "angles"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert np.allclose(a.zgr, before["z"] + SEDERO)

    # 2. one row, exact two-way split of the heat change
    assert len(a.diag.conservation) == 1
    row = a.diag.conservation[0]
    assert row["regridded"] == 1
    assert row["dE"] == pytest.approx(row["E_after"] - row["E_before"])
    assert row["dE_material"] + row["dE_interp"] == pytest.approx(row["dE"], rel=1e-12)
    assert row["dE_material"] == pytest.approx(row["E_added_top"] + row["E_added_bottom"] - row["E_removed_top"] - row["E_removed_bottom"])

    # 3. E_before from an independent evaluation of the pre-regrid state (loop over columns)
    T_melt = a.config.thermal.T_melt
    x = before["x"]
    wx = np.zeros(len(x)); wx[:-1] += 0.5 * np.diff(x); wx[1:] += 0.5 * np.diff(x)
    nz = a.config.thermal.grid_resolution
    e_before = 0.0
    for i in range(len(x)):
        e = before["rho"][i] * (before["H"][i] - before["Cs"][i] * T_melt)  # J/m3 per node
        dzv = before["abs_zgr"][i, 0] - before["abs_zgr"][i, 1]
        e_before += wx[i] * dzv * (e.sum() - 0.5 * e[0] - 0.5 * e[-1])
    assert row["E_before"] == pytest.approx(e_before, rel=1e-12)

    # 4. the regrid error is small compared with the heat that is added and removed
    assert row["n_filled_top_nodes"] > 0 and row["n_removed_top_nodes"] > 0
    moved = abs(row["E_added_top"]) + abs(row["E_removed_top"])
    assert abs(row["dE_interp"]) < 0.02 * moved, (row["dE_interp"], moved)  # observed 0.2 % for this 0.1 to 0.4 m bed change

    # 5. sediment budget: XBeach output and thermal grid share the x grid, so the volumes agree
    net = float(np.trapezoid(SEDERO, before["x"])) if hasattr(np, "trapezoid") else float(np.trapz(SEDERO, before["x"]))
    assert row["vol_xb_net"] == pytest.approx(net) and row["vol_grid_net"] == pytest.approx(net)
    assert row["vol_xb_zbdiff"] == pytest.approx(net)
    assert abs(row["vol_grid_minus_xb"]) < 1e-12
    assert row["vol_xb_eroded"] > 0 and row["vol_xb_deposited"] > 0
    # discrete node count of the thermal columns reproduces the bed change to within a node (dz = 0.1 m)
    assert row["vol_nodes_fill_top"] == pytest.approx(row["vol_grid_deposited"], rel=0.1)  # lattice: filled nodes count whole nodes (observed +3.6 % and +3.5 %)
    assert row["vol_nodes_removed_top"] == pytest.approx(row["vol_grid_eroded"], rel=0.1)  # lattice: filled nodes count whole nodes (observed +3.6 % and +3.5 %)

    # 6. csv written with the documented columns
    with open(Path(a.result_dir) / "conservation_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == dg.CONSERVATION_COLUMNS and len(rows) == 1
    assert float(rows[0]["dE_interp"]) == pytest.approx(row["dE_interp"])


def test_no_bed_change_logs_a_row_without_regrid(sims):
    a, _ = sims
    fake_xbeach_output(a, np.zeros(7))
    H = a.enthalpy_matrix.copy()
    a.update_grid(TID, "xboutput.nc")
    row = a.diag.conservation[0]
    assert row["regridded"] == 0 and row["dE"] == 0.0 and row["vol_grid_net"] == 0.0
    assert np.array_equal(a.enthalpy_matrix, H)


def test_helpers():
    x = np.array([0.0, 3.0, 6.0]); z = np.array([0.0, 4.0, 8.0])  # straight line of length 10
    assert dg.arc_length_weights(x, z).sum() == pytest.approx(10.0)
    old = np.array([[2.0, 1.0, 0.0]]); new = np.array([[3.0, 2.0, 1.0]])  # surface raised by 1 m
    m = dg.regrid_node_masks(old, old, new, new)  # (x grids are not used by the masks)
    assert m["filled_top"].tolist() == [[True, False, False]] and m["removed_bottom"].tolist() == [[False, False, True]]
    assert not m["filled_bottom"].any() and not m["removed_top"].any()
    H = np.full((1, 3), 100.0); rho = np.full((1, 3), 2.0); Cs = np.full((1, 3), 0.5)
    abs_z = np.array([[5.0, 4.9, 4.8]])  # vertical node spacing 0.1 m
    # rho * (H - Cs * T_melt) = 2 * (100 - 0.5 * 20) = 180 J/m3, trapezoid weights 0.5, 1, 0.5 over 0.1 m
    assert dg.column_energy(H, rho, Cs, 20.0, abs_z)[0] == pytest.approx(180 * 2 * 0.1)
    assert dg.x_weights(np.array([0.0, 2.0, 6.0])).tolist() == [1.0, 3.0, 2.0]


def test_material_heat_is_exact_for_a_uniform_column():
    """Uniform enthalpy: the heat change of a regrid is exactly the added thickness times e, whatever the new slope."""
    e0 = 5.0
    nz = 50
    old_z = 3.0 - 0.10 * np.arange(nz)[None, :]  # surface 3.0 m, vertical node spacing 0.10
    new_z = 3.4 - 0.09 * np.arange(nz)[None, :]  # surface raised 0.4 m, steeper column: spacing 0.09
    e_old, e_new = np.full((1, nz), e0), np.full((1, nz), e0)
    mat = dg.material_heat(old_z, e_old, new_z, e_new)
    dz_old, dz_new = 0.10, 0.09
    E_old, E_new = e0 * (nz - 1) * dz_old, e0 * (nz - 1) * dz_new  # trapezoid integral of a constant
    material = mat["added_top"][0] + mat["added_bottom"][0] - mat["removed_top"][0] - mat["removed_bottom"][0]
    assert material == pytest.approx(E_new - E_old, rel=1e-12)
    assert mat["added_top"][0] == pytest.approx(e0 * 0.4) and mat["removed_top"][0] == 0.0
    # erosion of a linear profile: removed cover is the exact integral of the old profile over the eroded layer
    z = 3.0 - 0.10 * np.arange(nz)[None, :]
    e_lin = 2.0 * z  # e = 2 z
    mat = dg.material_heat(z, e_lin, z - 0.5, 2.0 * (z - 0.5))
    assert mat["removed_top"][0] == pytest.approx(2.0 * (3.0**2 - 2.5**2) / 2)  # integral of 2z from 2.5 to 3.0


def test_summary_block_and_call_log(sims, tmp_path):
    a, _ = sims
    fake_xbeach_output(a, SEDERO)
    a.update_grid(TID, "xboutput.nc")
    for i, reason in enumerate(("storm", "inter", "both")):
        a.diag.record_call(dict(call_id=i, timestep_id=TID + i, time=str(a.timestamps[TID + i]), year=2010, reason=reason,
                                xbeach_times_value=1.0, hotstart=0, sim_hours=1, setup_wall_s=0.1, xbeach_wall_s=2.0,
                                grid_update_wall_s=0.3, success=1))
    a.diag.add_thermal_time(5.0, spinup=False)
    text = "\n".join(a.diag.summary_lines(a.timestamps, a.forcing_data["sea_ice_cover"].values))
    assert "XBeach calls: 3" in text and "simulated XBeach hours: 3" in text and "XBeach wall-clock: 6.0 s" in text
    assert "cumulative regrid error" in text and "2010" in text
    # ice-free hours: the synthetic sea ice falls below the 0.85 threshold after 3 days
    ice_free = int((a.forcing_data["sea_ice_cover"].values[:len(a.timestamps)] < 0.85).sum())
    assert str(ice_free) in text
    a.diag.close()
    with open(Path(a.result_dir) / "xbeach_calls.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["reason"] for r in rows] == ["storm", "inter", "both"] and list(rows[0].keys()) == dg.CALL_COLUMNS


def test_main_limits_openblas_threads():
    """Importing main first, as the calibration child scripts do, limits OpenBLAS unless the user set it."""
    code = "import os; import main; print(os.environ['OPENBLAS_NUM_THREADS'])"
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True).stdout.strip() == "1"
    env["OPENBLAS_NUM_THREADS"] = "4"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True).stdout.strip() == "4"
