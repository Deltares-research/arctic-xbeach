"""Run the Arctic-XBeach v1.0 thermal module on a flat, dry soil column at Barter Island
(ERA5 forcing, 2016) and export the middle column for comparison with the 2DV model.

The case (CASE_NAME/config.yaml) is driven through the normal v1.0 path (Simulation + main),
with XBeach off: v10_case uses the conservative flux-form conduction (default), v10_case_nodal
the legacy nodal form (v1.0 as published). The v1.0 source is not modified by this script. The only intervention is a diagnostic
observer around `sim.thermal_update` (instance attribute) that calls the original method
unchanged and records, per wrapper hour:
  - the initial temperature profile (before the first sub-step),
  - hourly means of the surface flux components over the 30 sub-steps (v1.0 itself only
    writes the flux of the last sub-step),
  - the exact end-of-hour temperature profile, from the final enthalpy with v1.0's own
    `_get_phase_masks` / `_temperature_from_enthalpy` (v1.0's `ground_temperature_distribution`
    output lags this by one 120 s sub-step, because temp_matrix is refreshed at the start of
    each sub-step).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parents[2]                      # worktree root (contains main.py)
CASE_NAME = "v10_case"                                # case folder; optional override: python run_v10_column.py v10_case_nodal
OUT_NAMES = {"v10_case": "v10_column.nc",             # flux-form conduction (default)
             "v10_case_nodal": "v10_column_nodal.nc", # legacy nodal conduction (v1.0 as published)
             "v10_case_barter_latent": "v10_column_barter_latent.nc"}
if len(sys.argv) > 1:
    CASE_NAME = sys.argv[1]
CASE_DIR = SCRIPT_DIR / CASE_NAME                     # config.yaml, x.grd, bed.dep, ne_layer.txt
ERA5_PATH = Path("D:/Git/arctic-xbeach/examples/case_studies/barter_island/database/era5.csv")  # full ERA5 series
STORM_FILE = CASE_DIR / "storms_dry.csv"              # written below: WL = 0, Hs = 0 every hour
OUT_NC = PROJ_DIR / "examples" / "2dv" / "output" / "compare_1d" / OUT_NAMES[CASE_NAME]
X_INDEX = 1                                           # middle x-point (edge point 0 always gets thaw_depth = 0 in v1.0)
PRINT_TO_SCREEN = False                               # v1.0 progress log (always written to v10_case/run.log)

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from arctic_xbeach.model import Simulation  # noqa: E402
from main import main  # noqa: E402


def write_dry_storm_file(path: Path, t_start: pd.Timestamp, t_end: pd.Timestamp) -> None:
    """Hourly 'no storm' conditions: WL = 0 and Hs = 0, so R2% = 0 and z = +5 m stays dry."""
    times = pd.date_range(t_start, t_end, freq="1h")  # v1.0 masks with <= t_end
    df = pd.DataFrame({"time": times, "Hs(m)": 0.0, "Tp(s)": 2.0, "Dp(deg)": 270.0, "WL(m)": 0.0})
    df.to_csv(path)


def zero_c_depth(T_c: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Depth of the first downward crossing of 0 degC (linear interpolation); 0 if the surface is frozen."""
    out = np.zeros(T_c.shape[0])
    for i, prof in enumerate(T_c):
        if prof[0] <= 0.0:
            continue
        j = np.argmax(prof <= 0.0)
        if prof[j] > 0.0:            # thawed over the whole column
            out[i] = depth[-1]
            continue
        out[i] = depth[j - 1] + prof[j - 1] / (prof[j - 1] - prof[j]) * (depth[j] - depth[j - 1])
    return out


# --- set up the simulation (v1.0 path) ------------------------------------------------------
runid = CASE_DIR.relative_to(PROJ_DIR).as_posix()
sim = Simulation(runid, proj_dir=PROJ_DIR)            # note: chdir's into CASE_DIR
sim.config.data.forcing_data_path = ERA5_PATH.as_posix()
sim.config.data.storm_data_path = STORM_FILE.as_posix()
write_dry_storm_file(STORM_FILE,
                     pd.to_datetime(sim.config.model.time_start, dayfirst=True),
                     pd.to_datetime(sim.config.model.time_end, dayfirst=True))

dt_thermal = sim.config.thermal.dt
step_seconds = sim.config.model.timestep * 3600
diag = {"T_init": None, "sums": np.zeros(6), "n": 0, "flux_mean": [], "T_end": [], "t_thermal": 0.0}
original_thermal_update = sim.thermal_update


def observed_thermal_update(timestep_id, subgrid_timestep_id):
    if diag["T_init"] is None:
        diag["T_init"] = sim.temp_matrix[X_INDEX].copy()
        # effective volumetric latent heat as the model uses it (per-mass latent x bulk density)
        diag["latent_vol"] = float(sim.latent_matrix[X_INDEX, 0] * sim.soil_density_matrix[X_INDEX, 0])
    t0 = time.perf_counter()
    original_thermal_update(timestep_id, subgrid_timestep_id)
    diag["t_thermal"] += time.perf_counter() - t0
    # flux components applied in this sub-step; temp_matrix[:, 0] is the surface T they used
    diag["sums"] += (sim.convective_flux[X_INDEX], sim.latent_flux[X_INDEX], sim.lw_flux[X_INDEX],
                     sim.sw_flux[X_INDEX], sim.heat_flux[X_INDEX], sim.temp_matrix[X_INDEX, 0])
    diag["n"] += 1
    if subgrid_timestep_id + dt_thermal >= step_seconds:          # last sub-step of the hour
        diag["flux_mean"].append(diag["sums"] / diag["n"])
        diag["sums"], diag["n"] = np.zeros(6), 0
        T_end = sim._temperature_from_enthalpy(*sim._get_phase_masks())
        diag["T_end"].append(T_end[X_INDEX].copy())


sim.thermal_update = observed_thermal_update

# --- run --------------------------------------------------------------------------------------
t_wall = time.perf_counter()
main(sim, print_to_screen=PRINT_TO_SCREEN)
wall_main = time.perf_counter() - t_wall
print(f"v1.0 main(): {wall_main:.1f} s wall, of which thermal_update {diag['t_thermal']:.1f} s")

# --- collect ----------------------------------------------------------------------------------
T_melt = sim.config.thermal.T_melt
depth = np.linspace(0.0, sim.config.thermal.max_depth, sim.config.thermal.grid_resolution)
assert np.allclose(depth, sim.thermal_zgr)

res = xr.open_dataset(CASE_DIR / "results" / "results.nc")
t_forcing = pd.DatetimeIndex(res["time"].values)                 # ERA5 row applied during the hour
t_state = t_forcing + pd.Timedelta(hours=sim.config.model.timestep)  # state valid at end of hour
flux = np.array(diag["flux_mean"])
T_end = np.array(diag["T_end"])
assert T_end.shape[0] == t_forcing.size

T_native = res["ground_temperature_distribution"].values[:, X_INDEX, :]
lag_diff = float(np.max(np.abs(T_native - T_end)))

T_c = T_end - T_melt
th = sim.config.thermal
ds = xr.Dataset(
    data_vars={
        "T": (("time", "depth"), T_c.astype("f8"),
              {"units": "degC", "long_name": "ground temperature at end of wrapper hour",
               "note": "from final enthalpy of the hour; v1.0's own ground_temperature_distribution lags this by one sub-step"}),
        "T_init": (("depth",), diag["T_init"] - T_melt,
                   {"units": "degC", "long_name": "initial ground temperature profile used by v1.0 (2016-01-01 00:00)",
                    "note": "ERA5 soil levels 1-4 at the first time step, linear between (0, stl1), (0.035, stl1), (0.175, stl2), (0.64, stl3), (1.945, stl4), (max_depth, stl4)"}),
        "thaw_depth": (("time",), res["thaw_depth"].values[:, X_INDEX].astype("f8"),
                       {"units": "m", "long_name": "thaw depth as computed by v1.0 find_thaw_depth",
                        "note": "depth of the first node of N_thaw_threshold=3 consecutive nodes with T <= T_melt (mushy nodes count as not thawed); quantised to dz; computed from temp_matrix one sub-step before end of hour"}),
        "thaw_depth_0C": (("time",), zero_c_depth(T_c, depth),
                          {"units": "m", "long_name": "depth of first 0 degC crossing, linear interpolation of T (diagnostic, not v1.0)"}),
        "Q_sensible": (("time",), flux[:, 0], {"units": "W m-2", "long_name": "sensible (convective) heat flux, hourly mean over sub-steps"}),
        "Q_latent": (("time",), flux[:, 1], {"units": "W m-2", "long_name": "latent heat flux (ERA5 mean_surface_latent_heat_flux)"}),
        "Q_lw_net": (("time",), flux[:, 2], {"units": "W m-2", "long_name": "net longwave flux (ERA5 mean_surface_net_long_wave_radiation_flux)"}),
        "Q_sw_net": (("time",), flux[:, 3], {"units": "W m-2", "long_name": "net shortwave flux (ERA5 mean_surface_net_short_wave_radiation_flux, factor 1)"}),
        "Q_total": (("time",), flux[:, 4], {"units": "W m-2", "long_name": "total surface heat flux applied to the ghost node, hourly mean"}),
        "Q_sensible_last_substep": (("time",), res["convective_heat_flux"].values[:, X_INDEX].astype("f8"),
                                    {"units": "W m-2", "long_name": "sensible flux of the last sub-step (v1.0 native output)"}),
        "Q_total_last_substep": (("time",), res["total_heat_flux"].values[:, X_INDEX].astype("f8"),
                                 {"units": "W m-2", "long_name": "total flux of the last sub-step (v1.0 native output)"}),
        "T_surface_mean": (("time",), flux[:, 5] - T_melt,
                           {"units": "degC", "long_name": "hourly mean surface-node temperature used in the sensible flux"}),
        "T_air": (("time",), res["air_temperature_2m"].values.astype("f8") - T_melt,
                  {"units": "degC", "long_name": "ERA5 2 m air temperature (held constant over the hour)"}),
        "wind_speed": (("time",), res["wind_velocity"].values.astype("f8"),
                       {"units": "m s-1", "long_name": "10 m wind speed sqrt(u10^2 + v10^2)"}),
        "forcing_time": (("time",), t_forcing.values,
                         {"long_name": "timestamp of the ERA5 row applied during the hour ending at time"}),
    },
    coords={
        "time": ("time", t_state.values, {"long_name": "end of wrapper hour (state valid time)"}),
        "depth": ("depth", depth, {"units": "m", "positive": "down", "long_name": "depth below surface of v1.0 nodes"}),
    },
    attrs={
        "title": "Arctic-XBeach v1.0 thermal-only column, Barter Island flat dry land, ERA5 2016",
        "case": str(CASE_DIR),
        "x_index": X_INDEX,
        "sign_convention": "all fluxes positive downward = into the ground (ERA5 convention; v1.0 adds ERA5 fluxes unchanged)",
        "flux_time_convention": "fluxes are means over [time-1h, time]; the ERA5 row used is forcing_time = time - 1h. ERA5 mean rates are labelled at the END of their averaging hour, so v1.0 applies them 1 h late.",
        "sensible_flux_formula": "Q = 0.0296 (U L_e/nu)^0.8 Pr^(1/3) k_air/L_e (T_air - T_surface_node); L_e=0.003 m, nu=1.33e-5 m2/s, Pr=0.71, k_air=0.024 W/m/K; i.e. h = 16.12 U^0.8 W m-2 K-1",
        "surface_bc": "2nd-order ghost node: T_ghost = T[1] + 2 dz Q_total / k[0]; node 0 at the surface is prognostic",
        "bottom_bc": "T_ghost_bottom = T[-2] + 2 dz G, G = geothermal_gradient",
        "dt_s": dt_thermal, "dz_m": float(depth[1] - depth[0]),
        "c_frozen_effective_J_m3_K": sim.config.thermal.c_soil_frozen,
        "c_unfrozen_effective_J_m3_K": sim.config.thermal.c_soil_unfrozen,
        "latent_effective_J_m3": diag["latent_vol"],
        "conduction_form": sim.conduction_form,
        "latent_heat_legacy": int(bool(th.get("latent_heat_legacy", False))),
        "k_frozen": sim.config.thermal.k_soil_frozen_max, "k_unfrozen": sim.config.thermal.k_soil_unfrozen_max,
        "geothermal_gradient_K_m": sim.config.thermal.geothermal_gradient,
        "wall_time_main_s": round(wall_main, 2),
        "wall_time_thermal_update_s": round(diag["t_thermal"], 2),
        "max_abs_diff_T_vs_v10_native_output_K": lag_diff,
    },
)
res.close()
OUT_NC.parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(OUT_NC)

# --- sanity report ----------------------------------------------------------------------------
n_nan = int(sum(int(ds[v].isnull().sum()) for v in ds.data_vars if v != "forcing_time"))
i_max = int(ds.thaw_depth.argmax())
print(f"Wrote {OUT_NC} (conduction_form = {sim.conduction_form}, "
      f"effective latent heat {diag['latent_vol']:.4e} J m-3)")
print(f"NaNs: {n_nan}; |T - v1.0 native T|max = {lag_diff:.3f} K (one-sub-step lag)")
print(f"T_air {float(ds.T_air.min()):.1f} .. {float(ds.T_air.max()):.1f} degC; "
      f"T_surface {float(ds.T_surface_mean.min()):.1f} .. {float(ds.T_surface_mean.max()):.1f} degC")
print(f"Max thaw depth (v1.0) {float(ds.thaw_depth.max()):.3f} m at {pd.Timestamp(ds.time.values[i_max])}; "
      f"max 0 degC depth {float(ds.thaw_depth_0C.max()):.3f} m")
print("Annual-mean fluxes [W m-2]: " + ", ".join(
    f"{v}={float(ds[v].mean()):.2f}" for v in ["Q_sensible", "Q_latent", "Q_lw_net", "Q_sw_net", "Q_total"]))
print(f"T at 10 m: {float(ds.T.isel(depth=-1)[0]):.2f} -> {float(ds.T.isel(depth=-1)[-1]):.2f} degC")
