"""Step 1 of the calibration: thermal-only Optuna calibration of the ground temperature (no XBeach).

Port of the published driver (thermal study 20260201, best trial 869). Objective, parameter ranges, penalty and metrics are
unchanged. What changed for v1.1: all paths are in the configuration block (pathlib), the sampler is seeded, a starting parameter
set can be enqueued, the number of trials / parallel workers are plain variables, the solver options of v1.1 are written into
every trial config, and each trial runs in a child process of the current Python interpreter (no .bat / conda activation, so it
runs on Windows and Linux).

Each trial: copy BASE_SIM_DIR, write the sampled thermal parameters into its config.yaml, run main.py (thermal-only: with_xbeach is
False in the base config), compare the modelled ground temperature in the column x-index 1 against the borehole record
(BI_T-1, 17 depths, 0 to 4.25 m) and return the mean RMSE over the depths. A trial whose surface temperature never reaches 0.01 K
(frozen all the time, the column did not thaw) gets its RMSE multiplied by 10. A failed run returns inf: a child process with a
nonzero exit code, or a results.nc that stops before the end of the scored period, is not scored.

Run:  python calibration/run_thermal_calibration.py        (see calibration/README.md)

On a Linux cluster the same script is configured through AXB_* environment variables (hpc/h7/thermal_calibration.sbatch). With
STORAGE = "journal" the study is an append-only file that is safe on NFS; with SCRATCH set each trial runs on that (local) disk and
is copied back to RUN_ROOT_DIR. Only ONE driver process per study is supported (the sbatch file uses --dependency=singleton): with
journal storage, trials still RUNNING at start are taken to be left by a killed earlier job and are marked FAIL.
"""
# === CONFIGURATION ===
# The AXB_* environment variables override the defaults below (used by hpc/h7/*.sbatch); without them nothing changes.
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]  # arctic-xbeach checkout that is imported by the trials
BASE_SIM_DIR = Path(os.environ.get("AXB_BASE_SIM_DIR", r"N:\arctic-xbeach-runs\thermal_calibration_runs\20260201_calibration_runs\base"))  # template run folder (config.yaml, x.grd, bed.dep, ne_layer.txt)
RUN_ROOT_DIR = Path(os.environ.get("AXB_RUN_ROOT", r"N:\arctic-xbeach-runs\v11_runs\thermal_calibration"))  # trial folders, optuna_study.db (or optuna_journal.log), logs are written here
OBSERVATION_CSV = Path(os.environ.get("AXB_OBS_CSV", r"D:\Git\thermo-morphological-model\database\raw_datasets\erikson2\Temp_arrays USGS-UCSC_Oberle\BI_T-1_processed.csv"))
FORCING_CSV = Path(os.environ.get("AXB_FORCING_CSV", r"N:\arctic-xbeach-runs\database\era5.csv"))   # written into every trial config (None = keep the path of the base config)
STORM_CSV = Path(os.environ.get("AXB_STORM_CSV", r"N:\arctic-xbeach-runs\database\storms.csv"))   # idem

N_TRIALS = int(os.environ.get("AXB_N_TRIALS", 1000))        # completed trials the study should reach; a rerun on an existing study runs only the rest
                       # (published study: 125 epochs x 8 workers = 1000)
N_PARALLEL = int(os.environ.get("AXB_N_PARALLEL", 8))         # trials running at the same time (one child process each)
SEED = 1               # seed of the TPE sampler. With N_PARALLEL > 1 the order in which trials finish still varies, so runs are not bit-reproducible
# enqueue this parameter set as the first trial of a new study (names of the suggest_* calls below); None = no start set
START_PARAMS = dict(nb_min=0.46951231603096133, nb_increase=0.24059594350455396, transition_top=0.41887032171820293,
                    transition_bottom=2.372186433986581, k_soil_frozen_min=0.9046457783637623, k_soil_unfrozen_min=0.3296227605203751,
                    k_compaction_factor=1.9129954173279924, c_soil_frozen=1574265.3175829535, c_ratio=1.9716403484401948,
                    geothermal_gradient=0.020426871031478094)   # thermal trial 869 of the published study (value 1.0589)

# v1.1 solver options written into every trial config (thermal.* and data.*). Use the legacy values (nodal, per_mass, 0) for the v1.0 solver.
SOLVER_OPTIONS = dict(conduction_form="flux", latent_heat_form="volumetric")
ERA5_FLUX_SHIFT_HOURS = 1
THERMAL_DT = int(os.environ.get("AXB_THERMAL_DT", 60))       # thermal substep [s]. 60 s: largest divisor of 3600 below 75 s, the limit dt <= (1 - 2Fo) C_frozen dz / (2 hc) of the wet surface node (Barter, 3 Nov)
                       # thaw depth does not depend on dt (column test); 360 s changed the objective by 0.0001 K. None = base config
OUTPUT_WINDOW = ("31-05-2016", "15-09-2016")  # netCDF output only over the borehole record (BI_T-1); None = whole run
SENSOR_DEPTH_OFFSET = float(os.environ.get("AXB_SENSOR_DEPTH_OFFSET", 0.0))  # [m] added to each sensor depth before the nearest-node lookup. Nodes are cell centres, so 0 is the true depth.
                          # 0.1 repeats the published objective and thermal trial 490 (0.82 K at dt 90 s); it is 2 cells of the 0.05 m grid

STUDY_NAME = os.environ.get("AXB_STUDY_NAME", "calibration_optuna")
STORAGE = os.environ.get("AXB_STORAGE", "sqlite")  # "sqlite" (RUN_ROOT_DIR/optuna_study.db) or "journal" (RUN_ROOT_DIR/optuna_journal.log, for NFS)
SCRATCH = Path(os.environ["AXB_SCRATCH"]) if os.environ.get("AXB_SCRATCH") else None  # trials run here and are copied back to RUN_ROOT_DIR; None = run in RUN_ROOT_DIR
KEEP_XB_FILES = os.environ.get("AXB_KEEP_XB_FILES", "1").lower() not in ("0", "false", "no")  # False = delete results/xb_files of a trial after scoring (no effect with SCRATCH: never copied back)
CONSTANT_LIAR = os.environ.get("AXB_CONSTANT_LIAR", "0").lower() in ("1", "true", "yes")  # with ~40 parallel trials this keeps TPE from proposing near-duplicates of the trials still running
MAKE_FIGURE = True     # temperature_layers.png in every trial folder
KEEP_RESULTS_NC = False  # the results.nc of a trial is deleted after the skill has been computed
# === END CONFIGURATION ===

# One BLAS / OpenMP thread per process: 40 trials would otherwise each try to use all cores. Set before numpy is imported;
# the child processes inherit them (main.py sets the same OpenBLAS default for its own runs).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import datetime
import gc
import shutil
import subprocess
import sys
import threading
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so that optuna_hpc is found when the file is not run as a script
sys.path.insert(0, str(REPO_DIR))  # for arctic_xbeach.config_loader
from arctic_xbeach.config_loader import load_config  # a duplicate key in a config.yaml is an error
import optuna_hpc as hpc  # calibration/optuna_hpc.py, next to this script
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import optuna
import pandas as pd
import yaml
from netCDF4 import Dataset

warnings.filterwarnings("ignore")

# Optuna runs the n_jobs trials as threads of this process; HDF5 (netCDF4) and pyplot are not thread-safe, so the
# post-processing of one trial at a time holds this lock (two reads at once gave "NetCDF: HDF error", trial 352).
POSTPROC_LOCK = threading.Lock()

# Parameter ranges [low, high]; low == high means fixed. Entries that the structured scheme of suggest_thermal_params()
# sets itself (nb_*, k_soil_*, c_soil_*, depth_constant_k) are skipped in objective(), as in the published driver.
PARAM_RANGES = {
    # ===== KEEP FIXED =====
    "max_depth": [10, 10],
    "grid_resolution": [200, 200],        # 5 cm resolution
    "T_melt": [273.15, 273.15],
    "L_water_ice": [334000, 334000],      # [J/kg]
    "rho_water": [1000, 1000],
    "rho_ice": [917, 917],
    "rho_particle": [2650, 2650],
    "hc_guess": [500, 500],               # heat transfer coefficient water to ground

    # ===== CALIBRATE THESE =====
    "k_soil_frozen_min": [0.7, 2.0],      # overridden by suggest_thermal_params
    "k_soil_frozen_max": [1.7, 2.7],      # overridden (= min * k_compaction_factor)
    "k_soil_unfrozen_min": [0.3, 0.8],    # overridden
    "k_soil_unfrozen_max": [0.6, 1.5],    # overridden
    "depth_constant_k": [0.5, 1.5],       # overridden (mean of the two transition depths)
    "nb_max": [0.40, 0.65],               # overridden (= nb_min + nb_increase)
    "nb_min": [0.40, 0.65],               # overridden
    "nb_max_depth": [0.5, 0.5],           # overridden (= transition_top)
    "nb_min_depth": [1.5, 1.5],           # overridden (= transition_bottom)
    "c_soil_frozen": [3000000, 5000000],  # overridden
    "c_soil_unfrozen": [5500000, 8000000],  # overridden (= c_soil_frozen * c_ratio)
    "geothermal_gradient": [0.02, 0.04],  # [K/m]
}


def suggest_thermal_params(trial: optuna.Trial) -> dict:
    """Structured sampling of the thermal parameters (porosity, transition depths, conductivity, heat capacity)."""
    params = {}

    # porosity: higher near the surface (organic layer)
    params["nb_min"] = trial.suggest_float("nb_min", 0.30, 0.50)
    params["nb_increase"] = trial.suggest_float("nb_increase", 0.05, 0.25)
    params["nb_max"] = params["nb_min"] + params["nb_increase"]

    # depth structure shared by porosity and conductivity
    params["transition_top"] = trial.suggest_float("transition_top", 0.3, 0.8)
    params["transition_bottom"] = trial.suggest_float("transition_bottom", 1.0, 2.5)
    params["nb_max_depth"] = params["transition_top"]
    params["nb_min_depth"] = params["transition_bottom"]
    params["depth_constant_k"] = (params["transition_top"] + params["transition_bottom"]) / 2

    # thermal conductivity, increasing with depth by the compaction factor
    params["k_soil_frozen_min"] = trial.suggest_float("k_soil_frozen_min", 0.7, 2.0)
    params["k_soil_unfrozen_min"] = trial.suggest_float("k_soil_unfrozen_min", 0.3, 0.8)
    params["k_compaction_factor"] = trial.suggest_float("k_compaction_factor", 1.0, 2.0)
    params["k_soil_frozen_max"] = params["k_soil_frozen_min"] * params["k_compaction_factor"]
    params["k_soil_unfrozen_max"] = params["k_soil_unfrozen_min"] * params["k_compaction_factor"]

    # volumetric heat capacity
    params["c_soil_frozen"] = trial.suggest_float("c_soil_frozen", 1.5e6, 2.5e6)
    params["c_ratio"] = trial.suggest_float("c_ratio", 1.2, 2.0)
    params["c_soil_unfrozen"] = params["c_soil_frozen"] * params["c_ratio"]
    return params


def update_config_yaml(sim_dir: Path, params: dict):
    """Write sampled parameters into thermal (only keys that exist there), then the v1.1 options and the forcing paths."""
    config_path = sim_dir / "config.yaml"
    config = load_config(config_path)

    for k, v in params.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        if "thermal" in config and k in config["thermal"]:
            config["thermal"][k] = v

    config["thermal"].update(SOLVER_OPTIONS)
    config["data"]["era5_flux_shift_hours"] = ERA5_FLUX_SHIFT_HOURS
    if THERMAL_DT is not None:
        config["thermal"]["dt"] = THERMAL_DT
    if OUTPUT_WINDOW is not None:
        config["output"]["output_start"], config["output"]["output_end"] = OUTPUT_WINDOW
    if FORCING_CSV is not None:
        config["data"]["forcing_data_path"] = str(FORCING_CSV)
    if STORM_CSV is not None:
        config["data"]["storm_data_path"] = str(STORM_CSV)
    config_path.write_text(yaml.dump(config, default_flow_style=False))


def compute_rmse_mae(obs_vals, model_vals):
    obs_vals, model_vals = np.array(obs_vals), np.array(model_vals)
    return np.sqrt(np.mean((obs_vals - model_vals) ** 2)), np.mean(np.abs(obs_vals - model_vals))


def compute_metrics_n(model_time, model_temp, obs_time, obs_temp):
    """RMSE, MAE, bias and number of valid pairs of the model at the observation times (nearest model time)."""
    df_model = pd.DataFrame({"time": pd.to_datetime(model_time), "temp": model_temp}).sort_values("time")
    df_obs = pd.DataFrame({"time": pd.to_datetime(obs_time), "temp": obs_temp}).sort_values("time")
    merged = pd.merge_asof(df_obs, df_model, on="time", direction="nearest", suffixes=("_obs", "_model"))
    valid = ~pd.isnull(merged["temp_obs"]) & ~pd.isnull(merged["temp_model"])
    obs_vals, model_vals = merged.loc[valid, "temp_obs"], merged.loc[valid, "temp_model"]
    rmse, mae = compute_rmse_mae(obs_vals, model_vals)
    return rmse, mae, (model_vals - obs_vals).mean(), int(valid.sum())


def compute_metrics(model_time, model_temp, obs_time, obs_temp):
    """RMSE, MAE and bias of the model at the observation times (nearest model time)."""
    return compute_metrics_n(model_time, model_temp, obs_time, obs_temp)[:3]


def results_end_problem(sim_dir: Path, last_time, output_window: tuple[str, str] | None = None) -> str | None:
    """None if the last results.nc record reaches the end of the scored period, else the reason.

    The scored period ends at model.time_end of the trial config or at the end of OUTPUT_WINDOW, whichever is earlier. Records are
    labelled at the end of their step and written every output_res steps, so a complete run ends within one record spacing of it.
    output_window=None uses OUTPUT_WINDOW."""
    cfg = load_config(sim_dir / "config.yaml")
    end = pd.to_datetime(cfg["model"]["time_end"], dayfirst=True)
    window = output_window or OUTPUT_WINDOW
    if window is not None:
        end = min(end, pd.to_datetime(window[1], dayfirst=True))
    spacing = pd.Timedelta(hours=cfg["model"]["timestep"] * cfg["output"]["output_res"])
    if pd.Timestamp(last_time) < end - spacing:
        return f"results.nc stops at {pd.Timestamp(last_time)}, before the end of the scored period ({end})"
    return None


def to_native_datetime(dt):
    """cftime / list of cftime to datetime.datetime."""
    if hasattr(dt, "to_datetime"):
        return dt.to_datetime()
    if isinstance(dt, (list, np.ndarray)):
        return [to_native_datetime(d) for d in dt]
    if hasattr(dt, "year"):
        return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    return dt


CHILD_SCRIPT = """\
from pathlib import Path
import sys
proj_dir = Path({repo!r})
sys.path.insert(0, str(proj_dir))
from main import main, Simulation
import gc
sim = Simulation({sim_dir!r}, proj_dir=proj_dir)
main(sim, print_to_screen=False)
gc.collect()
"""


def score_borehole_2016(sim_dir: Path, sensor_depth_offset: float = SENSOR_DEPTH_OFFSET, make_figure: bool = MAKE_FIGURE,
                        output_window: tuple[str, str] | None = None) -> tuple[float, pd.DataFrame]:
    """Score results/results.nc of sim_dir against the borehole record (BI_T-1, 17 depths) and return (objective, per-depth table).

    The objective is the mean RMSE over the depths, times 10 if the surface never gets above 0.01 degC. The table has the columns
    depth_m (sensor depth), rmse, mae, bias, n. Writes rmse_bias_layers.txt and, with make_figure, temperature_layers.png into sim_dir.
    A results.nc that stops before the end of the scored period gives (inf, empty table). output_window=None uses OUTPUT_WINDOW."""
    nc_path = sim_dir / "results" / "results.nc"
    df_val = pd.read_csv(OBSERVATION_CSV, parse_dates=["date_time"])

    with POSTPROC_LOCK, Dataset(nc_path, "r") as nc:
        time_units = nc.variables["time"].units
        time_dt = netCDF4.num2date(nc.variables["time"][:], units=time_units)
        time_pd = pd.to_datetime(np.array([to_native_datetime(t) for t in time_dt]))
        ground_temp = nc.variables["ground_temperature_distribution"][:]  # (nt, nx, nz)
        abs_zgr = nc.variables["abs_zgr"][:][0, 1, :]  # column of grid point 1, as in the published driver

    problem = results_end_problem(sim_dir, time_pd[-1], output_window)
    if problem:
        print(f"  => run incomplete, not scored: {problem}")
        return float("inf"), pd.DataFrame(columns=["depth_m", "rmse", "mae", "bias", "n"])

    depth_model = (abs_zgr - abs_zgr[0]) * -1
    depths_cm = [d for d in range(0, 426, 25) if d != 75]  # 0, 25, 50, 100, ..., 425 cm (17 depths)

    rmse_list, bias_list, rows, model_temps = [], [], [], {}
    surface_temp_model = None
    for d in depths_cm:
        col = f"dpth{d}cm"
        idx = np.abs(depth_model - (d / 100 + sensor_depth_offset)).argmin()
        model_temp = ground_temp[:, 1, idx] - 273.15  # K to degC, middle grid point
        model_temps[col] = model_temp
        if col == "dpth0cm":
            surface_temp_model = model_temp.copy()
        rmse, mae, bias, n = compute_metrics_n(time_pd, model_temp, df_val["date_time"], df_val[col])
        rmse_list.append(rmse)
        bias_list.append(bias)
        rows.append(dict(depth_m=d / 100, rmse=rmse, mae=mae, bias=bias, n=n))

    rmse_mean = np.mean(rmse_list)
    rmse_weight = np.average(rmse_list, weights=np.ones(len(rmse_list)))  # all depths equally weighted

    # penalty: a surface that never gets above freezing means the model never thawed
    if surface_temp_model is not None and np.all(surface_temp_model < 0.01):
        rmse_weight *= 10.0
        print("  WARNING: surface temperature (dpth0cm) is never above zero. Applying penalty factor 10x")

    with open(sim_dir / "rmse_bias_layers.txt", "w") as f:
        f.write(f"Weighted RMSE: {rmse_weight:.4f}\nMean RMSE: {rmse_mean:.4f}\n\n")
        for d, r, b in zip(depths_cm, rmse_list, bias_list):
            f.write(f"dpth{d}cm: RMSE={r:.4f}, Bias={b:.4f}\n")

    if make_figure:
        with POSTPROC_LOCK:
            selected = [0, 50, 100, 200, 300]
            fig, axs = plt.subplots(len(selected), 1, figsize=(18, 12), sharex=True)
            for ax, d, color in zip(axs, selected, ["C0", "C1", "C2", "C3", "C4"]):
                col = f"dpth{d}cm"
                rmse, mae, bias = compute_metrics(time_pd, model_temps[col], df_val["date_time"], df_val[col])
                ax.plot(time_pd, model_temps[col], color, label="Modelled")
                ax.plot(df_val["date_time"], df_val[col], "k", label="Measured")
                ax.text(0.95, 0.05, f"{d} cm depth\nRMSE = {rmse:.2f} °C\nMAE = {mae:.2f} °C\nBias = {bias:.2f} °C",
                        transform=ax.transAxes, fontsize=12, va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
                ax.set_xlim((pd.to_datetime("2016-05-15"), pd.to_datetime("2016-09-30")))
                ax.set_ylim((-10, 5))
                ax.legend(loc="upper left")
                ax.set_ylabel("Temperature [C]")
                ax.grid()
            axs[-1].set_xlabel("Time")
            fig.tight_layout()
            fig.savefig(sim_dir / "temperature_layers.png", dpi=300)
            plt.close(fig)

    return rmse_weight, pd.DataFrame(rows)


def run_simulation(sim_dir: Path) -> float:
    """Run the model of one trial in a child process and return the objective (mean RMSE over the observed depths)."""
    script = sim_dir / "run_simulation_subprocess.py"
    script.write_text(CHILD_SCRIPT.format(repo=str(REPO_DIR), sim_dir=str(sim_dir)))
    result = subprocess.run([sys.executable, str(script)], cwd=str(sim_dir))
    print(f" => simulation finished: {sim_dir.name}, exit code {result.returncode}")  # a native crash leaves only this

    nc_path = sim_dir / "results" / "results.nc"
    try:
        if result.returncode != 0:  # a killed run can leave a partial results.nc
            print(f"  => run failed (exit code {result.returncode}), not scored: {sim_dir.name}")
            return float("inf")
        rmse_weight, _ = score_borehole_2016(sim_dir, SENSOR_DEPTH_OFFSET, MAKE_FIGURE)  # module values read here, not at definition
        if np.isinf(rmse_weight):  # incomplete results.nc
            return rmse_weight
        gc.collect()
        print("=> finished this iteration")

    except Exception as e:
        print(f"  ERROR processing {sim_dir.name}: {e}")
        rmse_weight = float("inf")

    finally:
        try:
            if not KEEP_RESULTS_NC and nc_path.exists():
                nc_path.unlink()
        except Exception:
            pass
    return rmse_weight


def objective(trial: optuna.Trial) -> float:
    params = suggest_thermal_params(trial)
    for k, v in PARAM_RANGES.items():  # remaining (fixed or plain) parameters
        if k in params:
            continue
        if isinstance(v[0], int) and isinstance(v[1], int):
            params[k] = trial.suggest_int(k, int(v[0]), int(v[1]))
        else:
            params[k] = trial.suggest_float(k, float(v[0]), float(v[1]))

    work_dir, sim_dir = hpc.trial_dirs(RUN_ROOT_DIR, SCRATCH, trial.number)  # the same folder unless SCRATCH is set
    try:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(BASE_SIM_DIR, work_dir)
        update_config_yaml(work_dir, params)
        rmse = run_simulation(work_dir)
    finally:  # also after a failed run: run.log is needed
        hpc.finish_trial_dir(work_dir, sim_dir, KEEP_RESULTS_NC, KEEP_XB_FILES)
    trial.set_user_attr("sim_dir", str(sim_dir))
    trial.set_user_attr("rmse", float(rmse))

    hpc.append_log(RUN_ROOT_DIR / "calibration_prints.txt",
                   f"Trial {trial.number} completed.\n  Parameters: {params}\n  RMSE: {rmse}\n  Simulation directory: {sim_dir}\n")
    return float(rmse)


def main():
    RUN_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print_log = RUN_ROOT_DIR / "calibration_prints.txt"
    hpc.append_log(print_log, f"Calibration run started: {datetime.datetime.now()}\n")

    study = hpc.open_study(STUDY_NAME, RUN_ROOT_DIR, STORAGE, optuna.samplers.TPESampler(seed=SEED, constant_liar=CONSTANT_LIAR))
    if START_PARAMS is not None and len(study.trials) == 0:
        study.enqueue_trial(START_PARAMS)  # first trial of a new study

    n_complete = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    n_todo = max(N_TRIALS - n_complete, 0)
    print(f"Study has {n_complete} completed trials; running {n_todo} more to reach {N_TRIALS}")
    try:
        study.optimize(objective, n_trials=n_todo, n_jobs=N_PARALLEL, callbacks=[lambda study, trial: gc.collect()])
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
        hpc.append_log(print_log, "Optimization interrupted by user.\n")

    study.trials_dataframe().to_csv(RUN_ROOT_DIR / "calibration_results.csv", index=False)
    best = study.best_trial
    hpc.append_log(print_log, f"\nCalibration finished: {datetime.datetime.now()}\nBest parameters found: {best.params}\n"
                              f"Best trial number: {best.number}\nBest RMSE: {best.value}\n")
    print("Best parameters found:", best.params)
    print("Best trial number:", best.number)
    print("Best RMSE:", best.value)

    values = [t.value for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(values, "b-", alpha=0.5, label="Trial RMSE")
    ax.plot(pd.Series(values).cummin(), "r-", lw=2, label="Best RMSE")
    ax.set_xlabel("Trial")
    ax.set_ylabel("RMSE (°C)")
    ax.legend()
    ax.grid(True)
    fig.savefig(RUN_ROOT_DIR / "convergence.png", dpi=150)
    gc.collect()


if __name__ == "__main__":
    main()
