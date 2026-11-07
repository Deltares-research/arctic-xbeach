"""Step 2 of the calibration: coupled (thermal + XBeach) Optuna calibration of the XBeach and wrapper parameters, thermal parameters fixed.

Port of the published driver (coupled study 20260214, best trial 88). Objective, parameter ranges and metrics are unchanged:
the combined RMSE sqrt((RMSE_bluff_edge^2 + RMSE_shoreline^2) / 2) of the modelled bluff edge (z = BLUFF_ELEVATION) and shoreline
(z = SHORELINE_ELEVATION), both normalised at NORMALIZATION_DATE, over the calibration period. What changed for v1.1: paths in the
configuration block (pathlib), seeded sampler, optional starting parameter set, N_TRIALS / N_PARALLEL as variables, the v1.1 solver
options in every trial config, trials run in a child process of the current interpreter (Windows and Linux), and the 10-year thermal
spin-up is computed ONCE (RUN_ROOT_DIR/spinup_state.npz) and loaded by every trial (model.spinup_state_in), because XBeach does
not run during the spin-up and the thermal parameters are the same in all trials.

Workflow
  1. template: BASE_SIM_DIR is copied to RUN_ROOT_DIR/template; the thermal section is taken from THERMAL_CONFIG_YAML (the best config.yaml of
     step 1) if given, the v1.1 options and forcing paths are applied.
  2. spin-up: a run of the template without XBeach from time_start to the end of the spin-up writes spinup_state.npz (skipped if it exists
     and was made with the same thermal config; delete the file when forcing, bathymetry or thermal parameters change).
  3. trials: copy of the template + sampled parameters + spinup_state_in, run, score, delete results.nc.

Run:  python calibration/auto_calibrate_coupled_model.py        (see calibration/README.md)

On a Linux cluster the same script is configured through AXB_* environment variables (hpc/h7/coupled_calibration.sbatch). With
STORAGE = "journal" the study is an append-only file that is safe on NFS; with SCRATCH set each trial runs on that (local) disk and
is copied back to RUN_ROOT_DIR without its results/xb_files. The template and the spin-up state stay in RUN_ROOT_DIR. Only ONE driver
process per study is supported (the sbatch file uses --dependency=singleton): with journal storage, trials still RUNNING at start
are taken to be left by a killed earlier job and are marked FAIL.

A trial with more failed XBeach calls than MAX_FAILED_XBEACH_CALLS (counted in results/xbeach_calls.csv) is not scored (inf):
main.py logs a failed call and carries on, which would otherwise be scored as a run without that morphological change.
The same limit is written to xbeach.max_failed_calls of every config, so a trial stops (exit code 3, scored inf) at the first failed
call beyond it instead of running on for hours. AXB_XBEACH_LAUNCHER (xbeach.launcher) starts XBeach through a command prefix such as
"mpirun -np 2" for MPI-only builds.

Exit code: main() exits nonzero (so that sbatch --dependency=afterok holds the follow-up jobs) if no trial of the study has a finite
value. With TIME_END set (smoke run) a trial is scored only up to TIME_END and may be inf for lack of observations; the exit code then reports failed runs instead
(a trial with a nonzero child exit code, failed XBeach calls beyond MAX_FAILED_XBEACH_CALLS, or a FAIL state).
"""
# === CONFIGURATION ===
# The AXB_* environment variables override the defaults below (used by hpc/h7/*.sbatch); without them nothing changes.
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]  # arctic-xbeach checkout that is imported by the trials
BASE_SIM_DIR = Path(os.environ.get("AXB_BASE_SIM_DIR", r"N:\arctic-xbeach-runs\v11_runs\coupled_base_v11"))  # template run folder (config.yaml, x.grd, bed.dep, ne_layer.txt); the v1.0 base with CFL_xbeach 0.9 instead of 0.7
RUN_ROOT_DIR = Path(os.environ.get("AXB_RUN_ROOT", r"N:\arctic-xbeach-runs\v11_runs\coupled_calibration"))  # template, spin-up state, trial folders, optuna_study.db (or optuna_journal.log), logs
THERMAL_CONFIG_YAML = Path(os.environ.get("AXB_THERMAL_CONFIG_YAML", r"N:\arctic-xbeach-runs\v11_runs\thermal_calibration\optuna_trial_000490\config.yaml"))  # best thermal trial of step 1 (RMSE 0.8203 K); its thermal section replaces the template's. None = keep the template's thermal section
THERMAL_DT = int(os.environ.get("AXB_THERMAL_DT", 60))        # [s] thermal timestep of the coupled runs, replaces thermal.dt of the config above. 60 s, as the thermal calibration (wet surface node limit). None = keep it
BLUFF_EDGES_CSV = Path(os.environ.get("AXB_BLUFF_EDGES_CSV", r"D:\Git\thermo-morphological-model\database\ts_datasets\bluff_edges_without_geometry.csv"))  # columns time, distance_from_ref
SHORELINES_CSV = Path(os.environ.get("AXB_SHORELINES_CSV", r"D:\Git\thermo-morphological-model\database\ts_datasets\shore_lines_without_geometry.csv"))
FORCING_CSV = Path(os.environ.get("AXB_FORCING_CSV", r"N:\arctic-xbeach-runs\database\era5.csv"))   # written into every config (None = keep the path of the base config)
STORM_CSV = Path(os.environ.get("AXB_STORM_CSV", r"N:\arctic-xbeach-runs\database\storms.csv"))   # idem
XBEACH_EXE = os.environ.get("AXB_XBEACH_EXE") or None   # path of xbeach.exe written into every config (None = keep the base config's)
XBEACH_LAUNCHER = os.environ.get("AXB_XBEACH_LAUNCHER") or None   # command prefix written to xbeach.launcher of every config, e.g. "/opt/.../mpirun -np 2 --oversubscribe --bind-to none" for an MPI-only XBeach build (None = start the executable directly)

N_TRIALS = int(os.environ.get("AXB_N_TRIALS", 10))          # completed trials the study should reach; a rerun on an existing study runs only the rest
                       # (published study: 25 epochs x 8 workers = 200)
N_PARALLEL = int(os.environ.get("AXB_N_PARALLEL", 8))         # trials running at the same time (one child process each)
MAX_FAILED_XBEACH_CALLS = int(os.environ.get("AXB_MAX_FAILED_XBEACH_CALLS", 0))  # a trial with more failed XBeach calls than this is not scored (inf); also written to xbeach.max_failed_calls, so the run stops at the first failure beyond it
OUTPUT_RES = int(os.environ.get("AXB_OUTPUT_RES", 24))       # [h] thermal record spacing of the trial results.nc. Hourly 2D thermal fields reached ~4.6 GB per trial
                                                             # within 8 h and 43 trials filled a 138 GB H7 scratch (5 Nov 2026); the scorer needs only the XBeach records, written at every call
TIME_END = os.environ.get("AXB_TIME_END") or None   # 'DD-MM-YYYY', replaces model.time_end of the template (short smoke test); None = keep it
SEED = 1               # seed of the TPE sampler (with N_PARALLEL > 1 the order in which trials finish still varies)
START_PARAMS = None    # enqueue this parameter set as the first trial of a new study. Example: the published best (trial 88, value 4.1222)
# START_PARAMS = dict(facua=0.018829910025755153, bedfriccoef=0.033501187112818874, wetslp=0.15233427680340267,
#                     dryslp=0.5578092253749002, xb_threshold=2.012924093703792, lsgrad=0.0, N_thaw_threshold=3)

# v1.1 solver options written into every config (thermal.* and data.*). Legacy values (nodal, per_mass, 0) select the v1.0 solver.
SOLVER_OPTIONS = dict(conduction_form="flux", latent_heat_form="volumetric")
ERA5_FLUX_SHIFT_HOURS = 1
USE_SPINUP_STATE = True  # compute the spin-up once and let every trial load it

# skill definition (published)
SHORELINE_METHOD = "simple"
SHORELINE_ELEVATION = +2.0   # [m]
BLUFF_ELEVATION = 13.2       # [m]
CALIB_START, CALIB_END = "2012-01-01", "2016-01-01"   # observations in this period enter the RMSE
NORMALIZATION_DATE = "2012-07-11"                      # model and observations are matched at this date (offset removed)

STUDY_NAME = os.environ.get("AXB_STUDY_NAME", "calibration_optuna")
STORAGE = os.environ.get("AXB_STORAGE", "sqlite")  # "sqlite" (RUN_ROOT_DIR/optuna_study.db) or "journal" (RUN_ROOT_DIR/optuna_journal.log, for NFS)
SCRATCH = Path(os.environ["AXB_SCRATCH"]) if os.environ.get("AXB_SCRATCH") else None  # trials run here and are copied back to RUN_ROOT_DIR; None = run in RUN_ROOT_DIR
KEEP_XB_FILES = os.environ.get("AXB_KEEP_XB_FILES", "1").lower() not in ("0", "false", "no")  # False = delete results/xb_files of a trial after scoring (about 26,000 files per trial; no effect with SCRATCH: never copied back)
CONSTANT_LIAR = os.environ.get("AXB_CONSTANT_LIAR", "0").lower() in ("1", "true", "yes")  # with ~40 parallel trials this keeps TPE from proposing near-duplicates of the trials still running
PARAM_RANGES_JSON = os.environ.get("AXB_PARAM_RANGES") or None  # JSON {"name": [low, high], ...} replacing those entries of PARAM_RANGES, e.g. a narrowed range for a follow-up study; None = the published ranges
TIMEOUT_H = float(os.environ["AXB_TIMEOUT_H"]) if os.environ.get("AXB_TIMEOUT_H") else None  # [h] no new trial is started after this; running trials finish. Used to end a job before its SLURM --time so that no running trial is lost. None = no limit
SEED_FROM_RUN_ROOT = Path(os.environ["AXB_SEED_FROM_RUN_ROOT"]) if os.environ.get("AXB_SEED_FROM_RUN_ROOT") else None  # RUN_ROOT of an earlier study whose usable trials start this one (read only); needs SEED_FROM_STUDY. None = no seeding
SEED_FROM_STUDY = os.environ.get("AXB_SEED_FROM_STUDY") or None  # name of that study
SEED_FROM_STORAGE = os.environ.get("AXB_SEED_FROM_STORAGE") or STORAGE  # storage kind of that study ("sqlite" or "journal")
MAKE_FIGURE = True
REMOVE_RESULTS_NC = True
# === END CONFIGURATION ===

# One BLAS / OpenMP thread per process: 40 trials would otherwise each try to use all cores. Set before numpy is imported;
# the child processes (and XBeach) inherit them (main.py sets the same OpenBLAS default for its own runs).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import datetime
import gc
import json
import shutil
import subprocess
import sys
import threading
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so that optuna_hpc is found when the file is not run as a script
sys.path.insert(0, str(REPO_DIR))  # for arctic_xbeach.config_loader
from arctic_xbeach.config_loader import load_config  # a duplicate key in a config.yaml is an error
import optuna_hpc as hpc  # calibration/optuna_hpc.py, next to this script
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xarray as xr
import yaml

warnings.filterwarnings("ignore")

# Optuna runs the n_jobs trials as threads of this process; HDF5 (netCDF4) and pyplot are not thread-safe, so the
# post-processing of one trial at a time holds this lock (two reads at once gave "NetCDF: HDF error" in the thermal study).
POSTPROC_LOCK = threading.Lock()

def is_int_range(bounds) -> bool:
    """Both bounds int: the parameter is sampled with suggest_int, else with suggest_float."""
    return isinstance(bounds[0], int) and isinstance(bounds[1], int)


def override_param_ranges(ranges: dict, text: str | None) -> dict:
    """Copy of ranges with the entries named in the JSON text (AXB_PARAM_RANGES) replaced. Exits on a bad key or value.
    A parameter keeps its sampling type: float parameters get float bounds, an int parameter needs int bounds."""
    out = {k: list(v) for k, v in ranges.items()}
    if not text:
        return out
    try:
        new = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"AXB_PARAM_RANGES is not valid JSON: {e}")
    if not isinstance(new, dict):
        raise SystemExit(f"AXB_PARAM_RANGES must be a JSON object {{name: [low, high]}}, got {text!r}")
    for k, v in new.items():
        if k not in ranges:
            raise SystemExit(f"AXB_PARAM_RANGES: unknown parameter {k!r}; known parameters: {', '.join(ranges)}")
        if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) for x in v)):
            raise SystemExit(f"AXB_PARAM_RANGES: {k!r} must be two finite numbers [low, high], got {v!r}")
        if v[0] > v[1]:
            raise SystemExit(f"AXB_PARAM_RANGES: {k!r} has low > high: {v!r}")
        if is_int_range(ranges[k]):
            if not all(isinstance(x, int) for x in v):
                raise SystemExit(f"AXB_PARAM_RANGES: {k!r} is an integer parameter, its bounds must be integers, got {v!r}")
            out[k] = list(v)
        else:
            out[k] = [float(v[0]), float(v[1])]  # [2, 3] for xb_threshold must stay a float range (suggest_float, not suggest_int)
    return out


# Parameter ranges [low, high]; low == high means fixed. facSk and facAs are set equal to facua when the config is written.
PARAM_RANGES = {
    "facua": [0.00, 0.40],
    "bedfriccoef": [0.015, 0.035],
    "wetslp": [0.05, 0.3],
    "dryslp": [0.5, 1.0],
    "xb_threshold": [2.0, 2.5],     # [m + MSL]; ranges from ~1 % (2.5) to ~5 % (2.0) of the time
    "lsgrad": [0.0, 0.0],
    "N_thaw_threshold": [3, 3],
}
PARAM_RANGES = override_param_ranges(PARAM_RANGES, PARAM_RANGES_JSON)  # AXB_PARAM_RANGES

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


# --- observations (published: read with geopandas, here plain pandas; same columns and casts) --------------------------
def _read_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["distance_from_ref"] = df["distance_from_ref"].astype(np.float64)
    return df.sort_values("time").reset_index(drop=True)


DF_BE = _read_positions(BLUFF_EDGES_CSV)
DF_SL = _read_positions(SHORELINES_CSV)
_calib = lambda df: df[(df["time"] >= pd.Timestamp(CALIB_START)) & (df["time"] <= pd.Timestamp(CALIB_END))]
DF_BE_CALIB, DF_SL_CALIB = _calib(DF_BE), _calib(DF_SL)
NORM_DATE = pd.Timestamp(NORMALIZATION_DATE)


# --- config handling -------------------------------------------------------------------------------------------------
def apply_common_options(config: dict) -> dict:
    """Solver options, forcing paths and executable shared by all runs (template, spin-up run, trials)."""
    config["thermal"].update(SOLVER_OPTIONS)
    config["data"]["era5_flux_shift_hours"] = ERA5_FLUX_SHIFT_HOURS
    if FORCING_CSV is not None:
        config["data"]["forcing_data_path"] = str(FORCING_CSV)
    if STORM_CSV is not None:
        config["data"]["storm_data_path"] = str(STORM_CSV)
    if XBEACH_EXE is not None:
        config["xbeach"]["version"] = str(XBEACH_EXE)
    if XBEACH_LAUNCHER is not None:
        config["xbeach"]["launcher"] = XBEACH_LAUNCHER
    config["xbeach"]["max_failed_calls"] = MAX_FAILED_XBEACH_CALLS  # main.py stops a run at the first failed call beyond this (the spin-up run has no XBeach, so it is not affected)
    config.setdefault("output", {})["output_res"] = OUTPUT_RES  # the spin-up run overrides this with 10**9
    config.setdefault("output", {})["log_xbeach_calls"] = True  # xbeach_calls.csv is how failed XBeach calls are counted
    return config


def update_config_yaml(sim_dir: Path, params: dict, extra_model: dict | None = None):
    """Write the sampled XBeach / wrapper parameters (and extra model options) into config.yaml of a trial."""
    config_path = sim_dir / "config.yaml"
    config = load_config(config_path)
    clean = {}
    for k, v in params.items():
        clean[k] = int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v.tolist() if isinstance(v, np.ndarray) else v

    config.setdefault("xbeach", {})
    config.setdefault("wrapper", {})
    for k, v in clean.items():
        if k == "xb_threshold":
            config["wrapper"][k] = v
        elif k == "N_thaw_threshold":  # both sections, as in the published driver
            config["wrapper"][k] = v
            config["xbeach"][k] = v
        else:
            config["xbeach"][k] = v
    if "facua" in clean:  # one friction value for all three factors
        config["xbeach"]["facSk"] = clean["facua"]
        config["xbeach"]["facAs"] = clean["facua"]
    if extra_model:
        config["model"].update(extra_model)
    config_path.write_text(yaml.dump(config, default_flow_style=False))


def prepare_template() -> Path:
    """RUN_ROOT_DIR/template: BASE_SIM_DIR with the thermal section of step 1, the v1.1 options and the forcing paths."""
    template = RUN_ROOT_DIR / "template"
    if template.exists():
        shutil.rmtree(template)
    shutil.copytree(BASE_SIM_DIR, template)
    config = load_config(template / "config.yaml")
    if THERMAL_CONFIG_YAML is not None:
        config["thermal"] = load_config(Path(THERMAL_CONFIG_YAML))["thermal"]
    if THERMAL_DT is not None:
        config["thermal"]["dt"] = THERMAL_DT  # after the replacement above, which brings the dt of the thermal calibration
    if TIME_END is not None:
        config["model"]["time_end"] = TIME_END
    apply_common_options(config)
    (template / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))
    return template


CHILD_EXIT_CODES: dict[str, int] = {}  # sim_dir -> exit code of its last child process; objective() stores it as a trial attribute


def launch(sim_dir: Path) -> int:
    """Run main.py on sim_dir in a child process of the current interpreter and return its exit code."""
    script = sim_dir / "run_simulation_subprocess.py"
    script.write_text(CHILD_SCRIPT.format(repo=str(REPO_DIR), sim_dir=str(sim_dir)))
    result = subprocess.run([sys.executable, str(script)], cwd=str(sim_dir))
    CHILD_EXIT_CODES[str(sim_dir)] = result.returncode
    print(f" => child process exit code {result.returncode}: {sim_dir.name}")  # a native crash leaves only this
    return result.returncode


def ensure_spinup_state(template: Path) -> Path:
    """Create RUN_ROOT_DIR/spinup_state.npz once (spin-up without XBeach). An existing file must match the template's thermal config."""
    state = RUN_ROOT_DIR / "spinup_state.npz"
    config = load_config(template / "config.yaml")

    if state.exists():
        sys.path.insert(0, str(REPO_DIR))
        from arctic_xbeach.model import Simulation
        with np.load(state, allow_pickle=False) as z:
            stored = json.loads(str(z["__meta__"]))["thermal_hash"]
        if stored != Simulation.thermal_state_hash(config["thermal"]):
            raise SystemExit(f"{state} was made with a different thermal config than the template. Delete it to recompute the spin-up.")
        print(f"Using existing spin-up state {state}")
        return state

    spin_years = float(config["model"].get("spin_up_time", 0))
    if spin_years <= 0:
        raise SystemExit("model.spin_up_time is 0 in the template: nothing to save. Set USE_SPINUP_STATE = False.")
    run_dir = RUN_ROOT_DIR / "_spinup"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(template, run_dir)
    config["xbeach"]["with_xbeach"] = False                       # thermal only; the state is the thermal state on the initial bathymetry
    config["output"]["output_res"] = 10**9                        # no hourly NetCDF output
    end = pd.to_datetime(config["model"]["time_start"], dayfirst=True) + pd.Timedelta(days=spin_years * 365.25 + 2)
    config["model"]["time_end"] = end.strftime("%d-%m-%Y")        # a little beyond the spin-up so that its last step is run
    config["model"]["spinup_state_out"] = str(state)
    (run_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    print(f"Computing the spin-up once ({spin_years:g} years) -> {state}")
    t0 = time.time()
    if launch(run_dir) != 0:
        raise SystemExit(f"Spin-up run failed (nonzero exit code); see {run_dir / 'run.log'}")
    if not state.exists():
        raise SystemExit(f"Spin-up run did not write {state}; see {run_dir / 'run.log'}")
    print(f"Spin-up done in {(time.time() - t0) / 60:.1f} min")
    return state


# --- skill -----------------------------------------------------------------------------------------------------------
def compute_rmse_mae(obs_vals, model_vals):
    obs_vals, model_vals = np.array(obs_vals), np.array(model_vals)
    return np.sqrt(np.mean((obs_vals - model_vals) ** 2)), np.mean(np.abs(obs_vals - model_vals))


def find_shoreline(x, z, msl=0.0, method="simple"):
    """x of the first crossing of z = msl (linear interpolation), NaN if there is none."""
    if method != "simple":
        raise ValueError(f"Unknown method: {method}")
    crossings = np.where(np.diff(np.sign(z - msl)))[0]
    if len(crossings) == 0:
        return np.nan
    i = crossings[0]
    return x[i] + (msl - z[i]) * (x[i + 1] - x[i]) / (z[i + 1] - z[i])


def find_bluff_edge(x, z, bluff_elevation=13.2):
    """x of the first crossing of z = bluff_elevation, NaN if there is none."""
    return find_shoreline(x, z, msl=bluff_elevation, method="simple")


def results_end_problem(sim_dir: Path, last_time) -> str | None:
    """None if the last record of the thermal time variable reaches the end of the scored period, else the reason.

    The scored period ends at CALIB_END or model.time_end of the trial config, whichever is earlier. The XBeach records
    (time_xbeach) are only written at XBeach calls and can stop weeks before the end, so the thermal records are checked. They are
    labelled at the end of their step and written every output_res steps: a complete run ends within one record spacing of it."""
    cfg = load_config(sim_dir / "config.yaml")
    end = min(pd.to_datetime(cfg["model"]["time_end"], dayfirst=True), pd.Timestamp(CALIB_END))
    spacing = pd.Timedelta(hours=cfg["model"]["timestep"] * cfg["output"]["output_res"])
    if pd.Timestamp(last_time) < end - spacing:
        return f"results.nc stops at {pd.Timestamp(last_time)}, before the end of the scored period ({end})"
    return None


def count_failed_xbeach_calls(sim_dir: Path) -> int | None:
    """Number of XBeach calls with success = 0 in results/xbeach_calls.csv (written by main.py, one row per call), None if the
    file is missing. main.py logs a failed call and carries on, so only this file shows it."""
    path = sim_dir / "results" / "xbeach_calls.csv"
    if not path.exists():
        return None
    return int((pd.read_csv(path)["success"] == 0).sum())


def run_simulation(sim_dir: Path) -> tuple[float, int | None]:
    """Run one trial in a child process and return (combined RMSE, number of failed XBeach calls).

    The RMSE is inf if the run or the scoring failed: a child with a nonzero exit code, a results.nc that stops before the end of
    the scored period, or more than MAX_FAILED_XBEACH_CALLS failed XBeach calls is not scored. The count is None if it is unknown."""
    returncode = launch(sim_dir)
    print(f" => simulation finished: {sim_dir.name}")

    nc_path = sim_dir / "results" / "results.nc"
    n_failed = None
    try:
        if returncode != 0:  # a killed run can leave a partial results.nc
            print(f"  => run failed (exit code {returncode}), not scored: {sim_dir.name}")
            return float("inf"), n_failed
        if not nc_path.exists():
            print(f"  WARNING: results file not found: {nc_path}")
            return float("inf"), n_failed

        cfg = load_config(sim_dir / "config.yaml")
        if cfg.get("xbeach", {}).get("with_xbeach", True):
            n_failed = count_failed_xbeach_calls(sim_dir)
            if n_failed is None:
                print(f"  => xbeach_calls.csv not found in {sim_dir / 'results'} (no XBeach call was made, or output.log_xbeach_calls is false), "
                      f"failed calls cannot be counted, not scored: {sim_dir.name}")
                return float("inf"), n_failed
            if n_failed > MAX_FAILED_XBEACH_CALLS:
                print(f"  => {n_failed} failed XBeach calls (allowed: {MAX_FAILED_XBEACH_CALLS}), not scored: {sim_dir.name}")
                return float("inf"), n_failed
        else:
            n_failed = 0

        with POSTPROC_LOCK:  # variables are loaded inside the lock (xarray reads lazily), the loop below then works on memory
            with xr.open_dataset(nc_path, engine="netcdf4") as ds:
                zb_xbeach, x_xbeach = ds["zb_xbeach"].load(), ds["xgr"].load()
                time_datetime = pd.to_datetime(ds["time_xbeach"].values)
                last_time = pd.to_datetime(ds["time"].values[-1])
        problem = results_end_problem(sim_dir, last_time)
        if problem:
            print(f"  => run incomplete, not scored: {problem}")
            return float("inf"), n_failed
        n_times = len(time_datetime)

        shoreline_pos, bluff_pos = np.zeros(n_times), np.zeros(n_times)
        for i in range(n_times):
            zb, x = zb_xbeach.isel(time_xbeach=i).values, x_xbeach.values
            valid = ~np.isnan(zb)
            if valid.sum() > 2:
                shoreline_pos[i] = find_shoreline(x[valid], zb[valid], msl=SHORELINE_ELEVATION, method=SHORELINE_METHOD)
                bluff_pos[i] = find_bluff_edge(x[valid], zb[valid], bluff_elevation=BLUFF_ELEVATION)
            else:
                shoreline_pos[i] = bluff_pos[i] = np.nan

        # 1-day centred smoothing
        shoreline_pos = pd.Series(shoreline_pos, index=time_datetime).rolling(window="1D", center=True, min_periods=1).mean().values
        bluff_pos = pd.Series(bluff_pos, index=time_datetime).rolling(window="1D", center=True, min_periods=1).mean().values

        # normalise model and observations at the common anchor date
        norm = NORM_DATE.value
        model_t = time_datetime.values.astype("datetime64[ns]").astype(np.int64)
        obs_be_at_norm = float(np.interp(norm, DF_BE["time"].values.astype("datetime64[ns]").astype(np.int64), DF_BE["distance_from_ref"].values))
        obs_sl_at_norm = float(np.interp(norm, DF_SL["time"].values.astype("datetime64[ns]").astype(np.int64), DF_SL["distance_from_ref"].values))
        bluff_offset = obs_be_at_norm - float(np.interp(norm, model_t, bluff_pos, left=bluff_pos[0], right=bluff_pos[-1]))
        shore_offset = obs_sl_at_norm - float(np.interp(norm, model_t, shoreline_pos, left=shoreline_pos[0], right=shoreline_pos[-1]))
        bluff_norm, shore_norm = bluff_pos + bluff_offset, shoreline_pos + shore_offset

        be_t = DF_BE_CALIB["time"].values.astype("datetime64[ns]").astype(np.int64)
        rmse_bluff, mae_bluff = compute_rmse_mae(DF_BE_CALIB["distance_from_ref"].values, np.interp(be_t, model_t, bluff_norm))
        sl_t = DF_SL_CALIB["time"].values.astype("datetime64[ns]").astype(np.int64)
        rmse_shore, mae_shore = compute_rmse_mae(DF_SL_CALIB["distance_from_ref"].values, np.interp(sl_t, model_t, shore_norm))

        combined = np.sqrt((rmse_bluff**2 + rmse_shore**2) / 2) if not (np.isnan(rmse_bluff) or np.isnan(rmse_shore)) else np.nan

        with open(sim_dir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"Bluff Edge RMSE: {rmse_bluff:.4f} m\nBluff Edge MAE: {mae_bluff:.4f} m\n"
                    f"Shoreline RMSE: {rmse_shore:.4f} m\nShoreline MAE: {mae_shore:.4f} m\n"
                    f"Combined RMSE (sqrt((bluff^2+shore^2)/2)): {combined:.4f} m\n")

        if MAKE_FIGURE:
            with POSTPROC_LOCK:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                ax1.plot(time_datetime, bluff_norm, "b-", label="Model", lw=2)
                ax1.plot(DF_BE_CALIB["time"], DF_BE_CALIB["distance_from_ref"], "ko", label="Observed", ms=6)
                ax1.set_ylabel("Bluff edge position (m)", fontsize=12)
                ax1.legend(loc="upper left")
                ax1.grid(True, alpha=0.3)
                ax1.text(0.98, 0.05, f"RMSE = {rmse_bluff:.2f} m, MAE = {mae_bluff:.2f} m", transform=ax1.transAxes, ha="right", fontsize=12)
                ax2.plot(time_datetime, shore_norm, "r-", label="Model", lw=2)
                ax2.plot(DF_SL_CALIB["time"], DF_SL_CALIB["distance_from_ref"], "ko", label="Observed", ms=6)
                ax2.set_ylabel("Shoreline position (m)", fontsize=12)
                ax2.set_xlabel("Time", fontsize=12)
                ax2.legend(loc="upper left")
                ax2.grid(True, alpha=0.3)
                ax2.text(0.98, 0.05, f"RMSE = {rmse_shore:.2f} m, MAE = {mae_shore:.2f} m", transform=ax2.transAxes, ha="right", fontsize=12)
                fig.tight_layout()
                fig.savefig(sim_dir / "calibration_timeseries.png", dpi=300)
                plt.close(fig)

        gc.collect()
        return combined, n_failed

    except Exception as e:
        print(f"  ERROR processing results of {sim_dir.name}: {e}")
        return float("inf"), n_failed

    finally:
        if REMOVE_RESULTS_NC and nc_path.exists():
            for _ in range(3):  # Windows can hold the file briefly
                try:
                    os.remove(str(nc_path))
                    break
                except PermissionError:
                    time.sleep(0.5)
                    gc.collect()
                except Exception:
                    break
        gc.collect()


SPINUP_STATE = None  # set in main()
TEMPLATE = None


def param_distribution(bounds):
    """The distribution that objective() gets from suggest_int / suggest_float for these bounds (same rule: is_int_range)."""
    if is_int_range(bounds):
        return optuna.distributions.IntDistribution(int(bounds[0]), int(bounds[1]))
    return optuna.distributions.FloatDistribution(float(bounds[0]), float(bounds[1]))


def log_line(text: str) -> None:
    """Print, and append to calibration_prints.txt once RUN_ROOT_DIR exists."""
    print(text)
    if RUN_ROOT_DIR.exists():
        hpc.append_log(RUN_ROOT_DIR / "calibration_prints.txt", text + "\n")


def seed_study(study: optuna.Study, src_root: Path, src_name: str, src_storage: str) -> int:
    """Copy the usable trials of an earlier study into the new, empty study; returns how many. The source is only read.

    Usable: COMPLETE, finite value, a value for every key of PARAM_RANGES, each inside the current [low, high] (inclusive; a fixed
    parameter must equal its bound). Only the PARAM_RANGES parameters are copied, with the distributions objective() creates, so TPE
    treats seeded and new trials alike. A study that already has trials (a rerun) is left alone."""
    if study.get_trials(deepcopy=False):
        log_line(f"Study already has trials: not seeded from {src_name} ({src_root})")
        return 0
    src_file = src_root / ("optuna_journal.log" if src_storage == "journal" else "optuna_study.db")
    if not src_file.exists():  # opening a journal that is not there would create an empty one in the source folder
        raise SystemExit(f"Seeding: {src_file} not found (AXB_SEED_FROM_RUN_ROOT / AXB_SEED_FROM_STORAGE)")
    try:
        source = optuna.load_study(study_name=src_name, storage=hpc.make_storage(src_storage, src_root))
    except KeyError:
        raise SystemExit(f"Seeding: no study named {src_name!r} in {src_file}")
    if source.direction != study.direction:
        raise SystemExit(f"Seeding: {src_name} optimises {source.direction.name}, this study {study.direction.name}")
    distributions = {k: param_distribution(v) for k, v in PARAM_RANGES.items()}
    src_trials = source.get_trials(deepcopy=False)
    n = 0
    for t in src_trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None or not np.isfinite(t.value):
            continue
        if not all(k in t.params and lo <= t.params[k] <= hi for k, (lo, hi) in PARAM_RANGES.items()):
            continue
        study.add_trial(optuna.trial.create_trial(
            params={k: t.params[k] for k in PARAM_RANGES}, distributions=distributions, value=float(t.value),
            user_attrs={**t.user_attrs, "seeded_from": f"{src_name}#{t.number}"}))
        n += 1
    log_line(f"seeded {n} of {len(src_trials)} trials from {src_name} ({src_root})")
    return n


def objective(trial: optuna.Trial) -> float:
    params = {}
    for k, v in PARAM_RANGES.items():
        if is_int_range(v):
            params[k] = trial.suggest_int(k, int(v[0]), int(v[1]))
        else:
            params[k] = trial.suggest_float(k, float(v[0]), float(v[1]))

    work_dir, sim_dir = hpc.trial_dirs(RUN_ROOT_DIR, SCRATCH, trial.number)  # the same folder unless SCRATCH is set
    try:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(TEMPLATE, work_dir)
        update_config_yaml(work_dir, params, extra_model={"spinup_state_in": str(SPINUP_STATE)} if SPINUP_STATE else None)
        rmse, n_failed = run_simulation(work_dir)
        trial.set_user_attr("exit_code", CHILD_EXIT_CODES.pop(str(work_dir), None))  # None if no child was launched (stubs)
    finally:  # also after a failed run: run.log is needed
        hpc.finish_trial_dir(work_dir, sim_dir, not REMOVE_RESULTS_NC, KEEP_XB_FILES)
    trial.set_user_attr("sim_dir", str(sim_dir))
    trial.set_user_attr("rmse", float(rmse))
    trial.set_user_attr("n_failed_xbeach_calls", n_failed)  # None if the run did not get as far as counting

    try:
        hpc.append_log(RUN_ROOT_DIR / "calibration_prints.txt",
                       f"Trial {trial.number} completed.\n  Parameters: {params}\n  RMSE: {rmse}\n  Simulation directory: {sim_dir}\n")
    except Exception:
        pass
    return float(rmse)


def study_exit_problem(study: optuna.Study) -> str | None:
    """None if the study succeeded, else why the driver must exit nonzero. Prints which rule was applied.

    Without TIME_END: at least one COMPLETE trial must have a finite value. With TIME_END (smoke run, scored only up to TIME_END,
    so a clean trial can still be inf): no trial may have a nonzero child exit code, more failed XBeach calls than allowed, or a FAIL state."""
    trials = study.get_trials(deepcopy=False)
    if TIME_END is None:
        print("Exit code rule: the study needs at least one completed trial with a finite value")
        n_ok = sum(t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and np.isfinite(t.value) for t in trials)
        return None if n_ok else f"None of the {len(trials)} trials has a finite value (all failed or were not scored; see the run.log of the trials)"
    print(f"Exit code rule (AXB_TIME_END = {TIME_END}, smoke run): no trial may have failed")
    bad = [t.number for t in trials if t.state == optuna.trial.TrialState.FAIL
           or t.user_attrs.get("exit_code") not in (0, None)
           or (t.user_attrs.get("n_failed_xbeach_calls") or 0) > MAX_FAILED_XBEACH_CALLS]
    return f"Trials {bad} failed (nonzero exit code, failed XBeach calls beyond {MAX_FAILED_XBEACH_CALLS}, or FAIL state)" if bad else None


def main():
    global SPINUP_STATE, TEMPLATE
    RUN_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    print_log = RUN_ROOT_DIR / "calibration_prints.txt"
    hpc.append_log(print_log, f"Calibration run started: {datetime.datetime.now()}\n")
    log_line(f"Parameter ranges: {PARAM_RANGES}")

    TEMPLATE = prepare_template()
    if USE_SPINUP_STATE:
        SPINUP_STATE = ensure_spinup_state(TEMPLATE)
    if N_TRIALS == 0:  # spin-up only (e.g. on a small node while the study job waits with --dependency=afterok)
        hpc.append_log(print_log, f"N_TRIALS = 0: spin-up only, no study opened. Spin-up state: {SPINUP_STATE}\n")
        print(f"N_TRIALS = 0: spin-up only, no study opened. Spin-up state: {SPINUP_STATE}")
        return

    study =hpc.open_study(STUDY_NAME, RUN_ROOT_DIR, STORAGE, optuna.samplers.TPESampler(seed=SEED, constant_liar=CONSTANT_LIAR))  # a rerun continues the study
    if SEED_FROM_RUN_ROOT is not None and SEED_FROM_STUDY is not None:
        seed_study(study, SEED_FROM_RUN_ROOT, SEED_FROM_STUDY, SEED_FROM_STORAGE)  # before n_todo: seeded trials count toward N_TRIALS
    if START_PARAMS is not None and len(study.trials) == 0:
        study.enqueue_trial(START_PARAMS)  # first trial of a new study (not after seeding)

    n_complete = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    n_todo = max(N_TRIALS - n_complete, 0)
    print(f"Study has {n_complete} completed trials; running {n_todo} more to reach {N_TRIALS}")
    try:
        study.optimize(objective, n_trials=n_todo, n_jobs=N_PARALLEL, timeout=TIMEOUT_H * 3600 if TIMEOUT_H is not None else None,
                       callbacks=[lambda study, trial: gc.collect()])
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
        hpc.append_log(print_log, "Optimization interrupted by user.\n")

    try:
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
        ax.set_ylabel("RMSE (m)")
        ax.legend()
        ax.grid(True)
        fig.savefig(RUN_ROOT_DIR / "convergence.png", dpi=150)
    except Exception as e:
        print(f"No completed trials or failed to summarise the study: {e}")
    gc.collect()

    problem = study_exit_problem(study)
    if problem:  # a nonzero exit code keeps sbatch --dependency=afterok from starting the follow-up jobs
        hpc.append_log(print_log, f"STUDY FAILED: {problem}\n")
        print(f"ERROR: {problem}")
        sys.exit(1)


if __name__ == "__main__":
    main()
