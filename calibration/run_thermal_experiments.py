"""Thermal-only follow-up runs (no XBeach) of one calibrated parameter set: 2011 validation, one-at-a-time sensitivity, thermal dt test.

The parameter set is the best (or a chosen) trial of a calibration study, or START_PARAMS of run_thermal_calibration.py. Each run
folder is a copy of BASE_SIM_DIR with the parameters written exactly as a calibration trial does (update_config_yaml), then the
thermal dt, the time window and the output window of the experiment.

  validation_2011  one run 2001-2012, scored against the Erikson BL01 record (T50cm ... T295cm) at the true sensor depths, plus the
                   2016 bias by depth of the baseline run: bias_by_depth_2016_2011.csv
  sensitivity      baseline + one parameter at a time (SENSITIVITY_LEVELS) over 2006-2017, scored on 2016 (BI_T-1, 17 depths) with the
                   maximum thaw depth of the 2016 season: sensitivity_summary.csv
  dt_test          the baseline at other thermal dt: dt_test_summary.csv. The baseline (dt THERMAL_DT) is the run of the sensitivity
                   folder and is run once.

Max thaw depth = maximum over 2016 of the model's thaw_depth output (results.nc, variable thaw_depth, m below the surface) in column
x-index 1, the column that is scored. The model writes the depth of the first frozen node, so it moves in steps of the node spacing
(0.05 m): a change below that does not show.

Run:  python calibration/run_thermal_experiments.py      Each run takes about 20 min; a rerun skips the folders that have a scored.json.
Output per run folder: run.log, console.log, config.yaml, rmse_bias_layers.txt + temperature_layers.png (2016) or
validation_2011_metrics.csv + validation_2011.png (2011), scored.json. results.nc is deleted after scoring (KEEP_RESULTS_NC).
"""
# === CONFIGURATION ===
import os
from pathlib import Path

# parameter set: a calibration study (journal .log or sqlite .db) or, with None, START_PARAMS of run_thermal_calibration.py
STUDY_STORAGE = None   # e.g. Path(r"N:\arctic-xbeach-runs\thermal_calibration_runs\20261104_h7_truedepth_dt60\optuna_journal.log")
STUDY_NAME = "calibration_optuna"
TRIAL_NUMBER = None    # None = best trial of the study

BASE_SIM_DIR = Path(r"N:\arctic-xbeach-runs\thermal_calibration_runs\20261104_h7_truedepth_dt60\base")  # template run folder
OUT_ROOTS = {"validation_2011": Path(r"N:\arctic-xbeach-runs\thermal_validation_runs\20261104_v11_validation_2011"),
             "sensitivity": Path(r"N:\arctic-xbeach-runs\thermal_validation_runs\20261104_v11_sensitivity"),
             "dt_test": Path(r"N:\arctic-xbeach-runs\thermal_validation_runs\20261104_v11_dt_test")}
EXPERIMENTS = ["validation_2011", "sensitivity", "dt_test"]
TAG = ""               # appended to each OUT_ROOTS folder name, e.g. "_dryrun_trial894"
THERMAL_DT = 60        # [s] thermal substep of all runs except the dt test
DT_TEST = [30, 90, 300]  # [s] other dt of the dt test (the baseline at THERMAL_DT is shared with the sensitivity runs)
N_PARALLEL = 14        # child processes at the same time
RUN_MINUTES_GUESS = 20  # only for the time estimate printed at the start
KEEP_RESULTS_NC = False  # False = results.nc is deleted after scoring
KEEP_2011_NC_BELOW_MB = 200  # the 2011 results.nc is kept if it is smaller than this (None = follow KEEP_RESULTS_NC)

# 2016 runs: the time window of the base config (2006-2017), output over the whole thaw season (the calibration scores 31-05 to 15-09)
WINDOW_2016 = ("01-05-2016", "31-10-2016")
SENSOR_DEPTH_OFFSET_2016 = 0.0  # [m] nodes are cell centres, 0 is the true depth
# one parameter at a time: (parameter, level, mode, amount); mode "add" / "mult" act on the baseline value, "set" replaces it.
# The dependent values follow as in the calibration: nb_max = nb_min + nb_increase, k_*_max = k_*_min * k_compaction_factor,
# c_soil_unfrozen = c_soil_frozen * c_ratio
SENSITIVITY_LEVELS = [("nb_min", "low", "add", -0.1), ("nb_min", "high", "add", 0.1),
                      ("k_soil_frozen_min", "low", "mult", 0.75), ("k_soil_frozen_min", "high", "mult", 1.25),
                      ("k_soil_unfrozen_min", "low", "mult", 0.75), ("k_soil_unfrozen_min", "high", "mult", 1.25),
                      ("c_soil_frozen", "low", "mult", 0.75), ("c_soil_frozen", "high", "mult", 1.25),
                      ("geothermal_gradient", "low", "set", 0.020), ("geothermal_gradient", "high", "set", 0.040)]
EQUAL_TOL = 1e-4       # a level this close to the baseline value is flagged in the summary

# 2011 validation: Erikson BL01 hourly record 2011-07-08 to 2011-09-27
OBS_2011_CSV = Path(r"N:\arctic-xbeach-runs\datasets\ground_temperature_erikson.csv")
OBS_2011_SITE = "BL01"
OBS_2011_COLUMNS = {0.50: "T50cm", 1.00: "T100cm", 2.00: "T200cm", 2.95: "T295cm"}  # true sensor depth [m]: column
TIME_2011 = ("01-01-2001", "01-10-2012")    # model.time_start, model.time_end
WINDOW_2011 = ("01-05-2011", "31-12-2011")  # output window
# Forcing files, solver options and the era5 flux shift are those of run_thermal_calibration.py (FORCING_CSV, STORM_CSV, SOLVER_OPTIONS,
# ERA5_FLUX_SHIFT_HOURS), the 2016 observations its OBSERVATION_CSV.
# === END CONFIGURATION ===

# One BLAS / OpenMP thread per process, as in the drivers (the child processes inherit them).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import datetime
import json
import math
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so that run_thermal_calibration is found when the file is not run as a script
import run_thermal_calibration as rtc  # noqa: E402  (also puts the repo on sys.path)
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from arctic_xbeach.config_loader import load_config  # noqa: E402
from netCDF4 import Dataset  # noqa: E402
from optuna.trial import FixedTrial  # noqa: E402


class Job(NamedTuple):
    exp: str          # key of OUT_ROOTS the folder belongs to
    name: str         # folder name inside that root
    raw: dict         # Optuna parameters of the run (before the dependent values are derived)
    dt: int           # [s]
    year: int         # 2016 (borehole BI_T-1) or 2011 (Erikson BL01)
    info: dict        # sensitivity: parameter, level, value, baseline_value


# --- parameters ---------------------------------------------------------------------------------------------------------
def load_raw_params() -> tuple[dict, dict]:
    """(Optuna parameters, where they come from) of the calibrated set."""
    if STUDY_STORAGE is None:
        return dict(rtc.START_PARAMS), dict(source="START_PARAMS of run_thermal_calibration.py")
    path = Path(STUDY_STORAGE)
    if path.suffix == ".log":
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        storage = JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))
    elif path.suffix == ".db":
        storage = f"sqlite:///{path.as_posix()}"
    else:
        raise ValueError(f"STUDY_STORAGE must be an optuna_journal.log or a .db file, not {path}")
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    if TRIAL_NUMBER is None:
        trial = study.best_trial
    else:
        trials = [t for t in study.get_trials(deepcopy=False) if t.number == TRIAL_NUMBER]
        if not trials:
            raise ValueError(f"trial {TRIAL_NUMBER} not found in study {STUDY_NAME} ({path})")
        trial = trials[0]
        if trial.state != optuna.trial.TrialState.COMPLETE:
            raise ValueError(f"trial {TRIAL_NUMBER} is {trial.state.name}, not COMPLETE")
    return dict(trial.params), dict(source="study", storage=str(path), study_name=STUDY_NAME, trial_number=trial.number, value=trial.value)


def sample_fixed(raw: dict) -> tuple[dict, list[str]]:
    """(parameter set of a calibration trial with derived and fixed entries, as run_thermal_calibration.objective builds it;
    names of the Optuna parameters outside the range of the search)."""
    fixed = {k: v[0] for k, v in rtc.PARAM_RANGES.items() if v[0] == v[1]}
    trial = FixedTrial({**fixed, **raw})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # FixedTrial warns for a value outside the range; the summary flags it instead
        params = rtc.sample_params(trial)
    outside = [k for k, dist in trial.distributions.items() if not dist.low <= trial.params[k] <= dist.high]
    return params, outside


def full_params(raw: dict) -> dict:
    return sample_fixed(raw)[0]


def level_raw(raw: dict, parameter: str, mode: str, amount: float) -> dict:
    """Optuna parameters with one parameter changed."""
    base = raw[parameter]
    value = {"add": base + amount, "mult": base * amount, "set": amount}[mode]
    return {**raw, parameter: float(value)}


def make_jobs(raw: dict) -> list[Job]:
    """Folders of the selected experiments. The baseline (THERMAL_DT) belongs to the sensitivity root and is shared by all."""
    unknown = set(EXPERIMENTS) - set(OUT_ROOTS)
    if unknown or not EXPERIMENTS:
        raise ValueError(f"EXPERIMENTS must be a non-empty subset of {list(OUT_ROOTS)}, got {EXPERIMENTS}")
    jobs = []
    if "dt_test" in EXPERIMENTS:
        jobs += [Job("dt_test", f"dt{dt:03d}", raw, dt, 2016, {}) for dt in DT_TEST if dt != THERMAL_DT]
    jobs.append(Job("sensitivity", "baseline", raw, THERMAL_DT, 2016, {}))
    if "sensitivity" in EXPERIMENTS:
        for parameter, level, mode, amount in SENSITIVITY_LEVELS:
            new = level_raw(raw, parameter, mode, amount)
            jobs.append(Job("sensitivity", f"sens_{parameter}_{level}", new, THERMAL_DT, 2016,
                            dict(parameter=parameter, level=level, value=new[parameter], baseline_value=raw[parameter])))
    if "validation_2011" in EXPERIMENTS:
        jobs.append(Job("validation_2011", "validation_2011", raw, THERMAL_DT, 2011, {}))
    for job in jobs:
        if 3600 % job.dt:
            raise ValueError(f"thermal dt {job.dt} s ({job.name}) does not divide the model timestep of 3600 s")
    return sorted(jobs, key=lambda j: -(11.75 if j.year == 2011 else 11) / j.dt)  # longest first, so the last one to start is short


def root_of(exp: str) -> Path:
    root = OUT_ROOTS[exp]
    return root.with_name(root.name + TAG)


# --- one run folder -----------------------------------------------------------------------------------------------------
def build_run_folder(run_dir: Path, job: Job) -> None:
    """Copy of BASE_SIM_DIR with the parameters of the job, its thermal dt, time window and output window."""
    if run_dir.exists():
        shutil.rmtree(run_dir)  # an earlier attempt that was not scored
    shutil.copytree(BASE_SIM_DIR, run_dir)
    rtc.update_config_yaml(run_dir, full_params(job.raw))  # the same call as a calibration trial
    cfg = load_config(run_dir / "config.yaml")
    cfg["thermal"]["dt"] = job.dt
    if job.year == 2011:
        cfg["model"]["time_start"], cfg["model"]["time_end"] = TIME_2011
        cfg["output"]["output_start"], cfg["output"]["output_end"] = WINDOW_2011
    else:  # the time window of the base config must contain the thaw season
        start, end = (pd.to_datetime(cfg["model"][k], dayfirst=True) for k in ("time_start", "time_end"))
        if start > pd.Timestamp("2016-05-01") or end < pd.Timestamp("2016-10-31"):
            raise ValueError(f"model time window {start:%d-%m-%Y} to {end:%d-%m-%Y} of the base config does not cover 2016-05 to 2016-10")
        cfg["output"]["output_start"], cfg["output"]["output_end"] = WINDOW_2016
    (run_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))


def run_child(run_dir: Path) -> int:
    """Run main.py in a child process of this interpreter; its prints go to console.log (run.log is written by the model)."""
    script = run_dir / "run_simulation_subprocess.py"
    script.write_text(rtc.CHILD_SCRIPT.format(repo=str(rtc.REPO_DIR), sim_dir=str(run_dir)))
    with open(run_dir / "console.log", "w") as log:
        return subprocess.run([sys.executable, str(script)], cwd=str(run_dir), stdout=log, stderr=subprocess.STDOUT).returncode


# --- scoring ------------------------------------------------------------------------------------------------------------
def read_nc_time(nc: Dataset) -> pd.DatetimeIndex:
    time_dt = netCDF4.num2date(nc.variables["time"][:], units=nc.variables["time"].units)
    return pd.to_datetime(np.array([rtc.to_native_datetime(t) for t in time_dt]))


def max_thaw_depth(nc_path: Path, year: int = 2016) -> float:
    """Maximum over `year` of thaw_depth [m below the surface] in column x-index 1 (the column that is scored)."""
    with rtc.POSTPROC_LOCK, Dataset(nc_path, "r") as nc:
        time_pd = read_nc_time(nc)
        thaw = np.ma.filled(nc.variables["thaw_depth"][:].astype(float), np.nan)[:, 1]
    keep = np.asarray(time_pd.year == year)
    return float(np.nanmax(thaw[keep])) if keep.any() else float("nan")


def score_validation_2011(run_dir: Path, obs_csv: Path | None = None, make_figure: bool = True) -> pd.DataFrame:
    """Metrics of the run against the Erikson record at the true sensor depths (nearest node, column x-index 1, K to degC, nearest
    model time as in compute_metrics). One row per depth plus a pooled row; written to validation_2011_metrics.csv in run_dir."""
    obs = pd.read_csv(obs_csv or OBS_2011_CSV, parse_dates=["time"])
    if "SiteID" in obs.columns:
        obs = obs[obs["SiteID"] == OBS_2011_SITE]
    with rtc.POSTPROC_LOCK, Dataset(run_dir / "results" / "results.nc", "r") as nc:
        time_pd = read_nc_time(nc)
        temp = np.ma.filled(nc.variables["ground_temperature_distribution"][:].astype(float), np.nan)[:, 1, :] - 273.15
        abs_zgr = nc.variables["abs_zgr"][:][0, 1, :]
    problem = rtc.results_end_problem(run_dir, time_pd[-1], WINDOW_2011)
    if problem:
        raise RuntimeError(f"run incomplete: {problem}")
    node_depth = np.abs(np.asarray(abs_zgr) - abs_zgr[0])

    rows, pooled_err, panels = [], [], []
    for depth, col in OBS_2011_COLUMNS.items():
        model = pd.DataFrame({"time": time_pd, "model": temp[:, np.abs(node_depth - depth).argmin()]}).sort_values("time")
        o = obs[["time", col]].rename(columns={col: "obs"}).sort_values("time")
        j = pd.merge_asof(o, model, on="time", direction="nearest").dropna()
        err = j["model"] - j["obs"]
        pooled_err.append(err.values)
        rows.append(dict(depth_m=depth, rmse=float(np.sqrt((err ** 2).mean())), mae=float(err.abs().mean()), bias=float(err.mean()), n=len(err)))
        panels.append((depth, model, o, rows[-1]))
    err = np.concatenate(pooled_err)
    rows.append(dict(depth_m="pooled", rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()), bias=float(err.mean()), n=len(err)))
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "validation_2011_metrics.csv", index=False)

    if make_figure:
        with rtc.POSTPROC_LOCK:
            fig, axs = plt.subplots(len(panels), 1, figsize=(12, 9), sharex=True)
            for ax, (depth, model, o, r) in zip(np.atleast_1d(axs), panels):
                ax.plot(model["time"], model["model"], "C0", label="Modelled")
                ax.plot(o["time"], o["obs"], "k", label="Measured")
                ax.text(0.98, 0.05, f"{depth:.2f} m\nRMSE = {r['rmse']:.2f} °C\nBias = {r['bias']:.2f} °C", transform=ax.transAxes,
                        va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
                ax.set_xlim(o["time"].min(), o["time"].max())
                ax.set_ylabel("Temperature [°C]")
                ax.legend(loc="upper left")
                ax.grid()
            axs[-1].set_xlabel("Time")
            fig.tight_layout()
            fig.savefig(run_dir / "validation_2011.png", dpi=150)
            plt.close(fig)
    return table


def run_job(job: Job) -> dict:
    """Build, run and score one folder. Returns the scored.json content, or {"failed": reason}; a scored folder is not run again."""
    run_dir = root_of(job.exp) / job.name
    done = run_dir / "scored.json"
    if done.exists():
        return {**json.loads(done.read_text()), "skipped": True}

    nc_path = run_dir / "results" / "results.nc"
    t0 = time.time()
    try:
        build_run_folder(run_dir, job)
        returncode = run_child(run_dir)
        runtime_min = (time.time() - t0) / 60
        if returncode != 0:
            return {"failed": f"exit code {returncode}"}
        result = dict(run=job.name, dt=job.dt, year=job.year, runtime_min=round(runtime_min, 2),
                      finished=datetime.datetime.now().isoformat(timespec="seconds"))
        if job.year == 2016:
            objective, table = rtc.score_borehole_2016(run_dir, SENSOR_DEPTH_OFFSET_2016, True, output_window=WINDOW_2016)
            if not math.isfinite(objective):
                return {"failed": "results.nc incomplete or not scorable"}
            table.to_csv(run_dir / "bias_by_depth_2016.csv", index=False)
            result.update(objective=float(objective), max_thaw_depth_m=max_thaw_depth(nc_path, 2016))
        else:
            table = score_validation_2011(run_dir)
            pooled = table[table["depth_m"] == "pooled"].iloc[0]
            result.update(objective=float(pooled["rmse"]), bias=float(pooled["bias"]))
        done.write_text(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"  ERROR in {job.name}: {e}")
        return {"failed": str(e)}
    finally:
        try:
            keep = KEEP_RESULTS_NC
            if job.year == 2011 and KEEP_2011_NC_BELOW_MB is not None and nc_path.exists():
                keep = keep or nc_path.stat().st_size < KEEP_2011_NC_BELOW_MB * 1e6
            if not keep and nc_path.exists():
                nc_path.unlink()
        except OSError:
            pass


# --- summaries ----------------------------------------------------------------------------------------------------------
def write_summaries(jobs: list[Job], results: dict[str, dict], raw: dict, source: dict) -> dict[str, pd.DataFrame]:
    """Write the csv files of the selected experiments and return them."""
    out = {}
    nan = float("nan")
    base = results.get("baseline", {})
    b_obj, b_thaw = base.get("objective", nan), base.get("max_thaw_depth_m", nan)

    if "sensitivity" in EXPERIMENTS:
        rows = [dict(run="baseline", parameter="baseline", level="base", value=nan, baseline_value=nan, objective_rmse=b_obj, d_rmse=0.0,
                     max_thaw_depth_m=b_thaw, d_thaw_m=0.0, flag="" if "objective" in base else "failed")]
        for job in (j for j in jobs if j.exp == "sensitivity" and j.name != "baseline"):
            r, info = results.get(job.name, {}), job.info
            flags = []
            if abs(info["value"] - info["baseline_value"]) <= EQUAL_TOL:
                flags.append("equals_baseline")
            if sample_fixed(job.raw)[1]:
                flags.append("outside_search_range")
            if "objective" not in r:
                flags.append("failed")
            rows.append(dict(run=job.name, parameter=info["parameter"], level=info["level"], value=info["value"], baseline_value=info["baseline_value"],
                             objective_rmse=r.get("objective", nan), d_rmse=r.get("objective", nan) - b_obj, max_thaw_depth_m=r.get("max_thaw_depth_m", nan),
                             d_thaw_m=r.get("max_thaw_depth_m", nan) - b_thaw, flag=";".join(flags)))
        out["sensitivity"] = pd.DataFrame(rows)
        out["sensitivity"].to_csv(root_of("sensitivity") / "sensitivity_summary.csv", index=False)

    if "dt_test" in EXPERIMENTS:
        rows = [dict(dt=THERMAL_DT, objective_rmse=b_obj, d_rmse=0.0, max_thaw_depth_m=b_thaw, d_thaw_m=0.0, runtime_min=base.get("runtime_min", nan))]
        for job in (j for j in jobs if j.exp == "dt_test"):
            r = results.get(job.name, {})
            rows.append(dict(dt=job.dt, objective_rmse=r.get("objective", nan), d_rmse=r.get("objective", nan) - b_obj,
                             max_thaw_depth_m=r.get("max_thaw_depth_m", nan), d_thaw_m=r.get("max_thaw_depth_m", nan) - b_thaw,
                             runtime_min=r.get("runtime_min", nan)))
        out["dt_test"] = pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
        out["dt_test"].to_csv(root_of("dt_test") / "dt_test_summary.csv", index=False)

    if "validation_2011" in EXPERIMENTS:
        root = root_of("validation_2011")
        metrics_file = root / "validation_2011" / "validation_2011_metrics.csv"
        bias_file = root_of("sensitivity") / "baseline" / "bias_by_depth_2016.csv"
        if metrics_file.exists():
            m11 = pd.read_csv(metrics_file)
            m11.to_csv(root / "validation_2011_metrics.csv", index=False)
            out["validation_2011"] = m11
            b11 = m11[m11["depth_m"] != "pooled"].astype({"depth_m": float}).add_suffix("_2011").rename(columns={"depth_m_2011": "depth_m"})
            b16 = pd.read_csv(bias_file).add_suffix("_2016").rename(columns={"depth_m_2016": "depth_m"}) if bias_file.exists() else pd.DataFrame(columns=["depth_m"])
            both = b16.merge(b11, on="depth_m", how="outer").sort_values("depth_m").reset_index(drop=True)
            cols = ["depth_m", "bias_2016", "rmse_2016", "n_2016", "bias_2011", "rmse_2011", "n_2011"]
            both = both.reindex(columns=cols)
            both.to_csv(root / "bias_by_depth_2016_2011.csv", index=False)
            out["bias_by_depth"] = both

    params_used = dict(source=source, raw_params=raw, full_params=full_params(raw),
                       thermal_dt=THERMAL_DT, dt_test=DT_TEST, experiments=EXPERIMENTS, base_sim_dir=str(BASE_SIM_DIR),
                       created=datetime.datetime.now().isoformat(timespec="seconds"))
    for root in {root_of(j.exp) for j in jobs}:
        (root / "params_used.json").write_text(json.dumps(params_used, indent=2))
    return out


def main():
    raw, source = load_raw_params()
    print(f"Parameters from {source}")
    jobs = make_jobs(raw)
    for root in {root_of(j.exp) for j in jobs}:
        root.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (root_of(j.exp) / j.name / "scored.json").exists()]
    waves = math.ceil(len(todo) / N_PARALLEL)
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already scored, {len(todo)} to run at {N_PARALLEL} in parallel: "
          f"about {waves * RUN_MINUTES_GUESS} min ({waves} wave(s) of ~{RUN_MINUTES_GUESS} min; dt 30 s takes longer)")
    for root in sorted({root_of(j.exp) for j in jobs}):
        print(f"  output: {root}")

    results, n_done = {}, 0
    with ThreadPoolExecutor(max_workers=N_PARALLEL) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            results[job.name] = fut.result()
            n_done += 1
            r = results[job.name]
            what = r["failed"] if "failed" in r else f"objective {r['objective']:.4f}" + (f", max thaw {r['max_thaw_depth_m']:.3f} m" if "max_thaw_depth_m" in r else "")
            print(f"[{n_done}/{len(jobs)} {datetime.datetime.now():%H:%M}] {job.name}: {what}")

    tables = write_summaries(jobs, results, raw, source)
    for name, table in tables.items():
        print(f"\n{name}\n{table.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")
    failed = [n for n, r in results.items() if "failed" in r]
    if failed:
        print(f"\nFAILED (rerun to retry): {failed}")


if __name__ == "__main__":
    main()
