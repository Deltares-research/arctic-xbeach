"""Score a finished coupled (thermal + XBeach) Arctic-XBeach run against the observed bluff edge and shoreline.

    from score_coupled_run import score_run
    result = score_run(Path("<run_dir>"), "P2")     # period in {"P1", "P2", "P3"}; writes <run_dir>/score.json and returns the same dict

Reads <run_dir>/results/results.nc (never modified or deleted) and <run_dir>/results/xbeach_calls.csv, <run_dir>/run.log,
<run_dir>/config.yaml when present. Nothing in arctic_xbeach/ or in the calibration drivers is imported: the logic is replicated.

Two methods, one function (the difference is only in the dates, see PERIODS):
  P2  exactly run_simulation() of calibration/auto_calibrate_coupled_model.py (lines 301-416): model offset removed at
      NORMALIZATION_DATE (interpolated observations), RMSE over the observations in CALIB_START..CALIB_END.
  P1, P3  N:\\arctic-xbeach-runs\\_scripts\\validate_coupled_model.py (per1, per3, and the "normalize each run to the start of its
      period" block, lines 310-344; metrics lines 346-465): offset removed at the period start, RMSE over all observations in the period.

Result fields: see the dict built in score_run() and the docstring of score_run().
Run  python calibration/score_coupled_run.py  to score the folders in RUNS_TO_SCORE.
"""
# === CONFIGURATION ===
import os
from pathlib import Path

# Observations: the CSVs of the coupled driver (auto_calibrate_coupled_model.py lines 39-40; the validate script reads the same files with geopandas)
BLUFF_EDGES_CSV = Path(os.environ.get("AXB_BLUFF_EDGES_CSV", r"D:\Git\thermo-morphological-model\database\ts_datasets\bluff_edges_without_geometry.csv"))  # columns time, distance_from_ref
SHORELINES_CSV = Path(os.environ.get("AXB_SHORELINES_CSV", r"D:\Git\thermo-morphological-model\database\ts_datasets\shore_lines_without_geometry.csv"))

# Skill definition (driver lines 60-65; validate_coupled_model.py uses the same elevations)
SHORELINE_ELEVATION = +2.0   # [m] shoreline = first crossing of this z on the profile
BLUFF_ELEVATION = 13.2       # [m] bluff edge = first crossing of this z
SMOOTHING_WINDOW = "1D"      # centred rolling mean of the position series
CALIB_START, CALIB_END = "2012-01-01", "2016-01-01"   # P2: observations in this period enter the RMSE
NORMALIZATION_DATE = "2012-07-11"                      # P2: model and observations are matched at this date

# Per period: normalisation date, RMSE window (observations inclusive), window of the retreat numbers, and where the end of results.nc is checked
PERIODS = {
    "P1": dict(method="validate_coupled_model.py per1", norm="2009-08-02", rmse=("2009-08-02", "2011-07-15"), retreat=("2009-08-02", "2011-07-15")),
    "P2": dict(method="auto_calibrate_coupled_model.py run_simulation", norm=NORMALIZATION_DATE, rmse=(CALIB_START, CALIB_END), retreat=("2012-07-11", "2015-07-05")),  # retreat window = per2 of validate_coupled_model.py
    "P3": dict(method="validate_coupled_model.py per3", norm="2016-08-27", rmse=("2016-08-27", "2018-07-30"), retreat=("2016-08-27", "2018-07-30")),
}

WRITE_SERIES_CSV = True   # also write score_series.csv (model bluff / shoreline at every XBeach time, raw and normalised) next to score.json

# __main__ only: (run folder, period) pairs to score, and where to write score.json (None = in each run folder)
RUNS_TO_SCORE = [
    (Path(r"N:\arctic-xbeach-runs\coupled_validation_runs\20260303_validation_runs_trail88\validation_2012"), "P1"),
    (Path(r"N:\arctic-xbeach-runs\coupled_validation_runs\20260303_validation_runs_trail88\validation_2019"), "P3"),
]
OUT_DIR = None
# === END CONFIGURATION ===

import datetime
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
import yaml

SCORER_VERSION = 1


# --- observations ----------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _read_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["distance_from_ref"] = df["distance_from_ref"].astype(np.float64)
    return df.sort_values("time").reset_index(drop=True)


def _ns(times) -> np.ndarray:
    """Datetimes as int64 nanoseconds (np.interp needs numbers)."""
    return np.asarray(times).astype("datetime64[ns]").astype(np.int64)


# --- skill (driver lines 257-275) ------------------------------------------------------------------------------------
def find_crossing(x: np.ndarray, z: np.ndarray, level: float) -> float:
    """x of the first crossing of z = level (linear interpolation), NaN if there is none."""
    crossings = np.where(np.diff(np.sign(z - level)))[0]
    if len(crossings) == 0:
        return np.nan
    i = crossings[0]
    return x[i] + (level - z[i]) * (x[i + 1] - x[i]) / (z[i + 1] - z[i])


def positions_from_profiles(zb: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bluff edge and shoreline x for every profile (rows of zb); NaN where the profile has 2 or fewer valid points."""
    n = zb.shape[0]
    bluff, shore = np.zeros(n), np.zeros(n)   # zeros as in the driver; every entry is overwritten below
    for i in range(n):
        valid = ~np.isnan(zb[i])
        if valid.sum() > 2:
            shore[i] = find_crossing(x[valid], zb[i][valid], SHORELINE_ELEVATION)
            bluff[i] = find_crossing(x[valid], zb[i][valid], BLUFF_ELEVATION)
        else:
            shore[i] = bluff[i] = np.nan
    return bluff, shore


def _json_clean(obj):
    """NaN / inf -> None and numpy scalars -> Python, so that the file is plain JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --- run status ------------------------------------------------------------------------------------------------------
def results_end_problem(run_dir: Path, last_time, scored_end: pd.Timestamp) -> str | None:
    """None if the last record of the thermal time variable reaches the end of the scored period, else the reason.

    Same test as results_end_problem() of the driver (lines 278-289): the scored period ends at scored_end or model.time_end of the
    run's config.yaml, whichever is earlier; the thermal records are written every timestep * output_res hours. XBeach records
    (time_xbeach) can stop weeks before the end, so they are not used. Without a config.yaml a spacing of one day is assumed."""
    end, spacing = scored_end, pd.Timedelta(days=1)
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        end = min(pd.to_datetime(cfg["model"]["time_end"], dayfirst=True), scored_end)
        spacing = pd.Timedelta(hours=cfg["model"]["timestep"] * cfg["output"]["output_res"])
    if pd.Timestamp(last_time) < end - spacing:
        return f"results.nc stops at {pd.Timestamp(last_time)}, before the end of the scored period ({end})"
    return None


def _unwritten_time_note(nc_path: Path) -> str:
    """Why time_xbeach cannot be decoded: XBeach records are only written at output steps (output.output_res > 1 leaves most of
    them as NetCDF fill values, about 1e37), and the driver would fail on such a file in the same way."""
    with xr.open_dataset(nc_path, engine="netcdf4", decode_times=False) as ds:
        t = ds["time_xbeach"].values
    n_bad = int((~(np.abs(t) < 1e20)).sum())
    return f"{n_bad} of {len(t)} time_xbeach entries are unwritten fill values (output.output_res > 1?); not scorable"


# --- call statistics -------------------------------------------------------------------------------------------------
def call_statistics(run_dir: Path) -> dict:
    """Counts and wall times from results/xbeach_calls.csv and, if present, the start / finish lines of run.log."""
    stats: dict = {"calls_file_found": False}
    path = run_dir / "results" / "xbeach_calls.csv"
    if path.exists():
        calls = pd.read_csv(path)
        stats["calls_file_found"] = True
        by_reason = calls["reason"].value_counts().to_dict()
        stats["n_calls"] = int(len(calls))
        stats["n_calls_by_reason"] = {r: int(by_reason.get(r, 0)) for r in ("storm", "inter", "both")}
        stats.update({f"n_calls_reason_{r}": int(n) for r, n in by_reason.items() if r not in ("storm", "inter", "both")})
        stats["n_failed_calls"] = int((calls["success"] == 0).sum())
        stats["xbeach_wall_hours"] = float(calls["xbeach_wall_s"].sum() / 3600)
        stats["setup_wall_hours"] = float(calls["setup_wall_s"].sum() / 3600)
        stats["grid_update_wall_hours"] = float(calls["grid_update_wall_s"].sum() / 3600)
        stats["xbeach_sim_hours"] = float(calls["sim_hours"].sum())
        per_year = calls.groupby("year").agg(n_calls=("call_id", "size"), xbeach_wall_hours=("xbeach_wall_s", lambda s: s.sum() / 3600))
        stats["per_year"] = {int(y): {"n_calls": int(r.n_calls), "xbeach_wall_hours": float(r.xbeach_wall_hours)} for y, r in per_year.iterrows()}

    stats["run_log_found"] = False
    log = run_dir / "run.log"
    if log.exists():
        text = log.read_text(errors="replace")
        pat = r"Simulation {} at: (\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})"
        starts, ends = re.findall(pat.format("started"), text), re.findall(pat.format("finished"), text)
        if starts and ends:
            t0, t1 = pd.Timestamp(starts[-1]), pd.Timestamp(ends[-1])   # the last start and finish: a run.log can hold several attempts
            if t1 >= t0:
                stats["run_log_found"] = True
                stats["wall_hours"] = float((t1 - t0).total_seconds() / 3600)
                if stats["calls_file_found"]:
                    stats["non_xbeach_wall_hours"] = stats["wall_hours"] - stats["xbeach_wall_hours"]   # thermal solver, output, I/O ... plus setup and grid update
                    stats["other_wall_hours"] = stats["non_xbeach_wall_hours"] - stats["setup_wall_hours"] - stats["grid_update_wall_hours"]
    return stats


# --- scoring ---------------------------------------------------------------------------------------------------------
def _score_positions(period: str, time_xb: pd.DatetimeIndex, bluff_raw: np.ndarray, shore_raw: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Smoothing, normalisation and skill numbers (the method of the module docstring)."""
    p = PERIODS[period]
    df_be, df_sl = _read_positions(BLUFF_EDGES_CSV), _read_positions(SHORELINES_CSV)

    s = lambda a: pd.Series(a, index=time_xb).rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean().values
    bluff_pos, shore_pos = s(bluff_raw), s(shore_raw)

    model_t = _ns(time_xb.values)
    norm = pd.Timestamp(p["norm"]).value
    obs_be_norm = float(np.interp(norm, _ns(df_be["time"].values), df_be["distance_from_ref"].values))
    obs_sl_norm = float(np.interp(norm, _ns(df_sl["time"].values), df_sl["distance_from_ref"].values))
    bluff_offset = obs_be_norm - float(np.interp(norm, model_t, bluff_pos, left=bluff_pos[0], right=bluff_pos[-1]))
    shore_offset = obs_sl_norm - float(np.interp(norm, model_t, shore_pos, left=shore_pos[0], right=shore_pos[-1]))
    bluff_n, shore_n = bluff_pos + bluff_offset, shore_pos + shore_offset

    out: dict = {"bluff_offset_m": bluff_offset, "shore_offset_m": shore_offset}
    window = lambda df: df[(df["time"] >= pd.Timestamp(p["rmse"][0])) & (df["time"] <= pd.Timestamp(p["rmse"][1]))]
    for name, df, series in (("bluff", df_be, bluff_n), ("shore", df_sl, shore_n)):
        w = window(df)
        at_obs = np.interp(_ns(w["time"].values), model_t, series)
        err = at_obs - w["distance_from_ref"].values
        out[f"n_obs_{name}"] = int(len(w))
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(err**2))) if len(w) else float("nan")
        out[f"mae_{name}"] = float(np.mean(np.abs(err))) if len(w) else float("nan")
        out[f"bias_{name}"] = float(np.mean(err)) if len(w) else float("nan")   # model - observed
        out[f"obs_at_obs_times_{name}"] = {"time": [str(t) for t in w["time"].dt.strftime("%Y-%m-%d")],
                                           "observed_m": w["distance_from_ref"].round(3).tolist(), "model_m": np.round(at_obs, 3).tolist()}

        # retreat over the retreat window: model from the normalised series, observed from the observations nearest to the window ends
        t0, t1 = (pd.Timestamp(d).value for d in p["retreat"])
        obs_t, obs_v = _ns(df["time"].values), df["distance_from_ref"].values
        obs_start, obs_end = obs_v[np.argmin(np.abs(obs_t - t0))], obs_v[np.argmin(np.abs(obs_t - t1))]
        mod_start, mod_end = float(np.interp(t0, model_t, series)), float(np.interp(t1, model_t, series))
        out[f"obs_retreat_{name}_m"] = float(obs_end - obs_start)
        out[f"model_retreat_{name}_m"] = float(mod_end - mod_start)
        out[f"model_start_{name}_m"], out[f"model_end_{name}_m"] = mod_start, mod_end
        out[f"obs_start_{name}_m"], out[f"obs_end_{name}_m"] = float(obs_start), float(obs_end)

    both = out["rmse_bluff"], out["rmse_shore"]
    out["rmse_combined"] = float(np.sqrt((both[0] ** 2 + both[1] ** 2) / 2)) if not any(np.isnan(both)) else float("nan")
    series_df = pd.DataFrame({"time": time_xb, "bluff_raw_m": bluff_pos, "shore_raw_m": shore_pos, "bluff_norm_m": bluff_n, "shore_norm_m": shore_n})
    return out, series_df


def score_run(run_dir: Path, period: str, out_dir: Path | None = None) -> dict:
    """Score run_dir for period "P1", "P2" or "P3", write score.json (into out_dir if given, else run_dir) and return the same dict.

    Always returns a dict. Fields (None = not available, e.g. NaN):
      period, run_dir, timestamp, scorer_version, method, constants  what was scored and with which constants
      status          "ok" | "incomplete" (results.nc stops before the end of the scored period, no skill numbers) | "error"
      complete        True if results.nc reaches the end of the scored period;  results_end_problem: reason text or None
      error           error text if status == "error", else None
      n_xbeach_times, results_first_time, results_last_time (thermal 'time'), xbeach_first_time, xbeach_last_time
      rmse_bluff, rmse_shore, rmse_combined [m]   RMSE at the observations inside the RMSE window; combined = sqrt((b^2+s^2)/2)
      mae_*, bias_* (model - observed), n_obs_bluff, n_obs_shore
      bluff_offset_m, shore_offset_m               offset added to the model so that it equals the observation at the normalisation date
      obs_retreat_{bluff,shore}_m, model_retreat_{bluff,shore}_m   retreat over constants.retreat_window (positive = landward), with the start and end values
      obs_at_obs_times_{bluff,shore}               observation dates, observed and modelled position (m) at them
      calls           call statistics (see call_statistics): calls_file_found False when xbeach_calls.csv is missing
      series_csv      path of score_series.csv, or None
    """
    run_dir = Path(run_dir)
    if period not in PERIODS:
        raise ValueError(f"period must be one of {sorted(PERIODS)}, got {period!r}")
    p = PERIODS[period]
    result: dict = {
        "period": period, "run_dir": str(run_dir), "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "scorer_version": SCORER_VERSION,
        "method": p["method"],
        "constants": {"bluff_elevation_m": BLUFF_ELEVATION, "shoreline_elevation_m": SHORELINE_ELEVATION, "smoothing_window": SMOOTHING_WINDOW,
                      "normalization_date": p["norm"], "rmse_window": list(p["rmse"]), "retreat_window": list(p["retreat"]),
                      "bluff_edges_csv": str(BLUFF_EDGES_CSV), "shorelines_csv": str(SHORELINES_CSV)},
        "status": "error", "complete": False, "results_end_problem": None, "error": None, "series_csv": None,
    }
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    try:
        result["calls"] = call_statistics(run_dir)
    except Exception as e:   # a damaged CSV must not stop the skill numbers
        result["calls"] = {"calls_file_found": False, "error": f"{type(e).__name__}: {e}"}

    try:
        nc_path = run_dir / "results" / "results.nc"
        if not nc_path.exists():
            raise FileNotFoundError(f"results file not found: {nc_path}")
        try:
            with xr.open_dataset(nc_path, engine="netcdf4") as ds:
                zb = ds["zb_xbeach"].transpose("time_xbeach", ...).load().values
                x = ds["xgr"].load().values
                time_xb = pd.to_datetime(ds["time_xbeach"].values)
                times = pd.to_datetime(ds["time"].values)
        except (OverflowError, ValueError) as e:
            raise ValueError(f"time variables cannot be decoded ({e}); {_unwritten_time_note(nc_path)}") from e
        if len(times) == 0 or len(time_xb) == 0:
            raise ValueError(f"results.nc has no records (time: {len(times)}, time_xbeach: {len(time_xb)})")
        first_time, last_time = times[0], times[-1]
        result.update(n_xbeach_times=int(len(time_xb)), results_first_time=str(first_time), results_last_time=str(last_time),
                      xbeach_first_time=str(time_xb[0]), xbeach_last_time=str(time_xb[-1]))

        problem = results_end_problem(run_dir, last_time, pd.Timestamp(p["rmse"][1]))
        result["results_end_problem"], result["complete"] = problem, problem is None
        if problem:
            result["status"] = "incomplete"   # like the driver: not scored
        else:
            bluff_raw, shore_raw = positions_from_profiles(zb, x)
            scores, series_df = _score_positions(period, time_xb, bluff_raw, shore_raw)
            result.update(scores)
            result["status"] = "ok"
            if WRITE_SERIES_CSV:
                out_dir.mkdir(parents=True, exist_ok=True)
                series_path = out_dir / "score_series.csv"
                series_df.to_csv(series_path, index=False)
                result["series_csv"] = str(series_path)
    except Exception as e:
        result["status"], result["error"] = "error", f"{type(e).__name__}: {e}"

    result = _json_clean(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    for run, per in RUNS_TO_SCORE:
        r = score_run(run, per, out_dir=OUT_DIR)
        print(f"{run}  {per}  status={r['status']}  complete={r['complete']}")
        if r["status"] == "ok":
            print(f"  RMSE bluff {r['rmse_bluff']:.2f}  shore {r['rmse_shore']:.2f}  combined {r['rmse_combined']:.2f} m   "
                  f"retreat bluff obs {r['obs_retreat_bluff_m']:.2f} model {r['model_retreat_bluff_m']:.2f}, "
                  f"shore obs {r['obs_retreat_shore_m']:.2f} model {r['model_retreat_shore_m']:.2f} m")
        else:
            print(f"  {r['error'] or r['results_end_problem']}")
