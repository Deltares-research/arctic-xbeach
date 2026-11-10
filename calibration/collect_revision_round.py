"""Collector of the revision round: status, progress, ETA and skill of the coupled runs in design.csv (16 or 27), rewritten every COLLECT_INTERVAL_MIN.

Writes <ROOT>\\summary\\round_summary.csv, round_summary.md and quick-look PNGs, then waits; exits after a final write when all runs
are done or failed (or after MAX_RUNTIME_H). It only reads: it never stops, restarts or submits a run. A finished run without a
score.json is scored with score_run() of score_coupled_run.py (once per collector process).

State of a run: the local-runs registry (read-only, ~/.claude/runs/registry.json, through the job id in <ROOT>\\submitted_jobs.json
{run_id: job id}), refined by the folder (model_exit_code.txt written by run_revision_one.py, run.log). Progress and ETA are the last
'Progress NN% | ... | ETA ~ X' line of run.log; the ETA is the model's own average-pace estimate as of that log line.
Run:  python -u calibration/collect_revision_round.py
"""
# === CONFIGURATION ===
import os
from pathlib import Path

DRY_RUN = False        # True: root = <ROOT>_dryrun
ROOT_BASE = Path(os.environ.get("AXB_REVISION_ROOT", r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104"))
CALIB_DIR = Path(__file__).resolve().parent
REGISTRY_JSON = Path.home() / ".claude" / "runs" / "registry.json"   # local-runs registry, read only
COLLECT_INTERVAL_MIN = 60
MAX_RUNTIME_H = 72     # safety cap: write a last summary and exit even if runs are still open (e.g. a lost job id)
STALE_LOG_H = 3        # without a registry entry, a run with no file written for this long counts as failed
HEARTBEAT_FILES = ("run.log", "results/xbeach_calls.csv", "ne_layer.txt")   # run.log alone goes quiet for hours in storm stretches
RUN_ONCE = os.environ.get("AXB_COLLECT_ONCE", "0") == "1"  # one pass, then exit (hourly check by hand or from a session)
# === END CONFIGURATION ===

import csv
import datetime
import json
import re
import sys
import time
import traceback

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

sys.path.insert(0, str(CALIB_DIR))

ROOT = ROOT_BASE   # set in main() (<ROOT>_dryrun for DRY_RUN)
PROGRESS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| Progress (\d+)% \| avg_step=([\d.]+)s \| (\S+ \S+) \| ETA ~ ([\d.]+)(h|min)", re.M)
_score_tried: set[str] = set()   # runs for which this process has already called score_run


def log(msg: str) -> None:
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)


def load_design() -> list[dict]:
    """[{id, label, period, folder, change}] from design.csv of the prep job, else from the design table of run_revision_round.py."""
    path = ROOT / "design.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(id=r["id"], label=r["label"], period=r["period"], folder=Path(r["folder"]), change=r["changes"]) for r in csv.DictReader(f)]
    import run_revision_round as prep
    return [dict(id=r["id"], label=r["label"], period=r["period"], folder=ROOT / prep.folder_name(r), change=r["change"]) for r in prep.RUNS]


def read_json(path: Path):
    """JSON file or None; the registry is rewritten by other processes, so any read can fail."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def registry_entry(run_id: str, jobs: dict | None, registry: dict | None) -> dict | None:
    if not jobs or not registry or run_id not in jobs:
        return None
    job_id = jobs[run_id]["id"] if isinstance(jobs[run_id], dict) else jobs[run_id]
    return (registry.get("runs") or {}).get(job_id)


def last_progress(run_dir: Path) -> dict | None:
    """Last 'Progress' line of run.log: percent, model time, ETA [h] and the time at which the ETA was logged."""
    try:
        with open(run_dir / "run.log", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - 200_000, 0))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None
    stamp, pct, _, model_time, eta, unit = matches[-1]
    eta_h = float(eta) / (60 if unit == "min" else 1)
    logged = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    return dict(percent=int(pct), model_time=model_time, eta_h=eta_h, logged=logged, finish=logged + datetime.timedelta(hours=eta_h))


def run_state(run_dir: Path, entry: dict | None) -> str:
    """'not started' | 'running' | 'done' | 'failed'."""
    code_file = run_dir / "model_exit_code.txt"
    code = None
    if code_file.exists():
        try:
            code = int(code_file.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            code = None
    if code is not None:
        return "done" if code == 0 else "failed"
    if entry:
        reg, exit_code = entry.get("state"), entry.get("exit_code")
        if reg == "queued":
            return "not started"
        if reg == "running":
            return "running"
        if reg in ("done", "failed", "stopped"):
            return "done" if reg == "done" and exit_code in (0, None) else "failed"
    log_file = run_dir / "run.log"
    if log_file.exists():
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")[-20_000:]
            if "Arctic-XBeach simulation completed" in text:
                return "done"
            if "Simulation failed" in text:
                return "failed"
            # run.log only gets a line per 1 % of model time; in long storm stretches that is more than STALE_LOG_H,
            # so the newest of the files every XBeach call writes counts as the heartbeat
            newest = max((run_dir / f).stat().st_mtime for f in HEARTBEAT_FILES if (run_dir / f).exists())
            age_h = (time.time() - newest) / 3600
        except OSError:
            return "running"
        return "running" if age_h < STALE_LOG_H else "failed"
    return "not started"


def get_score(run: dict, state: str) -> dict | None:
    """score.json of the run; a finished run without one is scored once by this process."""
    path = run["folder"] / "score.json"
    if state == "done" and not path.exists() and run["id"] not in _score_tried:
        _score_tried.add(run["id"])
        try:
            from score_coupled_run import score_run
            log(f"{run['id']}: no score.json, scoring")
            score_run(run["folder"], run["period"])
        except Exception:
            log(f"{run['id']}: scoring failed:\n{traceback.format_exc()}")
    return read_json(path)


def collect_row(run: dict, jobs, registry) -> tuple[dict, dict | None]:
    entry = registry_entry(run["id"], jobs, registry)
    state = run_state(run["folder"], entry)
    prog = last_progress(run["folder"]) if state in ("running", "done") else None
    score = get_score(run, state) if state == "done" else None
    calls = (score or {}).get("calls") or {}
    by_reason = calls.get("n_calls_by_reason") or {}
    g = lambda key: (score or {}).get(key)
    row = dict(
        id=run["id"], label=run["label"], period=run["period"], change=run["change"], state=state, job=(entry or {}).get("id", ""),
        progress_pct=100 if state == "done" else (prog["percent"] if prog else None),
        model_time=prog["model_time"] if prog else None,
        eta_h=round(prog["eta_h"], 2) if prog and state == "running" else None,
        expected_finish=f"{prog['finish']:%Y-%m-%d %H:%M}" if prog and state == "running" else None,
        score_status=g("status"), rmse_bluff_m=g("rmse_bluff"), rmse_shore_m=g("rmse_shore"),
        model_retreat_bluff_m=g("model_retreat_bluff_m"), obs_retreat_bluff_m=g("obs_retreat_bluff_m"),
        model_retreat_shore_m=g("model_retreat_shore_m"), obs_retreat_shore_m=g("obs_retreat_shore_m"),
        storm_calls=(by_reason.get("storm", 0) + by_reason.get("both", 0)) if by_reason else None, calls_total=calls.get("n_calls"),
        xbeach_hours=calls.get("xbeach_wall_hours"), wall_hours=calls.get("wall_hours"),
    )
    return row, score


# --- outputs ---------------------------------------------------------------------------------------------------------------------
def fmt(v, nd=2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_md(rows: list[dict], notes: list[str], path: Path) -> None:
    done = [r for r in rows if r["state"] == "done"]
    failed = [r["id"] for r in rows if r["state"] == "failed"]
    running = [r for r in rows if r["state"] == "running" and r["expected_finish"]]
    latest = max((r["expected_finish"] for r in running), default=None)
    lines = [f"# Revision round, summary of {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
             f"- Done: {len(done)} of {len(rows)}; running: {sum(r['state'] == 'running' for r in rows)}; not started: {sum(r['state'] == 'not started' for r in rows)}",
             f"- Failed: {', '.join(failed) if failed else 'none'}",
             f"- Latest expected finish of a running run (model ETA at its last log line): {latest or 'n/a'}", ""]
    cols = [("id", "id"), ("label", "label"), ("period", "period"), ("state", "state"), ("progress_pct", "progress %"), ("eta_h", "ETA h"),
            ("rmse_bluff_m", "bluff RMSE m"), ("rmse_shore_m", "shore RMSE m"), ("model_retreat_bluff_m", "bluff retreat model m"),
            ("obs_retreat_bluff_m", "obs m"), ("model_retreat_shore_m", "shore retreat model m"), ("obs_retreat_shore_m", "obs m "),
            ("storm_calls", "storm calls"), ("xbeach_hours", "XBeach h"), ("wall_hours", "wall h")]
    lines += ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |")
    lines += ["", "Retreat is positive landward over the retreat window of the scorer (P1 2009-08-02 to 2011-07-15, P2 2012-07-11 to 2015-07-05, "
              "P3 2016-08-27 to 2018-07-30); RMSE over the observations of the period. XBeach h = summed wall time of the XBeach calls, wall h = "
              "run.log start to finish. Source: score.json per run (score_coupled_run.py); progress and ETA: run.log."]
    if notes:
        lines += ["", "Notes:"] + [f"- {n}" for n in notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_timeseries(runs: list[dict], scores: dict, out_dir: Path) -> list[str]:
    """One figure per period: normalised model bluff edge and shoreline of every scored run of the period against the observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    notes = []
    for period in ("P1", "P2", "P3"):
        members = [r for r in runs if r["period"] == period and scores.get(r["id"], {}) and scores[r["id"]].get("series_csv")]
        if not members:
            notes.append(f"{period} time-series figure skipped: no scored run with score_series.csv yet")
            continue
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for k, (ax, name, col) in enumerate(zip(axes, ("bluff edge", "shoreline"), ("bluff_norm_m", "shore_norm_m"))):
            for i, r in enumerate(members):
                try:
                    s = pd.read_csv(scores[r["id"]]["series_csv"], parse_dates=["time"])
                except Exception as e:
                    notes.append(f"{r['id']}: score_series.csv not readable ({e})")
                    continue
                centre = r["change"].startswith("none")
                ax.plot(s["time"], s[col], lw=2.2 if centre else 1.2, color="k" if centre else f"C{i % 10}", label=f"{r['id']} {r['label']}")
            obs = scores[members[0]["id"]].get("obs_at_obs_times_" + ("bluff" if k == 0 else "shore")) or {}
            if obs.get("time"):
                ax.plot(pd.to_datetime(obs["time"]), obs["observed_m"], "o", color="tab:red", ms=5, label="observed")
            ax.set_ylabel(f"{name} position [m]")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2 if len(members) <= 12 else 3)
        axes[0].set_title(f"{period}: modelled vs observed (model normalised as in the scorer)")
        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_{period}.png", dpi=150)
        plt.close(fig)
    return notes


def plot_p2_sensitivity(runs: list[dict], rows: dict, out_dir: Path) -> list[str]:
    """Bar chart: modelled bluff-edge (and shoreline) retreat of the P2 runs relative to the P2 centre run (the P2 run whose change is 'none ...')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base_id = next((r["id"] for r in runs if r["period"] == "P2" and r["change"].startswith("none")), None)
    base = rows.get(base_id)
    if not base or base["model_retreat_bluff_m"] is None:
        return [f"P2 sensitivity bars skipped: {base_id} (P2 centre) is not scored yet"]
    members = [r for r in runs if r["period"] == "P2" and r["id"] != base_id and rows[r["id"]]["model_retreat_bluff_m"] is not None]
    if not members:
        return ["P2 sensitivity bars skipped: no scored P2 sensitivity run yet"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(members) + 2), sharey=True)
    for ax, key, name in zip(axes, ("model_retreat_bluff_m", "model_retreat_shore_m"), ("bluff edge", "shoreline")):
        d = [rows[r["id"]][key] - base[key] if rows[r["id"]][key] is not None and base[key] is not None else float("nan") for r in members]
        ax.barh(range(len(members)), d, color=["tab:blue" if v >= 0 else "tab:orange" for v in d])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{name} retreat minus {base_id} [m]  ({base_id}: {base[key]:.1f} m)" if base[key] is not None else f"{name} retreat difference [m]")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(members)))
    axes[0].set_yticklabels([f"{r['id']} {r['change'].split(';')[0]}" for r in members], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("P2 sensitivity runs: modelled retreat relative to the centre run")
    fig.tight_layout()
    fig.savefig(out_dir / "p2_sensitivity_bluff_retreat.png", dpi=150)
    plt.close(fig)
    return []


def collect_once() -> list[dict]:
    """One pass: read, score finished runs, write csv / md / png. Returns the rows."""
    runs = load_design()
    jobs = read_json(ROOT / "submitted_jobs.json")
    registry = read_json(REGISTRY_JSON)
    notes = []
    if jobs is None:
        notes.append("submitted_jobs.json not found or unreadable: states come from the run folders only")
    if registry is None:
        notes.append(f"registry {REGISTRY_JSON} not readable this time: states come from the run folders only")
    rows, scores = [], {}
    for run in runs:
        row, score = collect_row(run, jobs, registry)
        rows.append(row)
        scores[run["id"]] = score
    out_dir = ROOT / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "round_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_id = {r["id"]: r for r in rows}
    for fn, args in ((plot_timeseries, (runs, scores, out_dir)), (plot_p2_sensitivity, (runs, by_id, out_dir))):
        try:
            notes += fn(*args)
        except Exception as e:
            notes.append(f"{fn.__name__} failed: {type(e).__name__}: {e}")
    write_md(rows, notes, out_dir / "round_summary.md")
    log(f"summary written: {sum(r['state'] == 'done' for r in rows)} of {len(rows)} done, "
        f"{sum(r['state'] == 'running' for r in rows)} running, {sum(r['state'] == 'failed' for r in rows)} failed -> {out_dir}")
    return rows


def main() -> int:
    global ROOT
    ROOT = ROOT_BASE.with_name(ROOT_BASE.name + "_dryrun") if DRY_RUN else ROOT_BASE
    t_start = time.time()
    while True:
        try:
            rows = collect_once()
            if all(r["state"] in ("done", "failed") for r in rows):
                log("all runs ended, final summary written")
                return 0
        except Exception:
            log(f"collect failed (will retry):\n{traceback.format_exc()}")
        if RUN_ONCE:
            return 0
        if time.time() - t_start > MAX_RUNTIME_H * 3600:
            log(f"MAX_RUNTIME_H ({MAX_RUNTIME_H} h) reached, exiting with runs still open")
            return 1
        time.sleep(COLLECT_INTERVAL_MIN * 60)


if __name__ == "__main__":
    sys.exit(main())
