"""One run of the revision round: run the coupled model in the folder <ROOT>\\rrNN_<label>, then score it.

    python -u calibration/run_revision_one.py rr05

The run job is submitted behind the prep job (run_revision_round.py), which writes <ROOT>\\READY when the folders are built and
checked. Without READY this script prints why and exits 2 at once (no waiting, no retry). Otherwise the model is started exactly as
the coupled driver's CHILD_SCRIPT does (Simulation(sim_dir, proj_dir=repo); main(sim, print_to_screen=False)) in one child process
of this interpreter, and score_run(run_dir, period) of score_coupled_run.py writes score.json. A failure of the scoring is logged
and does not fail the job. Exit code: 0 = model finished, 1 = model failed, 2 = round not READY / unknown run.
"""
# === CONFIGURATION ===
import os
from pathlib import Path

DRY_RUN = False        # True: run nothing, only print what would be done (root = <ROOT>_dryrun)
ROOT_BASE = Path(os.environ.get("AXB_REVISION_ROOT", r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104"))
REPO_DIR = Path(__file__).resolve().parents[1]
CALIB_DIR = Path(__file__).resolve().parent
# === END CONFIGURATION ===

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"   # one core per run; runs.py would set them to the declared cores, XBeach inherits them

import csv
import datetime
import subprocess
import sys
import time
import traceback

# Same child script as the coupled driver (calibration/auto_calibrate_coupled_model.py CHILD_SCRIPT)
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

ROOT = ROOT_BASE   # set in main() (<ROOT>_dryrun for DRY_RUN)


def log(msg: str) -> None:
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)


def find_run(run_id: str) -> tuple[Path, str]:
    """(run folder, period) of run_id from design.csv, which the prep job wrote from the folders it built."""
    with open(ROOT / "design.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] == run_id:
                return Path(row["folder"]), row["period"]
    raise KeyError(f"{run_id} is not in {ROOT / 'design.csv'}")


def main() -> int:
    global ROOT
    ROOT = ROOT_BASE.with_name(ROOT_BASE.name + "_dryrun") if DRY_RUN else ROOT_BASE
    if len(sys.argv) != 2:
        print("usage: run_revision_one.py <run id, e.g. rr05>")
        return 2
    run_id = sys.argv[1]
    log(f"{run_id}: root {ROOT}")

    if not (ROOT / "READY").exists():
        failed = ROOT / "FAILED"
        why = f"FAILED marker present:\n{failed.read_text(encoding='utf-8')}" if failed.exists() else "no READY and no FAILED marker (the prep job has not finished or did not run)"
        log(f"{run_id}: round is not READY, not running. {why}")
        return 2
    try:
        run_dir, period = find_run(run_id)
    except (OSError, KeyError) as e:
        log(f"{run_id}: cannot find the run: {e}")
        return 2
    if not (run_dir / "config.yaml").exists():
        log(f"{run_id}: {run_dir / 'config.yaml'} is missing")
        return 2

    status = run_dir / "model_exit_code.txt"   # also read by the collector
    # A run started outside the pool (4 Nov: rr15/rr16 as their own jobs) holds the folder's run.lock; the pool's later start of the
    # same id waits for it to end instead of failing on the lock, so the collector never sees a false failure. No second model run.
    sys.path.insert(0, str(REPO_DIR))
    from arctic_xbeach.run_lock import acquire_run_lock, release_run_lock
    waited = False
    while not DRY_RUN:
        try:
            release_run_lock(acquire_run_lock(run_dir))
            break
        except RuntimeError as e:
            if not waited:
                log(f"{run_id}: {e} Waiting for that run to end; this job does not start a second one.")
                waited = True
            time.sleep(60)
    if waited:
        code = status.read_text().split()[:1] if status.exists() else ["?"]
        log(f"{run_id}: the other run ended, model_exit_code {code[0]}; nothing more to do here")
        return 0 if code == ["0"] else 1

    rc = None
    if status.exists() and status.read_text().split()[:1] == ["0"]:
        rc = 0
        log(f"{run_id}: model already finished (model_exit_code.txt), only scoring")
    elif DRY_RUN:
        log(f"{run_id}: DRY RUN, would run the model in {run_dir} and score it as {period}")
        return 0
    else:
        script = run_dir / "run_simulation_subprocess.py"
        script.write_text(CHILD_SCRIPT.format(repo=str(REPO_DIR), sim_dir=str(run_dir)))
        log(f"{run_id}: starting the model in {run_dir} ({period})")
        t0 = datetime.datetime.now()
        with open(run_dir / "console.log", "w") as console:
            rc = subprocess.run([sys.executable, str(script)], cwd=str(run_dir), stdout=console, stderr=subprocess.STDOUT).returncode
        hours = (datetime.datetime.now() - t0).total_seconds() / 3600
        status.write_text(f"{rc}\nfinished {datetime.datetime.now():%Y-%m-%d %H:%M:%S}, {hours:.2f} h\n")   # a native crash leaves only this
        log(f"{run_id}: model exit code {rc} after {hours:.2f} h")

    if rc == 0:
        try:
            sys.path.insert(0, str(CALIB_DIR))
            from score_coupled_run import score_run   # lazy: a missing scorer breaks the scoring only
            result = score_run(run_dir, period)
            log(f"{run_id}: scored, status {result.get('status')}, RMSE bluff {result.get('rmse_bluff')} shore {result.get('rmse_shore')} "
                f"(score.json in {run_dir})")
        except Exception:
            log(f"{run_id}: SCORING FAILED (the model run is fine; the collector or a later call of score_run can redo it):\n{traceback.format_exc()}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
