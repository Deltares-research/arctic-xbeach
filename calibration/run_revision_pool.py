"""Runs the 16 coupled runs of the revision round inside one local-runs job, at most MAX_PARALLEL at a time, in design order.

The round job (launch_round.bat, declared with --cores 14) runs run_revision_round.py (prep) and then this script. Submitting the 16
runs as separate local-runs jobs failed on 4 Nov: with about 20 supervisors polling one registry lock, supervisors died on lock
timeouts. Here the runs are child processes of this job, so `runs.py stop <round job>` still stops all of them.
Each run is calibration/run_revision_one.py <id> in its own folder (it exits 2 at once without READY). A run that already has
model_exit_code.txt is skipped; a failed run is not retried. The collector makes one pass every COLLECT_EVERY_MIN and at the end.
"""
# === CONFIGURATION ===
import os
from pathlib import Path

ROOT = Path(os.environ.get("AXB_REVISION_ROOT", r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104"))
REPO = Path(__file__).resolve().parents[1]
MAX_PARALLEL = 14          # = the cores declared for the round job
COLLECT_EVERY_MIN = 60     # one collector pass this often
POLL_S = 30
# === END CONFIGURATION ===

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import csv
import datetime
import subprocess
import sys
import time


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with (ROOT / "pool.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_once() -> None:
    env = dict(os.environ, AXB_COLLECT_ONCE="1", AXB_REVISION_ROOT=str(ROOT))
    with (ROOT / "collector.log").open("a", encoding="utf-8") as out:
        rc = subprocess.run([sys.executable, "-u", str(REPO / "calibration" / "collect_revision_round.py")], cwd=ROOT, env=env,
                            stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"collector pass, exit {rc}")


def main() -> int:
    if not (ROOT / "READY").exists():
        log(f"no READY in {ROOT}; prep did not finish, no runs started")
        return 2
    with (ROOT / "design.csv").open(newline="", encoding="utf-8") as f:
        design = [(r["id"], Path(r["folder"])) for r in csv.DictReader(f)]
    todo = [(rid, d) for rid, d in design if not (d / "model_exit_code.txt").exists()]
    log(f"{len(design)} runs in design, {len(todo)} to run, {MAX_PARALLEL} at a time")
    running: dict[str, subprocess.Popen] = {}
    last_collect = 0.0
    while todo or running:
        for rid, p in list(running.items()):
            if p.poll() is not None:
                log(f"{rid} ended, exit {p.returncode}")
                d = dict(design)[rid]
                if not (d / "model_exit_code.txt").exists():  # ended before the model ran (exit 2): the collector reads this as failed
                    (d / "model_exit_code.txt").write_text(f"{p.returncode}\n", encoding="utf-8")
                del running[rid]
        while todo and len(running) < MAX_PARALLEL:
            rid, d = todo.pop(0)
            out = (d / "job.log").open("a", encoding="utf-8")
            running[rid] = subprocess.Popen([sys.executable, "-u", str(REPO / "calibration" / "run_revision_one.py"), rid],
                                            cwd=d, stdout=out, stderr=subprocess.STDOUT)
            log(f"{rid} started, pid {running[rid].pid}, {len(running)} running, {len(todo)} waiting")
        if time.time() - last_collect > COLLECT_EVERY_MIN * 60:
            collect_once()
            last_collect = time.time()
        time.sleep(POLL_S)
    collect_once()
    log("all runs ended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
