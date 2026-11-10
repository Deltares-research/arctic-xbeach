"""Export the history of an Optuna calibration study: trials CSV, best-so-far curve (PNG) and a short text summary.

The study database is copied to a temporary folder first (together with any -wal / -shm files) and read with sqlite3 in
read-only mode, so a database that a running calibration is writing to is never touched. Only the standard Optuna tables
(trials, trial_values, trial_params) are used, so this does not depend on the Optuna version.
A journal study (STORAGE = "journal", optuna_journal.log of the HPC runs) is copied to a temporary folder as well and loaded with Optuna.

Best-so-far is evaluated in two orders:
  by trial number     the Optuna convention (optimization history plot). With several parallel workers a lower number is not
                      always finished earlier.
  by completion time  the order in which trials actually finished; "completed trials" is the number of finished trials at that moment.
"""
# === CONFIGURATION ===
# The AXB_* environment variables override the defaults below (as in the drivers). AXB_RUN_ROOT exports the one study in that run folder instead of the two published ones.
import os

STORAGE = os.environ.get("AXB_STORAGE", "sqlite")  # "sqlite" (optuna_study.db) or "journal" (optuna_journal.log)
STUDY_NAME = os.environ.get("AXB_STUDY_NAME", "calibration_optuna")  # journal only; a sqlite database has one study (STUDY_ID)
STUDIES = [  # label (also the file stem of the outputs), path to optuna_study.db, objective label for the figure
    ("thermal_20260201", r"N:\arctic-xbeach-runs\thermal_calibration_runs\20260201_calibration_runs\optuna_study.db", "Ground temperature RMSE (K)"),
    ("coupled_20260214", r"N:\arctic-xbeach-runs\coupled_calibration_runs\20260214_calibration_runs\optuna_study.db", "Combined RMSE (m)"),
]
if os.environ.get("AXB_RUN_ROOT"):
    _root = os.environ["AXB_RUN_ROOT"]
    STUDIES = [(os.path.basename(_root.rstrip("/\\")), os.path.join(_root, "optuna_journal.log" if STORAGE == "journal" else "optuna_study.db"),
                os.environ.get("AXB_OBJECTIVE_LABEL", "Objective"))]
OUT_DIR = os.environ.get("AXB_EXPORT_DIR", r"D:\Git\arctic-xbeach-v11\calibration\published_history")
STUDY_ID = 1                 # study inside the database (the drivers create one study per database)
THRESHOLDS = (0.01, 0.001)   # relative distance to the final best at which "converged" is reported (1 % and 0.1 %)
DPI = 300
# === END CONFIGURATION ===

import shutil
import sqlite3
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_journal(path: Path, study_name: str) -> pd.DataFrame:
    """Same table as read_study, from a journal file (copied to a temporary folder first)."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / path.name
        shutil.copy2(path, copy)
        study = optuna.load_study(study_name=study_name, storage=JournalStorage(JournalFileBackend(str(copy))))
        trials = study.trials_dataframe(attrs=("number", "state", "value", "datetime_start", "datetime_complete", "params"))
    trials = trials.rename(columns={c: c.removeprefix("params_") for c in trials.columns})
    trials.loc[~np.isfinite(trials["value"]), "value"] = np.nan  # failed runs are stored as +inf by the drivers
    trials["duration_s"] = (trials["datetime_complete"] - trials["datetime_start"]).dt.total_seconds()
    trials["trial_id"] = trials["number"]
    trials["value_type"] = "FINITE"
    return trials


def read_study(db_path: Path, study_id: int) -> pd.DataFrame:
    """One row per trial: number, state, value (NaN if not finite), times and parameters."""
    if STORAGE == "journal":
        return read_journal(db_path, STUDY_NAME)
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(db_path) + suffix)
            if src.exists():
                shutil.copy2(src, Path(tmp) / src.name)
        con = sqlite3.connect(f"file:{(Path(tmp) / db_path.name).as_posix()}?mode=ro", uri=True)
        try:
            trials = pd.read_sql_query(
                "SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete, v.value, v.value_type "
                "FROM trials t LEFT JOIN trial_values v ON v.trial_id = t.trial_id AND v.objective = 0 "
                "WHERE t.study_id = ? ORDER BY t.number", con, params=(study_id,))
            params = pd.read_sql_query(
                "SELECT p.trial_id, p.param_name, p.param_value FROM trial_params p JOIN trials t ON t.trial_id = p.trial_id "
                "WHERE t.study_id = ?", con, params=(study_id,)).pivot(index="trial_id", columns="param_name", values="param_value")
        finally:
            con.close()
    trials.loc[trials["value_type"] != "FINITE", "value"] = np.nan  # failed runs are stored as +inf by the drivers
    trials["datetime_start"] = pd.to_datetime(trials["datetime_start"])
    trials["datetime_complete"] = pd.to_datetime(trials["datetime_complete"])
    trials["duration_s"] = (trials["datetime_complete"] - trials["datetime_start"]).dt.total_seconds()
    return trials.merge(params, left_on="trial_id", right_index=True, how="left")


def first_within(best_so_far: pd.Series, final_best: float, rel: float):
    """Index label of the first entry that is within rel of the final best (None if never)."""
    hit = best_so_far[best_so_far <= final_best * (1 + rel)]
    return None if hit.empty else hit.index[0]


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})

    for label, db, ylabel in STUDIES:
        df = read_study(Path(db), STUDY_ID)
        done = df[(df["state"] == "COMPLETE") & df["value"].notna()].copy()
        final_best = done["value"].min()
        best_row = done.loc[done["value"].idxmin()]

        # best so far by trial number
        by_number = done.sort_values("number")
        by_number["best_so_far"] = by_number["value"].cummin()
        # best so far by completion time (ties broken by trial number)
        by_time = done.sort_values(["datetime_complete", "number"]).reset_index(drop=True)
        by_time["completed_trials"] = np.arange(1, len(by_time) + 1)
        by_time["best_so_far_by_completion"] = by_time["value"].cummin()
        df = df.merge(by_number[["number", "best_so_far"]], on="number", how="left").merge(
            by_time[["number", "completed_trials", "best_so_far_by_completion"]], on="number", how="left")
        df = df.drop(columns=["trial_id", "value_type"])
        df.to_csv(out_dir / f"{label}_trials.csv", index=False)

        # figure: trial values and best-so-far curve, no title (caption carries it)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(by_number["number"], by_number["value"], ".", color="0.65", ms=4, label="Trial")
        ax.plot(by_number["number"], by_number["best_so_far"], "-", color="C3", lw=2, label="Best so far")
        ymax = np.nanpercentile(done["value"], 95)  # outliers (failed or poor trials) would squash the curve
        ax.set_ylim(final_best - 0.03 * (ymax - final_best), ymax)
        ax.set_xlim(0, df["number"].max())
        for rel, color in zip(THRESHOLDS, ("C0", "C2")):
            n = first_within(by_number.set_index("number")["best_so_far"], final_best, rel)
            if n is not None:
                ax.axvline(n, color=color, ls="--", lw=1.5, label=f"Within {rel * 100:g} % of final best (trial {n})")
        ax.set_xlabel("Trial number")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        fig.savefig(out_dir / f"{label}_best_so_far.png", dpi=DPI)
        plt.close(fig)

        # text summary
        lines = [f"Study: {label}", f"Database: {db}",
                 f"Trials in database: {len(df)} (complete with a finite value: {len(done)}, "
                 f"complete but not finite: {int(((df['state'] == 'COMPLETE') & df['value'].isna()).sum())}, "
                 f"running/other: {int((df['state'] != 'COMPLETE').sum())})",
                 f"Best value: {final_best:.6f}  at trial number {int(best_row['number'])}",
                 "Best parameters:"]
        param_cols = [c for c in df.columns if c not in ("number", "state", "datetime_start", "datetime_complete", "value", "duration_s",
                                                         "best_so_far", "completed_trials", "best_so_far_by_completion")]
        best_row = df.loc[df["number"] == best_row["number"]].iloc[0]  # the row with the merged curves, same trial
        lines += [f"  {c}: {float(best_row[c])!r}" for c in param_cols if pd.notna(best_row[c])]
        for rel in THRESHOLDS:
            n = first_within(by_number.set_index("number")["best_so_far"], final_best, rel)
            k = first_within(by_time.set_index("completed_trials")["best_so_far_by_completion"], final_best, rel)
            lines.append(f"Best-so-far within {rel * 100:g} % of the final best ({final_best * (1 + rel):.6f}):  "
                         f"trial number {n} (ordered by trial number);  after {k} completed trials (ordered by completion time)")
        lines.append(f"Median trial duration: {done['duration_s'].median() / 60:.1f} min")
        (out_dir / f"{label}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), "\n")


if __name__ == "__main__":
    main()
