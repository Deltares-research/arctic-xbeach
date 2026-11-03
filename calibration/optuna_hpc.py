"""Helpers shared by the two calibration drivers: study storage, trials left RUNNING, scratch folders, the shared print log.

Only one driver process per study is supported. The 'RUNNING' trials found at the start are taken to be left by a killed
earlier job and are marked FAIL; a second driver on the same study would have its live trials failed.
"""
import shutil
import threading
from pathlib import Path

import optuna

_LOG_LOCK = threading.Lock()  # the trials of one driver are threads that append to the same calibration_prints.txt


def append_log(path: Path, text: str) -> None:
    with _LOG_LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(text)


def make_storage(kind: str, run_root: Path):
    """'sqlite': RUN_ROOT/optuna_study.db. 'journal': RUN_ROOT/optuna_journal.log, an append-only file that Optuna documents as
    safe on NFS (SQLite locking is not reliable there)."""
    if kind == "sqlite":
        return f"sqlite:///{(run_root / 'optuna_study.db').as_posix()}"
    if kind == "journal":
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        path = str(run_root / "optuna_journal.log")
        return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    raise ValueError(f"STORAGE must be 'sqlite' or 'journal', not {kind!r}")


def open_study(study_name: str, run_root: Path, storage_kind: str, sampler) -> optuna.Study:
    """Create or load the study (a rerun continues it) and mark the trials that a killed job left RUNNING as FAIL."""
    run_root.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=study_name, direction="minimize", load_if_exists=True,
                                storage=make_storage(storage_kind, run_root), sampler=sampler)
    stale = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    for t in stale:
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    print(f"Marked {len(stale)} trials left RUNNING by an earlier job as FAIL")
    return study


def trial_dirs(run_root: Path, scratch: Path | None, number: int) -> tuple[Path, Path]:
    """(folder the model runs in, final folder in RUN_ROOT). They are the same folder without scratch."""
    name = f"optuna_trial_{number:06d}"
    return (scratch / name if scratch else run_root / name), run_root / name


def finish_trial_dir(work_dir: Path, final_dir: Path, keep_results_nc: bool, keep_xb_files: bool) -> None:
    """With scratch: copy the trial folder back to RUN_ROOT (also after a failed run: run.log is needed) without the XBeach
    call folders and, unless kept, results.nc, then delete the scratch copy. Without scratch: delete the XBeach call folders
    after scoring if keep_xb_files is False."""
    xb_files = work_dir / "results" / "xb_files"  # one folder per XBeach call (about 23 files each), written by arctic_xbeach/model.py
    if work_dir == final_dir:
        if not keep_xb_files:
            shutil.rmtree(xb_files, ignore_errors=True)
        return
    if not work_dir.exists():
        return

    def ignore(folder, names):
        if Path(folder) != work_dir / "results":
            return []
        return [n for n in names if n == "xb_files" or (n == "results.nc" and not keep_results_nc)]

    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(work_dir, final_dir, ignore=ignore)
    shutil.rmtree(work_dir, ignore_errors=True)  # only reached if the copy worked; after a failed copy the scratch folder stays
