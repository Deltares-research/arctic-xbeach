"""One run per folder (arctic_xbeach.run_lock): a second process cannot take the folder while the first holds it, and
the folder is free again once the holder ends, also when it is killed and leaves run.lock behind."""
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arctic_xbeach.run_lock import acquire_run_lock, release_run_lock, LOCK_NAME

ROOT = Path(__file__).resolve().parents[1]
HOLDER = """
import sys, time
sys.path.insert(0, {root!r})
from arctic_xbeach.run_lock import acquire_run_lock
lock = acquire_run_lock({run_dir!r})
print("locked", flush=True)
time.sleep(60)
"""


def _start_holder(run_dir):
    p = subprocess.Popen([sys.executable, "-c", HOLDER.format(root=str(ROOT), run_dir=str(run_dir))],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "locked"
    return p


def test_second_process_is_refused_and_folder_frees_after_kill(tmp_path):
    holder = _start_holder(tmp_path)
    try:
        with pytest.raises(RuntimeError, match=f"pid {holder.pid}"):
            acquire_run_lock(tmp_path)
    finally:
        holder.kill()
        holder.wait()
    assert (tmp_path / LOCK_NAME).exists()          # a killed run leaves the file behind ...
    lock = acquire_run_lock(tmp_path)               # ... but not the lock
    release_run_lock(lock)
    assert (tmp_path / LOCK_NAME).read_text().startswith("free")


def test_same_folder_can_be_reused_after_release(tmp_path):
    for _ in range(2):
        release_run_lock(acquire_run_lock(tmp_path))
