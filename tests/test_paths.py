"""Paths in config.yaml work on Windows and Linux (arctic_xbeach.miscellaneous.resolve_path), and a filesystem without
file locks is reported as such instead of as a second run (arctic_xbeach.run_lock)."""
import errno
from pathlib import Path

import pytest

from arctic_xbeach import run_lock
from arctic_xbeach.miscellaneous import resolve_path

PROJ = Path("proj") / "dir"


@pytest.mark.parametrize("path", [r"C:\x\y.csv","d:/x", "/u/x/y.csv"])
def test_absolute_paths_are_kept(path):
    assert resolve_path(path, PROJ) == str(Path(path))


def test_relative_path_is_joined_to_project_dir():
    assert resolve_path("examples/simple/database/era5.csv", PROJ) == str(PROJ / "examples/simple/database/era5.csv")


@pytest.mark.parametrize("err", [errno.ENOLCK, errno.EOPNOTSUPP])
def test_filesystem_without_locks_is_reported_as_such(tmp_path, monkeypatch, err):
    def no_locks(fh):
        raise OSError(err, "no locks available")
    monkeypatch.setattr(run_lock, "_try_lock", no_locks)
    with pytest.raises(RuntimeError, match="does not support file locks") as exc:
        run_lock.acquire_run_lock(tmp_path)
    assert str(tmp_path) in str(exc.value)
    assert "Another Arctic-XBeach run" not in str(exc.value)


def test_lock_held_by_another_run_is_still_reported_as_such(tmp_path, monkeypatch):
    def held(fh):
        raise OSError(errno.EAGAIN, "resource temporarily unavailable")
    monkeypatch.setattr(run_lock, "_try_lock", held)
    with pytest.raises(RuntimeError, match="Another Arctic-XBeach run is using"):
        run_lock.acquire_run_lock(tmp_path)
