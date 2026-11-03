"""
One run per folder: an OS file lock on <run folder>/run.lock.

A second run started in a folder that is in use would truncate run.log and recreate results.nc under the first run, which
corrupts the netCDF file of the first run ("NetCDF: HDF error"). The lock is held by the operating system, so it is
released when the process ends in any way (finished, exception, killed, terminal closed) and a lock file left behind
never blocks a later run. The file itself only records who holds the lock and may be deleted when no run is active.
"""
import errno
import os
import socket
from datetime import datetime
from pathlib import Path

LOCK_NAME   = "run.lock"
NO_LOCK_ERRNOS = (errno.ENOLCK, errno.EOPNOTSUPP)    # POSIX locks missing on the filesystem (e.g. some NFS mounts), not a second run
LOCK_OFFSET = 4096      # lock one byte past the text, so a second process can still read who holds the lock (Windows locks are mandatory)


def _try_lock(fh):
    fh.seek(LOCK_OFFSET)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)     # raises OSError when another process holds it
    else:
        import fcntl
        fcntl.lockf(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, LOCK_OFFSET)


def _write(fh, text):
    fh.seek(0)
    fh.truncate()
    fh.write(text.encode("utf-8"))
    fh.flush()


def acquire_run_lock(run_dir):
    """Lock run_dir for this process and return the open lock file. Raises RuntimeError if another run holds it."""
    path = Path(run_dir) / LOCK_NAME
    fh = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT), "r+b")    # no truncation: the holder's text stays readable
    try:
        _try_lock(fh)
    except OSError as err:
        if err.errno in NO_LOCK_ERRNOS:
            fh.close()
            raise RuntimeError(f"The filesystem of {Path(run_dir)} does not support file locks ({err.strerror}), so the one-run-per-folder "
                               f"check cannot work there. Use a run folder on a local or scratch disk, or one on a network "
                               f"filesystem that supports locking.")
        os.lseek(fh.fileno(), 0, os.SEEK_SET)
        text = os.read(fh.fileno(), 512)    # unbuffered: a buffered read would reach the locked byte and fail on Windows
        holder = text.decode("utf-8", errors="replace").strip() or "holder unknown"
        fh.close()
        raise RuntimeError(f"Another Arctic-XBeach run is using {Path(run_dir)} ({holder}). "
                           f"Wait for it to finish or stop it; starting a second run here would corrupt its output.")
    _write(fh, f"held by pid {os.getpid()} on {socket.gethostname()}, started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return fh


def release_run_lock(fh):
    """Release the lock. The file stays (deleting it could race with a run that is starting) and says it is free."""
    if fh is None or fh.closed:
        return
    try:
        _write(fh, f"free (last run pid {os.getpid()}, ended {datetime.now():%Y-%m-%d %H:%M:%S})\n")
    finally:
        fh.close()      # closing the handle releases the OS lock
