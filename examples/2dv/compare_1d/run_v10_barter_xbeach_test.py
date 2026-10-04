"""Test the cross-platform XBeach launcher (Simulation.start_xbeach) with a short coupled v1.0 run
of the Barter Island case study (2015-08-01 to 2015-08-03, storm triggers on 1-2 August).

The case (v10_barter_xbeach_test/config.yaml) is the case-study config with a 2-day window,
absolute paths to the full ERA5/storm files and `xbeach.executable`. The script wraps
`sim.start_xbeach` only to time each call, then checks every XBeach run folder for
xboutput.nc, zs/zb and sedero.
"""
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parents[2]                       # worktree root (contains main.py)
CASE_DIR = SCRIPT_DIR / "v10_barter_xbeach_test"
PRINT_TO_SCREEN = False                                # v1.0 log always goes to CASE_DIR/run.log

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from arctic_xbeach.model import Simulation  # noqa: E402
from main import main  # noqa: E402

sim = Simulation(CASE_DIR.relative_to(PROJ_DIR).as_posix(), proj_dir=PROJ_DIR)
print(f"XBeach executable: {sim.resolve_xbeach_executable()}")

calls = []
original_start_xbeach = sim.start_xbeach


def timed_start_xbeach(*args, **kwargs):
    t0 = time.perf_counter()
    ok = original_start_xbeach(*args, **kwargs)
    calls.append((kwargs.get("timestep_id"), ok, time.perf_counter() - t0))
    return ok


sim.start_xbeach = timed_start_xbeach

t_wall = time.perf_counter()
main(sim, print_to_screen=PRINT_TO_SCREEN)
print(f"main(): {time.perf_counter() - t_wall:.1f} s wall")

print(f"{len(calls)} XBeach calls")
for tid, ok, wall in calls:
    folder = Path(sim.result_dir) / "xb_files" / f"{int(tid):010d}"
    nc = folder / "xboutput.nc"
    msg = f"  step {tid:3d}: success={ok}, {wall:6.1f} s"
    if nc.is_file():
        with xr.open_dataset(nc) as ds:
            last = ds.isel(globaltime=-1).squeeze()
            sed = last["sedero"].values if "sedero" in ds else np.array([np.nan])
            zs = last["zs"].values
            msg += (f", zs max {np.nanmax(zs):.2f} m, zb present {'zb' in ds}, "
                    f"|sedero| max {np.nanmax(np.abs(sed)):.3e} m, storm={bool(sim.xbeach_storms[tid])}")
    else:
        msg += ", xboutput.nc MISSING"
    print(msg)
if calls:
    walls = np.array([c[2] for c in calls])
    print(f"wall per XBeach call: mean {walls.mean():.1f} s, min {walls.min():.1f} s, max {walls.max():.1f} s")
