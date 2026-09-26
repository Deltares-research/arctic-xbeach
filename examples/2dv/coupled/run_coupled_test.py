"""Coupled Arctic-XBeach run with the 2DV thermal model (thermal.model: "2dv"): Barter Island,
schematized profile, 15 June - 3 August 2015 (the 2DV model spins up the active layer; the
storm of 1-2 August triggers XBeach).

Checks: every XBeach call succeeds, the energy budget of the 2DV model closes including the
morphology term, and the bed handed to XBeach follows the 2DV state.

Outputs (in examples/2dv/output/coupled/): coupled_test.png, coupled_test_summary.txt; the run
folder holds results/thermal2dv.nc and the XBeach runs.
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parents[2]                       # worktree root (contains main.py)
CASE_DIR = SCRIPT_DIR / "barter_2dv_test"
OUT_DIR = PROJ_DIR / "examples" / "2dv" / "output" / "coupled"
PLOT_ONLY = False                                      # redraw from the run folder without rerunning

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from arctic_xbeach.model import Simulation  # noqa: E402
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402
from main import main  # noqa: E402


def run():
    sim = Simulation(CASE_DIR.relative_to(PROJ_DIR).as_posix(), proj_dir=PROJ_DIR)
    calls = []
    original = sim.start_xbeach

    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        ok = original(*args, **kwargs)
        calls.append((kwargs.get("timestep_id"), ok, time.perf_counter() - t0))
        return ok

    sim.start_xbeach = timed
    t0 = time.perf_counter()
    xgr, zgr = main(sim, print_to_screen=False)
    wall = time.perf_counter() - t0
    lines = [f"main(): {wall:.0f} s wall; {len(calls)} XBeach calls, "
             f"{sum(c[1] for c in calls)} successful, {sum(c[2] for c in calls):.0f} s in XBeach"]
    for tid, ok, w in calls:
        lines.append(f"  step {tid:5d} ({sim.timestamps[tid]}): success={ok}, {w:5.1f} s, storm={bool(sim.xbeach_storms[tid])}")
    return Path(sim.result_dir), lines


def summarise(result_dir, lines):
    ds = xr.load_dataset(result_dir / "thermal2dv.nc")
    terms = [k for k in ds.data_vars if k.startswith("energy_") and k != "energy_domain"]
    budget = sum(ds[k].values[-1] - ds[k].values[0] for k in terms)
    change = ds.energy_domain.values[-1] - ds.energy_domain.values[0]
    lines.append(f"2DV energy budget error {change - budget:.2e} J/m ({(change - budget) / max(abs(budget), 1):.1e} "
                 f"of the net exchange); terms: " + ", ".join(f"{k[7:]} {ds[k].values[-1] - ds[k].values[0]:.3e}" for k in terms))
    lines.append("sediment [m3/m]: " + ", ".join(f"{k[9:]} {v:.3f}" for k, v in ds.attrs.items() if k.startswith("sediment_")))
    lines.append(f"ablated bluff mineral {ds.eroded_sediment.values[-1]:.3f} m3/m")
    txt = "\n".join(lines)
    print(txt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "coupled_test_summary.txt").write_text(txt + "\n", encoding="utf-8")
    return ds


def plot(ds):
    pl.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    x_crest = float(ds.x.values[np.nanargmax(ds.zb_2dv.values)])
    for ax, it, label in ((axes[0], 0, "(a) start"), (axes[1], -1, "(b) end")):
        m = pl.plot_temperature(ax, ds, it) if hasattr(pl, "plot_temperature") else None
        if m is None:
            T = ds.T.isel(time=it).values
            m = ax.pcolormesh(ds.x_edge, ds.z_edge, T, cmap="RdBu_r", vmin=-10, vmax=10)
        ax.set_xlim(x_crest - 60.0, x_crest + 25.0)
        ax.set_ylim(-4.0, float(np.nanmax(ds.zb_2dv)) + 1.5)
        ax.set_ylabel("z (m)")
        ax.text(0.01, 0.95, f"{label}: {str(ds.time.values[it])[:16]}", transform=ax.transAxes, va="top")
    axes[1].step(ds.x, ds.zb_2dv, where="mid", color="k", lw=1.0, label="bed handed to XBeach")
    axes[1].legend(loc="lower right")
    axes[1].set_xlabel("x (m)")
    fig.colorbar(m, ax=axes, label="T (°C)", shrink=0.8)
    fig.savefig(OUT_DIR / "coupled_test.png", dpi=150)


if __name__ == "__main__":
    result_dir = CASE_DIR / "results"
    lines = ["(plot only)"]
    if not PLOT_ONLY:
        result_dir, lines = run()
    plot(summarise(result_dir, lines))
