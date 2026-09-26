"""Demo 1: a pure ice block melting in warm, well-mixed water (verification-grade).

A 1 m x 1 m ice block at 0 degC sits on an insulated floor in a tank of 5 degC water held
at that temperature by flushing. With no conduction into the (isothermal) block, each exposed
face retreats at v = h (T_w - T_f) / L_v. The block is 1 m wide and 1 m high with three
exposed faces, so the ice area is (W - 2vt)(H - vt) until it is gone.

Outputs (in output/ice_block/): ice_block.nc, ice_block_snapshots.png, ice_block_area.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from arctic_xbeach.thermal2dv import Grid, Model2DV, Recorder, Settings, constant_forcing, run  # noqa: E402
from arctic_xbeach.thermal2dv import materials as mat  # noqa: E402
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402

# === CONFIGURATION ===
OUT_DIR = REPO / "examples" / "2dv" / "output" / "ice_block"
DX = 0.05                 # grid spacing [m]
TANK_WIDTH = 4.0          # [m]
TANK_DEPTH = 2.0          # [m]
BLOCK_WIDTH = 1.0         # [m]
BLOCK_HEIGHT = 1.0        # [m]
T_WATER = 5.0             # ambient water temperature [degC]
H_WATER = 500.0           # water-ice heat transfer coefficient [W m-2 K-1]
DT = 600.0                # time step [s]
T_END = 30 * 3600.0       # run length [s]
SNAPSHOT_HOURS = [0, 8, 16, 24]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = Grid(np.arange(0.0, TANK_WIDTH + DX / 2, DX), np.arange(0.0, TANK_DEPTH + DX / 2, DX))
    settings = Settings(h_water=H_WATER, q_geo=0.0, tau_flush=600.0, n_sponge=0, S_sea=0.0,
                        Kx_surf=1.0, Kz_surf=1.0, erosion=False, collapse=False)
    model = Model2DV(grid, settings)
    model.phi_h2o[:] = 1.0  # every cell is H2O; the block is simply frozen H2O

    X, Z = grid.meshgrid()
    block = (np.abs(X - TANK_WIDTH / 2) < BLOCK_WIDTH / 2) & (Z < BLOCK_HEIGHT)
    model.set_temperature(np.where(block, 0.0, T_WATER), frozen_at_tf=True)
    forcing = constant_forcing(T_air=T_WATER, u10=0.0, lw_down=None, eta=TANK_DEPTH + 1.0, T_water=T_WATER)

    rec = Recorder(model, times=[h * 3600.0 for h in SNAPSHOT_HOURS])
    run(model, forcing, T_END, DT, recorder=rec)
    ds = rec.to_dataset()
    ds.to_netcdf(OUT_DIR / "ice_block.nc")

    # analytic ice area
    v = H_WATER * T_WATER / mat.LV_WATER
    t = np.array(rec.series["time"])
    area_exact = np.clip(BLOCK_WIDTH - 2 * v * t, 0, None) * np.clip(BLOCK_HEIGHT - v * t, 0, None)
    area_model = np.array(rec.series["ice_volume"])
    budget = sum(rec.series[f"energy_{k}"][-1] - rec.series[f"energy_{k}"][0] for k in model.energy)
    e_err = (rec.series["energy_domain"][-1] - rec.series["energy_domain"][0]) - budget
    i_half = np.argmin(np.abs(area_exact - 0.5))
    print(f"retreat rate v = {v * 86400:.3f} m/day; energy budget error = {e_err:.3e} J/m "
          f"({e_err / abs(budget):.1e} relative)")
    print(f"ice area at t = {t[i_half] / 3600:.1f} h: model {area_model[i_half]:.3f} m2, exact {area_exact[i_half]:.3f} m2")

    pl.apply_style()
    fig, axes = plt.subplots(1, len(SNAPSHOT_HOURS), figsize=(14, 3.4), sharey=True, constrained_layout=True)
    for k, ax in enumerate(axes):
        pl.plot_phase(ax, ds, k)
        ax.set_aspect("equal")
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0, 1.6)
        pl.panel_label(ax, f"({chr(97 + k)}) t = {float(ds.time[k]) / 3600:.0f} h")
        ax.set_xlabel("x (m)")
        # analytic block outline at the same time
        tk = float(ds.time[k])
        w, h = max(BLOCK_WIDTH - 2 * v * tk, 0), max(BLOCK_HEIGHT - v * tk, 0)
        if w > 0 and h > 0:
            ax.plot([2 - w / 2, 2 - w / 2, 2 + w / 2, 2 + w / 2], [0, h, h, 0], "r--", lw=1.2, label="analytic")
    axes[0].set_ylabel("z (m)")
    pl.phase_legend(axes[-1], loc="upper right", classes=[1, 2], ncol=1)
    pl.save(fig, OUT_DIR / "ice_block_snapshots.png")

    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    ax.plot(t / 3600, area_exact, "k-", lw=2, label="analytic, $v = h\\,\\Delta T / L_v$")
    ax.plot(t / 3600, area_model, "C3--", lw=1.8, label=f"2DV model, $\\Delta x$ = {DX * 100:.0f} cm")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("ice area (m$^2$ per m)")
    ax.set_xlim(0, T_END / 3600)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    pl.save(fig, OUT_DIR / "ice_block_area.png")


if __name__ == "__main__":
    main()
