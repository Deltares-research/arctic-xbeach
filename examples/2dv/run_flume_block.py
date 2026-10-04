"""Demo 1: an ice block at the end of a wave flume, standing in for a bluff.

A laboratory-scale analogue of thermal abrasion, after the cold-room flume experiments with
frozen blocks (Costard et al. 2003; Dupeyrat et al. 2011). A block of pure ice (by default; set
PHI_S_BLOCK > 0 for ice-rich frozen sediment) stands against the landward wall of a flume. Warm fresh water fills the flume to below the
block top, is kept at T_WATER by recirculation (a sponge at the wave-maker end plus flushing), and
is stirred by the waves. There is no hydrodynamics: the waves act only through the well-mixed water
and the face heat transfer coefficient h.

What should happen: the submerged face melts and cuts a niche at the waterline. The part above
the water stays nearly frozen and becomes an overhang. When the niche reaches D_CRIT, the overhang
drops into the niche (its enthalpy is kept) and the cycle repeats. Before the first failure the
face retreats at the quasi-steady rate

    v = h (T_w - T_f) / (L_v + C_f (T_f - T_0))

(no conduction into the cold interior, i.e. the idealised thermal-abrasion rate). For pure ice
C_f and L_v are those of ice (1.9 MJ m-3 K-1, 334 MJ m-3); for frozen sediment they are the
mixture values. Ice overhangs fail by the same niche rule (Settings.collapse_ice).

Outputs (in output/flume_block/): flume_block.nc, flume_block_snapshots.png, flume_block_retreat.png.
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
OUT_DIR = REPO / "examples" / "2dv" / "output" / "flume_block"
FLUME_LENGTH = 8.0        # [m]; the wave maker is at x = 0, the block against the wall at x = FLUME_LENGTH
FLUME_HEIGHT = 1.3        # model domain height above the flume floor [m]
DX = 0.05                 # grid spacing in x and z [m]
BLOCK_LENGTH = 2.0        # [m]
BLOCK_HEIGHT = 1.0        # [m]
WATER_DEPTH = 0.4         # still water depth [m]
PHI_S_BLOCK = 0.0         # mineral fraction of the block, rest ice (0 = pure ice; 0.35 = ice-rich silt)
T_BLOCK = -5.0            # initial block temperature [degC]
T_WATER = 5.0             # flume water temperature, held by recirculation [degC]
T_AIR = 5.0               # cold-room air temperature [degC]
U_AIR = 1.0               # air speed over the flume [m s-1] (bulk sensible heat only, no radiation)
H_WATER = 500.0           # water -> block heat transfer coefficient [W m-2 K-1]; waves raise it
D_CRIT = 0.3              # niche depth at which the overhang fails [m] (lab-scale cantilever)
DT = 600.0                # time step [s]
T_END = 48 * 3600.0       # run length [s]
SNAP_EVERY = 1800.0       # snapshot interval [s]


def face_position(ds, z_probe):
    """x of the seaward face of the block in the row containing z_probe, per snapshot."""
    j = int(np.argmin(np.abs(ds.z.values - z_probe)))
    x = ds.x.values
    if PHI_S_BLOCK > 0:
        solid = ds.phi_s.isel(z=j).values >= 0.1
    else:
        solid = ds.f_liq.isel(z=j).values < 0.5
    return np.array([x[np.argmax(row)] if row.any() else FLUME_LENGTH for row in solid])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = Grid(np.arange(0.0, FLUME_LENGTH + DX / 2, DX), np.arange(0.0, FLUME_HEIGHT + DX / 2, DX))
    settings = Settings(h_water=H_WATER, q_geo=0.0, emissivity=0.0, S_sea=0.0, n_sponge=8, tau_sponge=600.0,
                        tau_flush=3600.0, Kx_surf=1.0, Kz_surf=1e-2, d_crit=D_CRIT,
                        erosion=True, collapse=True, collapse_ice=True)
    model = Model2DV(grid, settings)

    X, Z = grid.meshgrid()
    x_face = FLUME_LENGTH - BLOCK_LENGTH
    block = (X >= x_face) & (Z < BLOCK_HEIGHT)
    water = ~block & (Z < WATER_DEPTH)
    model.phi_s[block] = PHI_S_BLOCK
    model.phi_h2o[block] = 1.0 - PHI_S_BLOCK
    model.phi_h2o[water] = 1.0
    T0 = T_BLOCK
    model.set_temperature(np.where(block, T0, T_WATER), frozen_at_tf=True)

    forcing = constant_forcing(T_air=T_AIR, u10=U_AIR, lw_down=None, eta=WATER_DEPTH, T_water=T_WATER)
    rec = Recorder(model, every=SNAP_EVERY)
    run(model, forcing, T_END, DT, recorder=rec)
    ds = rec.to_dataset()
    ds.to_netcdf(OUT_DIR / "flume_block.nc")

    # reference rate and diagnostics
    t_h = ds.time.values / 3600.0
    t_collapse = [e["time"] / 3600.0 for e in model.collapse_events]
    cf, _, lv = mat.heat_capacities(PHI_S_BLOCK, 1.0 - PHI_S_BLOCK)
    v_ref = H_WATER * T_WATER / (lv + cf * (0.0 - T0)) * 3600.0  # [m per hour]
    face = face_position(ds, 0.5 * WATER_DEPTH) - x_face
    crest = face_position(ds, BLOCK_HEIGHT - 0.05) - x_face
    t_fit = t_h[(t_h > 2.0) & (t_h < (t_collapse[0] if t_collapse else t_h[-1]))]
    v_model = np.polyfit(t_fit, np.interp(t_fit, t_h, face), 1)[0] if len(t_fit) > 2 else np.nan
    budget = sum(rec.series[f"energy_{k}"][-1] - rec.series[f"energy_{k}"][0] for k in model.energy)
    e_err = (rec.series["energy_domain"][-1] - rec.series["energy_domain"][0]) - budget
    print(f"niche retreat before the first failure: model {v_model * 24:.3f} m/day, reference {v_ref * 24:.3f} m/day "
          f"({(v_model / v_ref - 1) * 100:+.1f}%)")
    print(f"block failures at t = {', '.join(f'{t:.1f}' for t in t_collapse) or 'none'} h; "
          f"crest retreat after {T_END / 3600:.0f} h: {crest[-1]:.2f} m; eroded sediment {model.eroded_sediment:.3f} m3/m")
    print(f"energy budget error {e_err:.2e} J/m ({e_err / max(abs(budget), 1.0):.1e} of the net exchange)")

    # snapshots: start, just before and after the first failure, and the end; phase (top) and T (bottom)
    picks = [0.0]
    if t_collapse:
        picks += [t_collapse[0] - 0.4, t_collapse[0] + 0.6]
    else:
        picks += [T_END / 3600.0 / 3, 2 * T_END / 3600.0 / 3]
    picks.append(T_END / 3600.0)
    its = [int(np.argmin(np.abs(t_h - tp))) for tp in picks]

    pl.apply_style()
    fig, axes = plt.subplots(2, len(its), figsize=(4.0 * len(its), 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for k, it in enumerate(its):
        pl.plot_phase(axes[0, k], ds, it)
        m = pl.plot_temperature(axes[1, k], ds, it, vlim=6.0)
        for r in range(2):
            ax = axes[r, k]
            pl.plot_grid(ax, ds)
            ax.axhline(WATER_DEPTH, color="k", lw=0.6, ls=":")
            ax.set_xlim(x_face - 1.2, FLUME_LENGTH)
            ax.set_ylim(0, FLUME_HEIGHT)
            pl.panel_label(ax, f"({chr(97 + r * len(its) + k)}) t = {t_h[it]:.1f} h")
        axes[1, k].set_xlabel("x (m)")
    for r in range(2):
        axes[r, 0].set_ylabel("z (m)")
    pl.phase_legend(axes[0, 0], loc="upper left", classes=[1, 3, 4, 5] if PHI_S_BLOCK > 0 else [1, 2], ncol=1,
                    bbox_to_anchor=(0.0, 0.88), fontsize=9, handlelength=1.2)
    fig.colorbar(m, ax=axes[1, :], label="temperature (°C)", shrink=0.9, extend="both")
    pl.save(fig, OUT_DIR / "flume_block_snapshots.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.plot(t_h, face, "C0-", lw=2, label=f"face at z = {0.5 * WATER_DEPTH:.2f} m (niche)")
    ax.plot(t_h, crest, "C3-", lw=2, label=f"crest (z = {BLOCK_HEIGHT - 0.05:.2f} m)")
    tt = np.linspace(0, T_END / 3600.0, 50)
    ax.plot(tt, v_ref * tt, "k--", lw=1, label=f"quasi-steady rate {v_ref * 24:.2f} m/day")
    for k, tc in enumerate(t_collapse):
        ax.axvline(tc, color="0.5", lw=0.8, ls=":", label="block failure" if k == 0 else None)
    ax.set_xlim(0, T_END / 3600.0)
    ax.set_ylim(0, None)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("retreat (m)")
    ax.legend(frameon=False, loc="upper left")
    pl.save(fig, OUT_DIR / "flume_block_retreat.png")


if __name__ == "__main__":
    main()
