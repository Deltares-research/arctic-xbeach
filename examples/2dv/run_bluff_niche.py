"""Demo 2: thermal-abrasion niche and block failure at an ice-rich bluff.

A 5 m vertical bluff of ice-rich silt (65% ice by volume, -8 degC) fronts a narrow beach.
A storm raises the water level to +1 m for six days and holds 5 degC water against the bluff
toe. Heat crosses the water-bluff face at h (T_w - T_f); thawed, ice-rich cells lose their
matrix and become water, so a niche cuts into the bluff at the waterline. Once the niche is
deeper than d_crit the overhang drops into it as a block (its enthalpy is kept), the block's
base is wetted again and the cycle repeats. After the storm the water falls to +0.1 m.

Reference rate: without conduction into the cold interior, the waterline retreats at
v = h (T_w - T_f) / (L_v + C_f (T_f - T_0)).

Outputs (in output/bluff_niche/): bluff_niche.nc, bluff_niche_snapshots.png, bluff_niche_retreat.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from arctic_xbeach.thermal2dv import Grid, Model2DV, Recorder, Settings, run, stretched_edges  # noqa: E402
from arctic_xbeach.thermal2dv import materials as mat  # noqa: E402
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402
from arctic_xbeach.thermal2dv.forcing import Forcing  # noqa: E402

# === CONFIGURATION ===
OUT_DIR = REPO / "examples" / "2dv" / "output" / "bluff_niche"
X_FACE = 20.0             # initial bluff face position [m]
BLUFF_TOP = 5.0           # bluff crest elevation [m]
BEACH_Z = 0.0             # beach elevation in front of the bluff [m]
DX_FINE = 0.1             # fine grid spacing around the bluff toe [m]
PHI_S_BLUFF = 0.35        # mineral fraction of the bluff (rest is ice: very ice-rich silt)
PHI_S_BEACH = 0.60        # beach/seabed sediment (not ice-rich: left to XBeach in the coupled model)
PHI_S_BASE = 0.60         # permafrost below beach level: frozen but not ice-rich, so it is not ablated
T_BLUFF = -8.0            # initial bluff temperature [degC]
T_WATER = 5.0             # storm water temperature [degC]
T_AIR = 5.0               # air temperature [degC]
ETA_STORM = 1.0           # water level during the storm [m]
ETA_CALM = 0.1            # water level after the storm [m]
STORM_DAYS = 6.0
H_WATER = 500.0           # water -> soil heat transfer coefficient [W m-2 K-1]
D_CRIT = 1.5              # niche depth at which the overhang fails [m]
DT = 1800.0               # time step [s]
T_END = 10 * 86400.0      # run length [s]
SNAP_EVERY = 3 * 3600.0   # snapshot interval [s]


def forcing(t, zc):
    eta = ETA_STORM if t < STORM_DAYS * 86400.0 else ETA_CALM
    lw = mat.SIGMA * (T_AIR + mat.T0K) ** 4  # no net longwave for a surface at air temperature
    return Forcing(T_air=T_AIR, sw_down=0.0, lw_down=lw, u10=5.0, eta=eta,
                   T_ref=np.full(len(zc), T_WATER), albedo_land=0.18)


def waterline_face(ds, z_probe):
    """x of the first soil cell (from the sea) in the row containing z_probe, per snapshot."""
    j = int(np.argmin(np.abs(ds.z.values - z_probe)))
    soil = ds.phi_s.isel(z=j).values >= 0.1
    x = ds.x.values
    return np.array([x[np.argmax(row & (x > X_FACE - 5))] for row in soil])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xe = stretched_edges(0.0, 40.0, 15.0, 30.0, DX_FINE, 1.0, growth=1.15)
    ze = stretched_edges(-6.0, 7.0, -1.5, 6.0, DX_FINE, 1.0, growth=1.15)
    grid = Grid(xe, ze)
    settings = Settings(h_water=H_WATER, q_geo=0.06, tau_flush=3 * 3600.0, n_sponge=3, d_crit=D_CRIT,
                        erosion=True, collapse=True)
    model = Model2DV(grid, settings)

    X, Z = grid.meshgrid()
    beach_z = np.interp(grid.xc, [0.0, 10.0, X_FACE], [-2.0, BEACH_Z, BEACH_Z])
    bluff = (X >= X_FACE) & (Z < BLUFF_TOP)
    base = bluff & (Z < BEACH_Z)
    beach = (X < X_FACE) & (Z < beach_z[None, :])
    model.phi_s[bluff] = np.where(base, PHI_S_BASE, PHI_S_BLUFF)[bluff]
    model.phi_h2o[bluff] = 1.0 - model.phi_s[bluff]
    model.phi_s[beach] = PHI_S_BEACH
    model.phi_h2o[beach] = 1.0 - PHI_S_BEACH
    model.S[beach] = settings.S_sea
    model.set_temperature(np.where(bluff, T_BLUFF, 0.0))
    model.update_wet_dry(ETA_STORM, np.full(grid.nz, T_WATER))

    rec = Recorder(model, every=SNAP_EVERY)
    run(model, forcing, T_END, DT, recorder=rec)
    ds = rec.to_dataset()
    ds.to_netcdf(OUT_DIR / "bluff_niche.nc")

    t_days = ds.time.values / 86400.0
    t_collapse = [e["time"] / 86400.0 for e in model.collapse_events]
    cf, _, lv = mat.heat_capacities(PHI_S_BLUFF, 1.0 - PHI_S_BLUFF)
    v_ref = H_WATER * T_WATER / (lv + cf * (0.0 - T_BLUFF)) * 86400.0
    face = waterline_face(ds, 0.5 * ETA_STORM)
    crest = waterline_face(ds, BLUFF_TOP - 0.3)
    budget = sum(rec.series[f"energy_{k}"][-1] - rec.series[f"energy_{k}"][0] for k in model.energy)
    e_err = (rec.series["energy_domain"][-1] - rec.series["energy_domain"][0]) - budget
    print(f"collapse events at t = {', '.join(f'{t:.2f}' for t in t_collapse)} days; "
          f"crest retreat after {T_END / 86400:.0f} days: {crest[-1] - X_FACE:.2f} m")
    print(f"reference waterline retreat {v_ref:.2f} m/day; eroded sediment {model.eroded_sediment:.2f} m3/m; "
          f"energy budget error {e_err:.2e} J/m ({e_err / max(abs(budget), 1):.1e} relative)")

    # pick panels: start, niche just before and just after the first collapse, end of storm, end
    picks = [0.0]
    if t_collapse:
        picks += [t_collapse[0] - 0.05, t_collapse[0] + 0.1]
    picks += [STORM_DAYS - 0.05, T_END / 86400.0]
    its = [int(np.argmin(np.abs(t_days - tp))) for tp in picks]
    its = [its[0]] + [i for k, i in enumerate(its[1:]) if i != its[k]]

    pl.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        if k >= len(its):
            ax.set_visible(False)
            continue
        pl.plot_phase(ax, ds, its[k])
        ax.axhline(ETA_STORM if t_days[its[k]] < STORM_DAYS else ETA_CALM, color="w", lw=0.8, ls=":")
        pl.panel_label(ax, f"({chr(97 + k)}) t = {t_days[its[k]]:.2f} d")
        ax.set_xlim(12, 32)
        ax.set_ylim(-3, 6.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z (m)")
    pl.phase_legend(axes.flat[len(its) - 1], loc="lower right", classes=[1, 3, 4, 5], ncol=2)
    pl.save(fig, OUT_DIR / "bluff_niche_snapshots.png")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(t_days, face - X_FACE, "C0-", lw=2, label=f"face position at z = {0.5 * ETA_STORM:.1f} m (niche)")
    ax.plot(t_days, crest - X_FACE, "C3-", lw=2, label=f"crest retreat (z = {BLUFF_TOP - 0.3:.1f} m)")
    ax.plot(np.array(rec.series["time"]) / 86400.0, rec.series["eroded_sediment"], "C1-", lw=1.5,
            label="eroded sediment (m$^3$ per m)")
    tt = np.linspace(0, STORM_DAYS, 10)
    ax.plot(tt, v_ref * tt, "k--", lw=1, label=f"reference rate {v_ref:.2f} m/day")
    for k, tc in enumerate(t_collapse):
        ax.axvline(tc, color="0.5", lw=0.8, ls=":", label="block failure" if k == 0 else None)
    ax.axvspan(0, STORM_DAYS, color="C0", alpha=0.07, lw=0)
    ax.set_xlim(0, T_END / 86400.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("retreat (m), volume (m$^3$ per m)")
    ax.legend(frameon=False, loc="upper left")
    pl.save(fig, OUT_DIR / "bluff_niche_retreat.png")


if __name__ == "__main__":
    main()
