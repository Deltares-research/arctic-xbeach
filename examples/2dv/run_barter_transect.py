"""Demo 3: one year on the Barter Island profile with synthetic Arctic forcing.

The bed profile is the repo's Barter Island case (examples/case_studies/barter_island):
about -5 m at 650 m offshore rising to a 13 m ice-rich bluff. The grid spans the nearshore,
the bluff and 150 m of tundra, from 30 m below sea level (geothermal flux of 60 mW m-2 at
the base) to above the bluff crest. The landward and offshore soil boundaries have zero flux.
Water cells mix turbulently, are relaxed towards a stratified ocean profile in an offshore
sponge and by alongshore flushing, and exchange heat with the air through the same surface
energy balance as the land.

The forcing is synthetic (see synthetic_arctic_forcing): a Barter-like seasonal cycle of
air temperature and radiation, a sea at its freezing point in winter and stratified in
summer, a microtidal tide and one 1.6 m storm surge in mid-August. It starts on 1 January
from a steady conduction state (mean-annual land surface -9 degC, sea bed -1.5 degC), so
the first months carry some spin-up.

Outputs (in output/barter_transect/): barter_transect.nc, barter_snapshots.png,
barter_timeseries.png, barter_full_domain.png.
"""

import sys
import time as wallclock
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from arctic_xbeach.thermal2dv import (Grid, Model2DV, Recorder, Settings, stretched_edges,  # noqa: E402
                                      synthetic_arctic_forcing)
from arctic_xbeach.thermal2dv import materials as mat  # noqa: E402
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402

# === CONFIGURATION ===
PROFILE_DIR = REPO / "examples" / "case_studies" / "barter_island"
OUT_DIR = REPO / "examples" / "2dv" / "output" / "barter_transect"
X_START, X_END = 700.0, 1520.0      # cross-shore extent [m, profile coordinates]
FINE_X = (1320.0, 1400.0)           # fine-resolution band around the bluff [m]
DX_FINE, DX_MAX = 0.5, 10.0         # [m]
Z_BOTTOM, Z_TOP = -30.0, 15.0       # [m]
FINE_Z = (-6.0, 14.5)
DZ_FINE, DZ_MAX = 0.25, 3.0         # [m]
PHI_S_ICE_RICH = 0.35               # bluff/tundra: 65% ice by volume (Kanevskiy et al. 2013: ~77% in the upper 3 m)
Z_ICE_RICH = 0.3                    # ice-rich material only above this level, so the toe sits above calm water (beach in front)
PHI_S_SEDIMENT = 0.60               # deeper permafrost and seabed sediment
T_LAND_MEAN = -9.0                  # mean-annual ground surface temperature for the initial state [degC]
T_SEABED_MEAN = -1.5                # [degC]
DT = 3 * 3600.0                     # time step [s]
N_DAYS = 365
SNAPSHOT_DOYS = [74, 166, 201, 228, 258, 319]   # 15 Mar, 15 Jun, 20 Jul, 16 Aug (surge peak), 15 Sep, 15 Nov
PROBE_TUNDRA_OFFSET = 60.0          # tundra "borehole" this far landward of the crest [m]
PROBE_DEPTHS = [0.1, 0.5, 1.0, 2.0, 5.0]         # [m below the ground surface]
PROBE_WATER_OFFSETS = [50.0, 250.0]  # nearshore water/ice probes this far seaward of the shoreline [m]


def load_profile():
    x = np.loadtxt(PROFILE_DIR / "x.grd").ravel()
    z = np.loadtxt(PROFILE_DIR / "bed.dep").ravel()
    x = np.append(x, X_END + 50.0)
    z = np.append(z, z[-1])
    return x, z


def build_model():
    xp, zp = load_profile()
    grid = Grid(stretched_edges(X_START, X_END, *FINE_X, DX_FINE, DX_MAX, growth=1.1),
                stretched_edges(Z_BOTTOM, Z_TOP, *FINE_Z, DZ_FINE, DZ_MAX, growth=1.15))
    settings = Settings(h_water=500.0, q_geo=0.06, n_sponge=5, tau_sponge=3 * 3600.0, tau_flush=2 * 86400.0,
                        erosion=True, collapse=True, d_crit=1.5)
    model = Model2DV(grid, settings)

    X, Z = grid.meshgrid()
    x_shore = float(np.interp(0.0, zp[np.argmax(zp > -1):], xp[np.argmax(zp > -1):]))
    soil = grid.below_profile(xp, zp)
    land = X >= x_shore
    ice_rich = soil & land & (Z >= Z_ICE_RICH)
    model.phi_s[soil] = PHI_S_SEDIMENT
    model.phi_s[ice_rich] = PHI_S_ICE_RICH
    model.phi_h2o[soil] = 1.0 - model.phi_s[soil]
    model.set_salinity(settings.S_sea, soil & ~land)       # saline pore water offshore, fresh on land
    water = ~soil & (Z < 0.0)
    model.phi_h2o[water] = 1.0
    model.set_salinity(settings.S_sea, water)

    T0 = model.steady_state(T_air=T_LAND_MEAN, T_water=T_SEABED_MEAN)
    Tf_sea = float(mat.freezing_point(settings.S_sea))
    T0 = np.where(water, Tf_sea + 0.05, T0)
    model.set_temperature(T0)
    return model, xp, zp, x_shore


def column_index(grid, x):
    return int(np.argmin(np.abs(grid.xc - x)))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, xp, zp, x_shore = build_model()
    g = model.grid
    x_crest = float(xp[np.argmax(zp >= zp.max() - 0.01)])
    forcing = synthetic_arctic_forcing()
    print(f"grid {g.nx} x {g.nz} = {g.nx * g.nz} cells, {int(model.state().active.sum())} active; "
          f"shoreline x = {x_shore:.0f} m, crest x = {x_crest:.0f} m")

    i_tundra = column_index(g, x_crest + PROBE_TUNDRA_OFFSET)
    z_surf_tundra = float(np.interp(g.xc[i_tundra], xp, zp))
    j_probe = [int(np.argmin(np.abs(g.zc - (z_surf_tundra - d)))) for d in PROBE_DEPTHS]
    i_water = [column_index(g, x_shore - d) for d in PROBE_WATER_OFFSETS]

    series = {k: [] for k in ["t", "T_air", "eta", "T_ref_surf", "T_ref_bot", "active_layer", "iterations"]}
    series.update({f"T_tundra_{d}": [] for d in PROBE_DEPTHS})
    for d in PROBE_WATER_OFFSETS:
        series.update({f"T_water_top_{d:.0f}": [], f"T_water_bot_{d:.0f}": [], f"ice_{d:.0f}": [],
                       f"S_water_top_{d:.0f}": [], f"S_water_bot_{d:.0f}": [], f"S_ice_{d:.0f}": []})
    series["S_ref_surf"] = []

    def probe():
        p = model.state()
        F = forcing(model.time, g.zc)
        series["t"].append(model.time / 86400.0)
        series["T_air"].append(F.T_air)
        series["eta"].append(F.eta)
        series["T_ref_surf"].append(F.T_ref[np.argmin(np.abs(g.zc + 0.5))])
        series["T_ref_bot"].append(F.T_ref[np.argmin(np.abs(g.zc + 5.0))])
        series["S_ref_surf"].append(F.S_ref[np.argmin(np.abs(g.zc + 0.5))])
        col = p.soil[:, i_tundra]
        series["active_layer"].append(float(np.sum(g.dz * col * p.f_liq[:, i_tundra])))
        series["iterations"].append(model.last_step.get("iterations", 0))
        for d, j in zip(PROBE_DEPTHS, j_probe):
            series[f"T_tundra_{d}"].append(p.T[j, i_tundra])
        for d, i in zip(PROBE_WATER_OFFSETS, i_water):
            wet = np.nonzero(p.fluid[:, i])[0]
            series[f"T_water_top_{d:.0f}"].append(p.T[wet.max(), i] if len(wet) else np.nan)
            series[f"T_water_bot_{d:.0f}"].append(p.T[wet.min(), i] if len(wet) else np.nan)
            ice = p.open[:, i] & ~p.fluid[:, i]
            series[f"ice_{d:.0f}"].append(float(np.sum(g.dz * ice * (1 - p.f_liq[:, i]))))
            series[f"S_water_top_{d:.0f}"].append(p.S_bulk[wet.max(), i] if len(wet) else np.nan)
            series[f"S_water_bot_{d:.0f}"].append(p.S_bulk[wet.min(), i] if len(wet) else np.nan)
            w = g.dz * ice * model.phi_h2o[:, i]
            series[f"S_ice_{d:.0f}"].append(float(np.sum(w * p.S_bulk[:, i]) / w.sum()) if w.sum() > 0.02 else np.nan)

    rec = Recorder(model, times=[d * 86400.0 for d in SNAPSHOT_DOYS])
    rec(force=True)
    probe()
    t_wall = wallclock.time()
    t_end = N_DAYS * 86400.0
    n = 0
    while model.time < t_end - 1e-6:
        model.step(min(DT, t_end - model.time), forcing)
        rec()
        probe()
        n += 1
        if n % 240 == 0:
            print(f"  day {model.time / 86400:5.0f}  ({wallclock.time() - t_wall:5.0f} s wall, "
                  f"{model.last_step['iterations']} iterations)")
    rec(force=True)
    wall = wallclock.time() - t_wall

    ds = rec.to_dataset(t0="2021-01-01")
    for k, v in series.items():
        if k != "t":
            ds[f"probe_{k}"] = (("probe_time",), np.array(v, dtype=float))
    ds = ds.assign_coords(probe_time=np.datetime64("2021-01-01") + (np.array(series["t"]) * 86400e3).astype("timedelta64[ms]"))
    ds.to_netcdf(OUT_DIR / "barter_transect.nc")

    budget = sum(rec.series[f"energy_{k}"][-1] - rec.series[f"energy_{k}"][0] for k in model.energy)
    e_err = (rec.series["energy_domain"][-1] - rec.series["energy_domain"][0]) - budget
    print(f"{n} steps in {wall:.0f} s ({wall / n:.2f} s per step); mean iterations {np.mean(series['iterations'][1:]):.1f}")
    print(f"energy budget error {e_err:.2e} J/m ({e_err / max(abs(budget), 1.0):.1e} of the net exchange)")
    s_budget = sum(rec.series[f"salt_{k}"][-1] - rec.series[f"salt_{k}"][0] for k in model.salt_exchange)
    s_err = (rec.series["salt_domain"][-1] - rec.series["salt_domain"][0]) - s_budget
    print(f"salt budget error {s_err:.2e} psu m3/m ({s_err / rec.series['salt_domain'][0]:.1e} of the salt content)")
    print(f"max active layer at tundra probe {max(series['active_layer']):.2f} m; eroded sediment "
          f"{model.eroded_sediment:.2f} m3/m; block failures {len(model.collapse_events)}")
    print(f"max nearshore ice thickness: " + ", ".join(
        f"{d:.0f} m offshore {max(series[f'ice_{d:.0f}']):.2f} m" for d in PROBE_WATER_OFFSETS))
    plot(ds, series, x_shore, x_crest, g)


def plot(ds, series, x_shore, x_crest, g):
    pl.apply_style()
    labels = ["15 Mar", "15 Jun", "20 Jul", "16 Aug (surge)", "15 Sep", "15 Nov"]
    its = [int(np.argmin(np.abs((ds.time.values - ds.time.values[0]) / np.timedelta64(1, "D") - d)))
           for d in SNAPSHOT_DOYS]

    # snapshots around the bluff: temperature (left) and phase (right)
    fig, axes = plt.subplots(len(its), 2, figsize=(13, 2.3 * len(its)), sharex=True, sharey=True,
                             constrained_layout=True)
    for r, it in enumerate(its):
        m = pl.plot_temperature(axes[r, 0], ds, it, vlim=10.0)
        pl.plot_phase(axes[r, 1], ds, it)
        for c in range(2):
            pl.panel_label(axes[r, c], f"({chr(97 + 2 * r + c)}) {labels[r]}")
            axes[r, c].set_xlim(x_shore - 120, x_crest + 60)
            axes[r, c].set_ylim(-7, 14.5)
        axes[r, 0].set_ylabel("z (m)")
    for ax in axes[-1]:
        ax.set_xlabel("cross-shore distance (m)")
    fig.colorbar(m, ax=axes[:, 0], label="temperature (°C)", shrink=0.5, extend="both")
    pl.phase_legend(axes[0, 1], loc="lower left", classes=[1, 2, 3, 4, 5], ncol=2)
    pl.save(fig, OUT_DIR / "barter_snapshots.png")

    # the whole domain in August: stratified water, subsea permafrost, sponge
    it = its[3]
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    m = pl.plot_temperature(ax, ds, it, vlim=10.0)
    ax.axvspan(g.xe[0], g.xe[5], color="k", alpha=0.08, lw=0)
    ax.text(g.xe[0] + 5, -27, "sponge", fontsize=10)
    ax.set_xlim(g.xe[0], g.xe[-1])
    ax.set_ylim(g.ze[0], g.ze[-1])
    ax.set_xlabel("cross-shore distance (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(m, ax=ax, label="temperature (°C)", extend="both")
    pl.save(fig, OUT_DIR / "barter_full_domain.png")

    # time series
    t = np.array(series["t"])
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t, series["T_air"], color="0.6", lw=0.4, label="air (2 m)")
    ax.plot(t, series["T_ref_surf"], "C0-", lw=1.5, label="ocean reference, -0.5 m")
    ax.plot(t, series["T_ref_bot"], "C9-", lw=1.5, label="ocean reference, -5 m")
    ax.set_ylabel("forcing (°C)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(t, series["eta"], "k-", lw=0.4, alpha=0.7)
    ax2.set_ylabel("water level (m)")
    ax2.spines["right"].set_visible(True)
    pl.panel_label(ax, "(a)")

    ax = axes[1]
    for k, d in enumerate(PROBE_DEPTHS):
        ax.plot(t, series[f"T_tundra_{d}"], color=plt.cm.viridis(k / (len(PROBE_DEPTHS) - 1)), lw=1.2,
                label=f"{d:g} m")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("tundra ground T (°C)")
    ax.legend(frameon=False, ncol=5, loc="lower center", title="depth below surface")
    pl.panel_label(ax, "(b)")

    ax = axes[2]
    ax.plot(t, series["active_layer"], "C3-", lw=1.8)
    ax.set_ylabel("thaw depth, tundra (m)")
    ax.set_ylim(bottom=0)
    pl.panel_label(ax, "(c)")

    ax = axes[3]
    for k, d in enumerate(PROBE_WATER_OFFSETS):
        ax.plot(t, series[f"T_water_top_{d:.0f}"], color=f"C{k}", lw=1.2, label=f"surface, {d:.0f} m offshore")
        ax.plot(t, series[f"T_water_bot_{d:.0f}"], color=f"C{k}", lw=1.2, ls="--", label=f"bottom, {d:.0f} m offshore")
    ax.set_ylabel("water T (°C)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax3 = ax.twinx()
    for k, d in enumerate(PROBE_WATER_OFFSETS):
        ax3.fill_between(t, 0, series[f"ice_{d:.0f}"], color=f"C{k}", alpha=0.15, lw=0)
    ax3.set_ylabel("ice thickness (m, shaded)")
    ax3.spines["right"].set_visible(True)
    ax3.set_ylim(bottom=0)
    pl.panel_label(ax, "(d)")
    ax.set_xlabel("day of year")
    ax.set_xlim(0, t[-1])
    pl.save(fig, OUT_DIR / "barter_timeseries.png")

    # salinity: late winter (sea ice, brine-enriched water beneath it) and the August surge
    pick = [0, 3]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1.1]))
    for ax, k in zip(axes[:2], pick):
        m = pl.plot_salinity(ax, ds, its[k], vmin=0.0, vmax=40.0)
        ax.set_xlim(x_shore - 400, x_crest + 60)
        ax.set_ylim(-7, 14.5)
        ax.set_ylabel("z (m)")
        pl.panel_label(ax, f"({chr(97 + pick.index(k))}) {labels[k]}")
    axes[1].set_xlabel("cross-shore distance (m)")
    fig.colorbar(m, ax=axes[:2], label="bulk salinity (psu)", shrink=0.8, extend="max")
    ax = axes[2]
    ax.plot(t, series["S_ref_surf"], color="0.5", lw=1.2, label="ocean reference, -0.5 m")
    for k, d in enumerate(PROBE_WATER_OFFSETS):
        ax.plot(t, series[f"S_water_top_{d:.0f}"], color=f"C{k}", lw=1.2, label=f"water surface, {d:.0f} m offshore")
        ax.plot(t, series[f"S_water_bot_{d:.0f}"], color=f"C{k}", lw=1.2, ls="--", label=f"water bottom, {d:.0f} m offshore")
        ax.plot(t, series[f"S_ice_{d:.0f}"], color=f"C{k}", lw=2.2, ls=":", label=f"sea ice (bulk), {d:.0f} m offshore")
    ax.set(xlabel="day of year", ylabel="salinity (psu)", xlim=(0, t[-1]))
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=9)
    pl.panel_label(ax, "(c)")
    pl.save(fig, OUT_DIR / "barter_salinity.png")


if __name__ == "__main__":
    main()
