"""Demo 4: radiation geometry on the Barter Island bluff, with and without.

The Barter transect of Demo 3 (same grid, soil and initial state) is run for one year twice
with the same forcing: ERA5 2016 at Barter (direct and diffuse shortwave, longwave, air
temperature, wind, albedo, SST; still water at 0 m) or the synthetic year with the real sun path:

  legacy    upward faces get the full shortwave, bluff faces a fixed 0.5 x SW, every face
            sees the whole sky
  geometry  sun position from UTC and longitude, direct beam with cast shadows, diffuse and
            longwave weighted by each face's sky-view factor, terrain-reflected shortwave
            (radiation.py); the bluff faces north (bearing_seaward 1.86 deg, from the v1.0
            grid_orientation of 88.14 deg counter-clockwise from east)

Outputs (in output/radiation_demo/): radiation_demo_<forcing>.png, radiation_demo_<forcing>_summary.txt,
radiation_<forcing>_<case>.nc.
"""

import sys
import time as wallclock
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arctic_xbeach.thermal2dv import synthetic_arctic_forcing  # noqa: E402
from arctic_xbeach.thermal2dv.forcing import era5_forcing  # noqa: E402
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402
from arctic_xbeach.thermal2dv import radiation as rad  # noqa: E402
from run_barter_transect import build_model  # noqa: E402

# === CONFIGURATION ===
OUT_DIR = REPO / "examples" / "2dv" / "output" / "radiation_demo"  # results are suffixed by FORCING
LAT, LON = 70.13, -143.66           # Barter Island
FORCING = "era5"                    # "era5" (ERA5 2016 at Barter, era5/data/era5_barter_point.nc) or "synthetic"
ERA5_POINT = REPO / "examples" / "2dv" / "era5" / "data" / "era5_barter_point.nc"
START = "2016-01-01T00:00" if FORCING == "era5" else "2021-01-01T00:00"  # UTC at t = 0
BEARING_SEAWARD = 90.0 - 88.14      # compass bearing the bluff faces [deg]
DT = 3 * 3600.0                     # time step [s]
N_DAYS = 365
SUMMER = (152, 244)                 # 1 Jun - 31 Aug: window for the mean absorbed shortwave [day of year]
DIFF_DOY = 243                      # day of the temperature-difference snapshot (31 Aug)
WINDOW_X = (-45.0, 20.0)            # plotting window relative to the bluff crest [m]
WINDOW_Z = (-3.0, 14.0)             # [m]
Z_FACE_PROBE = 7.0                  # height of the horizontal thaw probe into the bluff face [m]
PLOT_ONLY = False                   # redraw from the saved NetCDF files


def run_case(case):
    model, xp, zp, x_shore = build_model()
    g = model.grid
    if case == "geometry":
        s = model.s
        s.latitude, s.longitude, s.start_time_utc, s.bearing_seaward = LAT, LON, START, BEARING_SEAWARD
    if FORCING == "era5":
        forcing = era5_forcing(ERA5_POINT, START, eta=0.0, S_sea=model.s.S_sea)
    else:
        forcing = synthetic_arctic_forcing(solar_site=(LAT, LON, START))
    x_crest = float(xp[np.argmax(zp >= zp.max() - 0.01)])
    iw = np.nonzero((g.xc > x_crest + WINDOW_X[0]) & (g.xc < x_crest + WINDOW_X[1]))[0]
    jw = np.nonzero((g.zc > WINDOW_Z[0]) & (g.zc < WINDOW_Z[1]))[0]
    j_probe = int(np.argmin(np.abs(g.zc - Z_FACE_PROBE)))

    p0 = model.state()
    z_top = np.array([g.ze[np.nonzero(p0.soil[:, i])[0].max() + 1] if p0.soil[:, i].any() else g.ze[0]
                      for i in range(g.nx)])  # initial soil surface per column
    sw_sum = {k: np.zeros(g.shape) for k in (rad.UP, rad.SEA)}
    days, T_daily, face_thaw, eroded, x_face = [], [], [], [], []
    t_wall = wallclock.time()
    t_end = N_DAYS * 86400.0
    while model.time < t_end - 1e-6:
        dt = min(DT, t_end - model.time)
        model.step(dt, forcing)
        doy = model.time / 86400.0
        if SUMMER[0] <= doy - dt / 86400.0 < SUMMER[1]:
            for k in sw_sum:
                sw_sum[k] += model.last_step["sw_faces"][k] * dt
        if abs(doy - round(doy)) < 1e-9:
            p = model.state()
            days.append(doy)
            T_daily.append(p.T[np.ix_(jw, iw)])
            solid = np.nonzero(p.soil[j_probe, :] & (g.xc > x_shore))[0]
            i0 = solid.min() if len(solid) else g.nx - 1
            thawed = p.soil[j_probe, i0:] & (p.f_liq[j_probe, i0:] > 0.5)
            n_th = int(np.argmin(thawed)) if not thawed.all() else len(thawed)
            face_thaw.append(float(np.sum(g.dx[i0:i0 + n_th])))
            x_face.append(float(g.xe[i0]))
            eroded.append(model.eroded_sediment)
    wall = wallclock.time() - t_wall
    n_sec = (SUMMER[1] - SUMMER[0]) * 86400.0
    p = model.state()
    ds = xr.Dataset(
        {"T": (("day", "z", "x"), np.array(T_daily)),
         "face_thaw": (("day",), np.array(face_thaw)), "x_face": (("day",), np.array(x_face)),
         "eroded": (("day",), np.array(eroded)),
         "sw_up": (("zg", "xg"), sw_sum[rad.UP] / n_sec), "sw_sea": (("zg", "xg"), sw_sum[rad.SEA] / n_sec),
         "z_top": (("xg",), z_top)},
        coords={"day": np.array(days), "z": g.zc[jw], "x": g.xc[iw] - x_crest, "zg": g.zc, "xg": g.xc - x_crest,
                "ze": g.ze, "xe": g.xe - x_crest})
    ds.attrs = dict(case=case, wall_s=wall, x_crest=x_crest, failures=len(model.collapse_events))
    ds.to_netcdf(OUT_DIR / f"radiation_{FORCING}_{case}.nc")
    return case


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = ["legacy", "geometry"]
    if not PLOT_ONLY:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for case in pool.map(run_case, cases):
                print(f"{case} done")
    ds = {c: xr.load_dataset(OUT_DIR / f"radiation_{FORCING}_{c}.nc") for c in cases}
    summarise(ds)
    plot(ds)


def surface_profile(d):
    """Jun-Aug mean absorbed shortwave per unit area of the true (sloping) surface, per column:
    the tread (up face) plus the riser (seaward face) of each step of the staircase, divided
    by the length of the surface segment they stand for. Also returns the local slope."""
    dx = np.diff(d.xe.values)
    rise = np.maximum(np.diff(d.z_top.values, prepend=d.z_top.values[0]), 0.0)
    absorbed = d.sw_up.values.sum(axis=0) + d.sw_sea.values.sum(axis=0)  # W per m alongshore
    return absorbed / np.hypot(dx, rise), np.degrees(np.arctan2(rise, dx))


def summarise(ds):
    lines = []
    for c, d in ds.items():
        q, slope = surface_profile(d)
        xg = d.xg.values
        bluff = (slope > 30.0) & (xg > -20.0) & (xg < 0.5)
        beach = (xg > -30.0) & (xg < -15.0)
        lines.append(f"{c:9s} Jun-Aug mean absorbed SW: bluff slope {np.mean(q[bluff]):6.1f} W m-2 (mean slope "
                     f"{np.mean(slope[bluff]):.0f} deg), beach {np.mean(q[beach]):6.1f}, tundra {np.mean(q[xg > 10.0]):6.1f}")
        lines.append(f"{'':9s} max horizontal thaw into the bluff at z = {Z_FACE_PROBE} m: {d.face_thaw.max().item():.2f} m; "
                     f"eroded {d.eroded.values[-1]:.2f} m3/m; block failures {d.attrs['failures']}; wall {d.attrs['wall_s']:.0f} s")
    txt = "\n".join(lines)
    print(txt)
    (OUT_DIR / f"radiation_demo_{FORCING}_summary.txt").write_text(txt + "\n", encoding="utf-8")


def plot(ds):
    pl.apply_style()
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    colors = {"legacy": "0.3", "geometry": "C3"}

    # (a) absorbed shortwave along the surface
    ax = fig.add_subplot(gs[0, 0])
    for c, d in ds.items():
        q, _ = surface_profile(d)
        ax.step(d.xg.values, q, where="mid", color=colors[c], lw=1.4, label=c)
    ax.set_xlim(-45.0, 30.0)
    ax.set_xlabel("x from the bluff crest (m)")
    ax.set_ylabel("absorbed SW, Jun-Aug mean (W m$^{-2}$ of surface)")
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, va="top")
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.legend(loc="lower right")

    # (b) the profile, for orientation
    ax = fig.add_subplot(gs[0, 1])
    d = ds["geometry"]
    ax.fill_between(d.xg.values, d.z_top.values, -3.0, step="mid", color="0.75", lw=0)
    ax.step(d.xg.values, d.z_top.values, where="mid", color="k", lw=0.8)
    ax.axhline(0.0, color="C0", lw=0.8)
    ax.set_xlim(-45.0, 30.0)
    ax.set_ylim(-3.0, 15.0)
    ax.set_xlabel("x from the bluff crest (m)")
    ax.set_ylabel("z (m)")
    ax.text(0.02, 0.95, f"(b) initial profile; the bluff faces {BEARING_SEAWARD:.0f} deg (north)",
            transform=ax.transAxes, va="top")

    # (c) horizontal thaw into the bluff and eroded volume
    ax = fig.add_subplot(gs[1, 0])
    ax2 = ax.twinx()
    for c, d in ds.items():
        ax.plot(d.day, d.face_thaw, color=colors[c], lw=1.4, label=f"{c}: thaw into face at z = {Z_FACE_PROBE:.0f} m")
        ax2.plot(d.day, d.eroded, color=colors[c], lw=1.0, ls="--", label=f"{c}: eroded")
    ax.set_xlim(120, 320)
    ax.set_xlabel("day of year")
    ax.set_ylabel("thaw depth into the face (m)")
    ax2.set_ylabel("eroded sediment (m$^3$ m$^{-1}$), dashed")
    ax.text(0.02, 0.95, "(c)", transform=ax.transAxes, va="top")
    ax.legend(loc="center right", fontsize=9)

    # (d) temperature difference geometry - legacy
    ax = fig.add_subplot(gs[1, 1])
    dg, dl = ds["geometry"], ds["legacy"]
    k = int(np.argmin(np.abs(dg.day.values - DIFF_DOY)))
    dT = dg.T.values[k] - dl.T.values[k]
    lim = max(np.nanpercentile(np.abs(dT), 99), 0.1)
    xe = dg.xe.values
    ze = dg.ze.values
    ix = np.searchsorted(xe, dg.x.values) - 1
    iz = np.searchsorted(ze, dg.z.values) - 1
    m = ax.pcolormesh(xe[ix[0]:ix[-1] + 2], ze[iz[0]:iz[-1] + 2], dT, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(m, ax=ax, label="T geometry - legacy (K)")
    ax.set_xlabel("x from the bluff crest (m)")
    ax.set_ylabel("z (m)")
    ax.text(0.02, 0.95, f"(d) day {DIFF_DOY} (31 Aug)", transform=ax.transAxes, va="top")
    ax.set_facecolor("0.93")
    fig.savefig(OUT_DIR / f"radiation_demo_{FORCING}.png", dpi=150)
    print(f"wrote {OUT_DIR / f'radiation_demo_{FORCING}.png'}")


if __name__ == "__main__":
    main()
