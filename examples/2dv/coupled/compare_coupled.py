"""Compare the coupled runs with the 2DV and the (fixed) v1.0 1D thermal module: Barter Island,
schematized profile, 15 June - 3 August 2015 (run_coupled_test.py, run_1d_reference.py).

Panels: (a) bed profiles at the start and the end; (b) erodible thickness handed to XBeach
(ne_layer) on 1 August, before the storm; (c) cumulative volume eroded above z = 0 from the
XBeach runs; (d) 2DV temperature around the bluff at the end.

Outputs (in examples/2dv/output/coupled/): compare_coupled.png, compare_coupled_summary.txt.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import sys
PROJ_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ_DIR))
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402

# === CONFIGURATION ===
HERE = Path(__file__).resolve().parent
RUN_2DV = HERE / "barter_2dv_test" / "results"
RUN_1D = HERE / "barter_1d_test" / "results"
OUT_DIR = PROJ_DIR / "examples" / "2dv" / "output" / "coupled"
NE_DATE = np.datetime64("2015-08-01T12:00")   # ne_layer snapshot before the storm
X_WINDOW = (1300.0, 1420.0)                   # plotting window [m]


def xbeach_erosion(result_dir):
    """Time of each XBeach call and the cumulative volume eroded above z = 0 [m3/m]."""
    t, vol = [], []
    total = 0.0
    for folder in sorted((result_dir / "xb_files").iterdir()):
        nc = folder / "xboutput.nc"
        if not nc.is_file():
            continue
        with xr.open_dataset(nc) as ds:
            ds = ds.squeeze()
            x = ds["globalx"].values.ravel()
            zb0 = ds["zb"].isel(globaltime=0).values.ravel()
            zb1 = ds["zb"].isel(globaltime=-1).values.ravel()
        dx = np.gradient(x)
        above = zb0 > 0.0
        total += float(np.sum(np.clip(zb0 - zb1, 0.0, None)[above] * dx[above]))
        t.append(int(folder.name))
        vol.append(total)
    return np.array(t), np.array(vol)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d2 = xr.load_dataset(RUN_2DV / "thermal2dv.nc")
    d1 = xr.open_dataset(RUN_1D / "results.nc")
    t1 = d1.time.values
    x1 = d1.xgr.values
    zgr1 = d1.zgr.isel(time=[0, -1]).values
    k1 = int(np.argmin(np.abs(t1 - NE_DATE)))
    ne1 = d1.thaw_depth.isel(time=k1).values
    k2 = int(np.argmin(np.abs(d2.time_xb.values - NE_DATE)))
    ne2 = d2.ne_xb.isel(time_xb=k2).values
    zb2 = d2.zb_xb.isel(time_xb=[0, -1]).values
    tx1, v1 = xbeach_erosion(RUN_1D)
    tx2, v2 = xbeach_erosion(RUN_2DV)

    lines = [f"XBeach calls: 1D {len(tx1)}, 2DV {len(tx2)}",
             f"volume eroded above z = 0 by XBeach: 1D {v1[-1] if len(v1) else 0:.2f} m3/m, "
             f"2DV {v2[-1] if len(v2) else 0:.2f} m3/m (2DV ablation adds {d2.eroded_sediment.values[-1]:.2f} m3/m of mineral)"]
    win = (x1 > X_WINDOW[0]) & (x1 < X_WINDOW[1])
    lines.append(f"ne_layer on {str(NE_DATE)[:10]} in the window: 1D mean {ne1[win].mean():.2f} m (max {ne1[win].max():.2f}), "
                 f"2DV mean {ne2[win].mean():.2f} m (max {ne2[win].max():.2f})")
    txt = "\n".join(lines)
    print(txt)
    (OUT_DIR / "compare_coupled_summary.txt").write_text(txt + "\n", encoding="utf-8")

    pl.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x1, zgr1[0], color="0.6", lw=1.0, label="start")
    ax.plot(x1, zgr1[1], color="0.2", lw=1.4, label="end, 1D")
    ax.plot(d2.x_xb, zb2[1], color="C3", lw=1.4, label="end, 2DV")
    ax.set_xlim(*X_WINDOW)
    ax.set_ylabel("bed level (m)")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.05, "(a)", transform=ax.transAxes, ha="right")

    ax = axes[0, 1]
    ax.plot(x1, ne1, color="0.2", lw=1.4, label="1D thaw depth")
    ax.plot(d2.x_xb, ne2, color="C3", lw=1.4, label="2DV erodible thickness")
    ax.set_xlim(*X_WINDOW)
    ax.set_ylim(0.0, 2.5)
    ax.set_ylabel(f"ne_layer on {str(NE_DATE)[:10]} (m)")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.92, "(b)", transform=ax.transAxes, ha="right")

    ax = axes[1, 0]
    ax.step(tx1 / 24.0, v1, where="post", color="0.2", lw=1.4, label="1D")
    ax.step(tx2 / 24.0, v2, where="post", color="C3", lw=1.4, label="2DV")
    ax.set_xlabel("days since 15 June 2015")
    ax.set_ylabel("eroded above z = 0 by XBeach (m$^3$ m$^{-1}$)")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.05, "(c)", transform=ax.transAxes, ha="right")

    ax = axes[1, 1]
    T = d2.T.isel(time=-1).values
    m = ax.pcolormesh(d2.x_edge, d2.z_edge, T, cmap="RdBu_r", vmin=-8, vmax=8)
    ax.step(d2.x, d2.zb_2dv, where="mid", color="k", lw=0.8)
    ax.set_xlim(*X_WINDOW)
    ax.set_ylim(-3.0, float(np.nanmax(d2.zb_2dv)) + 1.0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_facecolor("0.93")
    fig.colorbar(m, ax=ax, label="T, 2DV, end (°C)")
    ax.text(0.02, 0.92, "(d)", transform=ax.transAxes)
    fig.savefig(OUT_DIR / "compare_coupled.png", dpi=150)
    print(f"wrote {OUT_DIR / 'compare_coupled.png'}")


if __name__ == "__main__":
    main()
