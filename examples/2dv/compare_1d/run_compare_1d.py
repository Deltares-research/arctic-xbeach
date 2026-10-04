"""Compare the 2DV thermal model with Arctic-XBeach v1.0 on a flat, dry soil column at Barter Island.

Both models get the same one-year ERA5 forcing (2016), the same soil and the same initial state:

    soil        saturated silt, phi_s = 0.6, phi_h2o = 0.4 (c_f 1.96e6, c_u 2.88e6 J m-3 K-1,
                k_f 2.597, k_u 1.513 W m-1 K-1, latent heat rho_w L nb = 1.336e8 J m-3)
    column      10 m deep, geothermal gradient 0.034 K/m (flux k_f G = 0.088 W m-2)
    surface     v1.0's formulation: sensible heat h (T_air - T_s) with h = 16.12 U^0.8 (Man 2023,
                as coded in v1.0), plus ERA5 net SW + net LW + latent flux, prescribed hourly
    initial     v1.0's own profile from the ERA5 soil levels

So only the numerics differ: v1.0 is explicit FTCS on nodes (dt = 120 s, 200 nodes, mushy k =
mean of frozen and thawed), now by default in conservative flux form d/dz(k dT/dz) with harmonic
face k (thermal.conduction_form = flux); v1.0 legacy uses nodal k * d2T/dz2
(conduction_form = nodal, run v10_case_nodal). The 2DV model is implicit finite volume (dt = 1 h,
200 cells, conservative flux form, mushy k from the ice/water mix).

To attribute the difference, v1.0's scheme is also re-implemented here twice: exactly as coded in
the nodal form (it reproduces the nodal v1.0 output to ~0.03 K), and with only the conduction term
changed to the conservative flux form (arithmetic face k). A further line shows v1.0 with the legacy
latent-heat convention and L_water_ice = 3.34e8 (a units check).

Run run_v10_column.py (flux), run_v10_column.py v10_case_nodal (nodal) and optionally
run_v10_column_legacy_latent.py first; they write the v1.0 output.
Outputs (in output/compare_1d/): compare_1d.png, compare_1d_summary.txt, twodv_column.nc.
"""

import sys
import time as wallclock
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from arctic_xbeach.thermal2dv import Grid, Model2DV, Settings  # noqa: E402
from arctic_xbeach.thermal2dv import materials as mat  # noqa: E402
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402
from arctic_xbeach.thermal2dv.forcing import Forcing  # noqa: E402

# === CONFIGURATION ===
OUT_DIR = REPO / "examples" / "2dv" / "output" / "compare_1d"
V10_NC = OUT_DIR / "v10_column.nc"                           # v1.0, matched properties, flux-form conduction
V10_NODAL_NC = OUT_DIR / "v10_column_nodal.nc"               # v1.0, matched properties, legacy nodal conduction (optional)
V10_LEGACY_LATENT_NC = OUT_DIR / "v10_column_legacy_latent.nc"  # v1.0, legacy latent-heat convention (optional)
V10_RESULTS = REPO / "examples" / "2dv" / "compare_1d" / "v10_case" / "results" / "results.nc"  # v1.0 native output
V10_NODAL_RESULTS = REPO / "examples" / "2dv" / "compare_1d" / "v10_case_nodal" / "results" / "results.nc"
PHI_S, PHI_H2O = 0.6, 0.4
DEPTH = 10.0              # [m]
N_CELLS = 200
GEO_GRADIENT = 0.034      # [K m-1]
DT = 3600.0               # 2DV time step [s]; the forcing is piecewise constant per hour
PROBE_DEPTHS = [0.1, 0.5, 1.0, 2.0, 5.0]
DPI = 300


def thaw_bottom(T_c, depth, z_max=3.0):
    """Bottom of the thawed layer: the deepest point above 0 degC in the top z_max metres, linearly
    interpolated to the 0 degC crossing below it (0 if nothing is above 0 degC). Unlike the first
    crossing from the surface, this does not drop to zero when the surface refreezes in autumn
    while the layer below is still thawed."""
    out = np.zeros(T_c.shape[0])
    top = depth <= z_max
    for i, prof in enumerate(T_c[:, top]):
        warm = np.nonzero(prof > 0.0)[0]
        if len(warm) == 0:
            continue
        j = warm.max()
        if j + 1 >= len(prof):
            out[i] = depth[top][-1]
            continue
        out[i] = depth[j] + prof[j] / (prof[j] - prof[j + 1]) * (depth[j + 1] - depth[j])
    return out


def run_2dv(v10):
    grid = Grid(np.array([0.0, 1.0]), np.linspace(-DEPTH, 0.0, N_CELLS + 1))
    k_f = mat.soil_conductivity(PHI_S, PHI_H2O, 0.0)
    s = Settings(q_geo=k_f * GEO_GRADIENT, emissivity=0.0, erosion=False, collapse=False, salt_transport=False,
                 brine_drainage=False, tau_flush=0.0, n_sponge=0)
    m = Model2DV(grid, s)
    m.phi_s[:] = PHI_S
    m.phi_h2o[:] = PHI_H2O
    depth_c = -grid.zc
    m.set_temperature(np.interp(depth_c, v10.depth.values, v10.T_init.values)[:, None] + np.zeros(grid.shape))

    T_air = v10.T_air.values
    h_air = 16.12 * v10.wind_speed.values ** 0.8
    q_fix = (v10.Q_latent + v10.Q_lw_net + v10.Q_sw_net).values  # hourly means as v1.0 applied them
    zeros = np.zeros(grid.nz)

    def forcing(t, zc):
        i = min(int(t // 3600.0), len(T_air) - 1)
        return Forcing(T_air=T_air[i], sw_down=0.0, lw_down=0.0, u10=0.0, eta=-1e9, T_ref=zeros,
                       q_surface=q_fix[i], h_air=h_air[i])

    n = len(T_air)
    T_out = np.zeros((n, grid.nz))
    f_out = np.zeros((n, grid.nz))
    E0 = m.total_energy()
    t0 = wallclock.time()
    for i in range(n):
        m.step(DT, forcing)
        p = m.state()
        T_out[i] = p.T[:, 0]
        f_out[i] = p.f_liq[:, 0]
    wall = wallclock.time() - t0
    e_err = (m.total_energy() - E0) - m.energy["boundary"]
    # cells are stored bottom to top: flip so depth increases downward
    ds = xr.Dataset({"T": (("time", "depth"), T_out[:, ::-1]), "f_liq": (("time", "depth"), f_out[:, ::-1])},
                    coords={"time": v10.time.values, "depth": depth_c[::-1]})
    ds.attrs.update(wall_time_s=wall, energy_budget_error_J=e_err, energy_boundary_J=m.energy["boundary"])
    return ds


def v10_scheme(v10, conservative):
    """v1.0's explicit enthalpy scheme on its own nodes (dt = 120 s, ghost nodes at both ends),
    either as coded (nodal k times the discrete Laplacian) or in conservative flux form."""
    c_f, c_u, lv = mat.heat_capacities(PHI_S, PHI_H2O)
    k_f = float(v10.attrs.get("k_frozen", 2.597))
    k_u = float(v10.attrs.get("k_unfrozen", 1.513))
    n = len(v10.depth)
    dz, dt = DEPTH / (n - 1), 120.0
    w = np.full(n, dz)
    w[[0, -1]] = 0.5 * dz
    T = v10.T_init.values.copy()
    H = np.where(T < 0.0, c_f * T, lv + c_u * T)
    T_air = v10.T_air.values
    h_air = 16.12 * v10.wind_speed.values ** 0.8
    q_fix = (v10.Q_latent + v10.Q_lw_net + v10.Q_sw_net).values
    out = np.zeros((len(T_air), n))
    for i in range(len(T_air)):
        for _ in range(int(3600.0 / dt)):
            T = np.where(H < 0.0, H / c_f, np.where(H > lv, (H - lv) / c_u, 0.0))
            k = np.where(H < 0.0, k_f, np.where(H > lv, k_u, 0.5 * (k_f + k_u)))
            Q = h_air[i] * (T_air[i] - T[0]) + q_fix[i]
            if conservative:
                F = 0.5 * (k[1:] + k[:-1]) * (T[1:] - T[:-1]) / dz   # flux from node j+1 into node j
                div = np.zeros(n)
                div[:-1] += F
                div[1:] -= F
                div[0] += Q
                div[-1] += k[-1] * GEO_GRADIENT
                H = H + dt * div / w
            else:
                Te = np.concatenate(([T[1] + 2 * dz * Q / k[0]], T, [T[-2] + 2 * dz * GEO_GRADIENT]))
                H = H + k * dt / dz**2 * (Te[:-2] - 2 * Te[1:-1] + Te[2:])
        out[i] = np.where(H < 0.0, H / c_f, np.where(H > lv, (H - lv) / c_u, 0.0))
    return out


def v10_energy_error(v10, results_nc=V10_RESULTS):
    """Column energy change minus the energy that entered through the surface and the base, for v1.0
    (enthalpy per unit mass x rho_soil, trapezoidal over the nodes)."""
    if not results_nc.exists():
        return None
    r = xr.open_dataset(results_nc)
    x = int(v10.attrs.get("x_index", 1))
    E = r.ground_enthalpy_distribution.isel(xgr=x).values * r.rho.isel(xgr=x).values
    dz = DEPTH / (E.shape[1] - 1)
    w = np.full(E.shape[1], dz)
    w[[0, -1]] = 0.5 * dz
    col = E @ w
    k_f = float(v10.attrs.get("k_frozen", 2.597))
    q_in = np.cumsum(v10.Q_total.values) * 3600.0 + k_f * GEO_GRADIENT * 3600.0 * np.arange(1, len(v10.time) + 1)
    n = min(len(col), len(q_in))
    return (col[:n] - col[0]) - (q_in[:n] - q_in[0])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v10 = xr.load_dataset(V10_NC)
    print("running the 2DV column ...")
    d2 = run_2dv(v10)
    d2.to_netcdf(OUT_DIR / "twodv_column.nc")

    depth_v = v10.depth.values
    T2_on_v = np.array([np.interp(depth_v, d2.depth.values, row) for row in d2.T.values])
    thaw_v10 = thaw_bottom(v10.T.values, depth_v)
    thaw_2dv = thaw_bottom(T2_on_v, depth_v)
    thaw_legacy_latent = None
    if V10_LEGACY_LATENT_NC.exists():
        vb = xr.load_dataset(V10_LEGACY_LATENT_NC)
        thaw_legacy_latent = thaw_bottom(vb.T.values, vb.depth.values)
    e10 = v10_energy_error(v10)
    vn = xr.load_dataset(V10_NODAL_NC) if V10_NODAL_NC.exists() else None
    thaw_nodal = thaw_bottom(vn.T.values, vn.depth.values) if vn is not None else None
    e10_nodal = v10_energy_error(vn, V10_NODAL_RESULTS) if vn is not None else None
    print("re-implementing the v1.0 scheme (as coded, and conservative) ...")
    T_rep = v10_scheme(v10, conservative=False)
    T_cons = v10_scheme(v10, conservative=True)
    thaw_cons = thaw_bottom(T_cons, depth_v)

    t = v10.time.values
    doy = (t - np.datetime64("2016-01-01")) / np.timedelta64(1, "D")
    dT = T2_on_v - v10.T.values
    lines = [
        f"2DV: {d2.attrs['wall_time_s']:.0f} s wall for one year ({len(t)} steps of 1 h); energy budget error "
        f"{d2.attrs['energy_budget_error_J']:.2e} J m-2",
        f"v1.0: {v10.attrs.get('wall_time_thermal_update_s', np.nan):.0f} s in the thermal update "
        f"({v10.attrs.get('wall_time_main_s', np.nan):.0f} s in total, dt = 120 s)",
        f"max thaw depth (bottom of the thawed layer): v1.0 {thaw_v10.max():.2f} m, 2DV {thaw_2dv.max():.2f} m"
        + (f", v1.0 nodal (legacy) {thaw_nodal.max():.2f} m" if thaw_nodal is not None else "")
        + (f", v1.0 legacy latent heat (L = 3.34e8) {thaw_legacy_latent.max():.2f} m" if thaw_legacy_latent is not None else ""),
    ]
    if vn is not None:
        lines.append(f"v1.0 nodal scheme re-implemented as coded: max |T - v1.0 nodal output| = "
                     f"{np.abs(T_rep - vn.T.values).max():.3f} K")
    lines.append(f"v1.0 flux form vs its re-implementation (arithmetic face k): max |T diff| = "
                 f"{np.abs(T_cons - v10.T.values).max():.3f} K")
    lines.append(f"v1.0 scheme in conservative flux form: max thaw depth {thaw_cons.max():.2f} m; "
                 f"max |T - 2DV| = {np.abs(T_cons - T2_on_v).max():.2f} K, mean {np.mean(T_cons - T2_on_v):+.3f} K")
    for d in PROBE_DEPTHS:
        j = int(np.argmin(np.abs(depth_v - d)))
        lines.append(f"T at {d:g} m: mean difference 2DV - v1.0 {dT[:, j].mean():+.2f} K, "
                     f"RMS {np.sqrt(np.mean(dT[:, j] ** 2)):.2f} K, max |diff| {np.abs(dT[:, j]).max():.2f} K")
    if e10 is not None:
        lines.append(f"v1.0 column energy change minus boundary input after one year: {e10[-1] / 1e6:+.2e} MJ m-2 "
                     f"(max |.| over the year {np.abs(e10).max() / 1e6:.2e} MJ m-2)"
                     + (f"; nodal {e10_nodal[-1] / 1e6:+.1f} MJ m-2" if e10_nodal is not None else "")
                     + f" (2DV: {d2.attrs['energy_budget_error_J'] / 1e6:+.1e} MJ m-2)")

    # ------------------------------------------------------------------ figure
    pl.apply_style()
    fig = plt.figure(figsize=(12, 13), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1, 1.1, 0.9])
    ax = fig.add_subplot(gs[0])
    for k, d in enumerate(PROBE_DEPTHS):
        j = int(np.argmin(np.abs(depth_v - d)))
        col = plt.cm.viridis(k / (len(PROBE_DEPTHS) - 1))
        ax.plot(doy, v10.T.values[:, j], color=col, lw=1.6, label=f"{d:g} m")
        ax.plot(doy, T2_on_v[:, j], color=col, lw=1.0, ls="--")
    ax.plot([], [], color="k", lw=1.6, label="v1.0 flux form (solid)")
    ax.plot([], [], color="k", lw=1.0, ls="--", label="2DV (dashed)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(ylabel="ground T (°C)", xlim=(0, doy[-1]))
    ax.legend(frameon=False, ncol=7, loc="lower center", fontsize=9)
    pl.panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[1], sharex=ax)
    ax.plot(doy, thaw_v10, "C0-", lw=2, label="v1.0 (flux form)")
    ax.plot(doy, thaw_2dv, "C3--", lw=2, label="2DV")
    ax.plot(doy, thaw_cons, "k:", lw=1.5, label="v1.0 scheme in conservative flux form (re-implemented)")
    if thaw_nodal is not None:
        ax.plot(doy, thaw_nodal, "C1-.", lw=1.5, label="v1.0 nodal form (legacy)")
    if thaw_legacy_latent is not None:
        ax.plot(doy, thaw_legacy_latent, "C2:", lw=2, label="v1.0 legacy latent heat, L = 3.34e8")
    ax.set(ylabel="thaw depth (m)", ylim=(0, None))
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower left")
    pl.panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[2], sharex=ax)
    vmax = max(0.5, float(np.ceil(np.nanpercentile(np.abs(dT[:, depth_v <= 3.0]), 99) * 2) / 2))
    m = ax.pcolormesh(doy, depth_v, dT.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax.contour(doy, depth_v, v10.T.values.T, levels=[0.0], colors="C0", linewidths=1.0)
    ax.contour(doy, depth_v, T2_on_v.T, levels=[0.0], colors="k", linewidths=1.0, linestyles="--")
    ax.set(ylabel="depth (m)", ylim=(3, 0))
    fig.colorbar(m, ax=ax, label="T$_{2DV}$ − T$_{v1.0}$ (K)", extend="both", pad=0.01)
    pl.panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[3], sharex=ax)
    if e10 is not None:
        ax.plot(doy[:len(e10)], e10 / 1e6, "C0-", lw=2, label="v1.0 (flux form)")
    if e10_nodal is not None:
        ax.plot(doy[:len(e10_nodal)], e10_nodal / 1e6, "C1-.", lw=1.5, label="v1.0 nodal (legacy)")
    ax.plot(doy, np.zeros_like(doy) + d2.attrs["energy_budget_error_J"] / 1e6, "C3--", lw=2, label="2DV")
    ax.set(xlabel="day of year 2016", ylabel="energy error (MJ m$^{-2}$)")
    ax.legend(frameon=False, loc="center left")
    pl.panel_label(ax, "(d)")
    pl.save(fig, OUT_DIR / "compare_1d.png", dpi=DPI)

    summary = "\n".join(lines)
    (OUT_DIR / "compare_1d_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
