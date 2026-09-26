"""Analytical and conceptual verification suite for the 2DV thermal model, as one figure.

Each panel runs a small case and sets the model against a closed-form solution or a
conceptual expectation:

    (a) conduction          dry soil, surface step: erfc profile
    (b) Stefan thaw         saturated silt thawing from the surface: two-phase Neumann front
    (c) ice block           ice in warm, mixed water: faces retreat at h dT / Lv
    (d) cold ice-rich face  -8 degC permafrost under 5 degC water: quasi-steady retreat rate
    (e) sea-ice growth      fresh water: one-phase Neumann; sea water: same with dT = Tf(S) - Ts
    (f) salt diffusion      salinity step in open water and in pore water: erf profile
    (g) saline freezing     partly frozen saline sediment: liquid fraction on the liquidus
    (h) brine rejection     sea ice keeps brine on the liquidus, drains, and salts the water

The same checks (with tolerances) are in tests/test_thermal2dv.py.
Outputs (in output/analytical/): analytical_suite.png and a text summary.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from arctic_xbeach.thermal2dv import Grid, Model2DV, Settings, constant_forcing, run  # noqa: E402
from arctic_xbeach.thermal2dv import materials as mat  # noqa: E402
from arctic_xbeach.thermal2dv import plotting as pl  # noqa: E402

# === CONFIGURATION ===
OUT_DIR = REPO / "examples" / "2dv" / "output" / "analytical"
T_SURFACE_ICE = -10.0      # surface temperature for the sea-ice cases [degC]
S_SEA = 30.0               # sea salinity for the saline cases [psu]
S_PORE = [5.0, 15.0, 35.0] # pore-water salinities for the saline-freezing panel [psu]
DPI = 300


# ------------------------------------------------------------------ helpers
def column(depth, dz, phi_s, phi_h2o, **settings):
    """Single column with its top face exposed to the atmosphere; no geothermal flux."""
    grid = Grid(np.array([0.0, 1.0]), np.linspace(-depth, 0.0, int(round(depth / dz)) + 1))
    kw = dict(q_geo=0.0, emissivity=0.0, erosion=False, collapse=False, tau_flush=0.0, n_sponge=0)
    kw.update(settings)
    m = Model2DV(grid, Settings(**kw))
    m.phi_s[:] = phi_s
    m.phi_h2o[:] = phi_h2o
    return m


def dirichlet(T_s, eta=-1e9, T_water=0.0):
    # a huge wind speed makes the bulk exchange coefficient ~1e6 W m-2 K-1: a fixed surface temperature
    return constant_forcing(T_air=T_s, u10=1e6, lw_down=0.0, eta=eta, T_water=T_water)


def march(m, f, times, dt, probe):
    """Run to each time in `times` and collect probe(m)."""
    out = []
    for t in times:
        if t > m.time:
            run(m, f, t, dt=dt)
        out.append(probe(m))
    return np.array(out)


def neumann_two_phase(Ts, T0, Tf, k_u, k_f, c_u, c_f, lv):
    """Thaw front X = 2 lam sqrt(alpha_u t) for a surface held at Ts > Tf over frozen ground at T0."""
    a_u, a_f = k_u / c_u, k_f / c_f
    nu = np.sqrt(a_u / a_f)

    def f(lam):
        lhs = k_u * (Ts - Tf) * np.exp(-lam**2) / (erf(lam) * np.sqrt(np.pi * a_u))
        lhs -= k_f * (Tf - T0) * np.exp(-(nu * lam) ** 2) / (erfc(nu * lam) * np.sqrt(np.pi * a_f))
        return lhs - lv * lam * np.sqrt(a_u)

    return 2 * brentq(f, 1e-6, 5.0) * np.sqrt(a_u)


def neumann_one_phase(dT, k, c, lv):
    """Ice growing into water at its freezing point under a surface dT colder: h = 2 lam sqrt(alpha t)."""
    ste = c * dT / lv
    lam = brentq(lambda x: x * np.exp(x * x) * erf(x) - ste / np.sqrt(np.pi), 1e-6, 3.0)
    return 2 * lam * np.sqrt(k / c)


def frozen_thickness(m):
    return float(np.sum(m.grid.dz * (1.0 - m.state().f_liq[:, 0])))


# ------------------------------------------------------------------ cases
def case_conduction():
    m = column(10.0, 0.02, 1.0, 0.0)
    m.set_temperature(-5.0)
    t = 10 * 86400.0
    run(m, dirichlet(5.0), t, dt=3600.0)
    d = -m.grid.zc
    exact = -5.0 + 10.0 * erfc(d / (2 * np.sqrt(mat.K_MINERAL / mat.C_MINERAL * t)))
    return d, m.state().T[:, 0], exact


def case_stefan():
    phi_s, phi_h2o, T0, Ts = 0.6, 0.4, -5.0, 5.0
    m = column(6.0, 0.02, phi_s, phi_h2o)
    m.set_temperature(T0)
    t = np.linspace(0, 30, 16)[1:] * 86400.0
    X = march(m, dirichlet(Ts), t, 3600.0, lambda m: float(np.sum(m.state().f_liq[:, 0] * m.grid.dz)))
    c_f, c_u, lv = mat.heat_capacities(phi_s, phi_h2o)
    k_f = mat.soil_conductivity(phi_s, phi_h2o, 0.0)
    k_u = mat.soil_conductivity(phi_s, phi_h2o, 1.0)
    coef = neumann_two_phase(Ts, T0, 0.0, k_u, k_f, c_u, c_f, lv)
    return t, X, coef


def case_ice_block(h=500.0, T_w=5.0, dx=0.05):
    grid = Grid(np.arange(0.0, 4.0 + dx / 2, dx), np.arange(0.0, 2.0 + dx / 2, dx))
    s = Settings(h_water=h, q_geo=0.0, tau_flush=600.0, n_sponge=0, erosion=False, collapse=False,
                 Kx_surf=1.0, Kz_surf=1.0, S_sea=0.0)
    m = Model2DV(grid, s)
    m.phi_h2o[:] = 1.0
    X, Z = grid.meshgrid()
    m.set_temperature(np.where((np.abs(X - 2.0) < 0.5) & (Z < 1.0), 0.0, T_w), frozen_at_tf=True)
    f = constant_forcing(T_air=T_w, u10=0.0, lw_down=None, eta=10.0, T_water=T_w)
    t = np.arange(1, 31) * 3600.0
    A = march(m, f, t, 600.0, lambda m: float(np.sum(m.grid.volume * (1.0 - m.state().f_liq))))
    v = h * T_w / mat.LV_WATER
    return t, A, v


def case_cold_face(T_w=5.0, T0=-8.0, phi_s=0.35, dz=0.1):
    grid = Grid(np.array([0.0, 1.0]), np.arange(-4.0, 1.0 + dz / 2, dz))
    s = Settings(q_geo=0.0, emissivity=0.0, tau_flush=600.0, n_sponge=0)
    m = Model2DV(grid, s)
    soil = grid.zc[:, None] < 0.0
    m.phi_s[soil] = phi_s
    m.phi_h2o[:] = np.where(soil, 1.0 - phi_s, 1.0)
    m.set_salinity(s.S_sea, ~soil)
    m.set_temperature(np.where(soil, T0, T_w))
    f = constant_forcing(T_air=T_w, u10=0.0, lw_down=None, eta=1.0, T_water=T_w)

    def front(m):
        p = m.state()
        return float(np.sum(grid.dz * p.soil[:, 0] * (1.0 - p.f_liq[:, 0])))

    t = np.arange(1, 13) * 6 * 3600.0
    Z = march(m, f, t, 1800.0, front)
    cf, _, lv = mat.heat_capacities(phi_s, 1.0 - phi_s)
    v_ref = s.h_water * T_w / (lv + cf * (0.0 - T0))
    return t, 4.0 - Z, v_ref


def case_ice_growth():
    t = np.linspace(0, 10, 21)[1:] * 86400.0
    out = {}
    for S in (0.0, S_SEA):
        m = column(1.5, 0.01, 0.0, 1.0)
        m.set_salinity(S)
        Tf = float(mat.freezing_point(S))
        m.set_temperature(Tf, frozen_at_tf=False)
        out[S] = march(m, dirichlet(T_SURFACE_ICE, eta=10.0, T_water=Tf), t, 1800.0, frozen_thickness)
    coef = {S: neumann_one_phase(float(mat.freezing_point(S)) - T_SURFACE_ICE, mat.K_ICE, mat.C_ICE, mat.LV_WATER)
            for S in out}
    stefan = np.sqrt(2 * mat.K_ICE * (0.0 - T_SURFACE_ICE) / mat.LV_WATER)
    return t, out, coef, stefan


def case_salt_diffusion():
    res = {}
    # open water: turbulent diffusivity
    K = 1e-5
    m = column(10.0, 0.05, 0.0, 1.0, Kz_off=K, d_surf=0.0)
    m.set_salinity(np.where(m.grid.zc[:, None] > -5.0, 30.0, 20.0) + np.zeros(m.grid.shape))
    m.set_temperature(5.0)
    t = 2 * 86400.0
    run(m, constant_forcing(T_air=5.0, lw_down=None, eta=10.0, T_water=5.0), t, dt=3600.0)
    res["open water, K = 1e-5 m$^2$ s$^{-1}$, 2 days"] = (
        (m.grid.zc + 5.0) / (2 * np.sqrt((K + mat.D_SALT) * t)), (m.state().S_bulk[:, 0] - 25.0) / 5.0)
    # pore water of thawed sediment: molecular x tortuosity
    m = column(1.0, 0.01, 0.6, 0.4)
    m.set_salinity(np.where(m.grid.zc[:, None] > -0.5, 30.0, 0.0) + np.zeros(m.grid.shape))
    m.set_temperature(5.0)
    t = 365 * 86400.0
    run(m, constant_forcing(T_air=5.0, lw_down=None, eta=-1e9, T_water=5.0), t, dt=86400.0)
    D = mat.D_SALT * m.s.tortuosity
    res["pore water, D = 5e-10 m$^2$ s$^{-1}$, 1 year"] = (
        (m.grid.zc + 0.5) / (2 * np.sqrt(D * t)), (m.state().S_bulk[:, 0] - 15.0) / 15.0)
    return res


def case_saline_freezing():
    """Saturated silt with saline pore water, cooled from the top for 20 days."""
    out = {}
    for S in S_PORE:
        m = column(3.0, 0.02, 0.6, 0.4, salt_transport=False)
        m.set_salinity(S)
        m.set_temperature(float(mat.freezing_point(S)) + 0.5)
        run(m, dirichlet(-10.0), 20 * 86400.0, dt=3600.0)
        p = m.state()
        out[S] = (p.T[:, 0], p.f_liq[:, 0])
    return out


def case_brine():
    m = column(3.0, 0.02, 0.0, 1.0, tau_brine=5 * 86400.0, K_under_ice=1e-3)
    m.set_salinity(S_SEA)
    Tf = float(mat.freezing_point(S_SEA))
    m.set_temperature(Tf, frozen_at_tf=False)
    S0, E0 = m.total_salt(), m.total_energy()
    run(m, dirichlet(T_SURFACE_ICE, eta=10.0, T_water=Tf), 10 * 86400.0, dt=1800.0)
    p = m.state()
    err_salt = (m.total_salt() - S0) / S0
    err_energy = ((m.total_energy() - E0) - m.energy["boundary"]) / abs(m.energy["boundary"])
    return -m.grid.zc, p, m.s.S_ice_min, err_salt, err_energy


# ------------------------------------------------------------------ figure
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.apply_style()
    fig, axs = plt.subplots(4, 2, figsize=(12, 17), constrained_layout=True)
    ax = axs.ravel()
    lines = []
    model_kw = dict(ls="none", marker="o", ms=4, mfc="none", color="C0")
    exact_kw = dict(color="k", lw=1.5)

    print("(a) conduction")
    d, T, Tx = case_conduction()
    ax[0].plot(Tx, d, **exact_kw, label="erfc solution")
    ax[0].plot(T[::10], d[::10], **model_kw, label="model")
    ax[0].set(xlabel="Temperature (°C)", ylabel="Depth (m)", ylim=(3, 0), xlim=(-5.5, 5.5))
    lines.append(f"(a) conduction: max |T - erfc| = {np.max(np.abs(T - Tx)):.3f} K")

    print("(b) Stefan thaw")
    t, X, coef = case_stefan()
    tt = np.linspace(0, t[-1], 200)
    ax[1].plot(tt / 86400, coef * np.sqrt(tt), **exact_kw, label="Neumann (two-phase)")
    ax[1].plot(t / 86400, X, **model_kw, label="model")
    ax[1].set(xlabel="Time (days)", ylabel="Thaw depth (m)", xlim=(0, 30), ylim=(0, None))
    lines.append(f"(b) Stefan thaw: front error at 30 d = {(X[-1] / (coef * np.sqrt(t[-1])) - 1) * 100:+.1f}%")

    print("(c) ice block")
    t, A, v = case_ice_block()
    tt = np.linspace(0, t[-1], 200)
    A_x = np.clip(1.0 - 2 * v * tt, 0, None) * np.clip(1.0 - v * tt, 0, None)
    ax[2].plot(tt / 3600, A_x, **exact_kw, label=r"$(W-2vt)(H-vt)$, $v = h\Delta T/L_v$")
    ax[2].plot(t / 3600, A, **model_kw, label="model")
    ax[2].set(xlabel="Time (hours)", ylabel="Ice area (m$^2$)", xlim=(0, 30), ylim=(0, 1.05))
    i12 = np.argmin(np.abs(t - 12 * 3600))
    A12 = (1 - 2 * v * t[i12]) * (1 - v * t[i12])
    lines.append(f"(c) ice block: melted area at 12 h, model/analytic = {(1 - A[i12]) / (1 - A12):.3f}")

    print("(d) cold ice-rich face")
    t, R, v_ref = case_cold_face()
    rate = np.polyfit(t[2:], R[2:], 1)[0]
    ax[3].plot(t / 86400, R[2] + v_ref * (t - t[2]), **exact_kw, label=r"$h\Delta T / (L_v + C_f\,\Delta T_0)$")
    ax[3].plot(t / 86400, R, **model_kw, label="model")
    ax[3].set(xlabel="Time (days)", ylabel="Thaw-erosion depth (m)", xlim=(0, 3), ylim=(0, None))
    lines.append(f"(d) cold face: {rate * 86400:.3f} m/day against {v_ref * 86400:.3f} idealised "
                 f"({(rate / v_ref - 1) * 100:+.1f}%; the meltwater cools the water at the face)")

    print("(e) sea-ice growth")
    t, h, coef, stefan = case_ice_growth()
    tt = np.linspace(0, t[-1], 200)
    for S, col in zip(h, ("C0", "C3")):
        lab = "fresh" if S == 0 else f"S = {S:.0f}"
        ax[4].plot(tt / 86400, coef[S] * np.sqrt(tt), color=col, lw=1.5, label=f"Neumann, {lab}")
        ax[4].plot(t / 86400, h[S], ls="none", marker="o", ms=4, mfc="none", color=col, label=f"model, {lab}")
    ax[4].plot(tt / 86400, stefan * np.sqrt(tt), color="k", ls=":", lw=1.2, label="Stefan's law, fresh")
    ax[4].set(xlabel="Time (days)", ylabel="Ice thickness (m)", xlim=(0, 10), ylim=(0, None))
    lines.append(f"(e) fresh ice at 10 d: {(h[0.0][-1] / (coef[0.0] * np.sqrt(t[-1])) - 1) * 100:+.1f}% vs Neumann; "
                 f"sea ice {(h[S_SEA][-1] / (coef[S_SEA] * np.sqrt(t[-1])) - 1) * 100:+.1f}% vs Neumann with dT = Tf(S) - Ts")

    print("(f) salt diffusion")
    res = case_salt_diffusion()
    eta = np.linspace(-2.5, 2.5, 200)
    ax[5].plot(erf(eta), eta, **exact_kw, label="erf solution")
    for (lab, (x, y)), mk in zip(res.items(), ("o", "s")):
        sel = np.abs(x) < 2.5
        ax[5].plot(y[sel][::2], x[sel][::2], ls="none", marker=mk, ms=4, mfc="none", label=lab)
        lines.append(f"(f) salt diffusion, {lab.split(',')[0]}: max error {np.max(np.abs(y - erf(x))) * 100:.2f}% of the step")
    ax[5].set(xlabel=r"$(S - \bar S) / (\Delta S / 2)$", ylabel=r"$(z - z_0) / 2\sqrt{Dt}$", xlim=(-1.1, 1.1), ylim=(-2.5, 2.5))

    print("(g) saline freezing")
    out = case_saline_freezing()
    Tg = np.linspace(-10.5, 0.5, 400)
    for S, col in zip(out, ("C0", "C1", "C3")):
        f_x = np.where(Tg < -mat.M_LIQUIDUS * S, mat.M_LIQUIDUS * S / -np.minimum(Tg, -1e-9), 1.0)
        ax[6].plot(Tg, f_x, color=col, lw=1.5)
        T, f = out[S]
        ax[6].plot(T[::3], f[::3], ls="none", marker="o", ms=4, mfc="none", color=col, label=f"S = {S:.0f} psu")
        f_at_T = np.where(T < -mat.M_LIQUIDUS * S, mat.M_LIQUIDUS * S / -np.minimum(T, -1e-9), 1.0)
        lines.append(f"(g) saline freezing, S = {S:.0f}: max |f - liquidus| = {np.max(np.abs(f - f_at_T)):.1e}")
    ax[6].plot([], [], color="k", lw=1.5, label=r"liquidus, $f = m S_b / (-T)$")
    ax[6].set(xlabel="Temperature (°C)", ylabel="Liquid fraction of the pore H$_2$O", xlim=(-10.5, 0.5), ylim=(0, 1.05))

    print("(h) brine rejection")
    d, p, S_min, err_salt, err_energy = case_brine()
    ice = p.open[:, 0] & ~p.fluid[:, 0]
    ax[7].axvline(S_SEA, color="0.6", lw=1, ls="--", label="initial sea water")
    ax[7].axvline(S_min, color="0.6", lw=1, ls=":", label=r"$S_{ice,min}$")
    ax[7].plot(p.S_bulk[:, 0], d, color="C0", lw=2, label="bulk salinity")
    ax[7].plot(p.S_liq[ice, 0], d[ice], color="C3", lw=2, label="brine salinity")
    ax[7].plot(-p.T[ice, 0][::2] / mat.M_LIQUIDUS, d[ice][::2], ls="none", marker="o", ms=4, mfc="none",
               color="k", label=r"liquidus, $-T/m$")
    h_ice = float(np.sum(np.abs(np.diff(d)).mean() * (1 - p.f_liq[:, 0])))
    ax[7].axhline(h_ice, color="C2", lw=1, label="ice thickness")
    ax[7].set(xlabel="Salinity (psu)", ylabel="Depth (m)", ylim=(0.6, 0), xlim=(0, 200))
    lines.append(f"(h) brine: water under the ice {p.S_bulk[0, 0]:.2f} psu (from {S_SEA:.0f}); "
                 f"ice bulk {p.S_bulk[ice, 0].min():.1f}-{p.S_bulk[ice, 0].max():.1f} psu; "
                 f"salt budget {err_salt:.1e}, energy budget {err_energy:.1e}")

    for a, lab in zip(ax, "abcdefgh"):
        pl.panel_label(a, f"({lab})")
        a.legend(loc="lower right" if lab in "bdeg" else "lower left" if lab in "c" else "best", fontsize=9, frameon=False)
    ax[0].legend(loc="lower left", fontsize=9, frameon=False)
    ax[5].legend(loc="lower right", fontsize=9, frameon=False)
    ax[6].legend(loc="center left", fontsize=9, frameon=False)
    ax[7].legend(loc="lower right", fontsize=9, frameon=False)

    pl.save(fig, OUT_DIR / "analytical_suite.png", dpi=DPI)
    summary = "\n".join(lines)
    (OUT_DIR / "analytical_suite.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
