"""Verification tests for the 2DV thermal model (run: pytest tests/test_thermal2dv.py)."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf, erfc

from arctic_xbeach.thermal2dv import Grid, Model2DV, Settings, constant_forcing, run
from arctic_xbeach.thermal2dv import materials as mat
from arctic_xbeach.thermal2dv.erosion import undercut_columns


def column_model(depth, dz, phi_s, phi_h2o, T0, **settings):
    """Single soil column; the top face exchanges with the atmosphere, the base gets q_geo."""
    grid = Grid(np.array([0.0, 1.0]), np.linspace(-depth, 0.0, int(round(depth / dz)) + 1))
    s = Settings(q_geo=0.0, emissivity=0.0, erosion=False, collapse=False, **settings)
    m = Model2DV(grid, s)
    m.phi_s[:] = phi_s
    m.phi_h2o[:] = phi_h2o
    m.set_temperature(T0)
    return m


def fixed_surface(T_s):
    # a huge wind speed makes the bulk exchange coefficient ~1e6 W m-2 K-1: a Dirichlet surface
    return constant_forcing(T_air=T_s, u10=1e6, lw_down=0.0, eta=-1e9)


def test_conduction_matches_erfc():
    """Dry soil, surface temperature step: T = T0 + (Ts - T0) erfc(d / 2 sqrt(alpha t))."""
    m = column_model(depth=10.0, dz=0.02, phi_s=1.0, phi_h2o=0.0, T0=-5.0)
    t_end = 10 * 86400.0
    run(m, fixed_surface(5.0), t_end, dt=3600.0)
    alpha = mat.K_MINERAL / mat.C_MINERAL
    d = -m.grid.zc
    T_exact = -5.0 + 10.0 * erfc(d / (2 * np.sqrt(alpha * t_end)))
    T = m.state().T[:, 0]
    assert np.max(np.abs(T - T_exact)) < 0.05


def neumann_thaw_depth(Ts, T0, Tf, k_u, k_f, c_u, c_f, lv, t):
    """Two-phase Neumann (Stefan) solution for thaw from a surface held at Ts > Tf."""
    a_u, a_f = k_u / c_u, k_f / c_f
    nu = np.sqrt(a_u / a_f)

    def f(lam):
        lhs = k_u * (Ts - Tf) * np.exp(-lam**2) / (erf(lam) * np.sqrt(np.pi * a_u))
        lhs -= k_f * (Tf - T0) * np.exp(-(nu * lam) ** 2) / (erfc(nu * lam) * np.sqrt(np.pi * a_f))
        return lhs - lv * lam * np.sqrt(a_u)

    lam = brentq(f, 1e-6, 5.0)
    return 2 * lam * np.sqrt(a_u * t)


def test_stefan_thaw_front():
    """Saturated silt thawing from the surface: front position within 3% of Neumann."""
    phi_s, phi_h2o, T0, Ts = 0.6, 0.4, -5.0, 5.0
    m = column_model(depth=6.0, dz=0.02, phi_s=phi_s, phi_h2o=phi_h2o, T0=T0)
    t_end = 30 * 86400.0
    run(m, fixed_surface(Ts), t_end, dt=3600.0)
    p = m.state()
    X_model = float(np.sum(p.f_liq[:, 0] * m.grid.dz))
    c_f, c_u, lv = mat.heat_capacities(phi_s, phi_h2o)
    k_f = mat.soil_conductivity(phi_s, phi_h2o, 0.0)
    k_u = mat.soil_conductivity(phi_s, phi_h2o, 1.0)
    X_exact = neumann_thaw_depth(Ts, T0, 0.0, k_u, k_f, c_u, c_f, lv, t_end)
    assert abs(X_model - X_exact) / X_exact < 0.03


def ice_block_model(h=500.0, T_water=5.0, dx=0.05):
    """Ice block (1 x 1 m, at 0 degC) on an insulated floor in a well-mixed 5 degC tank."""
    grid = Grid(np.arange(0.0, 4.0 + dx / 2, dx), np.arange(0.0, 2.0 + dx / 2, dx))
    s = Settings(h_water=h, q_geo=0.0, tau_flush=600.0, n_sponge=0, erosion=False, collapse=False,
                 Kx_surf=1.0, Kz_surf=1.0, S_sea=0.0)
    m = Model2DV(grid, s)
    m.phi_h2o[:] = 1.0
    X, Z = grid.meshgrid()
    block = (np.abs(X - 2.0) < 0.5) & (Z < 1.0)
    T = np.where(block, 0.0, T_water)
    m.set_temperature(T, frozen_at_tf=True)
    f = constant_forcing(T_air=T_water, u10=0.0, lw_down=None, eta=10.0, T_water=T_water)
    return m, f


def test_ice_block_melt_rate():
    """Faces retreat at v = h dT / Lv: width W - 2vt, height Hb - vt."""
    m, f = ice_block_model()
    t_end = 12 * 3600.0
    run(m, f, t_end, dt=600.0)
    p = m.state()
    ice = float(np.sum(m.grid.volume * (1.0 - p.f_liq)))
    v = 500.0 * 5.0 / mat.LV_WATER
    ice_exact = (1.0 - 2 * v * t_end) * (1.0 - v * t_end)
    assert abs(ice - ice_exact) / (1.0 - ice_exact) < 0.10  # within 10% of the melted volume


def test_energy_budget_closes():
    """Domain energy change equals the boundary input (plus logged wet/dry changes)."""
    m, f = ice_block_model()
    m.s.q_geo = 0.05
    E0 = m.total_energy()
    run(m, f, 6 * 3600.0, dt=900.0)
    E1 = m.total_energy()
    budget = m.energy["boundary"] + m.energy["wetdry"] + m.energy["erosion"] + m.energy["collapse"]
    assert abs((E1 - E0) - budget) < 1e-6 * max(abs(budget), 1.0)


def test_undercut_detection():
    grid = Grid(np.arange(0.0, 5.01, 0.5), np.arange(0.0, 3.01, 0.5))
    m = Model2DV(grid, Settings())
    m.phi_s[:] = 0.6
    m.phi_s[2:4, 6:] = 0.0  # a 2 m deep niche from x = 3 m
    cols, depth = undercut_columns(m)
    assert depth == pytest.approx(2.0)
    assert all(jb == 2 and jt == 3 for _, jb, jt in cols)


def test_cold_ice_rich_face_retreat():
    """Warm water on frozen ice-rich soil (-8 degC): the thaw-erosion front descends at the
    quasi-steady rate v = h (T_w - T_f) / (L_v + C_f (T_f - T_0)) once past the start-up."""
    dz, T_w, T0, phi_s = 0.1, 5.0, -8.0, 0.35
    grid = Grid(np.array([0.0, 1.0]), np.arange(-4.0, 1.0 + dz / 2, dz))
    s = Settings(q_geo=0.0, emissivity=0.0, tau_flush=600.0, n_sponge=0)
    m = Model2DV(grid, s)
    soil = grid.zc[:, None] < 0.0
    m.phi_s[soil] = phi_s
    m.phi_h2o[:] = np.where(soil, 1.0 - phi_s, 1.0)
    m.S[~soil] = s.S_sea
    m.set_temperature(np.where(soil, T0, T_w))
    f = constant_forcing(T_air=T_w, u10=0.0, lw_down=None, eta=1.0, T_water=T_w)

    def front():  # frozen thickness, counting the part-thawed front cell fractionally
        p = m.state()
        return float(np.sum(grid.dz * p.soil[:, 0] * (1.0 - p.f_liq[:, 0])))

    run(m, f, 0.5 * 86400.0, dt=1800.0)
    z1 = front()
    run(m, f, 2.0 * 86400.0, dt=1800.0)
    z2 = front()
    cf, _, lv = mat.heat_capacities(phi_s, 1.0 - phi_s)
    v_ref = s.h_water * T_w / (lv + cf * (0.0 - T0))
    v_model = (z1 - z2) / (1.5 * 86400.0)
    # the water at the face runs ~0.25 K below T_ref (flushing), hence ~5% below v_ref
    assert -0.10 < (v_model - v_ref) / v_ref < 0.02
