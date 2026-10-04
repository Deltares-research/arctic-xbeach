"""Tests for the 2DV <-> XBeach coupler (run: pytest tests/test_coupling.py)."""

import numpy as np
import pytest

from arctic_xbeach.thermal2dv import Grid, Model2DV, Settings, constant_forcing, run
from arctic_xbeach.thermal2dv import materials as mat
from arctic_xbeach.thermal2dv.coupling import Thermal2DVCoupler, kobayashi_h, node_edges, remap


def beach_model(thaw_depth=1.0, dz=0.25, dx=0.5, erosion=False):
    """Flat ground at z = 0 (x 0-10 m, sea water to 5 m below 0 offshore of x = 3 m... no: all land),
    thawed to thaw_depth over frozen ground."""
    grid = Grid(np.arange(0.0, 10.0 + dx / 2, dx), np.arange(-5.0, 3.0 + dz / 2, dz))
    m = Model2DV(grid, Settings(q_geo=0.0, erosion=erosion, collapse=False))
    X, Z = grid.meshgrid()
    soil = Z < 0.0
    m.phi_s[soil] = 0.6
    m.phi_h2o[soil] = 0.4
    m.set_temperature(np.where(Z > -thaw_depth, 2.0, -5.0))
    return m


def test_remap_conserves_integral():
    rng = np.random.default_rng(1)
    x_src = np.cumsum(rng.uniform(0.5, 3.0, 40))
    e_src = node_edges(x_src)
    v = rng.normal(size=40)
    e_dst = np.linspace(e_src[0], e_src[-1], 137)
    w, cov = remap(e_src, v, e_dst)
    assert np.allclose(cov, 1.0)
    assert np.sum(w * np.diff(e_dst)) == pytest.approx(np.sum(v * np.diff(e_src)))


def test_bed_level_and_erodible_thickness():
    m = beach_model(thaw_depth=1.0)
    c = Thermal2DVCoupler(m)
    assert np.allclose(c.bed_level(), 0.0)
    assert np.allclose(c.erodible_thickness(), 1.0)


def test_frozen_overhang_is_not_erodible():
    m = beach_model(thaw_depth=0.0)
    X, Z = m.grid.meshgrid()
    niche = (X > 4.0) & (X < 6.0) & (Z > -3.0) & (Z < -1.0)
    m.phi_s[niche] = 0.0
    m.phi_h2o[niche] = 0.0
    m.H[niche] = 0.0
    c = Thermal2DVCoupler(m)
    assert np.allclose(c.bed_level(), 0.0)          # top envelope: XBeach sees the overhang's top
    assert np.allclose(c.erodible_thickness(), 0.0)


def test_erosion_and_deposition_exact_bed_and_budget():
    m = beach_model(thaw_depth=1.0)
    c = Thermal2DVCoupler(m)
    x_xb = np.linspace(0.25, 9.75, 20)
    dz = np.where(x_xb < 5.0, -0.6, 0.35)
    E0 = m.total_energy()
    c.apply_bed_change(x_xb, dz, T_water=1.0, S_water=30.0, eta=-10.0)
    zb = c.bed_level()
    xc = m.grid.xc
    assert np.allclose(zb[xc < 4.5], -0.6)
    assert np.allclose(zb[xc > 5.5], 0.35)
    # cells follow the bed to within half a cell
    top = np.array([m.grid.ze[np.nonzero(m.state().soil[:, i])[0].max() + 1] for i in range(m.grid.nx)])
    assert np.all(np.abs(top - zb) <= 0.5 * 0.25 + 1e-12)
    assert (m.total_energy() - E0) == pytest.approx(m.energy["morphology"], rel=1e-12, abs=1e-6)
    # the erodible layer shrank by the erosion and grew by the deposit
    ne = c.erodible_thickness()
    assert np.allclose(ne[xc < 4.5], 0.4, atol=0.13)
    assert np.allclose(ne[xc > 5.5], 1.35, atol=0.13)


def test_erosion_stops_at_frozen_ground():
    m = beach_model(thaw_depth=0.5)
    c = Thermal2DVCoupler(m)
    x_xb = np.linspace(0.25, 9.75, 20)
    c.apply_bed_change(x_xb, np.full(20, -2.0), T_water=1.0, S_water=30.0, eta=-10.0)
    assert np.all(c.bed_level() >= -0.5 - 0.125 - 1e-12)
    assert c.sediment["unapplied"] > 0.0


def test_to_xbeach_outside_domain():
    m = beach_model(thaw_depth=1.0)
    c = Thermal2DVCoupler(m, ne_offshore=7.0)
    x_xb = np.linspace(-20.0, 9.75, 60)
    zb_xb = np.where(x_xb < 0.0, -3.0, 0.0)
    zb, ne = c.to_xbeach(x_xb, zb_xb)
    off = x_xb < -1.0
    assert np.allclose(zb[off], -3.0) and np.allclose(ne[off], 7.0)
    inside = x_xb > 1.0
    assert np.allclose(zb[inside], 0.0) and np.allclose(ne[inside], 1.0)


def test_kobayashi_h_matches_hand_calculation():
    """H = 1 m, T = 8 s, d = 1 m: u_b = pi H / (T sinh kd), h = a fw c u_b / (1 + sqrt(fw/2) E)."""
    H, T, d = 1.0, 8.0, 1.0
    omega = 2 * np.pi / T
    k = 0.25
    for _ in range(50):
        k = omega**2 / (9.81 * np.tanh(k * d))
    u_b = np.pi * H / (T * np.sinh(k * d))
    u_star = np.sqrt(0.025) * u_b
    E = 0.52 * (u_star * 2.5 * 1.90e-3 / 1.848e-6) ** 0.45 * 13.4**0.8
    expected = 0.5 * 0.05 * mat.C_WATER * u_b / (1 + np.sqrt(0.025) * E)
    assert kobayashi_h(H, T, d) == pytest.approx(expected, rel=0.02)
    # each point uses its own depth (v1.0 used the last depth for all)
    h = kobayashi_h(np.array([1.0, 1.0]), T, np.array([1.0, 4.0]))
    assert h[0] > h[1]


def test_calm_coupled_run_equals_standalone():
    """With no storm and no bed change the coupler is transparent."""
    f = constant_forcing(T_air=5.0, sw_down=200.0, lw_down=300.0, u10=4.0, eta=-10.0)
    a, b = beach_model(), beach_model()
    c = Thermal2DVCoupler(b)
    for _ in range(12):
        a.step(3600.0, f)
        c.step(3600.0, f)
    assert np.array_equal(a.H, b.H)


def test_storm_sets_swash_level_and_h():
    m = beach_model(thaw_depth=1.0)
    c = Thermal2DVCoupler(m, h_bounds=(10.0, 1e5))
    x_xb = np.linspace(0.25, 9.75, 20)
    zb_xb = np.linspace(-1.0, 1.0, 20)
    t = np.arange(200)
    zs = 0.2 + 0.4 * np.sin(2 * np.pi * t / 20.0)[:, None] + 0.0 * x_xb[None, :]
    storm = c.set_storm(x_xb, zb_xb, zs, H=np.full(20, 0.5), T_wave=8.0)
    f_wet = ((zs - zb_xb[None, :]) > 0.01).mean(axis=0)
    assert storm["eta"] == pytest.approx(zb_xb[f_wet >= 0.1].max())
    # h decreases landward as the wetted fraction drops
    h = storm["h"]
    assert h[2] > h[12] > 0.0


def test_released_sediment_goes_to_toe():
    """Ablated bluff mineral is placed seaward of the toe with beach porosity."""
    m = beach_model(thaw_depth=1.0)
    c = Thermal2DVCoupler(m, toe_width=2.0, n_beach=0.4)
    m.eroded_sediment += 0.6  # m3 of mineral per m, e.g. from face ablation
    zb0 = c.bed_level().copy()
    c.place_released_sediment(T_water=1.0, S_water=30.0, eta=-0.5 + 1e-9)  # the whole ground is 'dry'
    dV = np.sum((c.bed_level() - zb0) * m.grid.dx)
    assert dV == pytest.approx(0.6 / 0.6)


def test_storm_tolerates_dry_nans():
    """XBeach writes NaN wave height (and water level) on dry nodes."""
    m = beach_model(thaw_depth=1.0)
    c = Thermal2DVCoupler(m)
    x_xb = np.linspace(0.25, 9.75, 20)
    zb_xb = np.linspace(-1.0, 1.0, 20)
    zs = np.full((50, 20), 0.3)
    zs[:, 15:] = np.nan
    H = np.where(zb_xb < 0.3, 0.4, np.nan)
    storm = c.set_storm(x_xb, zb_xb, zs, H, T_wave=6.0)
    assert np.all(np.isfinite(storm["h"])) and np.isfinite(storm["eta"])


def test_storm_stats_match_series():
    """Gaussian swash statistics give the same wetted fraction as a Gaussian time series."""
    m = beach_model(thaw_depth=1.0)
    c = Thermal2DVCoupler(m, h_bounds=(10.0, 1e5))
    x_xb = np.linspace(0.25, 9.75, 20)
    zb_xb = np.linspace(-1.0, 1.0, 20)
    rng = np.random.default_rng(0)
    zs = 0.2 + 0.3 * rng.standard_normal((20000, 1)) + 0.0 * x_xb[None, :]
    zs = np.maximum(zs, zb_xb[None, :])  # dry: zs = zb
    series = c.set_storm(x_xb, zb_xb, zs, H=np.full(20, 0.5), T_wave=8.0)
    f_series = c.storm["f_wet"].copy()
    stats = c.set_storm_stats(x_xb, zb_xb, np.full(20, 0.2), np.full(20, 0.3), zs.max(axis=0), np.full(20, 0.5), 8.0)
    assert np.allclose(stats["f_wet"], f_series, atol=0.02)
    assert stats["eta"] == pytest.approx(series["eta"], abs=0.11)
