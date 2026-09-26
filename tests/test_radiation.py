"""Tests for the radiation geometry of the 2DV model (run: pytest tests/test_radiation.py)."""

import numpy as np
import pytest

from arctic_xbeach.thermal2dv import Grid, Model2DV, Settings, constant_forcing, run
from arctic_xbeach.thermal2dv import radiation as rad

LAT, LON = 70.13, -143.66  # Barter Island


def minutes(day):
    t0 = np.datetime64(day + "T00:00", "s")
    return t0 + np.arange(0, 1440 * 60, 60).astype("timedelta64[s]")


@pytest.mark.parametrize("day, decl", [("2016-06-20", 23.43), ("2016-12-21", -23.44), ("2016-03-20", -0.05)])
def test_solar_noon_elevation_and_azimuth(day, decl):
    """Daily maximum elevation is 90 - lat + declination, reached with the sun due south."""
    el, az = rad.solar_position(minutes(day), LAT, LON)
    k = np.argmax(el)
    assert el[k] == pytest.approx(90.0 - LAT + decl, abs=0.15)
    assert az[k] == pytest.approx(180.0, abs=0.5)


def test_solar_noon_time_follows_longitude():
    """Solar noon at Barter is about 12:00 + 143.66/15 h UTC, shifted by the equation of time
    (about +1.5 min on 20 June): no time zone enters."""
    el, _ = rad.solar_position(minutes("2016-06-20"), LAT, LON)
    t_noon = np.argmax(el) / 60.0
    assert t_noon == pytest.approx(12.0 + 143.66 / 15.0 + 1.5 / 60.0, abs=2 / 60)


def test_polar_day_and_night():
    el_jun, _ = rad.solar_position(minutes("2016-06-20"), LAT, LON)
    el_dec, _ = rad.solar_position(minutes("2016-12-21"), LAT, LON)
    assert el_jun.min() > 0.0 and el_dec.max() < 0.0


def test_erbs_split_conserves_and_limits():
    t = np.datetime64("2016-07-01T21:00")
    el, _ = rad.solar_position(t, LAT, LON)
    for ghi in (0.0, 50.0, 300.0, 600.0):
        d, f = rad.erbs_split(ghi, el, t)
        assert d + f == pytest.approx(ghi)
        assert d >= 0.0 and f >= 0.0
    d, f = rad.erbs_split(200.0, 1.0, t)  # sun too low: all diffuse
    assert d == 0.0 and f == pytest.approx(200.0)


# ------------------------------------------------------------------ geometry
def uniform_grid(L, H, d):
    return Grid(np.linspace(0.0, L, int(round(L / d)) + 1), np.linspace(0.0, H, int(round(H / d)) + 1))


def step_geometry(H_low=1.0, H_high=6.0, x_step=10.0, L=20.0, d=0.25, **kw):
    """Low ground seaward of x_step, a vertical bluff face at x_step, high ground landward."""
    g = uniform_grid(L, H_high + 4.0, d)
    X, Z = g.meshgrid()
    active = Z < np.where(X < x_step, H_low, H_high)
    return g, active, rad.Geometry(g, active, **kw)


def test_svf_flat_ground_and_wall():
    g, active, geo = step_geometry(n_dirs=90)
    up, sea = geo.masks[rad.UP], geo.masks[rad.SEA]
    X, Z = g.meshgrid()
    # the bluff top and a vertical wall above lower ground
    assert np.allclose(geo.svf[rad.UP][up & (X > 10.0)], 1.0)
    assert np.allclose(geo.svf[rad.SEA][sea], 0.5)


def test_svf_beside_wall_matches_crossed_strings():
    """Flat ground at distance d from a wall of height H: SVF = (1 + cos h) / 2, h = atan(H/d)."""
    g, active, geo = step_geometry(n_dirs=720, d=0.25)
    up = geo.masks[rad.UP]
    X, _ = g.meshgrid()
    H = 5.0
    for j, i in zip(*np.nonzero(up & (X < 10.0) & (X > 2.0))):
        dist = 10.0 - g.xc[i]
        # the face is a strip; average the analytic value over its width
        xs = g.xe[i] + (np.arange(4) + 0.5) / 4 * g.dx[i]
        exact = np.mean(0.5 * (1.0 + np.cos(np.arctan(H / (10.0 - xs)))))
        assert geo.svf[rad.UP][j, i] == pytest.approx(exact, abs=0.005), dist


def test_svf_canyon_floor():
    """Floor at the centre of a symmetric canyon of half-width b and depth H: SVF = cos(atan(H/b))."""
    g = uniform_grid(20.0, 10.0, 0.25)
    X, Z = g.meshgrid()
    active = Z < np.where(np.abs(X - 10.0) < 2.0, 1.0, 5.0)
    geo = rad.Geometry(g, active, n_dirs=720, n_sub=1)
    i = np.argmin(np.abs(g.xc - 10.125))
    j = np.nonzero(geo.masks[rad.UP][:, i])[0][0]
    b = np.array([2.0 - 0.125, 2.0 + 0.125])  # distances to the two walls
    exact = 0.5 * np.sum(np.cos(np.arctan(4.0 / b)))
    assert geo.svf[rad.UP][j, i] == pytest.approx(exact, abs=0.005)


def test_overhang_underside_sees_ground_only():
    g = uniform_grid(10.0, 8.0, 0.25)
    X, Z = g.meshgrid()
    active = Z < 1.0
    active |= (X > 5.0) & (Z < 6.0)
    active &= ~((X > 5.0) & (X < 7.0) & (Z > 1.0) & (Z < 3.0))  # a niche 2 m deep, 2 m high
    geo = rad.Geometry(g, active, n_dirs=720, n_sub=1)
    down = geo.masks[rad.DOWN]
    assert down.any() and np.allclose(geo.svf[rad.DOWN][down], 0.0)
    # the back wall of the niche sees the sky only below the lip: elevations 0 to atan((3 - z) / 2)
    back = geo.masks[rad.SEA] & (X > 6.5) & (X < 7.5)
    exact = 0.5 * np.sin(np.arctan((3.0 - Z[back]) / 2.0))
    assert back.any() and np.allclose(geo.svf[rad.SEA][back], exact, atol=0.005)


def test_shadow_length_behind_step():
    """With the sun landward of a bluff of height H, the beach is shaded over H / tan(alpha_p)."""
    g, active, geo = step_geometry(d=0.1, n_sub=1, x_step=25.0, L=35.0)
    H = 5.0
    for el, rel in ((20.0, 0.0), (30.0, 40.0), (12.0, -60.0)):
        s = rad.sun_vector(el, 90.0 + rel, 90.0)  # landward bearing 90 deg, sun rel deg off the transect
        f = geo.direct_factor(s)[rad.UP]
        up = geo.masks[rad.UP]
        X, _ = g.meshgrid()
        beach = up & (X < 25.0)
        shaded = beach & (f == 0.0)
        tan_p = s[2] / s[0]
        L_shadow = 25.0 - g.xe[:-1][shaded.any(axis=0)].min()
        assert L_shadow == pytest.approx(H / tan_p, abs=g.dx[0])
        assert np.allclose(f[beach & ~shaded], 1.0)


def test_vertical_face_direct_factor():
    """cos(i) / sin(alpha) = cot(alpha) cos(A - A_face) on an unshaded vertical face."""
    g, active, geo = step_geometry()
    top = geo.masks[rad.SEA] & (g.meshgrid()[1] > 5.0)
    for el, az in ((20.0, 270.0), (35.0, 300.0), (10.0, 180.0)):
        s = rad.sun_vector(el, az, 90.0)  # the sea face looks towards 270
        f = geo.direct_factor(s)[rad.SEA][top]
        expected = max(np.cos(np.radians(az - 270.0)), 0.0) / np.tan(np.radians(el))
        assert np.allclose(f, expected)


def test_staircase_slope_matches_tilted_plane():
    """A 20 deg staircase slope receives, per unit horizontal length, what a tilted plane
    receives over the day: sin a - tan b cos a cos(A - A_up) summed with shading of the treads.
    This is the quantity v1.0's Buffo (1972) slope factor approximates (as a ratio of daily maxima)."""
    beta = np.radians(20.0)
    d = 0.2
    nx = 40
    g = Grid(np.arange(nx + 1) * d, np.arange(nx + 1 + 10) * d * np.tan(beta))
    jj, ii = np.meshgrid(np.arange(g.nz), np.arange(g.nx), indexing="ij")
    active = jj <= ii  # one riser per column, surface rising landward
    geo = rad.Geometry(g, active, n_sub=16)
    bearing_land = 180.0 + 30.0  # slope faces 30 deg east of north
    t = np.datetime64("2016-07-15T00:00") + np.arange(0, 24 * 60, 10).astype("timedelta64[m]")
    el, az = rad.solar_position(t, LAT, LON)
    mid = slice(5, nx - 5)  # away from the lateral boundaries
    stair = plane = 0.0
    for e, a in zip(el, az):
        s = rad.sun_vector(e, a, bearing_land)
        if s[2] <= 0.035:
            continue
        f = geo.direct_factor(s)
        per_col = f[rad.UP][:, mid].sum(0) * d + f[rad.SEA][:, mid].sum(0) * d * np.tan(beta)
        stair += per_col.mean() / d * s[2]
        plane += max(s[2] - np.tan(beta) * s[0], 0.0)
    assert stair == pytest.approx(plane, rel=0.02)


# ------------------------------------------------------------------ model
def column(radiation):
    grid = Grid(np.array([0.0, 1.0]), np.linspace(-5.0, 1.0, 61))
    site = dict(latitude=LAT, longitude=LON, start_time_utc="2016-07-01T00:00", bearing_seaward=2.0) if radiation else {}
    m = Model2DV(grid, Settings(q_geo=0.0, erosion=False, collapse=False, **site))
    X, Z = grid.meshgrid()
    m.phi_s[:] = np.where(Z < 0.0, 0.6, 0.0)
    m.phi_h2o[:] = np.where(Z < 0.0, 0.4, 0.0)
    m.set_temperature(np.where(Z < 0.0, -5.0, 5.0))
    return m


def test_flat_ground_reproduces_legacy():
    """On flat open ground the geometry gives exactly the legacy shortwave and longwave."""
    f = constant_forcing(T_air=5.0, sw_down=250.0, lw_down=300.0, u10=5.0, eta=-10.0)
    runs = []
    for radiation in (False, True):
        m = column(radiation)
        run(m, f, 3 * 86400.0, dt=3600.0)
        runs.append(m.state().T)
    assert np.nanmax(np.abs(runs[0] - runs[1])) < 1e-9


def bluff_model(radiation, bearing=2.0):
    grid = uniform_grid(30.0, 16.0, 0.5)
    site = dict(latitude=LAT, longitude=LON, start_time_utc="2016-07-15T00:00", bearing_seaward=bearing) if radiation else {}
    m = Model2DV(grid, Settings(q_geo=0.0, erosion=False, collapse=False, **site))
    X, Z = grid.meshgrid()
    land = Z < np.where(X < 15.0, 2.0, 12.0)
    m.phi_s[:] = np.where(land, 0.6, 0.0)
    m.phi_h2o[:] = np.where(land, 0.4, 0.0)
    m.set_temperature(np.where(land, -5.0, 5.0))
    return m


def test_energy_budget_closes_with_geometry():
    m = bluff_model(True)
    f = constant_forcing(T_air=5.0, sw_down=300.0, lw_down=280.0, u10=5.0, eta=-10.0)
    E0 = m.total_energy()
    run(m, f, 12 * 3600.0, dt=1800.0)
    budget = m.energy["boundary"] + m.energy["wetdry"] + m.energy["erosion"] + m.energy["collapse"]
    assert abs((m.total_energy() - E0) - budget) < 1e-6 * max(abs(budget), 1.0)


def clear_sky_forcing(start):
    """A clear July day: direct and diffuse from the sun's elevation, air at 0 degC."""
    t0 = np.datetime64(start, "s")

    def f(t, zc):
        el, _ = rad.solar_position(t0 + np.timedelta64(int(t), "s"), LAT, LON)
        sin_e = max(np.sin(np.radians(float(el))), 0.0)
        direct = 900.0 * 0.75 ** (1.0 / sin_e) * sin_e if sin_e > 0.035 else 0.0
        F = constant_forcing(T_air=0.0, sw_down=direct + 60.0 * sin_e, lw_down=280.0, u10=5.0, eta=-10.0)(t, zc)
        F.sw_direct, F.sw_diffuse = direct, 60.0 * sin_e
        return F

    return f


def test_north_facing_bluff_gets_less_sun_than_legacy():
    """Absorbed shortwave on the bluff face over a clear July day at 70 N: about 0.3 of flat
    ground facing north, above 0.6 facing south, against the legacy fixed 0.5."""
    f = clear_sky_forcing("2016-07-15T00:00")
    ratio = {}
    for key, radiation, bearing in (("legacy", False, 2.0), ("north", True, 2.0), ("south", True, 182.0)):
        m = bluff_model(radiation, bearing)
        p = m.state()
        face = rad.exposed_faces(p.active)[rad.SEA] & (m.grid.meshgrid()[1] > 3.0)
        face_sw = flat_sw = 0.0
        for t0 in np.arange(0.0, 86400.0, 1800.0):
            m._step_window = (t0, 1800.0)
            F = f(t0 + 900.0, m.grid.zc)
            _, sw, _ = m._shortwave(p, F)
            face_sw += sw[rad.SEA][face].mean()
            flat_sw += (1.0 - F.albedo_land) * F.sw_down
        ratio[key] = face_sw / flat_sw
    assert ratio["legacy"] == pytest.approx(0.5)
    assert 0.2 < ratio["north"] < 0.4 and ratio["south"] > 0.6
