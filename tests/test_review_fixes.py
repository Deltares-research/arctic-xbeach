"""Regression tests for the code review of 26 September 2026 (research/2026-09-26_2dv-temperature-
code-review.md): conservative collapse on stretched grids, connected niches, grid bounds, brine
source connectivity, atomic steps, air-exposed faces and Recorder validation."""

import numpy as np
import pytest

from arctic_xbeach.thermal2dv import Grid, Model2DV, Recorder, Settings, constant_forcing, stretched_edges
from arctic_xbeach.thermal2dv import materials as mat
from arctic_xbeach.thermal2dv import radiation as rad
from arctic_xbeach.thermal2dv.erosion import collapse_overhang, undercut_components


def overhang_model(ze, d_crit=0.5, width=1.0, n_cols=3):
    """Frozen ground in row 0, an air gap in row 1 and a frozen overhang above it, in the
    middle columns; the outer columns are solid throughout."""
    grid = Grid(np.linspace(0.0, width * n_cols, n_cols + 1), np.asarray(ze, dtype=float))
    m = Model2DV(grid, Settings(d_crit=d_crit, erosion=False, q_geo=0.0))
    m.phi_s[:] = 0.6
    m.phi_h2o[:] = 0.4
    m.phi_s[1, 1:-1] = 0.0
    m.phi_h2o[1, 1:-1] = 0.0
    m.set_salinity(5.0)
    T = -3.0 - np.arange(grid.nz)[:, None] * np.ones(grid.nx)  # different temperature per row
    m.set_temperature(T)
    return m


@pytest.mark.parametrize("ze", [[0.0, 1.0, 3.0, 6.0], [0.0, 1.0, 3.0, 6.0, 7.5], [0.0, 0.5, 1.0, 1.5]])
def test_collapse_conserves_mineral_h2o_salt_energy(ze):
    """Collapse of an overhang into an air gap is an internal rearrangement: mineral, H2O,
    salt and energy are conserved to round-off, also on a stretched grid."""
    m = overhang_model(ze)
    V = m.grid.volume
    before = [np.sum(V * a) for a in (m.phi_s, m.phi_h2o, m.salt, m.H)]
    assert collapse_overhang(m)
    after = [np.sum(V * a) for a in (m.phi_s, m.phi_h2o, m.salt, m.H)]
    assert np.allclose(after, before, rtol=1e-12, atol=1e-6)
    assert m.energy["collapse"] == pytest.approx(0.0, abs=1e-6)  # the gap held only air
    assert np.all(m.phi_s + m.phi_h2o <= 1.0 + 1e-12)


def test_collapse_into_water_logs_only_the_displaced_water():
    m = overhang_model([0.0, 1.0, 3.0, 6.0])
    m.phi_h2o[1, 1:-1] = 1.0
    m.set_salinity(30.0, np.zeros(m.grid.shape, bool) | (np.arange(m.grid.nz)[:, None] == 1))
    m.H[1, 1:-1] = mat.C_WATER * 2.0
    V = m.grid.volume
    water_E = float(np.sum(V[1, 1:-1] * m.H[1, 1:-1]))
    E0 = m.total_energy()
    collapse_overhang(m)
    assert m.total_energy() - E0 == pytest.approx(-water_E, rel=1e-12)
    assert m.energy["collapse"] == pytest.approx(-water_E, rel=1e-12)


def test_separate_subcritical_niches_do_not_add_up():
    """Two 1 m notches at opposite ends of the domain, d_crit = 1.5 m: no failure."""
    grid = Grid(np.linspace(0.0, 10.0, 11), np.linspace(0.0, 4.0, 5))
    m = Model2DV(grid, Settings(d_crit=1.5, erosion=False))
    m.phi_s[:] = 0.6
    m.phi_h2o[:] = 0.4
    m.phi_s[1, [1, 8]] = 0.0
    m.phi_h2o[1, [1, 8]] = 0.0
    m.set_temperature(-5.0)
    comps = undercut_components(m)
    assert [d for _, d in comps] == [1.0, 1.0]
    assert not collapse_overhang(m)
    # widen one notch to 2 m: only that overhang fails
    m.phi_s[1, 7] = m.phi_h2o[1, 7] = 0.0
    m.H[1, 7] = 0.0
    phi_left = m.phi_s[:, 1].copy()
    assert collapse_overhang(m)
    assert np.array_equal(m.phi_s[:, 1], phi_left)
    assert m.collapse_events[-1]["columns"] == [7, 8]


@pytest.mark.parametrize("args", [
    (0.0, 1.0, 0.95, 1.0, 0.1, 0.2),    # fine band narrower than d_fine
    (0.0, 1.0, 1.0, 1.0, 0.1, 0.2),     # empty fine band at the end
    (0.0, 1.0, 0.0, 0.0, 0.1, 0.2),     # empty fine band at the start
    (0.0, 10.0, 2.0, 7.33, 0.25, 1.0),  # non-integer number of fine cells
    (0.0, 10.0, -5.0, 20.0, 0.3, 1.0),  # fine band beyond the domain: clipped
    (-30.0, 15.0, -6.0, 14.5, 0.25, 3.0),
])
def test_stretched_edges_stay_in_domain(args):
    start, end, fs, fe, d, dmax = args
    e = stretched_edges(*args, growth=1.15)
    assert e[0] == start and e[-1] == end
    assert np.all(np.diff(e) > 0)
    assert np.diff(e).max() <= max(dmax, d) * 1.3 + 1e-9  # merged slivers aside, spacing capped


def test_brine_does_not_drain_a_separate_liquid_pocket():
    """Water / ice / water pocket / ice: only the ice resting on the water column drains."""
    grid = Grid(np.array([0.0, 1.0]), np.array([0.0, 1.0, 1.1, 1.2, 1.3]))
    m = Model2DV(grid, Settings(brine_drainage=True, tau_brine=3600.0))
    m.phi_h2o[:] = 1.0
    m.set_salinity(30.0)
    m.set_temperature(np.array([[1.0], [-5.0], [1.0], [-5.0]]))
    p = m.state()
    assert list(p.fluid[:, 0]) == [True, False, True, False]
    pocket = m.salt[2, 0]
    top_ice = m.salt[3, 0]
    m._drain_brine(3600.0)
    assert m.salt[2, 0] == pytest.approx(pocket)     # the pocket keeps its salt
    assert m.salt[3, 0] == pytest.approx(top_ice)    # ice above the pocket is not connected
    assert m.salt[1, 0] < 30.0                       # the attached ice drains


def test_failed_step_restores_state_and_budgets():
    """First half-step succeeds, the second fails: the model is back where it started."""
    grid = Grid(np.array([0.0, 1.0]), np.linspace(-2.0, 0.0, 21))
    m = Model2DV(grid, Settings(q_geo=0.05, erosion=False, collapse=False, max_halvings=1))
    m.phi_s[:] = 0.6
    m.phi_h2o[:] = 0.4
    m.set_temperature(-2.0)
    f = constant_forcing(T_air=5.0, u10=5.0, eta=-10.0)
    H0, E0, t0 = m.H.copy(), dict(m.energy), m.time
    orig = m._solve
    calls = []

    def flaky(dt, F):
        calls.append(dt)
        return orig(dt, F) if len(calls) == 2 else False

    m._solve = flaky
    with pytest.raises(RuntimeError):
        m.step(3600.0, f)
    assert len(calls) == 3
    assert np.array_equal(m.H, H0) and m.energy == E0 and m.time == t0


def test_isolated_block_exchanges_through_all_faces():
    """A block suspended in air has exposed up, down, seaward and landward faces, and its
    underside takes part in the sensible exchange with the air."""
    grid = Grid(np.linspace(0.0, 5.0, 6), np.linspace(0.0, 5.0, 6))
    m = Model2DV(grid, Settings(q_geo=0.0, erosion=False, collapse=False, emissivity=0.0))
    m.phi_s[2:4, 2:4] = 0.6
    m.phi_h2o[2:4, 2:4] = 0.4
    m.set_temperature(np.where(m.phi_s > 0, -10.0, np.nan))
    masks = rad.exposed_faces(m.state().active)
    assert [int(masks[k].sum()) for k in (rad.UP, rad.DOWN, rad.SEA, rad.LAND)] == [2, 2, 2, 2]
    _, _, ex = m._assemble(m.state())
    assert ex.A[rad.DOWN].sum() == pytest.approx(2.0)
    # warm air heats the block through all 8 faces equally: symmetric warming top and bottom
    m.step(600.0, constant_forcing(T_air=10.0, u10=5.0, eta=-10.0))
    T = m.state().T
    assert T[2, 2] == pytest.approx(T[3, 2], abs=1e-9)


def test_recorder_requires_every_or_times():
    m = Model2DV(Grid(np.array([0.0, 1.0]), np.array([0.0, 1.0])))
    with pytest.raises(ValueError):
        Recorder(m)
    with pytest.raises(ValueError):
        Recorder(m, every=10.0, times=[0.0])
    with pytest.raises(ValueError):
        Recorder(m, every=0.0)
    with pytest.raises(ValueError):
        Recorder(m, times=[10.0, 5.0])
