"""Tests for the thermal-solver options (conduction_form, latent_heat_form, era5_flux_shift_hours).

Run from the repository root with:  python -m pytest tests -q
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from legacy_scenario import SCENARIO as S, initial_temperature
from verification_common import ColumnSim, layer_index, layered_steady, run_stefan, stable_dt, stefan_exact, STEFAN
from arctic_xbeach.model import Simulation

DATA = Path(__file__).parent / "data"


def scenario_sim(**options):
    sim = ColumnSim(depth=S["depth"], N=S["N"], dt=S["dt"], T_init=initial_temperature(S["N"]),
                    k_frozen=S["k_frozen"], k_unfrozen=S["k_unfrozen"], c_frozen=S["c_frozen"], c_unfrozen=S["c_unfrozen"],
                    nb_max=S["nb_max"], nb_min=S["nb_min"], L_water_ice=S["L_water_ice"], rho_water=S["rho_water"],
                    rho_particle=S["rho_particle"], T_melt=S["T_melt"], geothermal_gradient=S["geothermal_gradient"],
                    **options)
    sim.surface_flux = np.full(1, S["surface_flux"])
    return sim


# --- 1. defaults are v1.0 -------------------------------------------------------------------------------
def test_defaults_reproduce_v10():
    """Default options give the original c6ab74b solver (reference made by tests/make_legacy_reference.py)."""
    ref = np.load(DATA / "legacy_reference.npz")
    sim = scenario_sim(conduction_form="nodal", latent_heat_form="per_mass")
    # a config that does not mention the options at all must behave the same
    del sim.config.thermal["conduction_form"], sim.config.thermal["latent_heat_form"]
    sim.define_soil_property_matrices(1, define_nb=True)  # recompute latent matrix with the defaults
    sim.k_frozen_matrix = S["k_frozen"].reshape(1, -1).copy()
    sim.k_unfrozen_matrix = S["k_unfrozen"].reshape(1, -1).copy()
    sim.enthalpy_matrix = sim._enthalpy_from_temperature(sim.temp_matrix)
    sim.advance(S["n_steps"])
    np.testing.assert_allclose(sim.enthalpy_matrix[0], ref["enthalpy"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(sim.temp_matrix[0], ref["temperature"], rtol=1e-12, atol=0)  # temp_matrix = T at start of last step
    # the melting band must actually have been exercised
    assert 0 < sim.liquid_fraction().max() <= 1 and (sim.temperature() > S["T_melt"]).any()


def test_invalid_option_raises():
    with pytest.raises(ValueError):
        scenario_sim(conduction_form="finite_difference").advance(1)
    with pytest.raises(ValueError):
        scenario_sim(latent_heat_form="per_kg")


# --- 2. latent heat form --------------------------------------------------------------------------------
def test_latent_heat_form_units():
    kw = dict(depth=1.0, N=11, dt=60.0, T_init=268.0, k_frozen=2.0, k_unfrozen=1.0, c_frozen=2e6, c_unfrozen=3e6,
              nb_max=0.5, nb_min=0.5)
    per_mass = ColumnSim(**kw, latent_heat_form="per_mass")
    volumetric = ColumnSim(**kw, latent_heat_form="volumetric")
    L, rho_w, nb = 334000.0, 1000.0, 0.5
    rho_soil = nb * rho_w + (1 - nb) * 2650.0
    np.testing.assert_allclose(volumetric.latent_matrix * volumetric.soil_density_matrix, rho_w * L * nb)  # J m-3
    np.testing.assert_allclose(per_mass.latent_matrix * per_mass.soil_density_matrix, rho_soil * L * nb)  # legacy per-mass form: rho_soil * L * nb
    np.testing.assert_allclose(per_mass.latent_matrix / volumetric.latent_matrix, rho_soil / rho_w)
    # the temperature <-> enthalpy round trip is consistent for both forms
    T = np.array([[250.0, 280.0, 265.0, 290.0, 270.0, 260.0, 285.0, 255.0, 275.0, 268.0, 281.0]])
    for sim in (per_mass, volumetric):
        sim.temp_matrix = T
        sim.enthalpy_matrix = sim._enthalpy_from_temperature(T)
        np.testing.assert_allclose(sim.temperature(), T, atol=1e-9)


# --- 3. Stefan problem ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stefan():
    ex = stefan_exact()
    N = 100
    dt = stable_dt(N, 0.25)
    out = {}
    for label, opt in {"legacy": dict(conduction_form="nodal", latent_heat_form="per_mass"),
                       "flux_volumetric": dict(conduction_form="flux", latent_heat_form="volumetric")}.items():
        t, front, *_ = run_stefan(N, dt, opt, n_record=10)
        out[label] = (t, front)
    return ex, out


def test_stefan_front_flux_volumetric(stefan):
    ex, out = stefan
    t, front = out["flux_volumetric"]
    rel = np.abs(front - ex.s(t)) / ex.s(t)
    assert rel[-1] < 0.03  # observed about 1.9 % at N = 100 (1.3 % at N = 200)
    assert rel[t >= STEFAN["t_end"] / 5].max() < 0.06


def test_stefan_legacy_front(stefan):
    ex, out = stefan
    t, front = out["legacy"]
    assert abs(front[-1] - ex.s(t[-1])) / ex.s(t[-1]) > 0.04  # legacy options: per-mass latent heat and nodal k at the front


# --- 4. conservation ------------------------------------------------------------------------------------
@pytest.mark.parametrize("latent", ["per_mass", "volumetric"])
def test_flux_form_conserves_energy_insulated(latent):
    """Insulated column (no surface flux, no geothermal flux), depth/phase-varying k and rho_soil, with melting."""
    sim = scenario_sim(conduction_form="flux", latent_heat_form=latent)
    sim.surface_flux = np.zeros(1)
    sim.config.thermal.geothermal_gradient = 0.0
    E0 = sim.stored_energy()[0]
    sim.advance_with_energy_balance(3000)
    phi = sim.liquid_fraction()
    assert phi.max() > 0 and (phi < 1).any()  # phase change occurred
    assert abs(sim.stored_energy()[0] - E0) / abs(E0) < 1e-12


def test_flux_form_energy_balance_with_boundary_fluxes():
    sim = scenario_sim(conduction_form="flux", latent_heat_form="volumetric")
    E0 = sim.stored_energy()[0]
    sim.advance_with_energy_balance(3000)
    dE, applied = sim.stored_energy()[0] - E0, sim.flux_integral[0]
    assert abs(dE - applied) / abs(applied) < 1e-10


def test_nodal_form_does_not_conserve_energy():
    sim = scenario_sim(conduction_form="nodal", latent_heat_form="volumetric")
    sim.surface_flux = np.zeros(1)
    sim.config.thermal.geothermal_gradient = 0.0
    E0 = sim.stored_energy()[0]
    sim.advance_with_energy_balance(3000)
    assert abs(sim.stored_energy()[0] - E0) / abs(E0) > 1e-5


def test_flux_form_uniform_k_matches_nodal():
    """For uniform k and a single phase the two forms are the same scheme (incl. the boundary treatment)."""
    kw = dict(depth=2.0, N=41, dt=100.0, T_init=np.linspace(262.0, 266.0, 41), k_frozen=1.7, k_unfrozen=1.7,
              c_frozen=2e6, c_unfrozen=2e6, geothermal_gradient=0.02)
    a, b = ColumnSim(**kw, conduction_form="nodal"), ColumnSim(**kw, conduction_form="flux")
    for s in (a, b):
        s.surface_flux = np.full(1, -25.0)
        s.advance(500)
    np.testing.assert_allclose(a.enthalpy_matrix, b.enthalpy_matrix, rtol=1e-11)


# --- 5. layered steady state ----------------------------------------------------------------------------
def test_layered_steady_state():
    depth, N, interfaces, k_layers = 4.0, 41, [1.05, 2.55], [1.0, 3.0, 1.5]
    z = np.linspace(0, depth, N)
    k = np.asarray(k_layers)[layer_index(z, interfaces)]
    exact = layered_steady(z, interfaces, k_layers, 268.15, 258.15)
    errs = {}
    for form in ("nodal", "flux"):
        sim = ColumnSim(depth=depth, N=N, dt=100.0, T_init=263.15, k_frozen=k, k_unfrozen=k, c_frozen=1e5,
                        c_unfrozen=1e5, conduction_form=form)
        sim.advance(12000, T_top=268.15, T_bottom=258.15)
        errs[form] = np.abs(sim.temperature()[0] - exact).max()
    assert errs["flux"] < 1e-3
    assert errs["nodal"] > 0.3


# --- 6. ERA5 flux shift ---------------------------------------------------------------------------------
def test_era5_flux_shift(tmp_path):
    t = pd.date_range("2016-01-01", periods=6, freq="h")
    df = pd.DataFrame({"time": t, "2m_temperature": np.arange(6.0),
                       "mean_surface_latent_heat_flux": 10.0 * np.arange(6),
                       "mean_surface_net_short_wave_radiation_flux": 100.0 + np.arange(6)})
    f = tmp_path / "era5.csv"
    df.to_csv(f, index=False)
    sim = Simulation.__new__(Simulation)
    t0, t1 = pd.Timestamp("2016-01-01 01:00"), pd.Timestamp("2016-01-01 04:00")
    legacy = sim._get_timeseries(t0, t1, f)
    shifted = sim._get_timeseries(t0, t1, f, flux_shift_hours=1)
    np.testing.assert_array_equal(legacy["mean_surface_latent_heat_flux"], [10, 20, 30])  # no shift: label t -> row t
    np.testing.assert_array_equal(shifted["mean_surface_latent_heat_flux"], [20, 30, 40])  # row t gets record labelled t+1h
    np.testing.assert_array_equal(shifted["2m_temperature"], legacy["2m_temperature"])  # instantaneous fields untouched
    last = sim._get_timeseries(pd.Timestamp("2016-01-01 04:00"), pd.Timestamp("2016-01-01 06:00"), f, flux_shift_hours=1)
    np.testing.assert_array_equal(last["mean_surface_latent_heat_flux"], [50, 50])  # last record held, not NaN


# --- 7. explicit stability of a wet surface node --------------------------------------------------------
def wet_surface_sim(dt, hc, T_init_C, wet=True):
    """Trial 490 surface values (k, C, dz = 10/199) in a ColumnSim; the wet attributes are set by hand, as the
    production _get_ghost_node_boundary_condition does. T_init_C < 0 gives a frozen, > 0 an unfrozen surface node."""
    sim = ColumnSim(depth=10.0, N=200, dt=dt, T_init=273.15 + T_init_C, k_frozen=1.94254, k_unfrozen=0.30034,
                    c_frozen=1.95696e6, c_unfrozen=1.95696e6 * 1.83791, nb_max=0.7, nb_min=0.48409,
                    conduction_form="flux", latent_heat_form="volumetric")
    sim.wet_mask = np.full(1, wet)
    sim.hc_water = hc
    return sim


@pytest.mark.parametrize("T_init_C, dt, hc, raises", [
    (-2.0, 180, 500.0, False),    # 1.97 < 2 (frozen)
    (-2.0, 360, 500.0, True),     # 3.94
    (-2.0, 180, 1210.0, True),    # about 4.5
    (-2.0, 60, 1210.0, False),    # about 1.5
    (3.0, 180, 500.0, False),     # 1.01 (unfrozen)
    (3.0, 360, 500.0, True),      # 2.02
    (3.0, 180, 1210.0, True),     # about 2.4 + conduction
    (3.0, 60, 1210.0, False),
])
def test_wet_surface_stability_check(T_init_C, dt, hc, raises):
    sim = wet_surface_sim(dt, hc, T_init_C)
    if raises:
        with pytest.raises(ValueError, match="wet surface node"):
            sim.thermal_update(0, 0)
    else:
        sim.thermal_update(0, 0)


def test_wet_surface_check_reports_stable_dt():
    sim = wet_surface_sim(360, 500.0, -2.0)
    with pytest.raises(ValueError) as err:
        sim.thermal_update(0, 0)
    coeff = float(re.search(r"= ([\d.]+) >= 2", str(err.value)).group(1))
    dt_max = float(re.search(r"<= ([\d.e+-]+) s", str(err.value)).group(1))
    assert coeff == pytest.approx(3.94, abs=0.01)
    assert dt_max == pytest.approx(2 * 360 / coeff, rel=1e-3)
    wet_surface_sim(dt_max * 0.99, 500.0, -2.0).thermal_update(0, 0)  # just below the limit passes


def test_wet_surface_check_skipped_when_dry_or_unset():
    wet_surface_sim(360, 1210.0, -2.0, wet=False).thermal_update(0, 0)  # dry node: no convective flux, no check
    sim = wet_surface_sim(360, 1210.0, -2.0)
    del sim.wet_mask, sim.hc_water  # a subclass that overrides the boundary condition never sets them
    sim.thermal_update(0, 0)
