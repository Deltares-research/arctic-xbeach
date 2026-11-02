"""Helpers to build a small thermal-only run folder with synthetic forcing, for the restart and diagnostics tests.

The run is a 7-node transect (some nodes dry, some wet) with a 4 m deep thermal grid of 40 nodes, the published thermal
parameter set, and a synthetic spring warming so that the active layer actually thaws during the spin-up.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# dt = 300 s: the wet surface node (hc 500, dz 0.1 m) needs 2 Fo + 2 hc dt / (C dz) < 2, which fails at 600 s
THERMAL = dict(
    L_water_ice=334000, MSL=0, N_thaw_threshold=3, T_melt=273.15, angle_max=89, angle_min=-89,
    c_soil_frozen=1574265.3175829535, c_soil_unfrozen=3103885.0192965684, delta_angle=1,
    depth_constant_k=1.395528377852392, dt=300, geothermal_gradient=0.020426871031478094,
    grid_interpolation="linear_interp_z", grid_resolution=40, hc_guess=500,
    k_soil_frozen_max=1.7305832283149918, k_soil_frozen_min=0.9046457783637623,
    k_soil_unfrozen_max=0.6305668303224798, k_soil_unfrozen_min=0.3296227605203751,
    max_depth=4, nb_max=0.7101082595355153, nb_max_depth=0.41887032171820293,
    nb_min=0.46951231603096133, nb_min_depth=2.372186433986581,
    rho_ice=917, rho_particle=2650, rho_soil="None", rho_water=1000,
    surface_flux_angle=90, surface_flux_factor=1, t_end="2011-01-01", t_start="2010-01-01",
    with_convective_transport_water_guess=True, with_latent=True, with_longwave=True, with_solar=True,
    with_solar_flux_calculator=False,
    conduction_form="flux", latent_heat_form="volumetric",
)


def write_forcing(run_dir: Path, start="2010-04-01", end="2010-08-01") -> tuple[Path, Path]:
    """Synthetic hourly ERA5-like forcing and storm table. Warming of 15 K over two months plus a daily cycle."""
    t = pd.date_range(start, end, freq="1h")
    day = np.arange(len(t)) / 24.0
    hour = np.arange(len(t)) % 24
    daily = np.sin(2 * np.pi * (hour - 9) / 24)
    n = len(t)
    forcing = pd.DataFrame(dict(
        time=t,
        **{"2m_temperature": 265.0 + 15.0 * np.clip(day / 90.0, 0, 1) + 3.0 * daily,
           "sea_surface_temperature": 272.0 + 6.0 * np.clip(day / 90.0, 0, 1),
           "sea_ice_cover": np.clip(0.9 - day / 60.0, 0.0, 1.0),
           "10m_u_component_of_wind": 3.0 + np.sin(day), "10m_v_component_of_wind": 2.0 + np.cos(day),
           "soil_temperature_level_1": np.full(n, 262.0), "soil_temperature_level_2": np.full(n, 263.0),
           "soil_temperature_level_3": np.full(n, 264.0), "soil_temperature_level_4": np.full(n, 265.0),
           "mean_surface_latent_heat_flux": -10.0 - 5.0 * daily,
           "mean_surface_net_long_wave_radiation_flux": -40.0 + 5.0 * np.sin(day),
           "mean_surface_net_short_wave_radiation_flux": 150.0 + 100.0 * daily}))
    storms = pd.DataFrame(dict(
        time=t, **{"WL(m)": 0.1 * np.sin(day), "Hs(m)": 0.5 + 0.3 * np.sin(day / 3), "Dp(deg)": 270.0, "Tp(s)": 6.0}))
    f, s = run_dir / "forcing.csv", run_dir / "storms.csv"
    forcing.to_csv(f, index=False)
    storms.to_csv(s, index=False)
    return f, s


def build_run(run_dir: Path, time_start="01-05-2010", time_end="04-07-2010", spin_up_days=61, thermal=None, **model) -> Path:
    """Write config.yaml, grid files and forcing of a thermal-only run into run_dir and return run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    forcing, storms = write_forcing(run_dir)
    x = np.linspace(0.0, 30.0, 7)
    z = np.linspace(3.0, -1.0, 7)  # dry on the left, wet on the right
    np.savetxt(run_dir / "x.grd", x)
    np.savetxt(run_dir / "bed.dep", z)
    np.savetxt(run_dir / "ne_layer.txt", np.zeros(7))
    cfg = dict(
        data=dict(forcing_data_path=str(forcing), storm_data_path=str(storms)),
        model=dict(longitude=-143.7, latitude=70.13, time_start=time_start, time_end=time_end, timestep=1,
                   call_xbeach_inter=168, time_zone_diff=-8, spin_up_time=spin_up_days / 365.25, **model),
        wrapper=dict(sea_ice_threshold=0.85, xb_threshold=2.0, CFL_thermal=0.5, CFL_xbeach=0.7, N_thaw_threshold=3),
        bathymetry=dict(xfile="x.grd", depfile="bed.dep", grid_orientation=88.14, ppwl=40, nx=7, with_nx=True,
                        with_schematized_bathymetry=False),
        xbeach=dict(with_xbeach=False, version="none", with_wind=False, with_ne_layer=True, thetamin=0, thetamax=360,
                    dtheta=360, thetanaut=1, wbctype="jons", bcfile="jonswap.txt", rho_sea_water=1025, dryslp=0.6,
                    wetslp=0.1, D50=3.6e-05, rho_solid=2650, bedfriccoef=0.02, lsgrad=0.0, facSk=0.0, facAs=0.0, facua=0.0),
        thermal=dict(THERMAL, **(thermal or {})),
        output=dict(output_res=24, use_default_output_path=True, output_path="none"),
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir


def reset_logger():
    """main.py attaches its file handler to a module-level logger only once; drop it so each run gets its own run.log."""
    log = logging.getLogger("thermo_model")
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


def run_main(run_dir: Path, repo_root: Path) -> dict:
    """Run main.main on run_dir and return copies of the final thermal state (cleanup() would delete it)."""
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from main import main
    from arctic_xbeach.model import Simulation

    reset_logger()
    cwd = os.getcwd()
    try:
        sim = Simulation(str(run_dir), proj_dir=str(repo_root))
        final = {}
        real_cleanup = sim.cleanup

        def capture():
            for name in ("enthalpy_matrix", "temp_matrix", "thaw_depth", "ghost_nodes_temperature", "heat_flux", "xgr", "zgr"):
                final[name] = np.array(getattr(sim, name), copy=True)
            final["diag"] = sim.diag
            real_cleanup()

        sim.cleanup = capture
        main(sim, print_to_screen=False)
    finally:
        reset_logger()
        os.chdir(cwd)
    return final
