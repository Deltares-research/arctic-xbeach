"""One fixed thermal scenario used to check that the default options reproduce v1.0 (commit c6ab74b) exactly.
It exercises depth-varying k, depth-varying nb (hence rho_soil), melting, a surface flux and a geothermal gradient."""
import numpy as np

SCENARIO = dict(
    depth=3.0, N=61, dt=120.0, n_steps=800,
    k_frozen=np.linspace(1.2, 2.4, 61), k_unfrozen=np.linspace(0.5, 1.0, 61),
    c_frozen=1.8e6, c_unfrozen=3.2e6, nb_max=0.6, nb_min=0.4,
    L_water_ice=334000.0, rho_water=1000.0, rho_particle=2650.0, T_melt=273.15,
    geothermal_gradient=0.03, surface_flux=40.0,
)


def initial_temperature(N=61):
    """Frozen column with a thawed layer in the top 0.3 m (K)."""
    z = np.linspace(0, SCENARIO["depth"], N)
    return np.where(z < 0.3, 275.0, 268.0 - 1.5 * z)
