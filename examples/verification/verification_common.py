"""Shared helpers for the thermal-solver verification scripts.

The verification cases drive the *real* Arctic-XBeach thermal solver (Simulation.thermal_update) on an
idealised 1D column. Only the surface boundary condition is replaced (the ERA5 energy balance is swapped for a
prescribed flux, or for an exact Dirichlet temperature that is re-imposed after each step), so everything else
(enthalpy formulation, phase masks, k of the in-between nodes, conduction scheme, latent heat form, Fourier check)
is exactly what the coupled model runs.

Contents:
    ColumnSim            thin Simulation subclass for a 1D column
    neumann_solution     analytical two-phase Stefan (Neumann) solution
    layered_steady       analytical steady profile of a layered column
    layered_reference    independent cell-centred finite-volume reference for the layered transient
"""

# === CONFIGURATION ===
SOLVER_VARIANTS = {  # name -> thermal options
    "legacy (nodal + per_mass)": dict(conduction_form="nodal", latent_heat_form="per_mass"),
    "flux + volumetric": dict(conduction_form="flux", latent_heat_form="volumetric"),
}
# === END CONFIGURATION ===

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, erfc
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply, spsolve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import arctic_xbeach.miscellaneous as um  # noqa: E402
from arctic_xbeach.model import Simulation  # noqa: E402


class AttrDict(dict):
    """dict with attribute access, same idea as the config object in Simulation.read_config."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class ColumnSim(Simulation):
    """1D thermal column driven through the production thermal_update().

    The surface boundary condition is a prescribed heat flux (`surface_flux`, W m-2 into the ground). The ghost
    node is built exactly like the production second-order construction, so the nodal scheme sees the same
    boundary treatment as in the coupled model. A Dirichlet temperature at the surface and/or bottom node can be
    imposed with `advance(..., T_top=..., T_bottom=...)` by resetting the enthalpy of that node after each step.
    """

    def __init__(self, *, depth, N, dt, T_init, k_frozen, k_unfrozen, c_frozen, c_unfrozen,
                 nb_max=0.4, nb_min=None, rho_soil="None", L_water_ice=334000.0, rho_water=1000.0,
                 rho_particle=2650.0, T_melt=273.15, conduction_form="nodal", latent_heat_form="per_mass",
                 geothermal_gradient=0.0, nx=1):
        nb_min = nb_max if nb_min is None else nb_min
        self.config = AttrDict(
            thermal=AttrDict(
                L_water_ice=L_water_ice, rho_water=rho_water, rho_particle=rho_particle, rho_soil=rho_soil,
                T_melt=T_melt, dt=dt, max_depth=depth, grid_resolution=N,
                c_soil_frozen=c_frozen, c_soil_unfrozen=c_unfrozen,
                k_soil_frozen_min=float(np.min(k_frozen)), k_soil_frozen_max=float(np.max(k_frozen)),
                k_soil_unfrozen_min=float(np.min(k_unfrozen)), k_soil_unfrozen_max=float(np.max(k_unfrozen)),
                depth_constant_k=1.0, nb_max=nb_max, nb_min=nb_min, nb_max_depth=0.5, nb_min_depth=1.5,
                geothermal_gradient=geothermal_gradient,
                conduction_form=conduction_form, latent_heat_form=latent_heat_form,
            ),
        )
        self.xgr = np.arange(nx, dtype=float)
        self.N = N
        self.dz = depth / (N - 1)
        self.thermal_zgr = np.linspace(0.0, depth, N)
        self.define_soil_property_matrices(nx, define_nb=True)
        # prescribed layered / uniform conductivity (overrides the linear-to-depth_constant_k profile of the model)
        self.k_frozen_matrix = np.tile(np.broadcast_to(k_frozen, (N,)), (nx, 1)).astype(float)
        self.k_unfrozen_matrix = np.tile(np.broadcast_to(k_unfrozen, (N,)), (nx, 1)).astype(float)
        self.A_matrix = um.get_A_matrix(N)
        self.temp_matrix = np.tile(np.broadcast_to(np.asarray(T_init, dtype=float), (N,)), (nx, 1))
        self.enthalpy_matrix = self._enthalpy_from_temperature(self.temp_matrix)
        self.surface_flux = np.zeros(nx)
        self.heat_flux = np.zeros(nx)
        self.time = 0.0
        self.flux_integral = np.zeros(nx)  # time integral of net boundary heat input [J m-2]

    # --- boundary condition used by thermal_update --------------------------------------------------------
    def _get_ghost_node_boundary_condition(self, timestep_id, subgrid_timestep_id):
        self.heat_flux = self.surface_flux.copy()
        return self.temp_matrix[:, 1] + 2 * self.dz * self.heat_flux / self.k_matrix[:, 0]

    # --- time stepping ------------------------------------------------------------------------------------
    def _dirichlet_enthalpy(self, T_value):
        return self._enthalpy_from_temperature(np.full(self.temp_matrix.shape, float(T_value)))

    def advance(self, n_steps, T_top=None, T_bottom=None, callback=None):
        """Take n_steps thermal steps of length config.thermal.dt. Optional Dirichlet values (K) at the end nodes."""
        dt = self.config.thermal.dt
        H_top = None if T_top is None else self._dirichlet_enthalpy(T_top)[:, 0]
        H_bot = None if T_bottom is None else self._dirichlet_enthalpy(T_bottom)[:, -1]
        for _ in range(int(n_steps)):
            self.thermal_update(0, 0)
            if H_top is not None:
                self.enthalpy_matrix[:, 0] = H_top
            if H_bot is not None:
                self.enthalpy_matrix[:, -1] = H_bot
            self.time += dt
            if callback is not None:
                callback(self)

    def advance_with_energy_balance(self, n_steps):
        """Flux-driven (no Dirichlet) stepping that also accumulates the boundary heat input actually applied."""
        dt = self.config.thermal.dt
        for _ in range(int(n_steps)):
            self.thermal_update(0, 0)
            # thermal_update has just used self.heat_flux (top) and k_bottom * gradient (bottom)
            self.flux_integral += dt * (self.heat_flux + self.k_matrix[:, -1] * self.config.thermal.geothermal_gradient)
            self.time += dt

    # --- diagnostics --------------------------------------------------------------------------------------
    def temperature(self):
        """Temperature [K] consistent with the current enthalpy, shape (nx, N)."""
        return self._temperature_from_enthalpy(*self._get_phase_masks())

    def liquid_fraction(self):
        """Melt fraction per node (0 frozen, 1 thawed), from the enthalpy relative to the melting band."""
        T_melt = self.config.thermal.T_melt
        H_solidus = self.Cs_matrix * T_melt
        return np.clip((self.enthalpy_matrix - H_solidus) / self.latent_matrix, 0.0, 1.0)

    def thaw_front(self):
        """Thaw depth [m] as the integrated melt fraction (half cell at the surface node), shape (nx,)."""
        phi = self.liquid_fraction()
        return self.dz * (phi[:, 1:].sum(axis=1) + 0.5 * phi[:, 0])

    def stored_energy(self):
        """Column enthalpy [J m-2] = sum_i w_i dz rho_i H_i with half-weight end nodes. For the flux
        scheme this is the quantity that balances the boundary fluxes exactly."""
        w = np.ones(self.N)
        w[0] = w[-1] = 0.5
        return (self.soil_density_matrix * self.enthalpy_matrix * w).sum(axis=1) * self.dz


# =======================================================================================================
# Analytical solutions
# =======================================================================================================
def neumann_solution(*, k_u, C_u, k_f, C_f, L_vol, T_surface, T_init, T_melt):
    """Two-phase Stefan (Neumann) solution for a frozen half-space (T_init < T_melt) with the surface held at
    T_surface > T_melt. k [W m-1 K-1], C = volumetric heat capacity [J m-3 K-1], L_vol = volumetric latent heat
    [J m-3]. Returns an object with lam, s(t) and T(z, t)."""
    a_u, a_f = k_u / C_u, k_f / C_f
    dTs, dT0 = T_surface - T_melt, T_melt - T_init

    def residual(lam):
        mu = lam * np.sqrt(a_u / a_f)
        thawed_side = k_u * dTs * np.exp(-lam**2) / (erf(lam) * np.sqrt(np.pi * a_u))
        frozen_side = k_f * dT0 * np.exp(-mu**2) / (erfc(mu) * np.sqrt(np.pi * a_f))
        return thawed_side - frozen_side - L_vol * lam * np.sqrt(a_u)

    lam = brentq(residual, 1e-8, 5.0, xtol=1e-14)

    class Solution:
        pass

    sol = Solution()
    sol.lam, sol.a_u, sol.a_f = lam, a_u, a_f
    sol.s = lambda t: 2.0 * lam * np.sqrt(a_u * np.asarray(t, dtype=float))

    def temperature(z, t):
        z = np.asarray(z, dtype=float)
        s = sol.s(t)
        mu = lam * np.sqrt(a_u / a_f)
        T_thawed = T_surface + (T_melt - T_surface) * erf(z / (2 * np.sqrt(a_u * t))) / erf(lam)
        T_frozen = T_init + (T_melt - T_init) * erfc(z / (2 * np.sqrt(a_f * t))) / erfc(mu)
        return np.where(z <= s, T_thawed, T_frozen)

    sol.T = temperature
    return sol


def layer_index(z, interfaces):
    """Layer number (0, 1, ...) of each depth z for the sorted list of interface depths."""
    return np.searchsorted(np.asarray(interfaces), z, side="right")


def layered_steady(z, interfaces, k_layers, T_top, T_bottom):
    """Steady piecewise-linear temperature of a layered column with fixed end temperatures."""
    z = np.asarray(z, dtype=float)
    edges = np.concatenate(([0.0], interfaces, [z.max() if z.max() > interfaces[-1] else interfaces[-1] + 1.0]))
    edges[-1] = max(edges[-1], z.max())
    thick = np.diff(edges)
    R_layers = thick / np.asarray(k_layers)
    q = (T_top - T_bottom) / R_layers.sum()  # downward conductive flux [W m-2]
    T = np.empty_like(z)
    for i, zi in enumerate(z):
        T[i] = T_top
        for j in range(len(thick)):
            seg = min(max(zi - edges[j], 0.0), thick[j])
            T[i] -= q * seg / k_layers[j]
    return T


def layered_reference(z_eval, interfaces, k_layers, C, T_top, T_bottom, T_init, t, depth, n_cells=1600):
    """Independent reference for the layered transient: cell-centred finite volumes on a fine grid with
    exact flux continuity at the layer interfaces (which sit on cell faces), integrated in time with the matrix
    exponential (no time-stepping error). Dirichlet temperatures at both ends (half-cell conductance)."""
    h = depth / n_cells
    zc = (np.arange(n_cells) + 0.5) * h
    k = np.asarray(k_layers)[layer_index(zc, interfaces)]
    k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])  # harmonic mean = exact series resistance for symmetric cells
    g = np.concatenate(([k[0] / (h / 2)], k_face / h, [k[-1] / (h / 2)]))  # face conductances [W m-2 K-1]
    main = -(g[:-1] + g[1:]) / (C * h)
    off = g[1:-1] / (C * h)
    A = diags([off, main, off], [-1, 0, 1], format="csc")
    b = np.zeros(n_cells)
    b[0] = g[0] * T_top / (C * h)
    b[-1] = g[-1] * T_bottom / (C * h)
    T_ss = spsolve(A, -b)
    T0 = np.full(n_cells, float(T_init))
    T_t = T_ss + expm_multiply(A * t, T0 - T_ss)
    return np.interp(z_eval, zc, T_t)


# =======================================================================================================
# The Stefan (two-phase melting) test case, shared by the Stefan and convergence scripts
# =======================================================================================================
STEFAN = dict(
    depth=8.0,            # [m] column depth (acts as semi-infinite: frozen-zone signal erfc(8/(2*1.6)) ~ 5e-4)
    t_end=30 * 86400.0,   # [s]
    T_melt=273.15,        # [K]
    T_surface=283.15,     # [K] surface stepped to +10 K above melting
    T_init=268.15,        # [K] initially frozen at -5 K
    k_frozen=2.0, k_unfrozen=1.0,          # [W m-1 K-1]
    c_frozen=2.0e6, c_unfrozen=3.0e6,      # [J m-3 K-1]
    nb=0.4, L_water_ice=334000.0, rho_water=1000.0, rho_particle=2650.0,  # rho_soil = nb*rho_w + (1-nb)*rho_p = 1990
)


def stefan_exact(p=STEFAN):
    """Neumann solution for the STEFAN case with the volumetric latent heat rho_w * L * nb."""
    return neumann_solution(
        k_u=p["k_unfrozen"], C_u=p["c_unfrozen"], k_f=p["k_frozen"], C_f=p["c_frozen"],
        L_vol=p["rho_water"] * p["L_water_ice"] * p["nb"],
        T_surface=p["T_surface"], T_init=p["T_init"], T_melt=p["T_melt"],
    )


def run_stefan(N, dt, options, p=STEFAN, n_record=60):
    """Run the Stefan case with the production solver. Returns (times, fronts, z, T_final, sim); the front is
    recorded n_record times (equally spaced in time, last = t_end)."""
    sim = ColumnSim(
        depth=p["depth"], N=N, dt=dt, T_init=p["T_init"],
        k_frozen=p["k_frozen"], k_unfrozen=p["k_unfrozen"], c_frozen=p["c_frozen"], c_unfrozen=p["c_unfrozen"],
        nb_max=p["nb"], L_water_ice=p["L_water_ice"], rho_water=p["rho_water"], rho_particle=p["rho_particle"],
        T_melt=p["T_melt"], **options,
    )
    n_steps = int(round(p["t_end"] / dt))
    if abs(n_steps * dt - p["t_end"]) > 1e-6:
        raise ValueError(f"dt={dt} does not divide t_end={p['t_end']}")
    record_at = set(np.unique(np.round(np.linspace(n_steps / n_record, n_steps, n_record)).astype(int)))
    times, fronts = [], []
    step = {"i": 0}

    def cb(s):
        step["i"] += 1
        if step["i"] in record_at:
            times.append(s.time)
            fronts.append(s.thaw_front()[0])

    sim.advance(n_steps, T_top=p["T_surface"], T_bottom=p["T_init"], callback=cb)
    return np.array(times), np.array(fronts), sim.thermal_zgr, sim.temperature()[0], sim


def stable_dt(N, fourier=0.25, p=STEFAN):
    """dt [s] giving the requested Fourier number (alpha_max dt / dz^2) for N nodes, rounded so it divides t_end."""
    dz = p["depth"] / (N - 1)
    alpha_max = max(p["k_frozen"] / p["c_frozen"], p["k_unfrozen"] / p["c_unfrozen"])
    dt = fourier * dz**2 / alpha_max
    n_steps = int(np.ceil(p["t_end"] / dt))
    return p["t_end"] / n_steps
