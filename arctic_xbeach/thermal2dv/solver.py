"""Implicit enthalpy-salinity solver for the coupled water-soil-ice 2DV thermal model.

Each cell holds volume fractions (mineral phi_s, total H2O phi_h2o, air), volumetric
enthalpy H and salt content. Sea water, sea ice, pore ice and ice wedges all share one
heat equation

    V dH/dt = sum_faces G_f (T_nb - T) + Q_sw + Q_air + Q_geo + V gamma C (T_ref - T)

discretised with backward Euler. The H-T relation (materials.py: sharp for fresh H2O, a
salt-exclusion mushy curve for saline H2O) is linearised per iteration (Newton, cf.
Swaminathan & Voller 1993), so each step converges to an energy-conserving solution
without a Fourier-number limit.

Salt is a second conserved quantity carried by the liquid only. After each heat step it is
diffused implicitly (turbulent diffusivity in water, molecular diffusivity x tortuosity in
pore water and brine) and relaxed towards S_ref in the same sponge / flushing as heat:

    V d(theta_w S)/dt = sum_faces G_s (S_nb - S) + V gamma (S_ref - S)

Brine drains from sea ice into the water below it (conceptual gravity drainage), so a
freezing sea rejects salt downward.

Water cells use a turbulent eddy diffusivity instead of molecular conduction. A face
between water and soil or ice carries an extra boundary-layer resistance 1/h, so thermal
abrasion of a bluff, a block or the seabed follows from the same face flux.

Melting faces: when warm water touches frozen material, the face itself sits at the
melting point and receives h (T_w - T_f). A plain series resistance through the half cell
to the cell centre would cap this flux at k / (dx/2) while the cell is still below T_f,
which underestimates melt rates of cold ice-rich bluffs by a factor of ~2 at dx = 0.1 m.
Such faces are therefore "clamped": the flux h (T_w - T_f) goes into the solid cell (as
sensible heat first, then latent heat), and energy stays conserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from . import materials as mat
from .grid import Grid


@dataclass
class Settings:
    # --- heat exchange with water, air and the deep ground
    h_water: float = 500.0            # water -> soil/ice heat transfer coefficient [W m-2 K-1] (v1.0 hc_guess)
    q_geo: float = 0.06               # geothermal heat flux into the base [W m-2] (North Slope 66 +/- 14 mW m-2)
    c_h: float = 1.3e-3               # bulk sensible-heat transfer coefficient (Delft3D heat model 5 default)
    rho_air: float = 1.3              # air density [kg m-3]
    cp_air: float = 1005.0            # air heat capacity [J kg-1 K-1]
    emissivity: float = 0.97          # surface emissivity for longwave emission and absorption
    albedo_water: float = 0.07
    albedo_ice: float = 0.6
    sw_beta: float = 0.55             # fraction of the shortwave absorbed in the top water cell (near infrared)
    kd: float = 0.7                   # attenuation of the penetrating shortwave [m-1] (Stefansson Sound, calm)
    sw_vertical_factor: float = 0.5   # shortwave on a vertical face relative to a horizontal one
    # --- water column and salt
    S_sea: float = 30.0               # default sea salinity [psu], used when the forcing gives no S_ref
    Kx_surf: float = 1.0              # horizontal eddy diffusivity in shallow water [m2 s-1] (Clark et al. 2010)
    Kz_surf: float = 1e-2             # vertical eddy diffusivity in shallow water [m2 s-1]
    Kx_off: float = 0.3               # horizontal eddy diffusivity offshore [m2 s-1]
    Kz_off: float = 1e-3              # vertical eddy diffusivity offshore [m2 s-1]
    K_under_ice: float = 1e-4         # vertical eddy diffusivity under an ice cover [m2 s-1]
    d_surf: float = 2.0               # water depth below which the shallow-water values apply [m]
    n_sponge: int = 5                 # number of offshore columns relaxed towards T_ref / S_ref
    tau_sponge: float = 3 * 3600.0    # relaxation time in the sponge [s]
    tau_flush: float = 2 * 86400.0    # alongshore flushing time elsewhere in the water [s]; 0 switches it off
    salt_transport: bool = True       # diffuse and relax salt; False keeps each cell's salt content fixed
    tortuosity: float = 0.5           # tortuosity factor on molecular salt diffusion in pores and brine [-]
    brine_drainage: bool = True       # drain brine from sea ice into the water below
    tau_brine: float = 10 * 86400.0   # e-folding time of brine drainage [s]
    S_ice_min: float = 5.0            # bulk salinity that drained sea ice tends to [psu] (first-year ice ~4-8)
    # --- cell classes and erosion
    open_threshold: float = 0.1       # cells with phi_s below this are open (water, ice or air)
    fluid_threshold: float = 0.999    # open cells with at least this liquid fraction mix turbulently; below it
                                      # they are ice, so the whole latent heat of a melting cell passes the 1/h layer
    erodible_ice_threshold: float = 0.5  # soil with phi_h2o at or above this is removed once thawed and wet
    erosion: bool = True              # remove thawed, ice-rich soil cells that touch water
    collapse: bool = True             # drop overhangs once the niche is deeper than d_crit
    d_crit: float = 1.5               # critical niche depth for block failure [m]
    collapse_ice: bool = False        # let ice overhangs fail too (ice bluffs, lab blocks); off by default
                                      # so that floating sea ice is never taken for an overhang
    # --- numerics
    tol_H: float = 10.0               # convergence tolerance on the enthalpy update [J m-3]
    max_iter: int = 40
    max_halvings: int = 6             # time-step halvings allowed when an iteration does not converge


def _face_terms(P, N, d_p, d_n, k_p, k_n, area, p, H, h):
    """Conductance of the faces between cells P and N (index tuples), and which of them are
    clamped melting faces (see module docstring).

    Returns G (series conductance), the clamp mask, which side is solid, and the solid's
    (bulk) freezing point."""
    r_p = 0.5 * d_p / k_p
    r_n = 0.5 * d_n / k_n
    fl_p, fl_n = p.fluid[P], p.fluid[N]
    interface = fl_p ^ fl_n
    G = area / (r_p + r_n + np.where(interface, 1.0 / h, 0.0))

    solid_is_p = interface & ~fl_p
    T_s = np.where(solid_is_p, p.T[P], p.T[N])
    T_w = np.where(solid_is_p, p.T[N], p.T[P])
    Tf_s = np.where(solid_is_p, p.Tf[P], p.Tf[N])
    below_liquidus = np.where(solid_is_p, H[P] < p.H_liq[P], H[N] < p.H_liq[N])
    r_s = np.where(solid_is_p, r_p, r_n)
    T_face = T_s + (T_w - T_s) * r_s / (r_s + 1.0 / h)
    clamp = interface & below_liquidus & (T_w > Tf_s) & (T_face >= Tf_s)
    return G, clamp, solid_is_p, Tf_s


class Model2DV:
    """State and time stepping of the 2DV thermal model."""

    def __init__(self, grid: Grid, settings: Settings | None = None):
        self.grid = grid
        self.s = settings or Settings()
        self.phi_s = np.zeros(grid.shape)
        self.phi_h2o = np.zeros(grid.shape)
        self.salt = np.zeros(grid.shape)   # phi_h2o x bulk salinity [psu]
        self.H = np.zeros(grid.shape)
        self.time = 0.0
        # cumulative exchange with the outside world [J, and psu m3, per m alongshore]
        self.energy = dict(boundary=0.0, wetdry=0.0, erosion=0.0, collapse=0.0)
        self.salt_exchange = dict(relaxation=0.0, wetdry=0.0, erosion=0.0, collapse=0.0)
        self.eroded_sediment = 0.0  # [m3 per m alongshore]
        self.collapse_events = []
        self.last_step = {}

    # ------------------------------------------------------------------ state
    def state(self):
        """Derived fields: freezing point, capacities, temperature, liquid content, salinity and cell classes."""
        s = self.s
        c = mat.closure(self.phi_s, self.phi_h2o, self.salt)
        active = (self.phi_s + self.phi_h2o) > 1e-9
        T, _, theta_w = mat.temperature_from_enthalpy(self.H, c)
        T = np.where(active, T, np.nan)
        f_liq = mat.liquid_fraction(theta_w, self.phi_h2o)
        with np.errstate(divide="ignore", invalid="ignore"):
            S_liq = np.where(active & (theta_w > 1e-9), self.salt / theta_w, np.nan)
        soil = active & (self.phi_s >= s.open_threshold)
        open_ = active & ~soil
        fluid = open_ & (f_liq >= s.fluid_threshold)
        return SimpleNamespace(Tf=c.Tf, H_liq=c.H_liq, cf=c.c_frozen, cu=c.c_liquid, lv=c.lv, T=T,
                               theta_w=theta_w, f_liq=f_liq, S_bulk=c.S_bulk, S_liq=S_liq,
                               active=active, soil=soil, open=open_, fluid=fluid)

    @property
    def S_bulk(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.phi_h2o > 0.0, self.salt / self.phi_h2o, 0.0)

    def set_salinity(self, S, mask=None):
        """Set the bulk salinity [psu] of the H2O in all cells, or in the cells of `mask`.
        Call after the fractions are set and before set_temperature()."""
        S = np.broadcast_to(np.asarray(S, dtype=float), self.grid.shape)
        mask = np.ones(self.grid.shape, dtype=bool) if mask is None else mask
        self.salt[mask] = (self.phi_h2o * S)[mask]

    def set_temperature(self, T, frozen_at_tf=True):
        """Initialise the enthalpy from a temperature field (fresh cells at exactly 0 degC start frozen)."""
        c = mat.closure(self.phi_s, self.phi_h2o, self.salt)
        T = np.asarray(T, dtype=float) + np.zeros(self.grid.shape)
        H = mat.enthalpy_from_temperature(T, c, frozen_at_tf)
        active = (self.phi_s + self.phi_h2o) > 1e-9
        self.H = np.where(active, H, 0.0)

    def total_energy(self):
        return float(np.sum(self.grid.volume * self.H))

    def total_salt(self):
        return float(np.sum(self.grid.volume * self.salt))

    # ------------------------------------------------------------- water level
    def update_wet_dry(self, eta, T_ref, S_ref=None):
        """Flood open cells below the water level and drain liquid cells above it."""
        s, g = self.s, self.grid
        p = self.state()
        zc = g.zc[:, None]
        V = g.volume
        S_ref = np.full(g.nz, s.S_sea) if S_ref is None else np.asarray(S_ref, dtype=float)

        drain = p.fluid & (zc >= eta)
        flood = ~p.active & (self.phi_s < s.open_threshold) & (zc < eta)

        e_removed = np.sum(V[drain] * self.H[drain])
        salt_removed = np.sum(V[drain] * self.salt[drain])
        self.phi_h2o[drain] = 0.0
        self.salt[drain] = 0.0
        self.H[drain] = 0.0

        if np.any(flood):
            self.phi_h2o[flood] = 1.0 - self.phi_s[flood]
            self.salt[flood] = (self.phi_h2o * S_ref[:, None])[flood]
            c = mat.closure(self.phi_s, self.phi_h2o, self.salt)
            T_new = np.maximum(np.broadcast_to(np.asarray(T_ref, dtype=float)[:, None], g.shape), c.Tf + 0.01)
            self.H[flood] = (c.c_liquid * T_new)[flood]
        self.energy["wetdry"] += np.sum(V[flood] * self.H[flood]) - e_removed
        self.salt_exchange["wetdry"] += np.sum(V[flood] * self.salt[flood]) - salt_removed

    # ------------------------------------------------------------ properties
    def _turbulence(self, p):
        """Eddy diffusivities per column: shallow (surf) or offshore values, damped under ice."""
        s, g = self.s, self.grid
        depth = (p.fluid * g.dz[:, None]).sum(axis=0)
        ice_covered = (p.open & ~p.fluid).any(axis=0)
        Kx = np.where(depth < s.d_surf, s.Kx_surf, s.Kx_off)
        Kz = np.where(depth < s.d_surf, s.Kz_surf, s.Kz_off)
        Kz = np.where(ice_covered, s.K_under_ice, Kz)
        return Kx, Kz

    def _relaxation_rate(self):
        """Relaxation rate [s-1] per column: sponge at the offshore boundary, flushing elsewhere."""
        s = self.s
        gamma = np.zeros(self.grid.nx)
        if s.tau_flush > 0:
            gamma[:] = 1.0 / s.tau_flush
        gamma[: s.n_sponge] = 1.0 / s.tau_sponge
        return gamma

    def _conductivities(self, p):
        Kx, Kz = self._turbulence(p)
        k_soil = mat.soil_conductivity(self.phi_s, self.phi_h2o, p.f_liq)
        k_open = p.f_liq * mat.K_WATER + (1.0 - p.f_liq) * mat.K_ICE
        # only liquid water mixes turbulently; ice and slush conduct molecularly
        turb = p.fluid * mat.C_WATER * self.phi_h2o
        kx = np.where(p.soil, k_soil, k_open + turb * Kx[None, :])
        kz = np.where(p.soil, k_soil, k_open + turb * Kz[None, :])
        return kx, kz

    def _assemble(self, p):
        """Conductance matrix over active cells (symmetric, except for clamped melting faces),
        plus exposed air-face areas and half-cell resistances per cell."""
        s, g = self.s, self.grid
        act = p.active
        n = int(act.sum())
        idx = -np.ones(g.shape, dtype=int)
        idx[act] = np.arange(n)
        kx, kz = self._conductivities(p)

        rows, cols, vals = [], [], []
        a_face = np.zeros(n)
        for P, N, d_p, d_n, k, area in (
            # faces between horizontal neighbours
            *[((jj, ii), (jj, ii + 1), g.dx[ii], g.dx[ii + 1], kx, g.dz[jj])
              for jj, ii in [np.nonzero(act[:, :-1] & act[:, 1:])]],
            # faces between vertical neighbours
            *[((jj, ii), (jj + 1, ii), g.dz[jj], g.dz[jj + 1], kz, g.dx[ii])
              for jj, ii in [np.nonzero(act[:-1, :] & act[1:, :])]],
        ):
            G, clamp, solid_is_p, Tf_s = _face_terms(P, N, d_p, d_n, k[P], k[N], area, p, self.H, s.h_water)
            a, b = idx[P], idx[N]
            # ordinary faces: symmetric conductance
            o = ~clamp
            rows += [a[o], b[o], a[o], b[o]]
            cols += [a[o], b[o], b[o], a[o]]
            vals += [G[o], G[o], -G[o], -G[o]]
            # clamped melting faces: flux h A (T_w - Tf) from the water cell into the solid cell
            Gc = (area * s.h_water)[clamp]
            i_s = np.where(solid_is_p, a, b)[clamp]
            i_w = np.where(solid_is_p, b, a)[clamp]
            rows += [i_s, i_w]
            cols += [i_w, i_w]
            vals += [-Gc, Gc]
            np.add.at(a_face, i_s, -Gc * Tf_s[clamp])
            np.add.at(a_face, i_w, Gc * Tf_s[clamp])
        L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()

        # faces exposed to the atmosphere: an inactive (air) neighbour or the top of the domain
        top = np.zeros(g.shape, dtype=bool)
        top[:-1, :] = act[:-1, :] & ~act[1:, :]
        top[-1, :] = act[-1, :]
        side = np.zeros(g.shape)
        side[:, 1:] += act[:, 1:] & ~act[:, :-1]
        side[:, :-1] += act[:, :-1] & ~act[:, 1:]
        exposure = SimpleNamespace(
            A_top=top * g.dx[None, :], r_top=0.5 * g.dz[:, None] / kz,     # area and half-cell resistance
            A_side=side * g.dz[:, None], r_side=0.5 * g.dx[None, :] / kx,
            a_face=a_face,                                                 # constant part of clamped-face fluxes
        )
        return idx, L, exposure

    def _shortwave(self, p, F):
        """Absorbed shortwave: per unit area at the top and side faces of soil and ice, and per
        cell [W per m alongshore] in the water column (two-band Beer-Lambert, with the light
        reaching the bed absorbed by the bed cell)."""
        s, g = self.s, self.grid
        q_vol = np.zeros(g.shape)
        sw_top = np.zeros(g.shape)
        sw_side = np.zeros(g.shape)
        if F.sw_down <= 0.0:
            return q_vol, sw_top, sw_side
        albedo = np.where(p.soil, F.albedo_land, np.where(p.fluid, s.albedo_water, s.albedo_ice))
        sw_side = np.where(p.fluid, 0.0, F.sw_down * s.sw_vertical_factor * (1.0 - albedo))
        for i in range(g.nx):
            js = np.nonzero(p.active[:, i])[0]
            if len(js) == 0:
                continue
            jt = js.max()
            if not p.fluid[jt, i]:
                sw_top[jt, i] = (1.0 - albedo[jt, i]) * F.sw_down
                continue
            I0 = (1.0 - albedo[jt, i]) * F.sw_down * g.dx[i]
            q_vol[jt, i] += s.sw_beta * I0
            pen = (1.0 - s.sw_beta) * I0
            depth, j = 0.0, jt
            while j >= 0 and p.fluid[j, i]:
                d0, depth = depth, depth + g.dz[j]
                q_vol[j, i] += pen * (np.exp(-s.kd * d0) - np.exp(-s.kd * depth))
                j -= 1
            if j >= 0 and p.active[j, i]:
                q_vol[j, i] += pen * np.exp(-s.kd * depth)
        return q_vol, sw_top, sw_side

    def _surface_exchange(self, T, faces, F):
        """Linearised surface energy balance of the air-exposed faces of each cell.

        Per unit area the balance is q = a' - b Ts, with Ts the surface temperature. Conduction
        from the surface to the cell centre over the half-cell resistance r gives the flux
        into the cell, Q = A (a' - b T) / (1 + b r). Returns g_ext, a_ext (Q = a_ext - g_ext T)
        and the updated surface temperatures."""
        s = self.s
        eps_sigma = s.emissivity * mat.SIGMA
        h_s = s.rho_air * s.cp_air * s.c_h * F.u10 if F.h_air is None else F.h_air
        g_ext = np.zeros_like(T)
        a_ext = np.zeros_like(T)
        Ts_new = []
        for A, r, sw, Ts in faces:
            TK3 = (Ts + mat.T0K) ** 3
            b = h_s + 4.0 * eps_sigma * TK3
            a = sw + s.emissivity * F.lw_down + h_s * F.T_air + eps_sigma * TK3 * (3.0 * Ts - mat.T0K)
            g_ext += A * b / (1.0 + b * r)
            a_ext += A * a / (1.0 + b * r)
            Ts_new.append((a * r + T) / (1.0 + b * r))
        return g_ext, a_ext, Ts_new

    # ------------------------------------------------------------ time step
    def step(self, dt, forcing_fn):
        """Advance by dt [s]: water level, implicit heat step (halved if needed), meltwater
        mixing, salt transport, brine drainage, then erosion."""
        from .erosion import collapse_overhang, mix_meltwater, remove_thawed_face_cells

        F = forcing_fn(self.time + 0.5 * dt, self.grid.zc)
        self.update_wet_dry(F.eta, F.T_ref, F.S_ref)
        fluid_before = self.state().fluid
        self._advance(dt, forcing_fn, self.time, 0)
        mix_meltwater(self, fluid_before)
        if self.s.salt_transport:
            self._solve_salt(dt, F)
        if self.s.brine_drainage:
            self._drain_brine(dt)
        self.time += dt
        if self.s.erosion:
            remove_thawed_face_cells(self)
        if self.s.collapse:
            collapse_overhang(self)

    def _advance(self, dt, forcing_fn, t0, depth):
        F = forcing_fn(t0 + 0.5 * dt, self.grid.zc)
        if self._solve(dt, F):
            return
        if depth >= self.s.max_halvings:
            raise RuntimeError(f"Enthalpy iteration did not converge at t = {t0:.0f} s, even with dt = {dt:.1f} s")
        self._advance(0.5 * dt, forcing_fn, t0, depth + 1)
        self._advance(0.5 * dt, forcing_fn, t0 + 0.5 * dt, depth + 1)

    def _solve(self, dt, F):
        """One backward-Euler heat step. Returns False (state untouched) if the iteration fails.

        The cell classes are taken from the start of the step, except that water which freezes
        during the step must not keep mixing turbulently: otherwise a column on the freezing
        plateau passes latent heat through its eddy conductances and makes frazil all the way
        down. As soon as a Newton iterate freezes a fluid cell, the pass restarts with that
        cell non-fluid. Classes only ever lose fluid cells, so this cannot oscillate; it
        typically costs one extra pass when a water surface starts to freeze."""
        not_fluid = np.zeros(self.grid.shape, dtype=bool)
        while True:
            Hk, froze, info = self._solve_pass(dt, F, not_fluid)
            if Hk is not None:
                break
            if froze is None or not np.any(froze & ~not_fluid):
                return False
            not_fluid |= froze
        act = info["active"]
        self.energy["boundary"] += info["energy_in"]
        self.H[act] = Hk
        self.last_step = dict(iterations=info["iterations"], dt=dt, energy_in=info["energy_in"],
                              energy_change=info["energy_change"],
                              energy_residual=info["energy_change"] - info["energy_in"])
        return True

    def _solve_pass(self, dt, F, not_fluid):
        s, g = self.s, self.grid
        p = self.state()
        p.fluid = p.fluid & ~not_fluid
        idx, L, ex = self._assemble(p)
        act = p.active
        V = g.volume[act]
        H0 = self.H[act]
        c = mat.closure(self.phi_s[act], self.phi_h2o[act], self.salt[act])

        # fixed sources and linear exchanges: flux into a cell = a - g * T
        q_vol, sw_top, sw_side = self._shortwave(p, F)
        a_const = q_vol[act] + ex.a_face
        q_geo = np.zeros(g.shape)
        q_geo[0, :] = s.q_geo * g.dx
        a_const += q_geo[act]

        g_const = (p.fluid * g.volume * p.cu * self._relaxation_rate()[None, :])[act]
        T_ref = np.broadcast_to(np.asarray(F.T_ref, dtype=float)[:, None], g.shape)[act]
        a_const += g_const * T_ref

        T, D, _ = mat.temperature_from_enthalpy(H0, c)
        faces = [[ex.A_top[act], ex.r_top[act], sw_top[act] + F.q_surface, T.copy()],
                 [ex.A_side[act], ex.r_side[act], sw_side[act], T.copy()]]

        Hk = H0.copy()
        converged = False
        for it in range(s.max_iter):
            T, D, _ = mat.temperature_from_enthalpy(Hk, c)
            g_air, a_air, Ts = self._surface_exchange(T, faces, F)
            for face, ts in zip(faces, Ts):
                face[3] = ts
            M = L + sp.diags(g_const + g_air)
            A = sp.diags(V / dt) + M @ sp.diags(D)
            rhs = V / dt * H0 + a_const + a_air - M @ (T - D * Hk)
            H_new = splu(A.tocsc()).solve(rhs)
            _, _, th = mat.temperature_from_enthalpy(H_new, c)
            fz = p.fluid[act] & (mat.liquid_fraction(th, c.phi_h2o) < s.fluid_threshold)
            if np.any(fz):  # restart the pass with these cells non-fluid
                froze = np.zeros(g.shape, dtype=bool)
                froze[act] = fz
                return None, froze, None
            if it >= 10:  # under-relax cells that keep hopping across a plateau edge
                H_new = Hk + 0.5 * (H_new - Hk)
            dH = np.max(np.abs(H_new - Hk)) if len(Hk) else 0.0
            Hk = H_new
            if dH < s.tol_H:
                converged = True
                break
        if not converged:
            return None, None, None

        T, _, _ = mat.temperature_from_enthalpy(Hk, c)
        g_air, a_air, _ = self._surface_exchange(T, faces, F)
        e_in = dt * float(np.sum(a_const + a_air - (g_const + g_air) * T))
        e_change = float(np.sum(V * (Hk - H0)))
        return Hk, None, dict(active=act, iterations=it + 1, energy_in=e_in, energy_change=e_change)

    def _solve_salt(self, dt, F):
        """Implicit diffusion and relaxation of salt in all cells that hold liquid."""
        s, g = self.s, self.grid
        p = self.state()
        liq = p.active & (p.theta_w > 1e-6)
        n = int(liq.sum())
        if n == 0:
            return
        idx = -np.ones(g.shape, dtype=int)
        idx[liq] = np.arange(n)
        Kx, Kz = self._turbulence(p)
        D_pore = mat.D_SALT * s.tortuosity * p.theta_w          # molecular, in pore water and brine
        Dx = np.where(p.fluid, Kx[None, :] + mat.D_SALT, D_pore)
        Dz = np.where(p.fluid, Kz[None, :] + mat.D_SALT, D_pore)

        rows, cols, vals = [], [], []
        for P, N, d_p, d_n, Dc, area in (
            *[((jj, ii), (jj, ii + 1), g.dx[ii], g.dx[ii + 1], Dx, g.dz[jj])
              for jj, ii in [np.nonzero(liq[:, :-1] & liq[:, 1:])]],
            *[((jj, ii), (jj + 1, ii), g.dz[jj], g.dz[jj + 1], Dz, g.dx[ii])
              for jj, ii in [np.nonzero(liq[:-1, :] & liq[1:, :])]],
        ):
            G = area / (0.5 * d_p / Dc[P] + 0.5 * d_n / Dc[N])
            a, b = idx[P], idx[N]
            rows += [a, b, a, b]
            cols += [a, b, b, a]
            vals += [G, G, -G, -G]
        L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()

        V = g.volume[liq]
        theta = p.theta_w[liq]
        S_ref = np.full(g.nz, s.S_sea) if F.S_ref is None else np.asarray(F.S_ref, dtype=float)
        S_ref_c = np.broadcast_to(S_ref[:, None], g.shape)[liq]
        g_rel = (p.fluid * g.volume * self._relaxation_rate()[None, :])[liq]
        A = sp.diags(V * theta / dt + g_rel) + L
        rhs = V * self.salt[liq] / dt + g_rel * S_ref_c
        S_new = splu(A.tocsc()).solve(rhs)
        self.salt[liq] = theta * S_new
        self.salt_exchange["relaxation"] += dt * float(np.sum(g_rel * (S_ref_c - S_new)))

    def _drain_brine(self, dt):
        """Conceptual gravity drainage: brine leaves the sea ice (and any brine pockets in it)
        so that the ice bulk salinity decays towards S_ice_min with tau_brine. The dense brine
        sinks and mixes, so it is spread by volume over the connected water column under the
        ice; dumping it into the first cell below would keep that cell liquid while the
        fresher water beneath it froze, trapping hypersaline pockets in the ice."""
        s, g = self.s, self.grid
        p = self.state()
        ice = p.open & ~p.fluid
        if not np.any(ice):
            return
        V = g.volume
        keep = 1.0 - np.exp(-dt / s.tau_brine)
        for i in np.nonzero(ice.any(axis=0))[0]:
            fl = np.nonzero(p.fluid[:, i])[0]
            if len(fl) == 0:
                continue  # grounded (bottom-fast) ice: nowhere to drain
            j0 = fl.min()
            j1 = j0
            while j1 + 1 < g.nz and p.fluid[j1 + 1, i]:
                j1 += 1                                     # top of the water column under the ice
            j_ice = np.nonzero(ice[:, i])[0].max()
            if j_ice <= j1:
                continue                                    # ice only below the water (anchor ice): skip
            src_rows = np.arange(j1 + 1, j_ice + 1)
            src_rows = src_rows[p.open[src_rows, i]]
            excess = keep * np.maximum(self.salt[src_rows, i] - s.S_ice_min * self.phi_h2o[src_rows, i], 0.0)
            total = float(np.sum(excess * V[src_rows, i]))
            if total <= 0.0:
                continue
            self.salt[src_rows, i] -= excess
            run = np.arange(j0, j1 + 1)
            w = V[run, i] * self.phi_h2o[run, i]
            self.salt[run, i] += total * self.phi_h2o[run, i] / np.sum(w)  # same salinity rise in every cell

    # -------------------------------------------------------------- helpers
    def steady_state(self, T_air, T_water, frozen=True):
        """Steady conduction field (no phase change) for initialisation: air-exposed faces are
        held at T_air, water cells at T_water, and the base receives the geothermal flux."""
        s, g = self.s, self.grid
        p = self.state()
        p.fluid = p.open.copy()  # before initialisation H is meaningless: every open cell is water here
        p.f_liq = np.where(p.soil, 0.0 if frozen else 1.0, 1.0)
        p.T = np.where(p.fluid, T_water, T_air)
        idx, L, ex = self._assemble(p)
        act = p.active
        big = 1e4  # W m-2 K-1: effectively a fixed temperature
        g_air = (ex.A_top / (1.0 / big + ex.r_top) + ex.A_side / (1.0 / big + ex.r_side)) * ~p.fluid  # water held at T_water
        g_ext = g_air[act] + (p.fluid * big * g.volume)[act]
        a_ext = (g_air * T_air)[act] + (p.fluid * big * g.volume * T_water)[act] + ex.a_face
        q_geo = np.zeros(g.shape)
        q_geo[0, :] = s.q_geo * g.dx
        a_ext = a_ext + q_geo[act]
        T = np.full(g.shape, np.nan)
        T[act] = splu((L + sp.diags(g_ext)).tocsc()).solve(a_ext)
        return T
