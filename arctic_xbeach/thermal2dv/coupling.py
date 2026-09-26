"""Coupling of the 2DV thermal model to XBeach (Arctic-XBeach v2).

Who owns what:
  XBeach  waves, water levels, transport of unfrozen sediment, avalanching of thawed material
  2DV     heat, salt and ice; thaw of bed and bluff; ablation of ice-rich material; niche
          formation and block failure

Hand-off at every XBeach call:
  2DV -> XBeach  bed level zb(x): the top envelope of the solid (an overhang's top; the niche
                 below it lives only in the 2DV model), and the erodible thickness ne(x): from
                 zb down to the first cell that is not fully thawed
  XBeach -> 2DV  bed change (sedero), the wetted fraction of the swash and the wave-dependent
                 heat transfer coefficient h(x) (Kobayashi et al. 1999)

Bed changes on the fixed 2DV grid: every column carries a sub-cell bed offset (the part of
the bed change smaller than half a cell). When it passes +-dz/2 a whole cell is added
(beach sediment at porosity n_beach, at the temperature of the water or surface it lands
in) or removed (replaced by sea water below the water level, by air above it). So the
thermal state never holds partly filled cells, the bed handed to XBeach is exact, and the
thermal bed is within half a cell of it. The energy and salt carried by added and removed
cells go into the model's `morphology` budget terms. Sediment released by ablation of
ice-rich cells (Model2DV.eroded_sediment) is placed at the bluff toe as loose, thawed beach
sediment for XBeach to move.

Grids: XBeach's x-grid (nodes) is variable, the 2DV grid is fixed. Both are treated as
piecewise-constant cells (XBeach cells between node midpoints) and remapped by overlap
weights, which conserves volume.
"""

from __future__ import annotations

import numpy as np

from . import materials as mat

G = 9.81


# ------------------------------------------------------------------ grids
def node_edges(x):
    """Cell edges around nodes x: midpoints, with half cells at both ends."""
    x = np.asarray(x, dtype=float)
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])


def overlap_matrix(src_edges, dst_edges):
    """W[i, k] = length of overlap of destination cell i with source cell k."""
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.maximum(hi - lo, 0.0)


def remap(src_edges, values, dst_edges):
    """Conservative remap of cell averages; NaN where a destination cell has no overlap."""
    W = overlap_matrix(src_edges, dst_edges)
    covered = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(covered > 0, W @ np.asarray(values, dtype=float) / covered, np.nan), covered / np.diff(dst_edges)


# ------------------------------------------------------------------ Kobayashi h
def wavenumber(omega, d):
    """Linear dispersion, Fenton & McKee (1990) explicit approximation (within ~1%)."""
    d = np.maximum(np.asarray(d, dtype=float), 1e-3)
    k0 = omega**2 / G
    return k0 / np.tanh((k0 * d) ** 0.75) ** (2.0 / 3.0)


def kobayashi_h(H, T, d, alpha=0.5, fw=0.05, nu=1.848e-6, ks=2.5 * 1.90e-3, Pr=13.4, c_water=mat.C_WATER):
    """Water -> soil heat transfer coefficient under breaking waves [W m-2 K-1], Kobayashi et
    al. (1999), with Kobayashi & Aktan (1986) and Jonsson (1966) parameter values, as in v1.0
    (which evaluated sinh(k d) with the depth of the last point for every point; here each
    point uses its own depth). H wave height [m], T period [s], d depth [m]."""
    H, d = np.asarray(H, dtype=float), np.maximum(np.asarray(d, dtype=float), 1e-3)
    k = wavenumber(2.0 * np.pi / T, d)
    u_b = np.pi * H / (T * np.sinh(k * d))
    u_star = np.sqrt(0.5 * fw) * u_b
    E = 0.52 * (u_star * ks / nu) ** 0.45 * Pr**0.8
    return alpha * fw * c_water * u_b / (1.0 + np.sqrt(0.5 * fw) * E)


# ------------------------------------------------------------------ coupler
class Thermal2DVCoupler:
    """Couples a Model2DV to XBeach runs on a (different) cross-shore grid.

    model        the 2DV model (its x-coordinates in the same frame as XBeach's)
    n_beach      porosity of deposited sediment
    ne_offshore  erodible thickness handed to XBeach offshore of the 2DV domain [m]
    toe_width    width over which released bluff sediment is spread seaward of the toe [m]
    h_factor     calibration factor on the Kobayashi h; h_bounds clip it [W m-2 K-1]
    wet_depth    water depth that counts as wet in the swash [m]
    f_wet_min    lowest wetted fraction that still floods the 2DV grid during a storm
    h_floor      h where XBeach never wets a column that the 2DV storm level floods [W m-2 K-1]
    """

    def __init__(self, model, n_beach=0.4, ne_offshore=10.0, toe_width=5.0, h_factor=1.0,
                 h_bounds=(50.0, 1.0e4), wet_depth=0.01, f_wet_min=0.1, h_floor=1.0):
        self.m = model
        self.n_beach = n_beach
        self.ne_offshore = ne_offshore
        self.toe_width = toe_width
        self.h_factor = h_factor
        self.h_bounds = h_bounds
        self.wet_depth = wet_depth
        self.f_wet_min = f_wet_min
        self.h_floor = h_floor
        g = model.grid
        self.offset = np.zeros(g.nx)            # sub-cell bed offset per column [m]
        self.released_placed = model.eroded_sediment
        self.storm = None                       # hydrodynamics of the active storm, or None
        model.energy.setdefault("morphology", 0.0)
        model.salt_exchange.setdefault("morphology", 0.0)
        self.sediment = dict(removed=0.0, added=0.0, released_to_toe=0.0, unapplied=0.0)  # [m3 bed per m]

    # -------------------------------------------------------------- 2DV -> XBeach
    def _top_solid(self, p=None):
        """Index of the highest soil cell per column (-1 if none): the top envelope."""
        p = self.m.state() if p is None else p
        has = p.soil.any(axis=0)
        j_top = p.soil.shape[0] - 1 - np.argmax(p.soil[::-1, :], axis=0)
        return np.where(has, j_top, -1)

    def bed_level(self):
        """zb per 2DV column: the top envelope of the soil plus the sub-cell offset."""
        g = self.m.grid
        j_top = self._top_solid()
        return np.where(j_top >= 0, g.ze[j_top + 1], g.ze[0]) + self.offset

    def erodible_thickness(self):
        """Thickness from zb down to the first soil cell that is not fully thawed, per column.
        A frozen overhang, a fallen frozen block or frozen ground gives 0."""
        m, g = self.m, self.m.grid
        p = m.state()
        j_top = self._top_solid(p)
        zb = self.bed_level()
        thawed = p.soil & (m.H >= p.H_liq)
        ne = np.zeros(g.nx)
        for i in range(g.nx):
            j = j_top[i]
            if j < 0:
                continue
            while j >= 0 and p.soil[j, i] and thawed[j, i]:
                j -= 1
            if j < 0:
                ne[i] = zb[i] - g.ze[0]
            elif p.soil[j, i]:
                ne[i] = zb[i] - g.ze[j + 1]
            else:
                ne[i] = zb[i] - g.ze[j + 1]  # a void (niche) under thawed cells: erodible down to it
        return np.maximum(ne, 0.0)

    def to_xbeach(self, x_xb, zb_xb):
        """Bed level and erodible thickness on XBeach's nodes. Inside the 2DV domain both come
        from the 2DV state (conservative remap); offshore of it XBeach keeps its own bed with
        ne_offshore; landward of it the last 2DV column is carried on."""
        g = self.m.grid
        e_xb = node_edges(x_xb)
        zb2, cov = remap(g.xe, self.bed_level(), e_xb)
        zne2, _ = remap(g.xe, self.bed_level() - self.erodible_thickness(), e_xb)
        inside = cov > 0.5
        zb_new = np.where(inside, zb2, zb_xb)
        ne = np.where(inside, zb2 - zne2, self.ne_offshore)
        land = np.asarray(x_xb) > g.xe[-1]
        ne = np.where(land, self.erodible_thickness()[-1], ne)
        zb_new = np.where(land & ~inside, zb_xb, zb_new)
        return zb_new, np.maximum(ne, 0.0)

    # -------------------------------------------------------------- XBeach -> 2DV
    def apply_bed_change(self, x_xb, dz_xb, T_water, S_water, eta):
        """Apply XBeach's bed change (sedero on its nodes) to the 2DV columns.
        T_water, S_water: the water that fills removed cells and wets deposited ones (scalar or
        per grid row); eta: the still water level deciding water or air."""
        g = self.m.grid
        dz, cov = remap(node_edges(x_xb), dz_xb, g.xe)
        dz = np.where(cov > 0.5, dz, 0.0)
        self._apply_columns(dz, T_water, S_water, eta)

    def place_released_sediment(self, T_water, S_water, eta):
        """Put the mineral released by ablation since the last call at the bluff toe as loose
        beach sediment (bed volume = mineral / (1 - n_beach))."""
        m, g = self.m, self.m.grid
        new = m.eroded_sediment - self.released_placed
        if new <= 0.0:
            return 0.0
        zb = self.bed_level()
        # toe: the most seaward column whose bed is above the water level, next to steep ground
        dry = np.nonzero(zb > eta)[0]
        if len(dry) == 0:
            return 0.0
        i_toe = dry[0]
        cols = np.nonzero((g.xc < g.xe[i_toe]) & (g.xc >= g.xe[i_toe] - self.toe_width))[0]
        if len(cols) == 0:
            cols = np.array([max(i_toe - 1, 0)])
        V_bed = new / (1.0 - self.n_beach)
        dz = np.zeros(g.nx)
        dz[cols] = V_bed / g.dx[cols].sum()
        self._apply_columns(dz, T_water, S_water, eta)
        self.released_placed = m.eroded_sediment
        self.sediment["released_to_toe"] += V_bed
        return V_bed

    def _apply_columns(self, dz, T_water, S_water, eta):
        m, g = self.m, self.m.grid
        V = g.volume
        e0, s0 = m.total_energy(), m.total_salt()
        T_row = np.broadcast_to(np.asarray(T_water, dtype=float), (g.nz,))
        S_row = np.broadcast_to(np.asarray(S_water, dtype=float), (g.nz,))
        for i in np.nonzero(dz)[0]:
            self.offset[i] += dz[i]
            while True:
                p = m.state()
                j = self._top_solid(p)[i]
                if j < 0:
                    break
                if self.offset[i] <= -0.5 * g.dz[j]:
                    if m.H[j, i] < p.H_liq[j, i]:  # not thawed: XBeach cannot take it
                        self.sediment["unapplied"] += -(self.offset[i] + 0.5 * g.dz[j]) * g.dx[i]
                        self.offset[i] = -0.5 * g.dz[j]
                        break
                    self._remove_cell(j, i, T_row[j], S_row[j], eta)
                    self.offset[i] += g.dz[j]
                    self.sediment["removed"] += V[j, i]
                elif j + 1 < g.nz and self.offset[i] > 0.5 * g.dz[j + 1]:
                    if not self._add_cell(j + 1, i, T_row[j + 1], S_row[j + 1], eta, p):
                        break
                    self.offset[i] -= g.dz[j + 1]
                    self.sediment["added"] += V[j + 1, i]
                else:
                    break
        m.energy["morphology"] += m.total_energy() - e0
        m.salt_exchange["morphology"] += m.total_salt() - s0

    def _remove_cell(self, j, i, T_w, S_w, eta):
        m = self.m
        if m.grid.zc[j] < eta:
            Tf = float(mat.freezing_point(S_w))
            m.phi_s[j, i], m.phi_h2o[j, i], m.salt[j, i] = 0.0, 1.0, S_w
            m.H[j, i] = mat.C_WATER * max(T_w, Tf + 0.01)
        else:
            m.phi_s[j, i] = m.phi_h2o[j, i] = m.salt[j, i] = m.H[j, i] = 0.0

    def _add_cell(self, j, i, T_w, S_w, eta, p):
        """Deposit a full cell of beach sediment at (j, i). Returns False if the cell is not free
        (soil or ice), which leaves the rest of the bed change in the offset."""
        m = self.m
        if p.active[j, i] and not p.fluid[j, i]:
            return False
        submerged = p.fluid[j, i] or m.grid.zc[j] < eta
        if submerged:
            T_dep = p.T[j, i] if p.fluid[j, i] else T_w
            S_dep = p.S_bulk[j, i] if p.fluid[j, i] else S_w
        else:
            T_dep = p.T[j - 1, i] if j > 0 and p.active[j - 1, i] else T_w  # the surface it lands on
            S_dep = S_w
        m.phi_s[j, i] = 1.0 - self.n_beach
        m.phi_h2o[j, i] = self.n_beach
        m.salt[j, i] = self.n_beach * S_dep
        c = mat.closure(m.phi_s[j:j + 1, i], m.phi_h2o[j:j + 1, i], m.salt[j:j + 1, i])
        m.H[j, i] = float(mat.enthalpy_from_temperature(np.array([T_dep]), c, True)[0])
        return True

    # -------------------------------------------------------------- hydrodynamics
    def set_storm(self, x_xb, zb_xb, zs, H, T_wave):
        """Hydrodynamics of an XBeach run from a water-level time series.

        zs: water level on XBeach's nodes, shape (nt, nx); H: wave height on the nodes (e.g.
        the run's mean H); T_wave: wave period [s]. See _storm for what is set."""
        zs = np.atleast_2d(np.asarray(zs, dtype=float))
        depth = zs - np.asarray(zb_xb, dtype=float)[None, :]
        wet = np.isfinite(depth) & (depth > self.wet_depth)
        n_wet = wet.sum(axis=0)
        d_mean = np.where(n_wet > 0, np.where(wet, depth, 0.0).sum(axis=0) / np.maximum(n_wet, 1), 0.0)
        return self._storm(x_xb, zb_xb, wet.mean(axis=0), d_mean, H, T_wave, float(np.nanmean(zs[:, 0])))

    def set_storm_stats(self, x_xb, zb_xb, zs_mean, zs_std, zs_max, H, T_wave):
        """Hydrodynamics of an XBeach run from its time-mean statistics (nmeanvar zs, H).

        The wetted fraction is the Gaussian exceedance P(zs > zb + wet_depth) with the run's
        mean and standard deviation of zs, and 0 where zs never rose above that level."""
        from scipy.special import ndtr

        zb = np.asarray(zb_xb, dtype=float)
        zs_mean, zs_std, zs_max = (np.asarray(v, dtype=float) for v in (zs_mean, zs_std, zs_max))
        level = zb + self.wet_depth
        with np.errstate(divide="ignore", invalid="ignore"):
            f_wet = np.where(zs_std > 1e-6, ndtr((zs_mean - level) / zs_std), (zs_mean > level).astype(float))
        f_wet = np.where(np.isfinite(zs_max) & (zs_max > level), np.nan_to_num(f_wet), 0.0)
        d_mean = np.maximum(np.nan_to_num(zs_mean - zb), 0.0)
        return self._storm(x_xb, zb_xb, f_wet, d_mean, H, T_wave, float(np.nanmean(zs_mean[:1])))

    def _storm(self, x_xb, zb_xb, f_wet, d_mean, H, T_wave, zs_offshore):
        """The 2DV grid is flooded up to the highest bed that is wet at least f_wet_min of the
        time, and h in each column is the wetted fraction times the (clipped, calibrated)
        Kobayashi value; h_floor where XBeach never wets it."""
        g = self.m.grid
        H = np.nan_to_num(np.asarray(H, dtype=float), nan=0.0)  # XBeach writes NaN/fill on dry nodes
        h = np.clip(self.h_factor * kobayashi_h(H, T_wave, np.maximum(d_mean, self.wet_depth)), *self.h_bounds)
        h_eff = np.maximum(np.where(np.isfinite(h), f_wet * h, 0.0), self.h_floor)
        e_xb = node_edges(x_xb)
        h_col, cov = remap(e_xb, h_eff, g.xe)
        f_col, _ = remap(e_xb, f_wet, g.xe)
        swash = f_wet >= self.f_wet_min
        eta = float(np.max(np.asarray(zb_xb)[swash])) if swash.any() else zs_offshore
        eta = max(eta, zs_offshore)
        h_col = np.where(cov > 0.5, h_col, self.m.s.h_water)
        self.storm = dict(eta=eta, h=h_col, f_wet=np.where(cov > 0.5, f_col, 1.0))
        return self.storm

    def end_storm(self):
        self.storm = None

    def forcing(self, base):
        """Wrap a forcing function: during a storm the water level is the swash level from
        set_storm, and the model uses the storm's h per column."""
        def f(t, zc):
            F = base(t, zc)
            if self.storm is not None:
                F.eta = self.storm["eta"]
            return F
        return f

    def step(self, dt, base_forcing):
        """One 2DV step, then the released bluff sediment goes to the toe."""
        m = self.m
        m.h_water_x = None if self.storm is None else self.storm["h"]
        f = self.forcing(base_forcing)
        m.step(dt, f)
        F = f(m.time, m.grid.zc)
        S = m.s.S_sea if F.S_ref is None else F.S_ref
        self.place_released_sediment(F.T_ref, S, F.eta)
