"""Radiation geometry for the 2DV thermal model: sun position, direct beam with cast shadows,
and sky-view factors of the air-exposed faces.

The transect is taken as uniform alongshore, so each face is a strip with an outward normal
in the x-z plane: up, down (the underside of an overhang), seaward (-x) or landward (+x).
With the sun direction s (unit vector, with its alongshore component kept in the incidence
angle), a face receives

    direct   I_b cos(i) if the ray from the face towards the sun reaches the sky, with
             I_b = direct-on-horizontal / sin(alpha) and cos(i) = n . s
    diffuse  SVF x diffuse-on-horizontal (isotropic sky)
    reflected (1 - SVF) x albedo_terrain x global-on-horizontal
    longwave only the sky-visible fraction SVF exchanges with the sky; the terrain it sees
             is taken to be at the face's own temperature, so the rest gives no net flux

The sky-view factor of a 2D strip is the cosine-weighted fraction of its half-plane of
directions that escapes to the sky: SVF = sum over escaping directions of (sin t2 - sin t1)/2
(Hottel's crossed strings for an alongshore-uniform geometry). Flat open ground has SVF = 1,
a vertical face above flat ground 0.5.

Rays are tested column by column: a ray is blocked if, within any column it crosses, its z
range overlaps an active cell (soil, ice or water). This is exact for the staircase grid,
overhangs included. Rays pointing below the horizon always count as blocked (the terrain
continues beyond the domain), rays leaving the domain upward reach the sky.
"""

from __future__ import annotations

import numpy as np

S0 = 1361.0  # solar constant [W m-2]


# ------------------------------------------------------------------ sun
def solar_position(time_utc, lat, lon):
    """Solar elevation and azimuth [deg] (azimuth clockwise from north) for UTC times.

    NOAA general solar position equations (Spencer 1971 series for the declination and the
    equation of time); accurate to a few tenths of a degree, which is ample for hourly
    forcing. `time_utc` is a numpy datetime64 (array)."""
    t = np.asarray(time_utc, dtype="datetime64[s]")
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    doy = (t - year_start).astype(float) / 86400.0  # fractional day of year, 0-based
    hour = (doy % 1.0) * 24.0
    n_days = np.where(((t.astype("datetime64[Y]").astype(int) + 1970) % 4) == 0, 366.0, 365.0)
    g = 2.0 * np.pi / n_days * (np.floor(doy) + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60.0 + eqtime + 4.0 * lon                     # true solar time [min]
    ha = np.radians(tst / 4.0 - 180.0)                          # hour angle
    phi = np.radians(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cos_zen, -1.0, 1.0))
    # azimuth from north, clockwise
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi))
    az = (np.degrees(az) + 180.0) % 360.0
    return 90.0 - np.degrees(zen), az


def extraterrestrial(time_utc):
    """Extraterrestrial normal irradiance [W m-2]."""
    t = np.asarray(time_utc, dtype="datetime64[s]")
    doy = (t - t.astype("datetime64[Y]").astype("datetime64[s]")).astype(float) / 86400.0
    return S0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0))


def erbs_split(ghi, elevation, time_utc, min_elevation=2.0):
    """Split global horizontal irradiance into direct and diffuse on the horizontal with the
    Erbs et al. (1982) diffuse fraction. Below min_elevation everything is diffuse."""
    ghi = np.maximum(np.asarray(ghi, dtype=float), 0.0)
    sin_a = np.sin(np.radians(elevation))
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(sin_a > 0, ghi / (extraterrestrial(time_utc) * sin_a), 0.0)
    kt = np.clip(kt, 0.0, 1.0)
    kd = np.where(kt <= 0.22, 1.0 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
                           0.165))
    kd = np.where(np.asarray(elevation) < min_elevation, 1.0, kd)
    return ghi * (1.0 - kd), ghi * kd


def sun_vector(elevation, azimuth, bearing_landward):
    """Unit vector towards the sun in transect coordinates: (s_x, s_y, s_z), with +x the
    landward direction (compass bearing `bearing_landward`), y alongshore, z up."""
    a = np.radians(elevation)
    rel = np.radians(np.asarray(azimuth) - bearing_landward)
    return np.cos(a) * np.cos(rel), np.cos(a) * np.sin(rel), np.sin(a)


# ------------------------------------------------------------------ faces
UP, DOWN, SEA, LAND = range(4)
NORMALS = {UP: (0.0, 1.0), DOWN: (0.0, -1.0), SEA: (-1.0, 0.0), LAND: (1.0, 0.0)}  # (n_x, n_z)


def exposed_faces(active):
    """Masks of the air-exposed faces of active cells, per direction (sea at low x).
    The domain top counts as exposed; the domain bottom and the lateral boundaries do not
    (they are the model's deep and lateral boundaries)."""
    up = np.zeros(active.shape, dtype=bool)
    up[:-1, :] = active[:-1, :] & ~active[1:, :]
    up[-1, :] = active[-1, :]
    down = np.zeros(active.shape, dtype=bool)
    down[1:, :] = active[1:, :] & ~active[:-1, :]
    sea = np.zeros(active.shape, dtype=bool)
    sea[:, 1:] = active[:, 1:] & ~active[:, :-1]
    land = np.zeros(active.shape, dtype=bool)
    land[:, :-1] = active[:, :-1] & ~active[:, 1:]
    return {UP: up, DOWN: down, SEA: sea, LAND: land}


def face_samples(grid, kind, jj, ii, n_sub):
    """Sample points spread evenly along each face, shape (n_faces, n_sub), nudged a hair
    outward so a ray starts outside its own cell. Several points per face resolve partial
    shading, e.g. of the treads of a staircase slope by the risers."""
    eps = 1e-6
    frac = (np.arange(n_sub) + 0.5) / n_sub
    if kind in (UP, DOWN):
        x = grid.xe[ii][:, None] + frac[None, :] * grid.dx[ii][:, None]
        z = grid.ze[jj + 1] + eps if kind == UP else grid.ze[jj] - eps
        return x, np.broadcast_to(np.asarray(z)[:, None], x.shape)
    z = grid.ze[jj][:, None] + frac[None, :] * grid.dz[jj][:, None]
    x = grid.xe[ii] - eps if kind == SEA else grid.xe[ii + 1] + eps
    return np.broadcast_to(np.asarray(x)[:, None], z.shape), z


class RayCaster:
    """Tests many rays against the active cells of the grid, column by column."""

    def __init__(self, grid, blocking):
        self.g = grid
        # cumulative count of blocking cells per column, for O(1) interval queries
        self.cum = np.zeros((grid.nx, grid.nz + 1), dtype=np.int32)
        self.cum[:, 1:] = np.cumsum(blocking.T, axis=1)

    def _blocked_in_column(self, cols, za, zb):
        """Whether column(s) `cols` hold a blocking cell overlapping the open interval (za, zb)."""
        ze = self.g.ze
        lo = np.clip(np.searchsorted(ze, za, side="right") - 1, 0, self.g.nz)
        hi = np.clip(np.searchsorted(ze, zb, side="left"), 0, self.g.nz)  # exclusive upper cell index
        ok = (zb > ze[0]) & (za < ze[-1]) & (hi > lo)
        cnt = np.where(ok, self.cum[cols, np.maximum(hi, lo)] - self.cum[cols, lo], 0)
        return cnt > 0

    def escapes(self, x0, z0, ux, uz):
        """Whether rays from (x0, z0) in direction (ux, uz) (x-z plane) leave the domain
        without crossing a blocking cell. All arguments are 1D arrays of equal length."""
        g = self.g
        x0, z0, ux, uz = (np.asarray(a, dtype=float) for a in (x0, z0, ux, uz))
        # a downward ray always meets the terrain, which extends beyond the domain on both sides
        blocked = uz < 0.0
        col0 = np.clip(np.searchsorted(g.xe, x0, side="right") - 1, 0, g.nx - 1)
        vert = np.abs(ux) < 1e-12
        # vertical rays: only their own column
        if np.any(vert & ~blocked):
            v = np.nonzero(vert & ~blocked)[0]
            za = np.where(uz[v] > 0, z0[v], g.ze[0])
            zb = np.where(uz[v] > 0, g.ze[-1], z0[v])
            blocked[v] = self._blocked_in_column(col0[v], za, zb)
        nv = np.nonzero(~vert & ~blocked)[0]
        if len(nv) == 0:
            return ~blocked
        # z of each ray at every column edge; columns behind the origin are masked out
        xe = g.xe[None, :]
        slope = (uz[nv] / ux[nv])[:, None]
        x_left = np.maximum(xe[:, :-1], np.where(ux[nv, None] > 0, x0[nv, None], -np.inf))
        x_right = np.minimum(xe[:, 1:], np.where(ux[nv, None] < 0, x0[nv, None], np.inf))
        ahead = x_right > x_left
        z1 = z0[nv, None] + (x_left - x0[nv, None]) * slope
        z2 = z0[nv, None] + (x_right - x0[nv, None]) * slope
        za = np.minimum(z1, z2)
        zb = np.maximum(z1, z2)
        cols = np.broadcast_to(np.arange(g.nx)[None, :], za.shape)
        hit = self._blocked_in_column(cols, za, zb) & ahead
        blocked[nv] |= hit.any(axis=1)
        return ~blocked


class Geometry:
    """Face inventory, sky-view factors and cast shadows for a given active mask.
    Rebuilt by the model whenever the active mask changes. Each face is sampled at `n_sub`
    points and its sky-view factor and sunlit fraction are averaged over them."""

    def __init__(self, grid, active, n_dirs=36, n_sub=4):
        self.g = grid
        self.n_sub = n_sub
        self.active = active.copy()
        self.masks = exposed_faces(active)
        self.caster = RayCaster(grid, active)
        self.samples = {}
        self.svf = {}
        edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_dirs + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        w = 0.5 * (np.sin(edges[1:]) - np.sin(edges[:-1]))  # 2D view-factor weights, sum to 1
        for kind, mask in self.masks.items():
            jj, ii = np.nonzero(mask)
            x0, z0 = face_samples(grid, kind, jj, ii, n_sub)
            self.samples[kind] = (jj, ii, x0.ravel(), z0.ravel())
            svf = np.ones(grid.shape)
            if len(jj):
                nx_, nz_ = NORMALS[kind]
                # the normal rotated by each angle in the x-z plane
                ux = nx_ * np.cos(mid) - nz_ * np.sin(mid)
                uz = nx_ * np.sin(mid) + nz_ * np.cos(mid)
                n_pts = x0.size
                esc = self._escapes_chunked(np.repeat(x0.ravel(), len(mid)), np.repeat(z0.ravel(), len(mid)),
                                            np.tile(ux, n_pts), np.tile(uz, n_pts))
                svf_pts = (esc.reshape(n_pts, len(mid)) * w[None, :]).sum(axis=1)
                svf[jj, ii] = svf_pts.reshape(len(jj), n_sub).mean(axis=1)
            self.svf[kind] = svf

    def _escapes_chunked(self, x0, z0, ux, uz, chunk=20000):
        out = np.empty(len(x0), dtype=bool)
        for k in range(0, len(x0), chunk):
            s = slice(k, k + chunk)
            out[s] = self.caster.escapes(x0[s], z0[s], ux[s], uz[s])
        return out

    def direct_factor(self, s_vec):
        """Per face kind: sunlit fraction x cos(incidence) / sin(elevation), i.e. the ratio of
        the direct irradiance on the face to that on open horizontal ground."""
        sx, _, sz = s_vec
        out = {}
        for kind, (jj, ii, x0, z0) in self.samples.items():
            f = np.zeros(self.g.shape)
            nx_, nz_ = NORMALS[kind]
            cos_i = nx_ * sx + nz_ * sz
            if sz > 0.035 and cos_i > 0.0 and len(jj):  # sun more than 2 degrees up and in front of the face
                norm = np.hypot(sx, sz)
                lit = self._escapes_chunked(x0, z0, np.full(len(x0), sx / norm), np.full(len(x0), sz / norm))
                f[jj, ii] = lit.reshape(len(jj), self.n_sub).mean(axis=1) * cos_i / sz
            out[kind] = f
        return out
