#!/usr/bin/env python3
"""
==============================================================================
Arctic-XBeach run diagnostics (read-only: never changes model state)
==============================================================================

RunDiagnostics collects, for one simulation:

  * an XBeach call log (xbeach_calls.csv): one row per coupling step in which XBeach runs
  * a conservation log (conservation_log.csv): one row per XBeach call, written by Simulation.update_grid
  * wall-clock of the thermal solver (whole run and spin-up separately)
  * the absolute and net surface heat exchange per calendar year (to put the regrid error in perspective)
  * a summary block for run.log

Both CSV files are written to the run's result folder. They are switched with output.log_xbeach_calls and
output.log_conservation in config.yaml (default: both on). Nothing here feeds back into the model.

Conventions of the heat budget
  * Stored heat of a thermal column is rho * (H - Cs * T_melt) [J/m3] integrated over elevation z with trapezoid
    weights [J/m2 of horizontal area], i.e. relative to the soil frozen at T_melt (the absolute enthalpy reference is
    arbitrary, this one keeps the numbers small). Integrating over z (node spacing dz * cos(slope), abs_zgr[:, 0] - abs_zgr[:, 1])
    matches how the regrid works: um.linear_interp_z carries each column over as a function of elevation.
  * The transect value [J per m alongshore] weights the columns with the trapezoid weights of the (unchanged) x grid.
    (The surface heat flux is per m2 of sloping surface, so for the exchange it is weighted with the arc length ds.)
  * At a regrid the change dE = E_after - E_before is split exactly in two:
        material = heat that the change of the column extent should add or remove, evaluated continuously:
                   added   = (fill enthalpy) * (thickness of new cover on top or of extension at the bottom)
                   removed = integral of the old profile over the eroded cover (top) or the truncated bottom part
                   (thicknesses in elevation: deposition/erosion of the bed, and the same shift at the column bottom)
        interp   = dE - material: what the regrid creates or destroys on top of that. It collects the interpolation of
                   the retained nodes and the node lattice (the fill/profile interface is placed at a node, not at the
                   exact old surface elevation). Zero for a perfect regrid.
"""

import csv
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None

CALL_COLUMNS = [
    "call_id", "timestep_id", "time", "year", "reason", "xbeach_times_value", "hotstart", "sim_hours",
    "setup_wall_s", "xbeach_wall_s", "grid_update_wall_s", "success",
]

CONSERVATION_COLUMNS = [
    "call_id", "timestep_id", "time", "regridded",
    # stored heat of the thermal columns, J per m alongshore (relative to frozen soil at T_melt)
    "E_before", "E_after", "dE", "dE_material", "dE_interp",
    "E_added_top", "E_added_bottom", "E_removed_top", "E_removed_bottom",
    "interp_pct_of_E_before",
    "n_filled_top_nodes", "n_filled_bottom_nodes", "n_removed_top_nodes", "n_removed_bottom_nodes",
    # sediment budget, m3 per m alongshore
    "vol_xb_net", "vol_xb_eroded", "vol_xb_deposited", "vol_xb_zbdiff",
    "vol_grid_net", "vol_grid_eroded", "vol_grid_deposited", "vol_grid_minus_xb",
    "vol_nodes_fill_top", "vol_nodes_removed_top",
]


# =============================================================================
# Pure helpers (also used by the tests)
# =============================================================================
def memory_status(path) -> str:
    """One line on process memory, system memory and free disk at path, for run.log. Must never fail a run."""
    parts = []
    if psutil is not None:
        try:
            mem = psutil.Process().memory_info()
            # On Windows, private bytes count against the commit limit and RSS does not
            private = getattr(mem, "private", mem.rss) / 1e6
            peak = getattr(mem, "peak_pagefile", 0) / 1e6
            parts.append(f"process {private:.0f} MB" + (f" (peak {peak:.0f} MB)" if peak else "") + f", RSS {mem.rss / 1e6:.0f} MB")
            swap = psutil.swap_memory()
            parts.append(f"system RAM available {psutil.virtual_memory().available / 1e9:.1f} GB, "
                         f"page file {swap.used / 1e9:.1f} of {swap.total / 1e9:.1f} GB")
        except Exception as e:
            parts.append(f"memory unknown ({e})")
    try:
        parts.append(f"disk free {shutil.disk_usage(path).free / 1e9:.1f} GB")
    except Exception as e:
        parts.append(f"disk unknown ({e})")
    return "Resources: " + ", ".join(parts)


def arc_length_weights(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Surface arc length [m] owned by each node of the profile (trapezoid weights along the profile)."""
    seg = np.hypot(np.diff(x), np.diff(z))
    w = np.zeros(len(x))
    w[:-1] += 0.5 * seg
    w[1:] += 0.5 * seg
    return w


def x_weights(x: np.ndarray) -> np.ndarray:
    """Horizontal width [m] owned by each node (trapezoid weights along x)."""
    dx = np.diff(x)
    w = np.zeros(len(x))
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def column_energy(H: np.ndarray, rho: np.ndarray, Cs: np.ndarray, T_melt: float, abs_zgr: np.ndarray,
                  node_mask: np.ndarray | None = None) -> np.ndarray:
    """Stored heat per column [J/m2 of horizontal area]: integral over elevation of rho * (H - Cs * T_melt), trapezoid
    weights with the vertical node spacing abs_zgr[:, 0] - abs_zgr[:, 1] of each column.

    node_mask (same shape as H) restricts the sum to a subset of nodes, e.g. the filled nodes of a regrid."""
    e = rho * (H - Cs * T_melt)
    w = np.ones(H.shape[1])
    w[0] = w[-1] = 0.5
    ew = e * w
    if node_mask is not None:
        ew = ew * node_mask
    dz_vertical = abs_zgr[:, 0] - abs_zgr[:, 1]
    return ew.sum(axis=1) * dz_vertical


def _integral_to(z_asc, e_asc, cum, z):
    """Integral of the piecewise-linear profile e(z) (nodes z_asc ascending) from z_asc[0] to z (z inside the range)."""
    z = np.clip(z, z_asc[0], z_asc[-1])
    j = np.clip(np.searchsorted(z_asc, z) - 1, 0, len(z_asc) - 2)
    ez = np.interp(z, z_asc, e_asc)
    return cum[j] + (z - z_asc[j]) * 0.5 * (e_asc[j] + ez)


def material_heat(abs_zgr_old, e_old, abs_zgr_new, e_new):
    """Heat [J/m2 horizontal] that the change of the column extent adds or removes, per column (continuous in elevation).

    e_old, e_new: rho * (H - Cs * T_melt) [J/m3] of the old and new columns. Columns run from the surface (index 0,
    highest z) downward. Returns dict of arrays: added_top, added_bottom, removed_top, removed_bottom."""
    n = abs_zgr_old.shape[0]
    out = {k: np.zeros(n) for k in ("added_top", "added_bottom", "removed_top", "removed_bottom")}
    for i in range(n):
        zo, zn = abs_zgr_old[i], abs_zgr_new[i]
        z_asc, e_asc = zo[::-1], e_old[i][::-1]
        cum = np.concatenate(([0.0], np.cumsum(0.5 * (e_asc[1:] + e_asc[:-1]) * np.diff(z_asc))))
        d_top = zn[0] - zo[0]  # >0: deposition / submergence fill, <0: eroded cover
        if d_top > 0:
            out["added_top"][i] = e_new[i][0] * d_top
        elif d_top < 0:
            out["removed_top"][i] = _integral_to(z_asc, e_asc, cum, zo[0]) - _integral_to(z_asc, e_asc, cum, zn[0])
        d_bot = zo[-1] - zn[-1]  # >0: column extends downward (erosion), <0: bottom part dropped (deposition)
        if d_bot > 0:
            out["added_bottom"][i] = e_new[i][-1] * d_bot
        elif d_bot < 0:
            out["removed_bottom"][i] = _integral_to(z_asc, e_asc, cum, zn[-1]) - _integral_to(z_asc, e_asc, cum, zo[-1])
    return out


def regrid_node_masks(abs_xgr_old, abs_zgr_old, abs_xgr_new, abs_zgr_new):
    """Which new nodes are filled (outside the z-range of the old column) and which old nodes are removed
    (outside the z-range of the new column), mirroring um.linear_interp_z. Columns i of old and new are matched
    (the x grid does not change in a regrid).

    Returns dict of boolean arrays (n_x, n_z): filled_top, filled_bottom (new nodes), removed_top, removed_bottom (old nodes).
    """
    z_old_max, z_old_min = abs_zgr_old.max(axis=1, keepdims=True), abs_zgr_old.min(axis=1, keepdims=True)
    z_new_max, z_new_min = abs_zgr_new.max(axis=1, keepdims=True), abs_zgr_new.min(axis=1, keepdims=True)
    return dict(
        filled_top=abs_zgr_new > z_old_max,
        filled_bottom=abs_zgr_new < z_old_min,
        removed_top=abs_zgr_old > z_new_max,
        removed_bottom=abs_zgr_old < z_new_min,
    )


# =============================================================================
# Diagnostics collector
# =============================================================================
class RunDiagnostics:

    def __init__(self, out_dir, dt_hours: float, sea_ice_threshold: float, spinup_seconds: float,
                 log_calls: bool = True, log_conservation: bool = True):
        self.out_dir = Path(out_dir)
        self.dt_hours = float(dt_hours)
        self.sea_ice_threshold = float(sea_ice_threshold)
        self.spinup_seconds = float(spinup_seconds)
        self.log_calls = bool(log_calls)
        self.log_conservation = bool(log_conservation)

        self.calls: list[dict] = []
        self.conservation: list[dict] = []
        self.call_id = -1  # id of the XBeach call in progress (set by begin_call)

        self.thermal_wall_s = 0.0
        self.spinup_wall_s = 0.0
        self.spinup_restored = False

        # surface heat exchange integrated over the transect, per calendar year: [abs, net] in J per m alongshore
        self.surface_exchange: dict[int, list[float]] = {}
        self.current_year: int | None = None

        self._call_file = None
        self._cons_file = None
        self._call_writer = None
        self._cons_writer = None

    # -- file handling ------------------------------------------------------------------------------------
    def _open(self, name, columns):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        f = open(self.out_dir / name, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        return f, w

    def close(self):
        for f in (self._call_file, self._cons_file):
            if f is not None and not f.closed:
                f.close()

    # -- recording ----------------------------------------------------------------------------------------
    def begin_call(self) -> int:
        self.call_id += 1
        return self.call_id

    def record_call(self, row: dict):
        self.calls.append(row)
        if not self.log_calls:
            return
        if self._call_writer is None:
            self._call_file, self._call_writer = self._open("xbeach_calls.csv", CALL_COLUMNS)
        self._call_writer.writerow(row)
        self._call_file.flush()

    def record_conservation(self, row: dict):
        self.conservation.append(row)
        if not self.log_conservation:
            return
        if self._cons_writer is None:
            self._cons_file, self._cons_writer = self._open("conservation_log.csv", CONSERVATION_COLUMNS)
        self._cons_writer.writerow({k: row.get(k, np.nan) for k in CONSERVATION_COLUMNS})
        self._cons_file.flush()

    def add_thermal_time(self, seconds: float, spinup: bool):
        self.thermal_wall_s += seconds
        if spinup:
            self.spinup_wall_s += seconds

    def add_surface_exchange(self, heat_flux: np.ndarray, ds_weights: np.ndarray, dt: float):
        """Called every thermal substep with the net surface flux [W/m2] applied to each node."""
        q = heat_flux * ds_weights  # W per m alongshore
        acc = self.surface_exchange.setdefault(self.current_year, [0.0, 0.0])
        acc[0] += float(np.abs(q).sum()) * dt
        acc[1] += float(q.sum()) * dt

    # -- summary ------------------------------------------------------------------------------------------
    def summary_lines(self, timestamps: pd.DatetimeIndex, sea_ice: np.ndarray) -> list[str]:
        """Text block for run.log. timestamps/sea_ice: model time stamps and forcing sea ice cover (same length)."""
        n = min(len(timestamps), len(sea_ice))
        ts = timestamps[:n]
        ice_free = np.asarray(sea_ice[:n]) < self.sea_ice_threshold
        elapsed = (ts - ts[0]) / pd.Timedelta("1s")
        after_spinup = np.asarray(elapsed) >= self.spinup_seconds
        years = sorted(set(ts.year) | {r["year"] for r in self.calls})

        df = pd.DataFrame(self.calls) if self.calls else pd.DataFrame(columns=CALL_COLUMNS)
        cons = pd.DataFrame(self.conservation) if self.conservation else pd.DataFrame(columns=CONSERVATION_COLUMNS)

        L = []
        bar = "=" * 78
        L.append(bar)
        L.append("RUN DIAGNOSTICS SUMMARY")
        L.append(bar)
        L.append(f"XBeach calls: {len(df)} (successful: {int(df['success'].sum()) if len(df) else 0}), "
                 f"simulated XBeach hours: {df['sim_hours'].sum() if len(df) else 0.0:.0f}")
        L.append(f"XBeach wall-clock: {df['xbeach_wall_s'].sum() if len(df) else 0.0:.1f} s "
                 f"(input writing {df['setup_wall_s'].sum() if len(df) else 0.0:.1f} s, "
                 f"grid update {df['grid_update_wall_s'].sum() if len(df) else 0.0:.1f} s)")
        L.append(f"Thermal-solver wall-clock (spin-up included): {self.thermal_wall_s:.1f} s")
        if self.spinup_restored:
            L.append("Spin-up wall-clock: 0.0 s (spin-up state restored from file, thermal time above is post spin-up only)")
        else:
            L.append(f"Spin-up wall-clock (thermal): {self.spinup_wall_s:.1f} s")
        L.append("")
        L.append("Per calendar year (reasons: storm = run-up criterion, inter = scheduled inter-storm call, both = coincide):")
        L.append(f"{'year':>6} {'storm':>6} {'inter':>6} {'both':>6} {'total':>6} {'xb_hours':>9} {'xb_wall_s':>10} "
                 f"{'ice_free_h':>11} {'ice_free_h_post_spinup':>23}")
        for y in years:
            d = df[df["year"] == y] if len(df) else df
            m = np.asarray(ts.year == y)
            cnt = {r: int((d["reason"] == r).sum()) if len(d) else 0 for r in ("storm", "inter", "both")}
            L.append(f"{y:>6} {cnt['storm']:>6} {cnt['inter']:>6} {cnt['both']:>6} {len(d):>6} "
                     f"{d['sim_hours'].sum() if len(d) else 0.0:>9.0f} {d['xbeach_wall_s'].sum() if len(d) else 0.0:>10.1f} "
                     f"{ice_free[m].sum() * self.dt_hours:>11.0f} {(ice_free & after_spinup)[m].sum() * self.dt_hours:>23.0f}")
        L.append(f"{'total':>6} {int((df['reason'] == 'storm').sum()) if len(df) else 0:>6} "
                 f"{int((df['reason'] == 'inter').sum()) if len(df) else 0:>6} {int((df['reason'] == 'both').sum()) if len(df) else 0:>6} "
                 f"{len(df):>6} {df['sim_hours'].sum() if len(df) else 0.0:>9.0f} {df['xbeach_wall_s'].sum() if len(df) else 0.0:>10.1f} "
                 f"{ice_free.sum() * self.dt_hours:>11.0f} {(ice_free & after_spinup).sum() * self.dt_hours:>23.0f}")

        L.append("")
        L.append("Conservation at grid updates (heat in J per m alongshore, relative to frozen soil at T_melt; volumes in m3 per m):")
        if len(cons) == 0:
            L.append("  no grid updates")
        else:
            cons = cons.copy()
            cons["year"] = pd.to_datetime(cons["time"]).dt.year
            L.append(f"  grid updates: {len(cons)}, with bed change: {int(cons['regridded'].sum())}")
            L.append(f"{'year':>6} {'n_regrid':>9} {'sum|interp|':>13} {'sum interp':>13} {'material':>13} "
                     f"{'surf |Q| int':>13} {'|interp|/|Q|':>13} {'interp/|Q|':>11} {'net vol xb':>11} {'vol mismatch':>13}")
            for y, g in cons.groupby("year"):
                g = g[g["regridded"] == 1]
                if g.empty:
                    continue
                ex = self.surface_exchange.get(int(y), [np.nan, np.nan])[0]
                abs_i, net_i = g["dE_interp"].abs().sum(), g["dE_interp"].sum()
                L.append(f"{y:>6} {len(g):>9} {abs_i:>13.3e} {net_i:>13.3e} {g['dE_material'].sum():>13.3e} "
                         f"{ex:>13.3e} {100 * abs_i / ex:>12.4f}% {100 * net_i / ex:>10.4f}% "
                         f"{g['vol_xb_net'].sum():>11.3f} {g['vol_grid_minus_xb'].sum():>13.3e}")
            g = cons[cons["regridded"] == 1]
            if len(g):
                tot_ex = sum(v[0] for k, v in self.surface_exchange.items() if k in set(g["year"]))
                L.append(f"  cumulative interpolation error: net {g['dE_interp'].sum():.3e} J/m, "
                         f"sum of absolute values {g['dE_interp'].abs().sum():.3e} J/m; "
                         f"surface heat exchange |Q| integrated over those years: {tot_ex:.3e} J/m "
                         f"(sum of |net flux * ds * dt| per substep, net flux = all surface terms of the thermal BC)")
                if self.spinup_restored:
                    L.append("  note: spin-up was restored from file, so the surface heat exchange above covers the post spin-up part of the run only")
                L.append(f"  cumulative regrid error: {100 * g['dE_interp'].sum() / tot_ex:.4f} % net, "
                         f"{100 * g['dE_interp'].abs().sum() / tot_ex:.4f} % absolute, of the absolute surface heat exchange")
                L.append(f"  net volume change XBeach {g['vol_xb_net'].sum():.3f} m3/m, thermal grid {g['vol_grid_net'].sum():.3f} m3/m, "
                         f"difference (grid - xbeach) {g['vol_grid_minus_xb'].sum():.3e} m3/m")
        L.append(bar)
        return L
