"""Atmospheric and ocean forcing for the 2DV thermal model.

A forcing function is any callable f(t, zc) -> Forcing, with t in seconds since the start
of the run and zc the cell-centre elevations (used to build the ocean reference profile).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import freezing_point


@dataclass
class Forcing:
    T_air: float           # 2 m air temperature [degC]
    sw_down: float         # downward shortwave at the surface [W m-2]
    lw_down: float         # downward longwave at the surface [W m-2]
    u10: float             # 10 m wind speed [m s-1], used for the bulk sensible heat flux
    eta: float             # still water level [m]
    T_ref: np.ndarray      # ocean reference temperature per grid row [degC] (sponge + flushing target)
    albedo_land: float = 0.18
    S_ref: np.ndarray | None = None  # ocean reference salinity per grid row [psu]; None uses Settings.S_sea
    sw_direct: float | None = None   # direct shortwave on the horizontal [W m-2]; with sw_diffuse it replaces
    sw_diffuse: float | None = None  # sw_down when radiation geometry is on (else sw_down is split with Erbs)
    q_surface: float = 0.0           # prescribed flux into upward-facing air-exposed faces [W m-2], e.g. reanalysis
                                     # net SW + net LW + latent when reproducing v1.0 (then set emissivity = 0)
    h_air: float | None = None       # air-side exchange coefficient [W m-2 K-1]; None uses the bulk rho cp C_h U


def constant_forcing(T_air=0.0, sw_down=0.0, lw_down=None, u10=0.0, eta=0.0, T_water=0.0, albedo_land=0.18,
                     S_water=None):
    """Forcing that does not change in time. lw_down=None gives a longwave that balances a
    black body at T_air, i.e. no net radiative exchange for a surface at air temperature."""
    if lw_down is None:
        lw_down = 5.670374419e-8 * (T_air + 273.15) ** 4

    def f(t, zc):
        S_ref = None if S_water is None else np.full(len(zc), S_water, dtype=float)
        return Forcing(T_air, sw_down, lw_down, u10, eta, np.full(len(zc), T_water, dtype=float), albedo_land, S_ref)

    return f


def synthetic_arctic_forcing(
    T_air_mean=-11.0,       # annual mean air temperature [degC] (Barter Island ~ -11)
    T_air_amp=16.0,         # seasonal amplitude [degC]
    T_air_diurnal=2.0,      # diurnal amplitude in the sunlit season [degC]
    sw_peak=290.0,          # peak daily-mean downward shortwave [W m-2]
    u10=6.0,                # background wind speed [m s-1]
    tide_amp=0.1,           # semi-diurnal tide amplitude [m] (Beaufort is microtidal)
    surge_peak=1.6,         # storm surge peak [m]
    surge_doy=228.0,        # day of year of the surge peak (mid-August)
    surge_width=1.5,        # e-folding half-width of the surge [days]
    storm_u10=15.0,         # wind speed during the storm [m s-1]
    S_sea=30.0,             # sea salinity at depth [psu]; sets the winter water temperature (freezing point)
    S_fresh_drop=6.0,       # summer freshening of the surface layer (river and melt water) [psu]
    T_surf_peak=7.5,        # peak summer surface-water temperature above freezing [K]
    T_bottom_peak=2.5,      # peak summer bottom-water temperature above freezing [K]
    z_thermocline=-3.0,     # centre of the warm-surface / cold-bottom transition [m]
    t0_doy=0.0,             # day of year at t = 0
    solar_site=None,        # (lat, lon, start_time_utc): shortwave from the real sun path instead of a cosine
    transmissivity=0.55,    # all-sky transmissivity with solar_site (0.55 gives ~270 W m-2 daily mean at the June solstice at 70 N)
):
    """Idealised seasonal forcing for a Beaufort Sea bluff coast.

    Air temperature is a cosine with its minimum near 20 January and a small diurnal cycle.
    Shortwave is zero during the polar night (roughly mid-November to late January). The sea
    is at its freezing point in winter; in summer it has a warm surface layer over cold,
    saline bottom water, as in the 2011 USGS Barter Island CTD record (bottom water about
    -1 to +1.5 degC at 4.7 m, lagoon about 5-6 degC). One storm surge hits in August.

    With solar_site the shortwave follows the sun: transmissivity x extraterrestrial x sin(elevation),
    consistent with the sun position the radiation geometry uses (Settings.latitude etc.).
    """
    Tf_sea = float(freezing_point(S_sea))
    if solar_site is not None:
        from . import radiation as rad
        lat, lon, start = solar_site
        start = np.datetime64(start, "s")

    def f(t, zc):
        doy = t0_doy + t / 86400.0
        day = doy % 365.0
        hour = (doy % 1.0) * 24.0

        sunlit = max(0.0, np.sin(np.pi * (day - 30.0) / 300.0))
        T_air = T_air_mean - T_air_amp * np.cos(2 * np.pi * (day - 20.0) / 365.0)
        T_air += T_air_diurnal * sunlit * np.cos(2 * np.pi * (hour - 15.0) / 24.0)

        if solar_site is None:
            sw = sw_peak * sunlit * (1.0 + 0.8 * np.cos(2 * np.pi * (hour - 13.0) / 24.0)) if sunlit > 0 else 0.0
        else:
            tt = start + np.timedelta64(int(round(t)), "s")
            el, _ = rad.solar_position(tt, lat, lon)
            sw = transmissivity * float(rad.extraterrestrial(tt)) * max(np.sin(np.radians(float(el))), 0.0)
        lw = 0.8 * 5.670374419e-8 * (T_air + 273.15) ** 4 + 20.0  # emissivity of a partly cloudy sky

        surge = surge_peak * np.exp(-(((day - surge_doy) / surge_width) ** 2))
        eta = tide_amp * np.sin(2 * np.pi * t / (12.42 * 3600.0)) + surge
        wind = u10 + (storm_u10 - u10) * surge / max(surge_peak, 1e-9)

        albedo_land = 0.18 if 160.0 <= day <= 275.0 else 0.8  # snow-free season only

        T_surf = Tf_sea + 0.05 + max(0.0, T_surf_peak * np.sin(np.pi * (day - 175.0) / 115.0)) * (175.0 <= day <= 290.0)
        T_bot = Tf_sea + 0.05 + max(0.0, T_bottom_peak * np.sin(np.pi * (day - 190.0) / 110.0)) * (190.0 <= day <= 300.0)
        w = 0.5 * (1.0 + np.tanh((np.asarray(zc) - z_thermocline) / 0.75))
        T_ref = T_bot + w * (T_surf - T_bot)
        fresh = max(0.0, np.sin(np.pi * (day - 170.0) / 120.0)) * (170.0 <= day <= 290.0)
        S_ref = S_sea - S_fresh_drop * fresh * w  # warm, fresher surface layer over cold, saline bottom water

        return Forcing(T_air, sw, lw, wind, eta, T_ref, albedo_land, S_ref)

    return f


ERA5_ACCUMULATED = ("ssrd", "fdir", "strd", "ssr", "str", "sf", "tp")


def era5_forcing(path, start_time_utc, end_time_utc=None, eta=0.0, water_profile=None, S_sea=30.0,
                 albedo_land=None, T_bottom=None):
    """Forcing from an ERA5 point file (examples/2dv/era5/merge_era5_barter.py output).

    Time is in seconds since `start_time_utc`, which should also be Settings.start_time_utc.
    ERA5 accumulated fields (radiation as mean W m-2) are stamped at the END of their hour;
    they are centred at t - 30 min before interpolating, so a value applies to the hour it
    describes (v1.0 applies them one hour late). Instantaneous fields are valid at their stamp.

    eta:           still water level [m]: a constant or a callable eta(t)
    water_profile: optional callable (t, zc) -> (T_ref, S_ref) for the ocean; by default the
                   ERA5 SST at the nearest ocean point (the freezing point under sea ice) over
                   the whole column, or linearly to T_bottom at the domain bottom if given
    albedo_land:   None uses ERA5 forecast albedo `fal` (snow included); else a constant
    """
    import xarray as xr

    t0 = np.datetime64(start_time_utc, "s")
    with xr.open_dataset(path) as ds:
        if end_time_utc is not None:
            ds = ds.sel(time=slice(t0 - np.timedelta64(2, "h"), np.datetime64(end_time_utc, "s") + np.timedelta64(2, "h")))
        stamp = (ds["time"].values.astype("datetime64[s]") - t0).astype(float)
        data = {v: ds[v].values.astype(float) for v in ds.data_vars if ds[v].dims == ("time",)}
    t_of = {v: stamp - 1800.0 if v in ERA5_ACCUMULATED else stamp for v in data}
    Tf_sea = float(freezing_point(S_sea))

    def at(v, t, keep_gaps=False):
        """Linear interpolation in time; gaps (NaN) are bridged unless keep_gaps, e.g. for SST,
        which ERA5 masks under sea ice."""
        y = data[v]
        if keep_gaps:
            return float(np.interp(t, t_of[v], y))
        ok = np.isfinite(y)
        return float(np.interp(t, t_of[v][ok], y[ok])) if ok.any() else np.nan

    def f(t, zc):
        T_air = at("t2m", t) - 273.15
        wind = float(np.hypot(at("u10", t), at("v10", t)))
        ghi = max(at("ssrd", t), 0.0)
        direct = min(max(at("fdir", t), 0.0), ghi) if "fdir" in data else None
        albedo = albedo_land if albedo_land is not None else float(np.clip(at("fal", t), 0.05, 0.9))
        zc = np.asarray(zc)
        if water_profile is not None:
            T_ref, S_ref = water_profile(t, zc)
        else:
            sst = at("sst_ocean", t, keep_gaps=True) - 273.15 if "sst_ocean" in data else np.nan
            ice = at("siconc_ocean", t) if "siconc_ocean" in data else 0.0
            T_s = Tf_sea if not np.isfinite(sst) or (np.isfinite(ice) and ice > 0.5) else max(sst, Tf_sea)
            if T_bottom is None:
                T_ref = np.full(len(zc), T_s)
            else:
                w = (zc - zc.min()) / max(np.ptp(zc), 1e-9)
                T_ref = max(T_bottom, Tf_sea) + w * (T_s - max(T_bottom, Tf_sea))
            S_ref = np.full(len(zc), S_sea)
        return Forcing(T_air, ghi, at("strd", t), wind, float(eta(t) if callable(eta) else eta), T_ref, albedo, S_ref,
                       sw_direct=direct, sw_diffuse=None if direct is None else ghi - direct)

    f.start_time_utc = str(t0)
    f.series = lambda v, t: at(v, t)  # raw access, e.g. f.series("siconc_ocean", t) for storm gating
    return f
