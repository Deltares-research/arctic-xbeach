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
):
    """Idealised seasonal forcing for a Beaufort Sea bluff coast.

    Air temperature is a cosine with its minimum near 20 January and a small diurnal cycle.
    Shortwave is zero during the polar night (roughly mid-November to late January). The sea
    is at its freezing point in winter; in summer it has a warm surface layer over cold,
    saline bottom water, as in the 2011 USGS Barter Island CTD record (bottom water about
    -1 to +1.5 degC at 4.7 m, lagoon about 5-6 degC). One storm surge hits in August.
    """
    Tf_sea = float(freezing_point(S_sea))

    def f(t, zc):
        doy = t0_doy + t / 86400.0
        day = doy % 365.0
        hour = (doy % 1.0) * 24.0

        sunlit = max(0.0, np.sin(np.pi * (day - 30.0) / 300.0))
        T_air = T_air_mean - T_air_amp * np.cos(2 * np.pi * (day - 20.0) / 365.0)
        T_air += T_air_diurnal * sunlit * np.cos(2 * np.pi * (hour - 15.0) / 24.0)

        sw = sw_peak * sunlit * (1.0 + 0.8 * np.cos(2 * np.pi * (hour - 13.0) / 24.0)) if sunlit > 0 else 0.0
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
