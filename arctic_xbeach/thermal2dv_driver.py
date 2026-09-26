"""The 2DV thermal model as the thermal module of Arctic-XBeach (config thermal.model: "2dv").

main.py calls, per wrapper time step (1 h):
    XBeach (if due) -> Thermal2DVDriver.update_grid  (bed change, storm hydrodynamics)
    Thermal2DVDriver.step                            (one 2DV step of the wrapper time step)
    Thermal2DVDriver.find_thaw_depth                 (zb and ne_layer for the next XBeach call)
    Thermal2DVDriver.write_output

The 2DV grid spans the profile from `x_offshore` (default: where the XBeach bed first rises
above z_offshore) to the landward end, with a fixed fine window around the bluff crest
(default 45 m seaward to 15 m landward; plan decision 4: wide enough for decades of retreat
at Barter's ~1-2 m/yr). Offshore of the 2DV grid XBeach keeps its own bed with ne_offshore.

Config block `thermal2dv` (all keys optional; defaults below):
    x_offshore, z_offshore     2DV offshore boundary [m], or the first x where zb > z_offshore (-6 m)
    fine_window                [x0, x1] relative to the crest (-45, 15) [m]
    dx_fine, dx_max            0.5, 10 [m];  dz_fine, dz_max 0.25, 3 [m]
    z_bottom                   -30 [m];  z_top: crest + 2 m
    phi_s_ice_rich, z_ice_rich  0.35 above z = 0.3 m landward of the shoreline; phi_s_sediment 0.6
    T_land_mean, T_seabed_mean -9, -1.5 [degC] for the initial steady state
    h_water                    500 (background, calm) [W m-2 K-1]; h_factor 1.0; h_bounds [50, 1e4]
    erosion, collapse, d_crit  true, true, 1.5
    radiation_geometry         true (sun, shadows, sky view; needs model.latitude/longitude and
                               bathymetry.grid_orientation, the seaward direction ccw from east)
    era5_point                 path to an ERA5 point file (examples/2dv/era5); if absent the v1.0
                               ERA5 csv is used with downward fluxes derived from the net fluxes
    S_sea                      30 [psu]
    q_geo                      0.06 [W m-2] geothermal flux at the base (v1.0 prescribes a gradient instead)
    output_interval            24 [h] between 2DV snapshots in thermal2dv.nc
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .thermal2dv import Grid, Model2DV, Recorder, Settings, stretched_edges
from .thermal2dv import materials as mat
from .thermal2dv.coupling import Thermal2DVCoupler
from .thermal2dv.forcing import Forcing, era5_forcing

logger = logging.getLogger("thermo_model")
SIGMA = 5.670374419e-8

DEFAULTS = dict(x_offshore=None, z_offshore=-6.0, fine_window=(-45.0, 15.0), dx_fine=0.5, dx_max=10.0,
                dz_fine=0.25, dz_max=3.0, z_bottom=-30.0, z_top=None, phi_s_ice_rich=0.35, z_ice_rich=0.3,
                phi_s_sediment=0.6, T_land_mean=-9.0, T_seabed_mean=-1.5, h_water=500.0, h_factor=1.0,
                h_bounds=(50.0, 1.0e4), erosion=True, collapse=True, d_crit=1.5, radiation_geometry=True,
                era5_point=None, S_sea=30.0, output_interval=24, ne_offshore=10.0, n_beach=0.4, q_geo=0.06)


class Thermal2DVDriver:
    def __init__(self, sim):
        self.sim = sim
        cfg = dict(DEFAULTS)
        if "thermal2dv" in sim.config:
            cfg.update({k: v for k, v in dict(sim.config.thermal2dv).items()})
        self.cfg = cfg
        self.t0 = np.datetime64(pd.Timestamp(sim.timestamps[0]).to_datetime64(), "s")

    # -------------------------------------------------------------- set-up
    def initialize(self):
        sim, cfg = self.sim, self.cfg
        xp, zp = np.asarray(sim.xgr, dtype=float), np.asarray(sim.zgr, dtype=float)
        i_crest = int(np.argmax(zp >= zp.max() - 0.01))
        x_crest, z_crest = float(xp[i_crest]), float(zp.max())
        x0 = cfg["x_offshore"] if cfg["x_offshore"] is not None else float(xp[np.argmax(zp > cfg["z_offshore"])])
        x1 = float(xp[-1])
        z_top = cfg["z_top"] if cfg["z_top"] is not None else z_crest + 2.0
        fw = cfg["fine_window"]
        grid = Grid(stretched_edges(x0, x1, x_crest + fw[0], x_crest + fw[1], cfg["dx_fine"], cfg["dx_max"], growth=1.1),
                    stretched_edges(cfg["z_bottom"], z_top, -6.0, z_crest + 1.5, cfg["dz_fine"], cfg["dz_max"], growth=1.15))

        site = {}
        if cfg["radiation_geometry"]:
            site = dict(latitude=float(sim.config.model.latitude), longitude=float(sim.config.model.longitude),
                        start_time_utc=str(self.t0),
                        bearing_seaward=(90.0 - float(sim.config.bathymetry.grid_orientation)) % 360.0)
        s = Settings(h_water=cfg["h_water"], q_geo=cfg["q_geo"],
                     erosion=cfg["erosion"], collapse=cfg["collapse"], d_crit=cfg["d_crit"], S_sea=cfg["S_sea"],
                     n_sponge=5, tau_sponge=3 * 3600.0, tau_flush=2 * 86400.0, **site)
        m = Model2DV(grid, s)
        X, Z = grid.meshgrid()
        wl0 = float(sim.water_levels[0]) if hasattr(sim, "water_levels") else 0.0
        x_shore = float(xp[np.argmax(zp > wl0)])
        soil = grid.below_profile(xp, zp)
        land = X >= x_shore
        ice_rich = soil & land & (Z >= cfg["z_ice_rich"])
        m.phi_s[soil] = cfg["phi_s_sediment"]
        m.phi_s[ice_rich] = cfg["phi_s_ice_rich"]
        m.phi_h2o[soil] = 1.0 - m.phi_s[soil]
        m.set_salinity(cfg["S_sea"], soil & ~land)
        water = ~soil & (Z < wl0)
        m.phi_h2o[water] = 1.0
        m.set_salinity(cfg["S_sea"], water)
        T0 = m.steady_state(T_air=cfg["T_land_mean"], T_water=cfg["T_seabed_mean"])
        T0 = np.where(water, float(mat.freezing_point(cfg["S_sea"])) + 0.05, T0)
        m.set_temperature(T0)

        self.model = m
        self.coupler = Thermal2DVCoupler(m, n_beach=cfg["n_beach"], ne_offshore=cfg["ne_offshore"],
                                         h_factor=cfg["h_factor"], h_bounds=tuple(cfg["h_bounds"]))
        self.forcing = self._build_forcing()
        self.recorder = Recorder(m, every=cfg["output_interval"] * 3600.0)
        self.recorder(force=True)
        self.handoff = []
        self.find_thaw_depth()
        self._keep_handoff()
        logger.info(f"2DV thermal model: {grid.nx} x {grid.nz} cells, x {x0:.0f}-{x1:.0f} m, "
                    f"fine window {x_crest + fw[0]:.0f}-{x_crest + fw[1]:.0f} m, crest at {x_crest:.0f} m; "
                    f"radiation geometry {'on' if site else 'off'}")

    def _build_forcing(self):
        sim, cfg = self.sim, self.cfg
        eta_t = (np.arange(len(sim.timestamps)) * sim.dt * 3600.0)
        wl = np.asarray(sim.water_levels, dtype=float) if hasattr(sim, "water_levels") else np.zeros(len(eta_t))

        def eta(t):
            return float(np.interp(t, eta_t, wl[: len(eta_t)]))

        if cfg["era5_point"]:
            path = Path(cfg["era5_point"])
            if not path.is_absolute():
                path = Path(sim.proj_dir) / path
            logger.info(f"2DV forcing from ERA5 point file {path}")
            return era5_forcing(path, str(self.t0), str(np.datetime64(pd.Timestamp(sim.timestamps[-1]).to_datetime64(), "s")
                                                        + np.timedelta64(int(sim.dt), "h")),
                                eta=eta, S_sea=cfg["S_sea"])
        return self._v1_csv_forcing(eta)

    def _v1_csv_forcing(self, eta):
        """Forcing from the v1.0 ERA5 csv, which has net rather than downward radiation. The
        downward fluxes are recovered with the albedo (0.18 snow-free, 0.8 with snow) and the
        ERA5 skin/soil-level-1 temperature; the latent heat flux enters as a prescribed flux on
        upward faces. ERA5 mean fluxes are centred on their hour (t - 30 min)."""
        df = self.sim.forcing_data
        t = (pd.to_datetime(df["time"]).values.astype("datetime64[s]") - self.t0).astype(float)
        col = lambda name: df[name].values.astype(float)  # noqa: E731
        snow = col("snow_depth") if "snow_depth" in df else np.zeros(len(t))
        albedo = np.where(snow > 0.01, 0.8, 0.18)
        sw_down = np.maximum(col("mean_surface_net_short_wave_radiation_flux"), 0.0) / (1.0 - albedo)
        T_skin = col("soil_temperature_level_1") if "soil_temperature_level_1" in df else col("2m_temperature")
        lw_down = col("mean_surface_net_long_wave_radiation_flux") + SIGMA * T_skin**4
        latent = col("mean_surface_latent_heat_flux")
        u = np.hypot(col("10m_u_component_of_wind"), col("10m_v_component_of_wind"))
        T_air = col("2m_temperature") - 273.15
        sst = col("sea_surface_temperature") - 273.15
        ice = col("sea_ice_cover")
        Tf = float(mat.freezing_point(self.cfg["S_sea"]))
        tc = t - 1800.0  # mean fluxes describe the hour ending at their stamp

        def f(tt, zc):
            T_w = Tf if np.interp(tt, t, ice) > 0.5 else max(float(np.interp(tt, t, sst)), Tf)
            return Forcing(float(np.interp(tt, t, T_air)), float(np.interp(tt, tc, sw_down)),
                           float(np.interp(tt, tc, lw_down)), float(np.interp(tt, t, u)), eta(tt),
                           np.full(len(zc), T_w), float(np.interp(tt, t, albedo)), None,
                           q_surface=float(np.interp(tt, tc, latent)))
        return f

    # -------------------------------------------------------------- per time step
    def step(self, timestep_id):
        self.coupler.step(self.sim.dt * 3600.0, self.forcing)
        self.coupler.end_storm()
        n_snap = len(self.recorder.snap["time"])
        self.recorder()
        if len(self.recorder.snap["time"]) > n_snap:
            self._keep_handoff()

    def _keep_handoff(self):
        """Bed level and erodible thickness on the XBeach grid, at every 2DV snapshot."""
        zb, ne = self.coupler.to_xbeach(self.sim.xgr, self.sim.zgr)
        self.handoff.append((self.model.time, zb, ne))

    def find_thaw_depth(self):
        """Hand the 2DV bed and erodible thickness to the XBeach grid (sim.zgr, sim.thaw_depth)."""
        sim = self.sim
        zb, ne = self.coupler.to_xbeach(sim.xgr, sim.zgr)
        sim.zgr = zb
        sim.thaw_depth = ne
        return ne

    def update_grid(self, timestep_id, fp_xbeach_output="xboutput.nc"):
        """Apply XBeach's bed change to the 2DV model and keep its hydrodynamics for this step."""
        sim, m = self.sim, self.model
        path = Path(sim.result_dir) / "xb_files" / f"{int(timestep_id):010d}" / fp_xbeach_output
        if not path.is_file():
            logger.warning(f"2DV: XBeach output not found ({path}); no bed update")
            return
        with xr.open_dataset(path) as ds:
            ds = ds.squeeze()
            x = ds["globalx"].values.ravel() if "globalx" in ds else ds["x"].values.ravel()
            last = ds.isel(globaltime=-1)
            sedero = last["sedero"].values.ravel()
            zb_end = last["zb"].values.ravel()
            stats = None
            if "zs_mean" in ds:  # time-mean output (nmeanvar zs, H; written for thermal.model 2dv)
                stats = [ds[v].values.ravel() for v in ("zs_mean", "zs_var", "zs_max", "H_mean")]
                stats[1] = np.sqrt(np.maximum(stats[1], 0.0))
            zs = ds["zs"].values.reshape(ds.sizes["globaltime"], -1) if "zs" in ds else None
            H = ds["H"].values.reshape(ds.sizes["globaltime"], -1).mean(axis=0) if "H" in ds else None
        F = self.forcing(m.time + 0.5 * sim.dt * 3600.0, m.grid.zc)
        S = m.s.S_sea if F.S_ref is None else F.S_ref
        self.coupler.apply_bed_change(x, sedero, F.T_ref, S, F.eta)
        # outside the 2DV domain XBeach's own bed evolves
        sim.zgr = np.interp(sim.xgr, x, zb_end)
        if sim.xbeach_storms[timestep_id] and (stats is not None or (zs is not None and H is not None)):
            T_p = float(sim.conditions[timestep_id]["Tp(s)"])
            if stats is not None:
                st = self.coupler.set_storm_stats(x, zb_end, *stats[:3], stats[3], T_p)
            else:  # only snapshots: a crude wetted fraction
                st = self.coupler.set_storm(x, zb_end, zs, H, T_p)
            logger.debug(f"2DV storm: swash level {st['eta']:.2f} m, max h {np.max(st['h']):.0f} W m-2 K-1")

    def write_output(self, timestep_id, t_start=None):
        pass  # snapshots are taken in step(); written by finalize()

    def finalize(self):
        ds = self.recorder.to_dataset(t0=str(self.t0))
        ds["zb_2dv"] = (("x",), self.coupler.bed_level())
        t = np.array([h[0] for h in self.handoff])
        ds = ds.assign_coords(x_xb=np.asarray(self.sim.xgr, dtype=float),
                              time_xb=self.t0 + (t * 1e3).astype("timedelta64[ms]"))
        ds["zb_xb"] = (("time_xb", "x_xb"), np.array([h[1] for h in self.handoff]),
                       {"long_name": "bed level handed to XBeach", "units": "m"})
        ds["ne_xb"] = (("time_xb", "x_xb"), np.array([h[2] for h in self.handoff]),
                       {"long_name": "erodible thickness handed to XBeach (ne_layer)", "units": "m"})
        ds.attrs.update({f"sediment_{k}": v for k, v in self.coupler.sediment.items()})
        out = Path(self.sim.result_dir) / "thermal2dv.nc"
        ds.to_netcdf(out)
        logger.info(f"2DV output written to {out}")
