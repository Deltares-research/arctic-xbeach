"""Snapshot recording of the 2DV state and conversion to an xarray Dataset."""

import numpy as np
import xarray as xr

# phase classes used for plotting and output
AIR, WATER, ICE, SOIL_FROZEN, SOIL_THAWING, SOIL_THAWED = range(6)
CLASS_NAMES = ["air", "water", "ice", "frozen soil", "thawing soil", "thawed soil"]


def phase_class(model, p=None):
    p = p or model.state()
    c = np.full(model.grid.shape, AIR, dtype=np.int8)
    c[p.fluid] = WATER
    c[p.open & ~p.fluid] = ICE
    c[p.soil & (p.f_liq < 0.01)] = SOIL_FROZEN
    c[p.soil & (p.f_liq >= 0.01) & (p.f_liq < 0.99)] = SOIL_THAWING
    c[p.soil & (p.f_liq >= 0.99)] = SOIL_THAWED
    return c


class Recorder:
    """Stores fields every `every` seconds, or at the given `times`, and scalars every call.

    Snapshots are taken when the recorder is called: each holds the first model state at or
    after its requested time (the actual model time is stored). Requested times crossed by one
    long step give a single snapshot."""

    def __init__(self, model, every=None, times=None):
        if (every is None) == (times is None):
            raise ValueError("Recorder: give exactly one of `every` or `times`")
        if every is not None and not every > 0:
            raise ValueError("Recorder: `every` must be positive")
        if times is not None and np.any(np.diff(np.asarray(times, dtype=float)) < 0):
            raise ValueError("Recorder: `times` must be non-decreasing")
        self.model = model
        self.every = every
        self.times = sorted(times) if times is not None else None
        self.next_time = 0.0 if times is None else (self.times[0] if self.times else np.inf)
        self.snap = dict(time=[], T=[], f_liq=[], phi_s=[], phi_h2o=[], S_bulk=[], S_liq=[], cls=[])
        self.series = dict(time=[], energy_domain=[], energy_boundary=[], energy_wetdry=[], energy_erosion=[],
                           energy_collapse=[], salt_domain=[], salt_relaxation=[], salt_wetdry=[], salt_erosion=[],
                           salt_collapse=[], eroded_sediment=[], ice_volume=[], iterations=[])

    def __call__(self, force=False):
        m = self.model
        p = m.state()
        V = m.grid.volume
        self.series["time"].append(m.time)
        self.series["energy_domain"].append(m.total_energy())
        # every budget term the model keeps (the XBeach coupler adds "morphology")
        for key, v in m.energy.items():
            self.series.setdefault(f"energy_{key}", []).append(v)
        self.series["salt_domain"].append(m.total_salt())
        for key, v in m.salt_exchange.items():
            self.series.setdefault(f"salt_{key}", []).append(v)
        self.series["eroded_sediment"].append(m.eroded_sediment)
        self.series["ice_volume"].append(float(np.sum(V * m.phi_h2o * (1.0 - p.f_liq) * p.active)))
        self.series["iterations"].append(m.last_step.get("iterations", 0))
        if self.snap["time"] and self.snap["time"][-1] == m.time:
            return
        if force or m.time >= self.next_time - 1e-6:
            self.snap["time"].append(m.time)
            self.snap["T"].append(p.T.copy())
            self.snap["f_liq"].append(np.where(p.active, p.f_liq, np.nan))
            self.snap["phi_s"].append(m.phi_s.copy())
            self.snap["phi_h2o"].append(m.phi_h2o.copy())
            self.snap["S_bulk"].append(np.where(p.active, p.S_bulk, np.nan))
            self.snap["S_liq"].append(p.S_liq.copy())
            self.snap["cls"].append(phase_class(m, p))
            if self.times is None:
                self.next_time = m.time + self.every
            else:
                while self.times and self.times[0] <= m.time + 1e-6:
                    self.times.pop(0)
                self.next_time = self.times[0] if self.times else np.inf

    def to_dataset(self, t0=None):
        g = self.model.grid
        t_snap = np.array(self.snap["time"])
        t_ser = np.array(self.series["time"])
        if t0 is not None:
            t_snap = np.datetime64(t0) + (t_snap * 1e3).astype("timedelta64[ms]")
            t_ser = np.datetime64(t0) + (t_ser * 1e3).astype("timedelta64[ms]")
        ds = xr.Dataset(
            {
                "T": (("time", "z", "x"), np.array(self.snap["T"]), {"units": "degC", "long_name": "temperature"}),
                "f_liq": (("time", "z", "x"), np.array(self.snap["f_liq"]), {"long_name": "liquid fraction of H2O"}),
                "phi_s": (("time", "z", "x"), np.array(self.snap["phi_s"]), {"long_name": "mineral volume fraction"}),
                "phi_h2o": (("time", "z", "x"), np.array(self.snap["phi_h2o"]), {"long_name": "total H2O volume fraction"}),
                "S_bulk": (("time", "z", "x"), np.array(self.snap["S_bulk"]), {"units": "psu", "long_name": "bulk salinity of the H2O"}),
                "S_liq": (("time", "z", "x"), np.array(self.snap["S_liq"]), {"units": "psu", "long_name": "salinity of the liquid (brine)"}),
                "phase": (("time", "z", "x"), np.array(self.snap["cls"]),
                          {"flag_values": list(range(6)), "flag_meanings": " ".join(n.replace(" ", "_") for n in CLASS_NAMES)}),
            },
            coords={"time": t_snap, "x": g.xc, "z": g.zc, "x_edge": g.xe, "z_edge": g.ze},
        )
        for key, val in self.series.items():
            if key != "time":
                ds[key] = (("time_series",), np.array(val))
        ds = ds.assign_coords(time_series=t_ser)
        ds["energy_domain"].attrs["units"] = "J m-1"
        ds["salt_domain"].attrs["units"] = "psu m3 m-1"
        ds["eroded_sediment"].attrs["units"] = "m3 m-1"
        ds["ice_volume"].attrs["units"] = "m3 m-1"
        return ds
