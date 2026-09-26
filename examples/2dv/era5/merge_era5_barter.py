"""Merge ERA5 downloads for Barter Island into model-ready NetCDF files.

Two sources are combined:
  - point time series from reanalysis-era5-single-levels-timeseries
    (download_era5_barter_timeseries.py -> data/raw/era5_ts_land.nc, era5_ts_ocean.nc):
    ssrd, fdir, strd, t2m, d2m, u10, v10, sp, tp, skt at the land point and sst, skt at the
    ocean point, whole record;
  - per-year 6-point box downloads from reanalysis-era5-single-levels
    (download_era5_barter.py -> data/raw/era5_<year>.nc): the variables the time-series
    dataset lacks (ssr, str, sf, fal, sd, rsn, siconc), plus data/raw/lsm_static.nc.
Both are the same ERA5 fields on the same 0.25 deg grid; over 2016 they agree to float32
rounding with the same time stamps (checked 2026-09-26). The time-series file wins for the
variables it holds; box variables are NaN for box years not downloaded yet.

The new-CDS layout is harmonised (valid_time -> time, number/expver dropped), accumulated
radiation is converted from J m-2 to W m-2, and two files are written:

  data/era5_barter_<y0>-<y1>.nc   box points for the box years present, dims (time, latitude,
                                   longitude), plus land_sea_mask(latitude, longitude)
  data/era5_barter_point.nc       the grid point nearest to Barter Island with
                                   land_sea_mask >= 0.5 (falls back to nearest point) over the
                                   full time-series record, plus SST / sea-ice cover / skin
                                   temperature at the nearest ocean point (*_ocean)

Each point variable carries `source` and `coverage` attributes; the global `coverage`
attribute lists them all.

TIME CONVENTION: accumulated fields (ssrd, fdir, strd, ssr, str, sf, tp) keep the ERA5
time stamp t but represent the mean (radiation) or total (sf, tp) over the PRECEDING HOUR
[t-1h, t]; centre them at t-30 min when interpolating. Instantaneous fields are valid at t.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# === CONFIGURATION ===
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"          # raw downloads
OUT_DIR = HERE / "data"                  # merged outputs
SITE_LAT, SITE_LON = 70.13, -143.7       # Barter Island (Kaktovik)
LAND_THRESHOLD = 0.5                     # land_sea_mask >= this counts as land
TS_FILES = {"land": RAW_DIR / "era5_ts_land.nc", "ocean": RAW_DIR / "era5_ts_ocean.nc"}
BOX_AREA = "70.25, -144.0, 70.0, -143.5"
TS_DATASET = "reanalysis-era5-single-levels-timeseries"
BOX_DATASET = "reanalysis-era5-single-levels"
GRID_TOL = 1e-3                          # time-series point must sit on the selected grid point

ACCUM_RAD = ["ssrd", "fdir", "strd", "ssr", "str"]   # J m-2 per hour -> W m-2
ACCUM_DEPTH = ["sf", "tp"]                             # m per hour, kept as depth
ACCUM_NOTE = ("accumulated over the preceding hour [t-1h, t] and stamped at t; "
              "centre at t-30 min when interpolating")
OCEAN_VARS = ("sst", "siconc", "skt")                 # also written at the nearest ocean point

# units / long names written to the output (overrides whatever the raw file carries)
META = {
    "ssrd": ("W m-2", "surface solar radiation downwards (global, all-sky)", "surface_downwelling_shortwave_flux_in_air"),
    "fdir": ("W m-2", "total-sky direct solar radiation at surface (horizontal plane)", None),
    "strd": ("W m-2", "surface thermal radiation downwards", "surface_downwelling_longwave_flux_in_air"),
    "ssr": ("W m-2", "surface net solar radiation", "surface_net_downward_shortwave_flux"),
    "str": ("W m-2", "surface net thermal radiation", "surface_net_downward_longwave_flux"),
    "sf": ("m", "snowfall (water equivalent) over the preceding hour", None),
    "tp": ("m", "total precipitation (water equivalent) over the preceding hour", None),
    "fal": ("1", "forecast albedo", "surface_albedo"),
    "t2m": ("K", "2 m air temperature", "air_temperature"),
    "d2m": ("K", "2 m dewpoint temperature", "dew_point_temperature"),
    "u10": ("m s-1", "10 m eastward wind", "eastward_wind"),
    "v10": ("m s-1", "10 m northward wind", "northward_wind"),
    "sp": ("Pa", "surface pressure", "surface_air_pressure"),
    "sd": ("m", "snow depth (water equivalent)", "lwe_thickness_of_surface_snow_amount"),
    "rsn": ("kg m-3", "snow density", None),
    "siconc": ("1", "sea ice area fraction", "sea_ice_area_fraction"),
    "sst": ("K", "sea surface temperature", "sea_surface_temperature"),
    "skt": ("K", "skin temperature", "surface_temperature"),
    "lsm": ("1", "land-sea mask (fraction of land)", "land_binary_mask"),
}


def open_raw(path: Path) -> xr.Dataset:
    ds = xr.load_dataset(path)
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        # Old-style ERA5/ERA5T split along an expver dimension: first non-NaN wins
        ds = ds.bfill("expver").isel(expver=0, drop=True)
    ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.variables], errors="ignore")
    return ds


def load_box() -> tuple[xr.Dataset | None, list[int]]:
    """All box years present, concatenated in time; variables missing in a year are NaN."""
    files = sorted(RAW_DIR.glob("era5_[0-9][0-9][0-9][0-9].nc"))
    years = [int(f.stem.split("_")[1]) for f in files]
    if not files:
        return None, years
    parts = [open_raw(f) for f in files]
    names = sorted({v for p in parts for v in p.data_vars})
    for i, p in enumerate(parts):  # 2016 holds all variables, later years only the box-only ones
        for v in names:
            if v not in p:
                p[v] = xr.full_like(next(q[v] for q in parts if v in q).isel(time=0, drop=True)
                                    .broadcast_like(p.time), np.nan)
        parts[i] = p[names]
    ds = xr.concat(parts, dim="time", data_vars="all", coords="minimal", compat="override", join="outer")
    ds = ds.sortby("time")
    ds = ds.isel(time=~ds.get_index("time").duplicated())
    return ds.sortby("latitude"), years  # ERA5 delivers north-to-south


def load_lsm() -> xr.DataArray:
    lsm = open_raw(RAW_DIR / "lsm_static.nc")["lsm"]
    if "time" in lsm.dims:
        lsm = lsm.isel(time=0, drop=True)
    return lsm.drop_vars("time", errors="ignore").sortby("latitude")


def load_ts(name: str, point: dict) -> xr.Dataset | None:
    """Time-series file for one point, checked to sit on the selected grid point."""
    path = TS_FILES[name]
    if not path.exists():
        print(f"warning: {path.name} not found; {name}-point variables fall back to the box")
        return None
    ds = open_raw(path)
    lat, lon = float(ds["latitude"]), float(ds["longitude"])
    if abs(lat - point["latitude"]) > GRID_TOL or abs(lon - point["longitude"]) > GRID_TOL:
        raise SystemExit(f"{path.name} holds ({lat}, {lon}) but the {name} point is "
                         f"({point['latitude']}, {point['longitude']}); re-download for that point")
    return ds.drop_vars(["latitude", "longitude"])


def to_output(da: xr.DataArray, name: str) -> xr.DataArray:
    """Unit conversion, float32 and output attributes for variable `name`."""
    units, long_name, std_name = META[name]
    if name in ACCUM_RAD:
        da = da / 3600.0
    da = da.astype("float32")
    da.attrs = {"units": units, "long_name": long_name}
    if std_name:
        da.attrs["standard_name"] = std_name
    if name in ACCUM_RAD:
        da.attrs["cell_methods"] = "time: mean (interval: 1 hour)"
        da.attrs["time_convention"] = f"mean flux {ACCUM_NOTE}; converted from J m-2 by /3600"
    elif name in ACCUM_DEPTH:
        da.attrs["cell_methods"] = "time: sum (interval: 1 hour)"
        da.attrs["time_convention"] = f"total {ACCUM_NOTE}"
    elif name != "lsm":
        da.attrs["time_convention"] = "instantaneous, valid at t"
    return da


def convert_box(ds: xr.Dataset, lsm: xr.DataArray) -> xr.Dataset:
    out = xr.Dataset(coords={"time": ds.time, "latitude": ds.latitude, "longitude": ds.longitude})
    for name in META:
        if name in ds and name != "lsm":
            out[name] = to_output(ds[name], name)
    out["land_sea_mask"] = to_output(lsm, "lsm")
    set_coord_attrs(out)
    return out


def set_coord_attrs(ds: xr.Dataset) -> None:
    for c, attrs in (("latitude", {"units": "degrees_north", "standard_name": "latitude"}),
                     ("longitude", {"units": "degrees_east", "standard_name": "longitude"}),
                     ("time", {"standard_name": "time", "long_name": "ERA5 time stamp (UTC)"})):
        if c in ds.coords:
            ds[c].attrs = attrs


def year_ranges(years: list[int]) -> str:
    """[1979, 1980, 1981, 2016] -> '1979-1981, 2016'."""
    if not years:
        return "none"
    out, start = [], years[0]
    for a, b in zip(years, years[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return ", ".join(out)


def check_completeness(t: pd.DatetimeIndex, label: str) -> str:
    full = pd.date_range(t[0], t[-1], freq="h")
    msg = f"{label}: {t[0]} .. {t[-1]}, {len(t)} steps, {len(full.difference(t))} missing hours within span"
    print(msg)
    return msg


def pick_points(lsm: xr.DataArray) -> tuple[dict, dict]:
    """Nearest land point (lsm >= threshold) and nearest ocean point to the site."""
    lat2, lon2 = xr.broadcast(lsm.latitude, lsm.longitude)
    dist_km = 111.2 * np.hypot(lat2 - SITE_LAT, (lon2 - SITE_LON) * np.cos(np.deg2rad(SITE_LAT)))

    def nearest(mask: xr.DataArray) -> dict | None:
        d = dist_km.where(mask)
        if not bool(d.notnull().any()):
            return None
        i, j = np.unravel_index(int(d.fillna(np.inf).argmin()), d.shape)
        return {"latitude": float(lsm.latitude[i]), "longitude": float(lsm.longitude[j]),
                "lsm": float(lsm[i, j]), "dist_km": float(d[i, j])}

    land = nearest(lsm >= LAND_THRESHOLD) or nearest(lsm.notnull())
    ocean = nearest(lsm < LAND_THRESHOLD)
    return land, ocean


def encoding_for(ds: xr.Dataset) -> dict:
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}
           for v in ds.data_vars}
    enc["time"] = {"units": "hours since 1979-01-01 00:00:00", "calendar": "standard", "dtype": "int32"}
    return enc


def overlap_report(ts: xr.Dataset, box_pt: xr.Dataset | None, label: str) -> list[str]:
    """Max |time series - box| per shared variable over the box years (output units)."""
    lines = []
    if box_pt is None:
        return lines
    common = ts.get_index("time").intersection(box_pt.get_index("time"))
    for v in ts.data_vars:
        if v not in box_pt or len(common) == 0:
            continue
        a = to_output(ts[v].sel(time=common), v).values.astype(float)
        b = to_output(box_pt[v].sel(time=common), v).values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            lines.append(f"{label} {v}: max|ts-box| {np.max(np.abs(a[ok] - b[ok])):.3g} "
                         f"{META[v][0]} over {ok.sum()} h")
    for line in lines:
        print(f"overlap {line}")
    return lines


def main() -> None:
    box, box_years = load_box()
    lsm = load_lsm()
    land, ocean = pick_points(lsm)
    print(f"land point:  lat {land['latitude']:.2f}, lon {land['longitude']:.2f}, "
          f"lsm {land['lsm']:.3f}, {land['dist_km']:.1f} km from site")
    if ocean:
        print(f"ocean point: lat {ocean['latitude']:.2f}, lon {ocean['longitude']:.2f}, "
              f"lsm {ocean['lsm']:.3f}, {ocean['dist_km']:.1f} km from site")
    print(f"box years present: {year_ranges(box_years)}")
    common = {
        "title": "ERA5 hourly single-level fields around Barter Island, Alaska",
        "source": (f"Copernicus Climate Change Service, {TS_DATASET} and {BOX_DATASET} "
                   "(Hersbach et al., 2020)"),
        "area_NWSE": BOX_AREA,
        "years_present": ",".join(str(y) for y in box_years),
        "time_convention": (f"Accumulated fields ({', '.join(ACCUM_RAD + ACCUM_DEPTH)}) are {ACCUM_NOTE}. "
                            "Instantaneous fields are valid at t."),
        "history": f"{pd.Timestamp.now():%Y-%m-%d %H:%M} merge_era5_barter.py",
    }

    # --- box file (box years only) ---
    if box is not None:
        y0, y1 = min(box_years), max(box_years)
        target = OUT_DIR / f"era5_barter_{y0}-{y1}.nc"
        for old in OUT_DIR.glob("era5_barter_*-*.nc"):  # stale files with another year range
            if old != target:
                print(f"removing stale {old.name}")
                old.unlink()
        ds = convert_box(box, lsm)
        ds.attrs = dict(common, source=f"Copernicus Climate Change Service, {BOX_DATASET} (Hersbach et al., 2020)",
                        completeness=check_completeness(ds.get_index("time"), "box"),
                        note=("years after the switch to the time-series dataset hold only the box-only "
                              "variables (ssr, str, sf, fal, sd, rsn, siconc); others are NaN there"))
        ds.to_netcdf(target, encoding=encoding_for(ds))
        print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")

    # --- point file (full time-series record) ---
    ts = {"land": load_ts("land", land), "ocean": load_ts("ocean", ocean) if ocean else None}
    def box_at(p: dict | None) -> xr.Dataset | None:
        if box is None or p is None:
            return None
        return box.sel(latitude=p["latitude"], longitude=p["longitude"]).drop_vars(["latitude", "longitude"])

    box_pt = {"land": box_at(land), "ocean": box_at(ocean)}
    times = [d.get_index("time") for d in (ts["land"], ts["ocean"], box_pt["land"]) if d is not None]
    time = times[0]
    for t in times[1:]:
        time = time.union(t)
    time = pd.DatetimeIndex(time).sort_values()

    pt_label = {k: (None if p is None else f"lat {p['latitude']}, lon {p['longitude']}")
                for k, p in (("land", land), ("ocean", ocean))}
    ts_span = {}
    for k, d in ts.items():
        if d is not None:
            ti = d.get_index("time")
            ts_span[k] = f"{ti[0]:%Y-%m-%d %H:%M} .. {ti[-1]:%Y-%m-%d %H:%M}"
    box_cov = f"box years {year_ranges(box_years)}; NaN in other years"

    def pick(name: str, where: str) -> xr.DataArray | None:
        """Variable `name` at point `where`, from the time series if it has it, else the box."""
        if ts[where] is not None and name in ts[where]:
            da = to_output(ts[where][name], name).reindex(time=time)
            da.attrs.update(source=f"{TS_DATASET}, {pt_label[where]}", coverage=ts_span[where])
            return da
        if box_pt[where] is not None and name in box_pt[where]:
            da = to_output(box_pt[where][name], name).reindex(time=time)
            da.attrs.update(source=f"{BOX_DATASET} (box), {pt_label[where]}", coverage=box_cov)
            return da
        return None

    pt = xr.Dataset(coords={"time": time})
    for name in META:
        if name == "lsm":
            continue
        da = pick(name, "land")
        if da is None:
            print(f"warning: {name} not available at the land point")
            continue
        pt[name] = da
    pt["land_sea_mask"] = to_output(lsm.sel(latitude=land["latitude"], longitude=land["longitude"]).drop_vars(
        ["latitude", "longitude"]), "lsm")
    if ocean:
        # SST and sea ice are masked over land; carry the nearest ocean point alongside
        for v in OCEAN_VARS:
            da = pick(v, "ocean")
            if da is None:
                continue
            da.attrs.update(long_name=f"{META[v][1]} at nearest ocean point",
                            point=f"lat {ocean['latitude']}, lon {ocean['longitude']}, lsm {ocean['lsm']:.3f}")
            pt[f"{v}_ocean"] = da
    pt = pt.assign_coords(latitude=land["latitude"], longitude=land["longitude"])
    set_coord_attrs(pt)

    overlap = overlap_report(ts["land"], box_pt["land"], "land") if ts["land"] is not None else []
    if ts["ocean"] is not None:
        overlap += overlap_report(ts["ocean"], box_pt["ocean"], "ocean")
    coverage = "; ".join(f"{v}: {pt[v].attrs['coverage']} ({pt[v].attrs['source'].split(',')[0]})"
                         for v in pt.data_vars if "coverage" in pt[v].attrs)
    pt.attrs = dict(common,
                    completeness=check_completeness(pt.get_index("time"), "point"),
                    coverage=coverage,
                    overlap_check=("; ".join(overlap) if overlap else "no overlap between sources"),
                    site=f"Barter Island, lat {SITE_LAT}, lon {SITE_LON}",
                    point_selection=(f"nearest grid point with land_sea_mask >= {LAND_THRESHOLD}: "
                                     f"lat {land['latitude']}, lon {land['longitude']}, lsm {land['lsm']:.3f}, "
                                     f"{land['dist_km']:.1f} km from site"),
                    ocean_point=("none in box" if not ocean else
                                 f"lat {ocean['latitude']}, lon {ocean['longitude']}, lsm {ocean['lsm']:.3f}, "
                                 f"{ocean['dist_km']:.1f} km from site"))
    for v in pt.data_vars:
        print(f"  {v:13s} {pt[v].attrs.get('coverage', '')}")
    point_file = OUT_DIR / "era5_barter_point.nc"
    tmp = point_file.with_suffix(".tmp.nc")
    pt.to_netcdf(tmp, encoding=encoding_for(pt))
    tmp.replace(point_file)
    print(f"wrote {point_file} ({point_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
