"""Merge per-year ERA5 downloads for Barter Island into model-ready NetCDF files.

Reads every data/raw/era5_<year>.nc present plus data/raw/lsm_static.nc, harmonises the
new-CDS layout (valid_time -> time, drops number/expver), converts accumulated radiation
from J m-2 to W m-2, and writes

  data/era5_barter_<y0>-<y1>.nc   all box points, dims (time, latitude, longitude),
                                   plus land_sea_mask(latitude, longitude)
  data/era5_barter_point.nc       the grid point nearest to Barter Island with
                                   land_sea_mask >= 0.5 (falls back to nearest point),
                                   plus SST / sea-ice cover at the nearest ocean point

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
RAW_DIR = HERE / "data" / "raw"          # per-year raw downloads
OUT_DIR = HERE / "data"                  # merged outputs
SITE_LAT, SITE_LON = 70.13, -143.7       # Barter Island (Kaktovik)
LAND_THRESHOLD = 0.5                     # land_sea_mask >= this counts as land

ACCUM_RAD = ["ssrd", "fdir", "strd", "ssr", "str"]   # J m-2 per hour -> W m-2
ACCUM_DEPTH = ["sf", "tp"]                             # m per hour, kept as depth
ACCUM_NOTE = ("accumulated over the preceding hour [t-1h, t] and stamped at t; "
              "centre at t-30 min when interpolating")

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


def open_year(path: Path) -> xr.Dataset:
    ds = xr.load_dataset(path)
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        # Old-style ERA5/ERA5T split along an expver dimension: first non-NaN wins
        ds = ds.bfill("expver").isel(expver=0, drop=True)
    ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.variables], errors="ignore")
    return ds


def load_all() -> tuple[xr.Dataset, list[int]]:
    files = sorted(RAW_DIR.glob("era5_*.nc"))
    if not files:
        raise SystemExit(f"no raw files in {RAW_DIR}")
    years = [int(f.stem.split("_")[1]) for f in files]
    ds = xr.concat([open_year(f) for f in files], dim="time", data_vars="all",
                   coords="minimal", compat="override", join="outer")
    ds = ds.sortby("time")
    ds = ds.isel(time=~ds.get_index("time").duplicated())
    static = RAW_DIR / "lsm_static.nc"
    if static.exists():
        ds["lsm"] = open_year(static)["lsm"].isel(time=0, drop=True)
    ds = ds.sortby("latitude")  # ERA5 delivers north-to-south
    return ds, years


def convert(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset(coords={"time": ds.time, "latitude": ds.latitude, "longitude": ds.longitude})
    for name, (units, long_name, std_name) in META.items():
        if name not in ds:
            print(f"warning: {name} not in raw data")
            continue
        da = ds[name]
        if name == "lsm" and "time" in da.dims:
            da = da.isel(time=0, drop=True)
        elif name in ACCUM_RAD:
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
        out[name if name != "lsm" else "land_sea_mask"] = da
    out["latitude"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    out["longitude"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    out["time"].attrs = {"standard_name": "time", "long_name": "ERA5 time stamp (UTC)"}
    return out


def check_completeness(ds: xr.Dataset, years: list[int]) -> str:
    t = ds.get_index("time")
    full = pd.date_range(t[0], t[-1], freq="h")
    n_missing = len(full.difference(t))
    missing_years = sorted(set(range(min(years), max(years) + 1)) - set(years))
    msg = (f"time {t[0]} .. {t[-1]}, {len(t)} steps, {n_missing} missing hours within span; "
           f"years present {len(years)}; years missing in span: {missing_years or 'none'}")
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


def main() -> None:
    raw, years = load_all()
    completeness = check_completeness(raw, years)
    ds = convert(raw)
    y0, y1 = min(years), max(years)
    common = {
        "title": "ERA5 hourly single-level fields around Barter Island, Alaska",
        "source": "Copernicus Climate Change Service, reanalysis-era5-single-levels (Hersbach et al., 2020)",
        "area_NWSE": "70.25, -144.0, 70.0, -143.5",
        "years_present": ",".join(str(y) for y in years),
        "completeness": completeness,
        "time_convention": (f"Accumulated fields ({', '.join(ACCUM_RAD + ACCUM_DEPTH)}) are {ACCUM_NOTE}. "
                            "Instantaneous fields are valid at t."),
        "history": f"{pd.Timestamp.now():%Y-%m-%d %H:%M} merge_era5_barter.py",
    }

    # Remove stale merged files from earlier runs with a different year range
    target = OUT_DIR / f"era5_barter_{y0}-{y1}.nc"
    for old in OUT_DIR.glob("era5_barter_*-*.nc"):
        if old != target:
            print(f"removing stale {old.name}")
            old.unlink()

    ds.attrs = dict(common)
    ds.to_netcdf(target, encoding=encoding_for(ds))
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")

    land, ocean = pick_points(ds["land_sea_mask"])
    print(f"land point:  lat {land['latitude']:.2f}, lon {land['longitude']:.2f}, "
          f"lsm {land['lsm']:.3f}, {land['dist_km']:.1f} km from site")
    if ocean:
        print(f"ocean point: lat {ocean['latitude']:.2f}, lon {ocean['longitude']:.2f}, "
              f"lsm {ocean['lsm']:.3f}, {ocean['dist_km']:.1f} km from site")

    pt = ds.sel(latitude=land["latitude"], longitude=land["longitude"])
    if ocean:
        # SST and sea ice are masked over land; carry the nearest ocean point alongside
        op = ds.sel(latitude=ocean["latitude"], longitude=ocean["longitude"])
        for v in ("sst", "siconc", "skt"):
            if v not in op:
                continue
            da =op[v].drop_vars(["latitude", "longitude"])
            da.attrs = dict(op[v].attrs, long_name=f"{op[v].attrs['long_name']} at nearest ocean point",
                            point=f"lat {ocean['latitude']}, lon {ocean['longitude']}, lsm {ocean['lsm']:.3f}")
            pt[f"{v}_ocean"] = da
    pt.attrs = dict(common,
                    site=f"Barter Island, lat {SITE_LAT}, lon {SITE_LON}",
                    point_selection=(f"nearest grid point with land_sea_mask >= {LAND_THRESHOLD}: "
                                     f"lat {land['latitude']}, lon {land['longitude']}, lsm {land['lsm']:.3f}, "
                                     f"{land['dist_km']:.1f} km from site"),
                    ocean_point=("none in box" if not ocean else
                                 f"lat {ocean['latitude']}, lon {ocean['longitude']}, lsm {ocean['lsm']:.3f}, "
                                 f"{ocean['dist_km']:.1f} km from site"))
    point_file = OUT_DIR / "era5_barter_point.nc"
    pt.to_netcdf(point_file, encoding=encoding_for(pt))
    print(f"wrote {point_file} ({point_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
