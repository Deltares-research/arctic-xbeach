"""Download ERA5 hourly point time series for Barter Island from the CDS time-series dataset.

`reanalysis-era5-single-levels-timeseries` stores ERA5 on the regular 0.25 deg grid in a
layout built for long single-point series: the whole record of one point comes back in one
request, in minutes instead of the ~5 h per (year x 2 variables) of the gridded dataset.
It holds only a subset of the single-level variables; the rest (fal, ssr, str, sd, rsn, sf,
siconc) still come from download_era5_barter.py (6-point box, per year).

Two requests:
  land point  (70.00, -143.75): ssrd, fdir, strd, t2m, d2m, u10, v10, sp, tp, skt
  ocean point (70.25, -143.75): sst, skt
These are the grid points merge_era5_barter.py selects from the box (nearest land point with
land_sea_mask >= 0.5, and nearest ocean point). The dataset snaps a requested location to
the nearest grid point; exact grid coordinates are requested, and the returned coordinates
are checked.

Output: data/raw/era5_ts_land.nc, data/raw/era5_ts_ocean.nc (raw CDS content, J m-2 for
radiation, ERA5 time stamps). Existing valid files are kept unless REFRESH is True.
Requires ~/.cdsapirc for the new CDS (https://cds.climate.copernicus.eu/api).
"""
import logging
import sys
import time
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

# === CONFIGURATION ===
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
LOG_FILE = HERE / "data" / "download_timeseries.log"
DATASET = "reanalysis-era5-single-levels-timeseries"
START = "1979-01-01"
END = None                                   # None: latest date the dataset offers (read from its constraints)
REFRESH = False                              # True: re-download even if the files exist (e.g. to extend the record)
POINTS = {
    "land": {
        "lat": 70.00, "lon": -143.75,
        "variables": {
            "surface_solar_radiation_downwards": "ssrd",
            "total_sky_direct_solar_radiation_at_surface": "fdir",
            "surface_thermal_radiation_downwards": "strd",
            "2m_temperature": "t2m",
            "2m_dewpoint_temperature": "d2m",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "surface_pressure": "sp",
            "total_precipitation": "tp",
            "skin_temperature": "skt",
        },
    },
    "ocean": {
        "lat": 70.25, "lon": -143.75,
        "variables": {
            "sea_surface_temperature": "sst",
            "skin_temperature": "skt",
        },
    },
}
GRID_TOL = 1e-3                              # returned point must equal the requested grid point
MAX_ATTEMPTS = 3
RETRY_WAIT_S = 300

log = logging.getLogger("era5_barter_ts")


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        root.addHandler(h)


def latest_date() -> str:
    """Last date offered by the dataset, from its public constraints."""
    import json
    import urllib.request
    base = f"https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{DATASET}"
    links = json.load(urllib.request.urlopen(base))["links"]
    href = next(lk["href"] for lk in links if lk["rel"] == "constraints")
    ends = [c["date"][0].split("/")[1] for c in json.load(urllib.request.urlopen(href))]
    return max(ends)


def unzip_if_needed(path: Path) -> None:
    """Replace a CDS zip at `path` by one merged NetCDF (CSV/NetCDF members)."""
    with open(path, "rb") as f:
        if f.read(2) != b"PK":
            return
    zpath = path.with_suffix(".zip")
    path.replace(zpath)
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".nc")]
        parts = []
        for n in names:
            member = zpath.with_name(f"{zpath.stem}__{Path(n).name}")
            member.write_bytes(zf.read(n))
            parts.append(member)
    log.info(f"{path.name}: CDS returned a zip ({names}); merging members")
    ds = xr.merge([xr.load_dataset(p) for p in parts], compat="override", join="outer",
                  combine_attrs="drop_conflicts")
    ds.to_netcdf(path)
    for p in parts:
        p.unlink()
    zpath.unlink()


def check_file(path: Path, pt: dict) -> None:
    """Raise if the file lacks variables or holds a different grid point than requested."""
    with xr.open_dataset(path) as ds:
        missing = [v for v in pt["variables"].values() if v not in ds.data_vars]
        lat = float(ds["latitude"].values.ravel()[0])
        lon = float(ds["longitude"].values.ravel()[0])
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        t = ds[tname].values
        log.info(f"{path.name}: lat {lat}, lon {lon}, {len(t)} steps {t[0]} .. {t[-1]}, "
                 f"variables {list(ds.data_vars)}")
    if missing:
        raise RuntimeError(f"{path.name}: missing variables {missing}")
    if abs(lat - pt["lat"]) > GRID_TOL or abs(lon - pt["lon"]) > GRID_TOL:
        raise RuntimeError(f"{path.name}: returned point ({lat}, {lon}) is not the requested grid point "
                           f"({pt['lat']}, {pt['lon']})")


def retrieve(name: str, pt: dict, end: str) -> Path:
    out = RAW_DIR / f"era5_ts_{name}.nc"
    if out.exists() and not REFRESH:
        check_file(out, pt)
        log.info(f"{out.name}: present, skipping (REFRESH = False)")
        return out
    req = {
        "variable": list(pt["variables"]),
        "location": {"longitude": pt["lon"], "latitude": pt["lat"]},
        "date": [f"{START}/{end}"],
        "data_format": "netcdf",
    }
    tmp = out.with_suffix(".part")
    wait = RETRY_WAIT_S
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log.info(f"{name}: submitting {START}/{end}, {len(req['variable'])} variables (attempt {attempt})")
            t0 = time.time()
            cdsapi.Client(progress=False).retrieve(DATASET, req, str(tmp))
            unzip_if_needed(tmp)
            tmp.replace(out)
            log.info(f"{name}: done in {(time.time() - t0) / 60:.1f} min, {out.stat().st_size / 1e6:.1f} MB")
            break
        except Exception as exc:
            log.error(f"{name}: attempt {attempt} failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(wait)
                wait *= 2
    else:
        raise RuntimeError(f"{name}: giving up after {MAX_ATTEMPTS} attempts")
    check_file(out, pt)  # outside the retry loop: a wrong point or missing variable stops the script
    return out


def main() -> None:
    setup_logging()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = END or latest_date()
    log.info(f"=== {DATASET}: {START} .. {end} ===")
    for name, pt in POINTS.items():
        retrieve(name, pt, end)


if __name__ == "__main__":
    main()
