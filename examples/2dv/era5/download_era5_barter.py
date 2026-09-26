"""Download ERA5 hourly single-level data for Barter Island, Alaska (one file per year).

Output: data/raw/era5_<year>.nc (all variables for one year) and data/raw/lsm_static.nc
(land-sea mask, one time step). Years already present and valid are skipped, so re-running
resumes an interrupted download. Failed requests are retried with a growing back-off.

CDS cost limit: the new CDS charges (fields x grid points) against a limit of 121 000 per
request. The 6-point box costs 6 x 8784 = 52 704 per variable per (leap) year, so a full
year of all variables in one request is refused ("cost limits exceeded"). Each year is
therefore split into chunks of VARS_PER_REQUEST variables (9 requests per year). All
chunks of all years go through one pool of CONCURRENT_REQUESTS in-flight requests, in year
order, so CDS queue waits (often 1 h+) overlap; each year is merged into one file as soon
as its chunks are in.

The new CDS writes instantaneous and accumulated fields to separate NetCDF files and may
return a zip even when `download_format: unarchived` is requested; zips are detected and
their members merged.

After the loop, merge_era5_barter.py is run to (re)build the merged output files.
Requires ~/.cdsapirc for the new CDS (https://cds.climate.copernicus.eu/api).
"""
import logging
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import xarray as xr

# === CONFIGURATION ===
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"                     # per-year raw downloads
PART_DIR = RAW_DIR / "parts"                         # per-chunk temporary downloads
LOG_FILE = HERE / "data" / "download.log"            # appended log
# Validation period first, then the rest of the record
YEARS = list(range(2010, 2026)) + list(range(1979, 2010))
AREA = [70.25, -144.0, 70.0, -143.5]                 # N, W, S, E around Barter Island (70.13 N, 143.7 W)
# Ordered so that chunks mostly hold one CDS stream (accumulated vs instantaneous)
VARIABLES = {
    # accumulated (J m-2 or m over the hour ending at the time stamp)
    "surface_solar_radiation_downwards": "ssrd",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
    "surface_thermal_radiation_downwards": "strd",
    "surface_net_solar_radiation": "ssr",
    "surface_net_thermal_radiation": "str",
    "snowfall": "sf",
    "total_precipitation": "tp",
    "forecast_albedo": "fal",                        # instantaneous from here on
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "surface_pressure": "sp",
    "snow_depth": "sd",
    "snow_density": "rsn",
    "sea_ice_cover": "siconc",
    "sea_surface_temperature": "sst",
    "skin_temperature": "skt",
}
STATIC_VARIABLE = "land_sea_mask"                    # downloaded once, one time step
VARS_PER_REQUEST = 2                                 # 2 x 52 704 < 121 000 cost limit
CONCURRENT_REQUESTS = 10                             # CDS requests in flight at once (all years share one pool)
MAX_ATTEMPTS = 5                                     # per chunk
RETRY_WAIT_S = 120                                   # first back-off; doubles each attempt
RUN_MERGE_AFTER = True                               # run merge_era5_barter.py at the end

log = logging.getLogger("era5_barter")


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        root.addHandler(h)


def expected_hours(year: int) -> int:
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return (366 if leap else 365) * 24


def file_is_valid(path: Path, short_names: list[str], n_time: int | None) -> bool:
    """File opens, holds the variables and (optionally) the expected number of time steps."""
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as ds:
            tname = "valid_time" if "valid_time" in ds.dims else "time"
            missing = [v for v in short_names if v not in ds.data_vars]
            nt = ds.sizes.get(tname, 0)
    except Exception as exc:  # corrupt / truncated file
        log.warning(f"{path.name}: cannot open ({exc})")
        return False
    if missing:
        log.warning(f"{path.name}: missing variables {missing}")
        return False
    if n_time is not None and nt != n_time:
        log.warning(f"{path.name}: {nt} time steps, expected {n_time}")
        return False
    return True


def request(year: int, variables: list[str], static: bool = False) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{year}"],
        "month": ["01"] if static else [f"{m:02d}" for m in range(1, 13)],
        "day": ["01"] if static else [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00"] if static else [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def unzip_if_needed(path: Path) -> None:
    """Replace a CDS zip (instant + accum members) at `path` by one merged NetCDF."""
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


def retrieve(label: str, req: dict, out: Path, short_names: list[str], n_time: int | None) -> bool:
    """Retrieve one request to `out` with retries; returns True when a valid file exists."""
    if file_is_valid(out, short_names, n_time):
        return True
    tmp = out.with_suffix(".part")
    wait = RETRY_WAIT_S
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log.info(f"{label}: submitting (attempt {attempt}/{MAX_ATTEMPTS})")
            t0 = time.time()
            cdsapi.Client(progress=False).retrieve("reanalysis-era5-single-levels", req, str(tmp))
            unzip_if_needed(tmp)
            tmp.replace(out)
            if file_is_valid(out, short_names, n_time):
                log.info(f"{label}: done in {(time.time() - t0) / 60:.1f} min")
                return True
            log.warning(f"{label}: downloaded file failed validity check")
        except Exception as exc:
            log.error(f"{label}: attempt {attempt} failed: {exc}")
        if attempt < MAX_ATTEMPTS:
            log.info(f"{label}: retrying in {wait} s")
            time.sleep(wait)
            wait *= 2
    log.error(f"{label}: giving up after {MAX_ATTEMPTS} attempts")
    return False


def download_static() -> bool:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return retrieve("lsm", request(2016, [STATIC_VARIABLE], static=True),
                    RAW_DIR / "lsm_static.nc", ["lsm"], None)


def year_jobs(year: int) -> list[tuple]:
    """(label, request, part_path, short_names, n_time) for each variable chunk of a year."""
    n_time = expected_hours(year)
    names = list(VARIABLES)
    jobs = []
    for i in range(0, len(names), VARS_PER_REQUEST):
        chunk = names[i:i + VARS_PER_REQUEST]
        shorts = [VARIABLES[v] for v in chunk]
        part = PART_DIR / f"era5_{year}_{'_'.join(shorts)}.nc"
        jobs.append((f"{year} [{','.join(shorts)}]", request(year, chunk), part, shorts, n_time))
    return jobs


def year_done(year: int) -> bool:
    return file_is_valid(RAW_DIR / f"era5_{year}.nc", list(VARIABLES.values()), expected_hours(year))


def assemble_year(year: int, parts: list[Path]) -> bool:
    """Merge the chunk files of one year into data/raw/era5_<year>.nc and remove the chunks."""
    out = RAW_DIR / f"era5_{year}.nc"
    ds = xr.merge([xr.load_dataset(p) for p in parts], compat="override", join="outer",
                  combine_attrs="drop_conflicts")
    tmp = out.with_suffix(".part")
    ds.to_netcdf(tmp)
    tmp.replace(out)
    if not year_done(year):
        log.error(f"{year}: merged year file failed validity check")
        return False
    for p in parts:
        p.unlink()
    log.info(f"{year}: complete, {out.stat().st_size / 1e6:.1f} MB")
    return True


def download_years(years: list[int]) -> list[int]:
    """Download the given years through one shared pool of CDS requests; returns failed years.

    All chunks of all years are queued in year order, so queue waits at CDS overlap across
    years; each year is assembled as soon as its own chunks are in.
    """
    PART_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for y in years:
        if year_done(y):
            log.info(f"{y}: present and valid, skipping")
        else:
            todo.append(y)
    failed = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
        futures = {y: [(j[2], pool.submit(retrieve, *j)) for j in year_jobs(y)] for y in todo}
        for y in todo:  # assemble in queue order; later years keep downloading meanwhile
            ok = [f.result() for _, f in futures[y]]
            if not all(ok):
                log.error(f"{y}: {ok.count(False)} of {len(ok)} chunks failed; parts kept for resume")
                failed.append(y)
            elif not assemble_year(y, [p for p, _ in futures[y]]):
                failed.append(y)
    return failed


def download_year(year: int) -> bool:
    return not download_years([year])


def main() -> None:
    setup_logging()
    log.info(f"=== start: {len(YEARS)} years, order {YEARS[0]}..{YEARS[-1]}, "
             f"{VARS_PER_REQUEST} variables per request, {CONCURRENT_REQUESTS} concurrent ===")
    download_static()
    failed = download_years(YEARS)
    present = sorted(int(p.stem.split("_")[1]) for p in RAW_DIR.glob("era5_*.nc"))
    log.info(f"=== download loop finished; {len(present)} years present; failed: {failed or 'none'} ===")
    if RUN_MERGE_AFTER:
        log.info("running merge_era5_barter.py")
        res = subprocess.run([sys.executable, str(HERE / "merge_era5_barter.py")],
                             capture_output=True, text=True)
        for line in (res.stdout + res.stderr).splitlines():
            log.info(f"[merge] {line}")
        log.info(f"merge exit code {res.returncode}")


if __name__ == "__main__":
    main()
