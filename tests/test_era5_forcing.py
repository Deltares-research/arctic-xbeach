"""ERA5 forcing adapter: time centring and unit handling (run: pytest tests/test_era5_forcing.py)."""

import numpy as np
import pytest
import xarray as xr

from arctic_xbeach.thermal2dv.forcing import era5_forcing


def fake_point_file(path):
    time = np.datetime64("2016-07-01T00:00") + np.arange(6).astype("timedelta64[h]")
    ssrd = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])  # mean over the hour ENDING at the stamp
    ds = xr.Dataset(
        {"ssrd": ("time", ssrd), "fdir": ("time", 0.6 * ssrd), "strd": ("time", np.full(6, 280.0)),
         "t2m": ("time", 273.15 + np.arange(6.0)), "u10": ("time", np.full(6, 3.0)),
         "v10": ("time", np.full(6, 4.0)), "fal": ("time", np.full(6, 0.2)),
         "sst_ocean": ("time", np.array([np.nan, 275.15, 276.15, 276.15, 276.15, 276.15])),
         "siconc_ocean": ("time", np.zeros(6))},
        coords={"time": time})
    ds.to_netcdf(path)
    return path


def test_accumulations_centred_on_their_hour(tmp_path):
    f = era5_forcing(fake_point_file(tmp_path / "pt.nc"), "2016-07-01T00:00")
    zc = np.array([-2.0, -1.0])
    # the value stamped 02:00 describes 01:00-02:00, so it applies at 01:30
    F = f(1.5 * 3600.0, zc)
    assert F.sw_down == pytest.approx(200.0)
    assert F.sw_direct == pytest.approx(120.0) and F.sw_diffuse == pytest.approx(80.0)
    # instantaneous fields are not shifted
    assert F.T_air == pytest.approx(1.5)
    assert F.u10 == pytest.approx(5.0)
    assert F.albedo_land == pytest.approx(0.2)


def test_missing_sst_falls_back_to_freezing(tmp_path):
    f = era5_forcing(fake_point_file(tmp_path / "pt.nc"), "2016-07-01T00:00", S_sea=30.0)
    F = f(0.0, np.array([-2.0]))
    assert F.T_ref[0] == pytest.approx(-1.63, abs=0.02)  # masked SST (sea ice): freezing point at 30 psu
    F = f(3 * 3600.0, np.array([-2.0]))
    assert F.T_ref[0] == pytest.approx(3.0)


def test_missing_box_years_filled_by_climatology(tmp_path):
    """fal missing for a whole year is filled from the day-of-year climatology, not bridged."""
    time = np.arange(np.datetime64("2015-01-01T00:00"), np.datetime64("2018-01-01T00:00"), np.timedelta64(1, "D"))
    doy = (time - time.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    fal = np.where((doy > 160) & (doy < 270), 0.15, 0.75)
    fal = np.where(time.astype("datetime64[Y]") == np.datetime64("2016", "Y"), np.nan, fal)  # 2016 missing
    n = len(time)
    ds = xr.Dataset({"ssrd": ("time", np.zeros(n)), "strd": ("time", np.full(n, 250.0)),
                     "t2m": ("time", np.full(n, 270.0)), "u10": ("time", np.ones(n)), "v10": ("time", np.zeros(n)),
                     "fal": ("time", fal)}, coords={"time": time})
    ds.to_netcdf(tmp_path / "pt.nc")
    f = era5_forcing(tmp_path / "pt.nc", "2016-07-15T00:00", end_time_utc="2016-07-20T00:00")
    assert f(0.0, np.array([-1.0])).albedo_land == pytest.approx(0.15)
    f = era5_forcing(tmp_path / "pt.nc", "2016-01-15T00:00", end_time_utc="2016-01-20T00:00")
    assert f(0.0, np.array([-1.0])).albedo_land == pytest.approx(0.75)
    assert f.filled["fal"] == pytest.approx(1.0)
