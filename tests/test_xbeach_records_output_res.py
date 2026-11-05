"""Every successful XBeach call gets a record in results.nc, whatever output.output_res is. Thermal records stay every output_res
steps. Before 5 Nov 2026 XBeach records were written only at thermal output steps, so a coarse output_res lost most calls, and the
coupled calibration had to keep hourly 2D thermal fields (~4.6 GB per trial on H7).

XBeach itself is stubbed: calls at fixed timesteps write a small xboutput.nc; set-up and grid update do nothing."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from run_helpers import build_run, run_main
from arctic_xbeach.model import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
CALL_STEPS = (5, 30)  # neither is a multiple of output_res = 24


@pytest.fixture()
def xbeach_at_call_steps(monkeypatch):
    def start_xbeach(self, *args, timestep_id=None, **kwargs):
        folder = Path(self.result_dir) / "xb_files" / f"{timestep_id:010d}"
        folder.mkdir(parents=True, exist_ok=True)
        n = len(self.xgr)
        fields = {v: (("globaltime", "x"), np.full((2, n), 0.1 * timestep_id, dtype="f4"))
                  for v in ("H", "zb", "zs", "E", "Sxx", "Sxy", "Syy", "vmag", "urms")}
        xr.Dataset(fields, coords=dict(globaltime=[0.0, 3600.0])).to_netcdf(folder / "xboutput.nc")
        return True

    monkeypatch.setattr(Simulation, "check_xbeach", lambda self, i: 1.0 if i in CALL_STEPS else 0.0)
    monkeypatch.setattr(Simulation, "write_ne_layer", lambda self: None)
    monkeypatch.setattr(Simulation, "xbeach_setup", lambda self, i: None)
    monkeypatch.setattr(Simulation, "update_grid", lambda self, *a, **k: None)
    monkeypatch.setattr(Simulation, "start_xbeach", start_xbeach)


def test_every_successful_call_is_recorded_with_daily_thermal_output(tmp_path, xbeach_at_call_steps):
    run_dir = build_run(tmp_path / "run", time_start="01-05-2010", time_end="03-05-2010", spin_up_days=0)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["xbeach"]["with_xbeach"] = True
    cfg["output"]["output_res"] = 24
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    run_main(run_dir, REPO_ROOT)

    t0 = pd.Timestamp("2010-05-01")
    with xr.open_dataset(run_dir / "results" / "results.nc") as ds:
        step = ds.timestep_id.values
        time_xb = pd.to_datetime(ds.time_xbeach.values)
        zb = ds.zb_xbeach.values
    assert list(step) == [0, 24]  # thermal records only at the output steps
    assert list(time_xb) == [t0 + pd.Timedelta(hours=s + 1) for s in CALL_STEPS]  # both calls, labelled at the end of their step
    assert np.allclose(zb[:, 0], [0.1 * s for s in CALL_STEPS])
