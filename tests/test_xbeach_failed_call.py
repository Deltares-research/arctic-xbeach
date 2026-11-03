"""After a failed XBeach call (xbeach_times set, xbeach_success False) the next step must not use the missing output:
no hotstart request and no attempt to read the water level from the failed call's xboutput.nc."""
import logging
from pathlib import Path

import pytest

from run_helpers import build_run
from test_hydro_forcing import make_forcing_sim

LOGGER = "arctic_xbeach.model"
T = 5  # the step under test; the previous call (step 4) is the one that failed or succeeded
PARAMS = "%% some section\nzs0 = 0\n\n%% Output variables\n"


def xbeach_sim(tmp_path):
    sim = make_forcing_sim(build_run(tmp_path / "run"))
    sim.timesteps_with_xbeach_active()
    sim.config.xbeach.with_xbeach = True
    sim.xbeach_times[T - 1] = 1.0  # a call was made in the previous step
    sim.xbeach_storms[[T - 1, T]] = 1.0
    return sim


def configure(sim, success):
    sim.xbeach_success[T - 1] = success
    previous = Path(sim.result_dir) / "xb_files" / f"{T - 1:010d}"
    previous.mkdir(parents=True)
    (previous / "hotstart_zs_000001.dat").write_text("x")
    folder = Path(sim.result_dir) / "xb_files" / f"{T:010d}"
    folder.mkdir()
    (folder / "params.txt").write_text(PARAMS)
    sim._configure_xbeach_files(T, str(folder), 0.0)
    return folder


def test_no_hotstart_after_a_failed_call(tmp_path):
    sim = xbeach_sim(tmp_path)
    folder = configure(sim, success=False)
    text = (folder / "params.txt").read_text()
    assert "hotstart       = 0" in text and "hotstartfileno = 0" in text
    assert not list(folder.glob("hotstart_*")) and sim.last_xbeach_hotstart is False


def test_hotstart_after_a_successful_storm_call(tmp_path):
    sim = xbeach_sim(tmp_path)
    folder = configure(sim, success=True)
    text = (folder / "params.txt").read_text()
    assert "hotstart       = 1" in text and "hotstartfileno = 1" in text
    assert (folder / "hotstart_zs_000001.dat").exists() and sim.last_xbeach_hotstart is True


def test_water_level_does_not_read_the_output_of_a_failed_call(tmp_path, caplog):
    sim = xbeach_sim(tmp_path)
    sim.xbeach_success[T - 1] = False
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert sim._update_water_level(T) == sim.water_levels[T]
    assert not caplog.records  # the old code tried to load the missing file and warned
