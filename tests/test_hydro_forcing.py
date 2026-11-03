"""Hydrodynamic forcing: hours with missing waves get placeholder waves but keep their own water level."""
import os
from pathlib import Path

import pandas as pd

from run_helpers import build_run
from arctic_xbeach.model import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_missing_wave_hours_keep_their_own_water_level(tmp_path):
    run_dir = build_run(tmp_path / "run")
    storms = pd.read_csv(run_dir / "storms.csv", parse_dates=["time"])
    gap = (storms["time"] >= "2010-05-02") & (storms["time"] < "2010-05-03")
    storms.loc[gap, "Hs(m)"] = float("nan")
    storms.to_csv(run_dir / "storms.csv", index=False)

    cwd = os.getcwd()
    sim = Simulation(str(run_dir), proj_dir=str(REPO_ROOT))
    os.chdir(cwd)
    cfg = sim.config
    sim.set_temporal_params(cfg.model.time_start, cfg.model.time_end, cfg.model.timestep)
    sim.load_forcing(cfg.data.forcing_data_path)
    sim.initialize_hydro_forcing(cfg.data.storm_data_path)

    gap_ids = [i for i, t in enumerate(sim.timestamps) if pd.Timestamp("2010-05-02") <= t < pd.Timestamp("2010-05-03")]
    assert len(gap_ids) == 24
    assert all(sim.conditions[i]["Hs(m)"] == 0.05 for i in gap_ids)
    assert all(sim.conditions[i]["WL(m)"] == sim.water_levels[i] for i in gap_ids)

    # the placeholder set is overwritten at inter-storm XBeach calls, which must not reach the stored hours
    sim.zero_conditions["WL(m)"] = 99.0
    assert all(sim.conditions[i]["WL(m)"] == sim.water_levels[i] for i in gap_ids)
