"""Hydrodynamic forcing: hours with missing waves get placeholder waves but keep their own water level."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

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


def make_forcing_sim(run_dir, forcing=True, storms=True):
    """Simulation of a build_run folder up to the forcing; restores the cwd that Simulation() changes."""
    cwd = os.getcwd()
    sim = Simulation(str(run_dir), proj_dir=str(REPO_ROOT))
    os.chdir(cwd)
    cfg = sim.config
    sim.set_temporal_params(cfg.model.time_start, cfg.model.time_end, cfg.model.timestep)
    if forcing:
        sim.load_forcing(cfg.data.forcing_data_path)
    if storms:
        sim.initialize_hydro_forcing(cfg.data.storm_data_path)
    return sim


def set_timestep(run_dir, hours):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["model"]["timestep"] = hours
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def drop_hour(csv_path, when):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df[df["time"] != pd.Timestamp(when)].to_csv(csv_path, index=False)


def test_hourly_forcing_is_the_file_rows_of_the_run_window(tmp_path):
    run_dir = build_run(tmp_path / "run")
    sim = make_forcing_sim(run_dir)
    window = lambda df: df[(df["time"] >= sim.t_start) & (df["time"] < sim.t_end)].reset_index(drop=True)
    forcing = window(pd.read_csv(run_dir / "forcing.csv", parse_dates=["time"]))
    storms = window(pd.read_csv(run_dir / "storms.csv", parse_dates=["time"]))

    assert len(sim.forcing_data) == len(sim.timestamps)
    for col in ("time", "2m_temperature", "sea_ice_cover", "soil_temperature_level_1"):
        assert np.array_equal(sim.forcing_data[col].values, forcing[col].values), col
    assert np.array_equal(sim.water_levels[:len(sim.timestamps)], storms["WL(m)"].values)
    assert all(sim.conditions[i]["Hs(m)"] == storms["Hs(m)"][i] for i in sim.T)


def test_three_hour_steps_read_forcing_at_their_own_time(tmp_path):
    run_dir = build_run(tmp_path / "run")
    set_timestep(run_dir, 3)
    sim = make_forcing_sim(run_dir)
    forcing = pd.read_csv(run_dir / "forcing.csv", parse_dates=["time"]).set_index("time")
    storms = pd.read_csv(run_dir / "storms.csv", parse_dates=["time"]).set_index("time")

    assert len(sim.forcing_data) == len(sim.timestamps)
    for k in (1, 2, 17):
        t = sim.timestamps[k]
        assert t == sim.t_start + pd.Timedelta(hours=3 * k)
        assert sim.forcing_data["time"].iloc[k] == t
        assert sim.forcing_data["2m_temperature"].iloc[k] == forcing.loc[t, "2m_temperature"]
        assert sim.water_levels[k] == storms.loc[t, "WL(m)"]
        assert sim.conditions[k]["WL(m)"] == storms.loc[t, "WL(m)"]


def test_forcing_with_a_missing_hour_is_an_error(tmp_path):
    run_dir = build_run(tmp_path / "run")
    drop_hour(run_dir / "forcing.csv", "2010-05-10 05:00")
    with pytest.raises(ValueError, match="2010-05-10 05:00"):
        make_forcing_sim(run_dir, storms=False)


def test_storms_with_a_missing_hour_is_an_error(tmp_path):
    run_dir = build_run(tmp_path / "run")
    drop_hour(run_dir / "storms.csv", "2010-05-10 05:00")
    with pytest.raises(ValueError, match="2010-05-10 05:00"):
        make_forcing_sim(run_dir)
