"""calibration/run_thermal_experiments.py without running the model: the run folders it builds, the 2011 scorer, the thaw depth, the
parameter source and the run/score/resume cycle with a stubbed child process and a small hand-made results.nc."""
import json
import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "calibration"))
import run_thermal_experiments as exp  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
RAW = dict(exp.rtc.START_PARAMS)  # nb_min 0.4695, nb_increase 0.2406, k_soil_frozen_min 0.9046, geothermal 0.0204 ...
N_Z = 100


@pytest.fixture()
def setup(monkeypatch, tmp_path):
    """A base folder (barter_island config with the 2006-2017 window of the real base) and OUT_ROOTS in tmp_path."""
    base = tmp_path / "base"
    base.mkdir()
    cfg = yaml.safe_load((REPO_ROOT / "examples" / "case_studies" / "barter_island" / "config.yaml").read_text())
    cfg["model"]["time_start"], cfg["model"]["time_end"] = "01-01-2006", "01-01-2017"
    cfg["output"]["output_res"] = 1
    (base / "config.yaml").write_text(yaml.dump(cfg))
    for name in ("x.grd", "bed.dep", "ne_layer.txt"):
        shutil.copy(REPO_ROOT / "examples" / "case_studies" / "barter_island" / name, base / name)
    monkeypatch.setattr(exp, "BASE_SIM_DIR", base)
    monkeypatch.setattr(exp, "OUT_ROOTS", {k: tmp_path / k for k in ("validation_2011", "sensitivity", "dt_test")})
    monkeypatch.setattr(exp, "TAG", "")
    monkeypatch.setattr(exp, "EXPERIMENTS", ["validation_2011", "sensitivity", "dt_test"])
    monkeypatch.setattr(exp, "THERMAL_DT", 60)
    monkeypatch.setattr(exp, "DT_TEST", [30, 90, 300])
    return tmp_path


def build(job) -> dict:
    run_dir = exp.root_of(job.exp) / job.name
    exp.build_run_folder(run_dir, job)
    return yaml.safe_load((run_dir / "config.yaml").read_text())


def test_jobs_cover_all_experiments(setup):
    names = {j.name for j in exp.make_jobs(RAW)}
    assert names == {"baseline", "validation_2011", "dt030", "dt090", "dt300",
                     *(f"sens_{p}_{lvl}" for p in ("nb_min", "k_soil_frozen_min", "k_soil_unfrozen_min", "c_soil_frozen", "geothermal_gradient") for lvl in ("low", "high"))}


def test_dt_that_does_not_divide_the_timestep_is_refused(setup, monkeypatch):
    monkeypatch.setattr(exp, "DT_TEST", [700])
    with pytest.raises(ValueError, match="does not divide"):
        exp.make_jobs(RAW)


def test_baseline_folder_has_the_trial_parameters(setup):
    jobs = {j.name: j for j in exp.make_jobs(RAW)}
    t = build(jobs["baseline"])
    th = t["thermal"]
    assert th["dt"] == 60 and th["conduction_form"] == "flux" and th["latent_heat_form"] == "volumetric"
    assert th["nb_min"] == pytest.approx(RAW["nb_min"]) and th["nb_max"] == pytest.approx(RAW["nb_min"] + RAW["nb_increase"])
    assert th["k_soil_frozen_max"] == pytest.approx(RAW["k_soil_frozen_min"] * RAW["k_compaction_factor"])
    assert th["c_soil_unfrozen"] == pytest.approx(RAW["c_soil_frozen"] * RAW["c_ratio"])
    assert th["depth_constant_k"] == pytest.approx((RAW["transition_top"] + RAW["transition_bottom"]) / 2)
    assert (th["max_depth"], th["grid_resolution"], th["rho_ice"]) == (10, 200, 917)
    assert th["geothermal_gradient"] == pytest.approx(RAW["geothermal_gradient"])
    assert (t["output"]["output_start"], t["output"]["output_end"]) == exp.WINDOW_2016
    assert (t["model"]["time_start"], t["model"]["time_end"]) == ("01-01-2006", "01-01-2017")
    assert t["data"]["era5_flux_shift_hours"] == exp.rtc.ERA5_FLUX_SHIFT_HOURS


@pytest.mark.parametrize("name, check", [
    ("sens_nb_min_low", lambda th: (th["nb_min"], th["nb_max"]) == pytest.approx((RAW["nb_min"] - 0.1, RAW["nb_min"] - 0.1 + RAW["nb_increase"]))),
    ("sens_nb_min_high", lambda th: (th["nb_min"], th["nb_max"]) == pytest.approx((RAW["nb_min"] + 0.1, RAW["nb_min"] + 0.1 + RAW["nb_increase"]))),
    ("sens_k_soil_frozen_min_low", lambda th: (th["k_soil_frozen_min"], th["k_soil_frozen_max"], th["k_soil_unfrozen_min"])
        == pytest.approx((RAW["k_soil_frozen_min"] * 0.75, RAW["k_soil_frozen_min"] * 0.75 * RAW["k_compaction_factor"], RAW["k_soil_unfrozen_min"]))),
    ("sens_k_soil_unfrozen_min_high", lambda th: (th["k_soil_unfrozen_min"], th["k_soil_unfrozen_max"], th["k_soil_frozen_min"])
        == pytest.approx((RAW["k_soil_unfrozen_min"] * 1.25, RAW["k_soil_unfrozen_min"] * 1.25 * RAW["k_compaction_factor"], RAW["k_soil_frozen_min"]))),
    ("sens_c_soil_frozen_low", lambda th: (th["c_soil_frozen"], th["c_soil_unfrozen"])
        == pytest.approx((RAW["c_soil_frozen"] * 0.75, RAW["c_soil_frozen"] * 0.75 * RAW["c_ratio"]))),
    ("sens_c_soil_frozen_high", lambda th: th["c_soil_frozen"] == pytest.approx(RAW["c_soil_frozen"] * 1.25)),
    ("sens_geothermal_gradient_low", lambda th: th["geothermal_gradient"] == 0.020),
    ("sens_geothermal_gradient_high", lambda th: th["geothermal_gradient"] == 0.040),
])
def test_sensitivity_levels_change_one_parameter(setup, name, check):
    jobs = {j.name: j for j in exp.make_jobs(RAW)}
    th = build(jobs[name])["thermal"]
    assert check(th)
    # everything else equals the baseline (the transition depths never move)
    assert th["nb_min_depth"] == pytest.approx(RAW["transition_bottom"]) and th["nb_max_depth"] == pytest.approx(RAW["transition_top"])
    assert th["dt"] == 60


def test_levels_outside_the_search_range_are_reported(setup):
    assert exp.sample_fixed(RAW)[1] == []
    jobs = {j.name: j for j in exp.make_jobs(RAW)}
    assert exp.sample_fixed(jobs["sens_nb_min_high"].raw)[1] == ["nb_min"]  # 0.57 > 0.50
    assert exp.sample_fixed(jobs["sens_c_soil_frozen_low"].raw)[1] == ["c_soil_frozen"]


def test_sensitivity_job_records_value_and_baseline_value(setup):
    job = {j.name: j for j in exp.make_jobs(RAW)}["sens_nb_min_low"]
    assert job.info == dict(parameter="nb_min", level="low", value=pytest.approx(RAW["nb_min"] - 0.1), baseline_value=RAW["nb_min"])


@pytest.mark.parametrize("name, dt", [("dt030", 30), ("dt090", 90), ("dt300", 300)])
def test_dt_test_folders(setup, name, dt):
    jobs = {j.name: j for j in exp.make_jobs(RAW)}
    t = build(jobs[name])
    assert t["thermal"]["dt"] == dt and jobs[name].exp == "dt_test"
    assert t["thermal"]["nb_min"] == pytest.approx(RAW["nb_min"])


def test_dt_test_does_not_repeat_the_baseline_dt(setup, monkeypatch):
    monkeypatch.setattr(exp, "DT_TEST", [30, 60, 90])
    assert "dt060" not in {j.name for j in exp.make_jobs(RAW)}


def test_validation_2011_folder_has_its_time_and_output_window(setup):
    t = build({j.name: j for j in exp.make_jobs(RAW)}["validation_2011"])
    assert (t["model"]["time_start"], t["model"]["time_end"]) == exp.TIME_2011 == ("01-01-2001", "01-10-2012")
    assert (t["output"]["output_start"], t["output"]["output_end"]) == exp.WINDOW_2011 == ("01-05-2011", "31-12-2011")


def test_base_config_that_misses_the_2016_season_is_refused(setup):
    cfg_path = exp.BASE_SIM_DIR / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["model"]["time_end"] = "01-01-2016"
    cfg_path.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="does not cover"):
        build({j.name: j for j in exp.make_jobs(RAW)}["baseline"])


def test_tag_is_appended_to_every_root(setup, monkeypatch):
    monkeypatch.setattr(exp, "TAG", "_dryrun")
    assert exp.root_of("sensitivity").name == "sensitivity_dryrun"


def test_unknown_experiment_is_refused(setup, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIMENTS", ["sensitivty"])
    with pytest.raises(ValueError, match="EXPERIMENTS"):
        exp.make_jobs(RAW)


# --- parameter source ---------------------------------------------------------------------------------------------------
def test_params_come_from_start_params_without_a_study(monkeypatch):
    monkeypatch.setattr(exp, "STUDY_STORAGE", None)
    raw, source = exp.load_raw_params()
    assert raw == RAW and "START_PARAMS" in source["source"]


def test_params_come_from_best_or_chosen_trial(monkeypatch, tmp_path):
    db = tmp_path / "study.db"
    study = optuna.create_study(study_name="s", direction="minimize", storage=f"sqlite:///{db.as_posix()}")
    dist = {"x": optuna.distributions.FloatDistribution(0, 1)}
    for x, value in ((0.1, 2.0), (0.2, 1.0), (0.3, 3.0)):
        study.add_trial(optuna.trial.create_trial(params={"x": x}, distributions=dist, value=value))
    monkeypatch.setattr(exp, "STUDY_STORAGE", db)
    monkeypatch.setattr(exp, "STUDY_NAME", "s")
    monkeypatch.setattr(exp, "TRIAL_NUMBER", None)
    raw, source = exp.load_raw_params()
    assert raw == {"x": 0.2} and source["trial_number"] == 1 and source["value"] == 1.0
    monkeypatch.setattr(exp, "TRIAL_NUMBER", 2)
    assert exp.load_raw_params()[0] == {"x": 0.3}
    monkeypatch.setattr(exp, "TRIAL_NUMBER", 7)
    with pytest.raises(ValueError, match="not found"):
        exp.load_raw_params()


# --- scoring ------------------------------------------------------------------------------------------------------------
def write_results(run_dir: Path, start: str, n_records: int, thaw=None, temp_of_depth=lambda z: -z) -> None:
    """results.nc with hourly records labelled at the end of their step; the temperature of a node [degC] is temp_of_depth(depth)."""
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    depth = 0.05 * np.arange(N_Z)
    with netCDF4.Dataset(run_dir / "results" / "results.nc", "w") as nc:
        for dim, size in (("time", None), ("xgr", 3), ("depth_id", N_Z)):
            nc.createDimension(dim, size)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = f"seconds since {start}"
        t[:] = 3600.0 * np.arange(0, n_records)
        nc.createVariable("abs_zgr", "f4", ("time", "xgr", "depth_id"))[:] = np.tile(15.0 - depth, (n_records, 3, 1))
        nc.createVariable("ground_temperature_distribution", "f4", ("time", "xgr", "depth_id"))[:] = np.tile(273.15 + temp_of_depth(depth), (n_records, 3, 1))
        if thaw is not None:
            nc.createVariable("thaw_depth", "f4", ("time", "xgr"))[:] = thaw


def write_config(run_dir: Path, time_end: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dict(model=dict(time_start="01-01-2011", time_end=time_end, timestep=1), output=dict(output_res=1))))


def test_2011_scorer_uses_true_depths_and_pools(monkeypatch, tmp_path):
    monkeypatch.setattr(exp, "WINDOW_2011", ("01-07-2011", "03-07-2011"))
    run_dir = tmp_path / "run"
    write_config(run_dir, "01-10-2012")
    write_results(run_dir, "2011-07-01 00:00:00", 48)  # model temperature = -depth, constant in time
    times = pd.date_range("2011-07-01 12:00", periods=10, freq="h")
    offsets = {0.50: 0.2, 1.00: -0.4, 2.00: 0.0, 2.95: 1.0}  # observation = model - offset, so the bias is +offset
    obs = pd.DataFrame({"SiteID": "BL01", "time": times, **{col: -d - offsets[d] for d, col in exp.OBS_2011_COLUMNS.items()}})
    other = obs.assign(SiteID="XX99", T50cm=99.0)  # another site must be ignored
    obs_csv = tmp_path / "obs.csv"
    pd.concat([obs, other]).to_csv(obs_csv, index=False)

    table = exp.score_validation_2011(run_dir, obs_csv, make_figure=True)
    depths = table["depth_m"].tolist()
    assert depths[:4] == [0.5, 1.0, 2.0, 2.95] and depths[4] == "pooled"
    assert table["n"].tolist() == [10, 10, 10, 10, 40]
    assert table["bias"].iloc[:4].tolist() == pytest.approx([0.2, -0.4, 0.0, 1.0], abs=1e-3)  # float32 temperatures
    assert table["rmse"].iloc[:4].tolist() == pytest.approx([0.2, 0.4, 0.0, 1.0], abs=1e-3)
    assert table["mae"].iloc[:4].tolist() == pytest.approx([0.2, 0.4, 0.0, 1.0], abs=1e-3)
    assert table["bias"].iloc[4] == pytest.approx(np.mean([0.2, -0.4, 0.0, 1.0]), abs=1e-3)
    assert table["rmse"].iloc[4] == pytest.approx(np.sqrt(np.mean([0.04, 0.16, 0.0, 1.0])), abs=1e-3)
    assert (run_dir / "validation_2011_metrics.csv").exists() and (run_dir / "validation_2011.png").exists()


def test_2011_scorer_refuses_a_run_that_stops_early(monkeypatch, tmp_path):
    monkeypatch.setattr(exp, "WINDOW_2011", ("01-07-2011", "03-07-2011"))
    run_dir = tmp_path / "run"
    write_config(run_dir, "01-10-2012")
    write_results(run_dir, "2011-07-01 00:00:00", 30)
    pd.DataFrame({"time": [pd.Timestamp("2011-07-01 12:00")], **{c: [0.0] for c in exp.OBS_2011_COLUMNS.values()}}).to_csv(tmp_path / "obs.csv", index=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        exp.score_validation_2011(run_dir, tmp_path / "obs.csv", make_figure=False)


def test_max_thaw_depth_is_the_2016_maximum_of_column_1(tmp_path):
    n = 72
    thaw = np.zeros((n, 3))
    thaw[:, 0] = 9.0                              # edge columns are not scored
    thaw[:, 2] = 9.0
    thaw[10, 1], thaw[40, 1] = 0.30, 0.55
    thaw[60, 1] = 0.9                             # after the turn of the year: not 2016
    write_results(tmp_path, "2016-12-30 00:00:00", n, thaw=thaw)
    assert exp.max_thaw_depth(tmp_path / "results" / "results.nc", 2016) == pytest.approx(0.55)


# --- run, score, resume ---------------------------------------------------------------------------------------------------
def test_run_job_scores_writes_scored_json_and_resumes(setup, monkeypatch):
    tmp_path = setup
    window = ("01-06-2016", "03-06-2016")
    monkeypatch.setattr(exp, "WINDOW_2016", window)
    obs = pd.DataFrame({"date_time": pd.date_range("2016-06-01 06:00", periods=24, freq="h"),
                        **{f"dpth{d}cm": -d / 100 + 0.5 for d in range(0, 426, 25) if d != 75}})  # observation = model + 0.5 at every depth
    obs.to_csv(tmp_path / "obs.csv", index=False)
    monkeypatch.setattr(exp.rtc, "OBSERVATION_CSV", tmp_path / "obs.csv")
    thaw = np.zeros((49, 3))
    thaw[20, 1] = 0.35
    calls = []

    def fake_child(run_dir):
        calls.append(run_dir.name)
        write_results(run_dir, "2016-06-01 00:00:00", 49, thaw=thaw)  # model = -depth degC; the surface is never above 0.01: penalty x10
        return 0

    monkeypatch.setattr(exp, "run_child", fake_child)
    job = {j.name: j for j in exp.make_jobs(RAW)}["baseline"]
    result = exp.run_job(job)
    run_dir = exp.root_of("sensitivity") / "baseline"
    assert calls == ["baseline"] and "failed" not in result
    assert result["max_thaw_depth_m"] == pytest.approx(0.35, abs=1e-6)
    assert result["objective"] == pytest.approx(5.0, abs=1e-3)  # rmse 0.5 at all depths, x10 because the surface never thaws
    assert json.loads((run_dir / "scored.json").read_text())["objective"] == pytest.approx(result["objective"])
    assert (run_dir / "rmse_bias_layers.txt").exists() and (run_dir / "temperature_layers.png").exists()
    table = pd.read_csv(run_dir / "bias_by_depth_2016.csv")
    assert list(table.columns) == ["depth_m", "rmse", "mae", "bias", "n"] and len(table) == 17
    assert table["bias"].tolist() == pytest.approx([-0.5] * 17, abs=1e-3)
    assert not (run_dir / "results" / "results.nc").exists()  # KEEP_RESULTS_NC is False

    # a second call neither builds nor runs the folder again
    again = exp.run_job(job)
    assert calls == ["baseline"] and again["skipped"] and again["objective"] == pytest.approx(result["objective"])


def test_failed_child_is_not_scored_and_is_retried(setup, monkeypatch):
    codes = iter([1, 1])
    monkeypatch.setattr(exp, "run_child", lambda run_dir: next(codes))
    job = {j.name: j for j in exp.make_jobs(RAW)}["baseline"]
    assert exp.run_job(job) == {"failed": "exit code 1"}
    assert not (exp.root_of("sensitivity") / "baseline" / "scored.json").exists()
    assert exp.run_job(job) == {"failed": "exit code 1"}  # not skipped


def test_summaries(setup, monkeypatch):
    jobs = exp.make_jobs(RAW)
    results = {"baseline": dict(objective=1.0, max_thaw_depth_m=0.50, runtime_min=20.0),
               "sens_nb_min_low": dict(objective=1.2, max_thaw_depth_m=0.55),
               "sens_geothermal_gradient_low": dict(objective=1.0, max_thaw_depth_m=0.50),  # not failed
               "dt030": dict(objective=1.01, max_thaw_depth_m=0.50, runtime_min=40.0),
               "dt300": dict(objective=1.1, max_thaw_depth_m=0.45, runtime_min=6.0)}
    for j in jobs:
        exp.root_of(j.exp).mkdir(parents=True, exist_ok=True)
    # a level that equals the baseline value is flagged
    monkeypatch.setattr(exp, "SENSITIVITY_LEVELS", [("geothermal_gradient", "low", "set", RAW["geothermal_gradient"] + 5e-5), ("nb_min", "low", "add", -0.1)])
    jobs = exp.make_jobs(RAW)
    monkeypatch.setattr(exp, "EXPERIMENTS", ["sensitivity", "dt_test"])
    tables = exp.write_summaries(jobs, results, RAW, dict(source="test"))
    s = tables["sensitivity"].set_index("run")
    assert s.loc["baseline", "d_rmse"] == 0.0
    assert s.loc["sens_nb_min_low", "d_rmse"] == pytest.approx(0.2) and s.loc["sens_nb_min_low", "d_thaw_m"] == pytest.approx(0.05)
    assert s.loc["sens_geothermal_gradient_low", "flag"] == "equals_baseline"
    assert s.loc["baseline", "flag"] == "" and s.loc["sens_nb_min_low", "flag"] == ""
    assert list(tables["sensitivity"].columns) == ["run", "parameter", "level", "value", "baseline_value", "objective_rmse", "d_rmse", "max_thaw_depth_m", "d_thaw_m", "flag"]
    dt = tables["dt_test"]
    assert dt["dt"].tolist() == [30, 60, 90, 300]
    assert dt.set_index("dt").loc[300, "d_thaw_m"] == pytest.approx(-0.05) and np.isnan(dt.set_index("dt").loc[90, "objective_rmse"])
    assert (exp.root_of("sensitivity") / "sensitivity_summary.csv").exists() and (exp.root_of("dt_test") / "dt_test_summary.csv").exists()
    used = json.loads((exp.root_of("sensitivity") / "params_used.json").read_text())
    assert used["raw_params"]["nb_min"] == RAW["nb_min"] and used["thermal_dt"] == 60 and used["source"] == {"source": "test"}


def test_bias_by_depth_joins_2016_and_2011(setup, monkeypatch):
    jobs = exp.make_jobs(RAW)
    for j in jobs:
        exp.root_of(j.exp).mkdir(parents=True, exist_ok=True)
    base_dir = exp.root_of("sensitivity") / "baseline"
    base_dir.mkdir()
    pd.DataFrame(dict(depth_m=[0.5, 1.0, 2.0], rmse=[1, 2, 3], mae=[1, 2, 3], bias=[0.1, 0.2, 0.3], n=[5, 5, 5])).to_csv(base_dir / "bias_by_depth_2016.csv", index=False)
    run11 = exp.root_of("validation_2011") / "validation_2011"
    run11.mkdir()
    pd.DataFrame(dict(depth_m=[0.5, 2.95, "pooled"], rmse=[4, 5, 6], mae=[4, 5, 6], bias=[-0.1, -0.5, -0.3], n=[7, 7, 14])).to_csv(run11 / "validation_2011_metrics.csv", index=False)
    monkeypatch.setattr(exp, "EXPERIMENTS", ["validation_2011"])
    tables = exp.write_summaries(jobs, {}, RAW, {})
    b = tables["bias_by_depth"].set_index("depth_m")
    assert b.index.tolist() == [0.5, 1.0, 2.0, 2.95]
    assert b.loc[0.5, "bias_2016"] == 0.1 and b.loc[0.5, "bias_2011"] == -0.1
    assert np.isnan(b.loc[2.95, "bias_2016"]) and b.loc[2.95, "bias_2011"] == -0.5
    assert (exp.root_of("validation_2011") / "validation_2011_metrics.csv").exists()
    assert (exp.root_of("validation_2011") / "bias_by_depth_2016_2011.csv").exists()
