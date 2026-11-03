"""The calibration drivers on a cluster: AXB_* environment variables, journal storage, scratch folders, trials left RUNNING.

No model is run: the child process / scoring function of a driver is replaced by a stub that writes the files a run leaves behind.
"""
import importlib.util
import sys
from pathlib import Path

import optuna
import pandas as pd
import pytest
import yaml

CALIB_DIR = Path(__file__).resolve().parents[1] / "calibration"
AXB_VARIABLES = [
    "AXB_BASE_SIM_DIR", "AXB_RUN_ROOT", "AXB_OBS_CSV", "AXB_BLUFF_EDGES_CSV", "AXB_SHORELINES_CSV", "AXB_THERMAL_CONFIG_YAML",
    "AXB_FORCING_CSV", "AXB_STORM_CSV", "AXB_XBEACH_EXE", "AXB_N_TRIALS", "AXB_N_PARALLEL", "AXB_STUDY_NAME", "AXB_THERMAL_DT",
    "AXB_SENSOR_DEPTH_OFFSET", "AXB_STORAGE", "AXB_SCRATCH", "AXB_KEEP_XB_FILES", "AXB_CONSTANT_LIAR",
]

sys.path.insert(0, str(CALIB_DIR))  # the drivers import optuna_hpc from their own folder when run as scripts
import optuna_hpc as hpc  # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_driver(name):
    """Execute the driver file again, so that the configuration block reads the environment as it is now."""
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", CALIB_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for var in AXB_VARIABLES:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def observations(tmp_path):
    """Small observation csv files for the import-time reading of the coupled driver."""
    obs = pd.DataFrame({"time": pd.to_datetime(["2012-07-11", "2013-07-11"]), "distance_from_ref": [4.0, 5.0]})
    obs.to_csv(tmp_path / "be.csv", index=False)
    obs.to_csv(tmp_path / "sl.csv", index=False)
    return tmp_path / "be.csv", tmp_path / "sl.csv"


# --- (a) environment variables ---------------------------------------------------------------------------------------
def test_thermal_defaults_are_the_desktop_values():
    m = load_driver("run_thermal_calibration")
    assert str(m.RUN_ROOT_DIR) == r"N:\arctic-xbeach-runs\v11_runs\thermal_calibration"
    assert str(m.FORCING_CSV) == r"N:\arctic-xbeach-runs\database\era5.csv"
    assert (m.N_TRIALS, m.N_PARALLEL, m.THERMAL_DT, m.SENSOR_DEPTH_OFFSET) == (1000, 8, 360, 0.0)
    assert (m.STUDY_NAME, m.STORAGE, m.SCRATCH, m.KEEP_XB_FILES, m.CONSTANT_LIAR) == ("calibration_optuna", "sqlite", None, True, False)


def test_thermal_environment_overrides_the_configuration(monkeypatch, tmp_path):
    for var, value in dict(AXB_BASE_SIM_DIR=tmp_path / "base", AXB_RUN_ROOT=tmp_path / "run", AXB_OBS_CSV=tmp_path / "obs.csv",
                           AXB_FORCING_CSV=tmp_path / "era5.csv", AXB_STORM_CSV=tmp_path / "storms.csv", AXB_SCRATCH=tmp_path / "scratch",
                           AXB_N_TRIALS=5, AXB_N_PARALLEL=42, AXB_THERMAL_DT=90, AXB_SENSOR_DEPTH_OFFSET=0.1, AXB_STUDY_NAME="s",
                           AXB_STORAGE="journal", AXB_KEEP_XB_FILES=0, AXB_CONSTANT_LIAR=1).items():
        monkeypatch.setenv(var, str(value))
    m = load_driver("run_thermal_calibration")
    assert (m.BASE_SIM_DIR, m.RUN_ROOT_DIR, m.OBSERVATION_CSV) == (tmp_path / "base", tmp_path / "run", tmp_path / "obs.csv")
    assert (m.FORCING_CSV, m.STORM_CSV, m.SCRATCH) == (tmp_path / "era5.csv", tmp_path / "storms.csv", tmp_path / "scratch")
    assert (m.N_TRIALS, m.N_PARALLEL, m.THERMAL_DT, m.SENSOR_DEPTH_OFFSET) == (5, 42, 90, 0.1)
    assert (m.STUDY_NAME, m.STORAGE, m.KEEP_XB_FILES, m.CONSTANT_LIAR) == ("s", "journal", False, True)


def test_coupled_environment_overrides_the_configuration(monkeypatch, tmp_path, observations):
    be, sl = observations
    for var, value in dict(AXB_BLUFF_EDGES_CSV=be, AXB_SHORELINES_CSV=sl, AXB_THERMAL_CONFIG_YAML=tmp_path / "thermal.yaml",
                           AXB_XBEACH_EXE="/opt/xbeach/bin/xbeach", AXB_RUN_ROOT=tmp_path / "run", AXB_N_PARALLEL=42).items():
        monkeypatch.setenv(var, str(value))
    m = load_driver("auto_calibrate_coupled_model")
    assert (m.BLUFF_EDGES_CSV, m.SHORELINES_CSV, m.THERMAL_CONFIG_YAML) == (be, sl, tmp_path / "thermal.yaml")
    assert (m.XBEACH_EXE, m.RUN_ROOT_DIR, m.N_PARALLEL) == ("/opt/xbeach/bin/xbeach", tmp_path / "run", 42)
    assert len(m.DF_BE) == 2 and m.DF_SL["distance_from_ref"].tolist() == [4.0, 5.0]  # the observations come from the files named by the environment


def test_coupled_without_xbeach_exe_keeps_the_base_config(monkeypatch, observations):
    be, sl = observations
    monkeypatch.setenv("AXB_BLUFF_EDGES_CSV", str(be))
    monkeypatch.setenv("AXB_SHORELINES_CSV", str(sl))
    monkeypatch.setenv("AXB_XBEACH_EXE", "")
    assert load_driver("auto_calibrate_coupled_model").XBEACH_EXE is None


# --- (b) a driver run with a stub child ------------------------------------------------------------------------------
def fake_run(sim_dir: Path) -> float:
    """What a run leaves in its folder: run.log, results.nc and XBeach call folders. The value depends on the trial folder."""
    (sim_dir / "run.log").write_text(f"stub run in {sim_dir}\n")
    (sim_dir / "metrics.txt").write_text("stub\n")
    (sim_dir / "results" / "xb_files" / "0000000001").mkdir(parents=True)
    (sim_dir / "results" / "xb_files" / "0000000001" / "params.txt").write_text("x")
    (sim_dir / "results" / "results.nc").write_text("not netcdf")
    return 1.0 + int(sim_dir.name[-6:])


def thermal_driver(monkeypatch, tmp_path, n_trials):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "config.yaml").write_text(yaml.safe_dump(dict(thermal={"dt": 360}, data={}, output={})))
    for var, value in dict(AXB_BASE_SIM_DIR=base, AXB_RUN_ROOT=tmp_path / "run", AXB_SCRATCH=tmp_path / "scratch", AXB_STORAGE="journal",
                           AXB_N_TRIALS=n_trials, AXB_N_PARALLEL=2, AXB_CONSTANT_LIAR=1, AXB_FORCING_CSV=tmp_path / "era5.csv",
                           AXB_STORM_CSV=tmp_path / "storms.csv").items():
        monkeypatch.setenv(var, str(value))
    (tmp_path / "scratch").mkdir(exist_ok=True)
    m = load_driver("run_thermal_calibration")
    monkeypatch.setattr(m, "START_PARAMS", None)
    monkeypatch.setattr(m, "run_simulation", fake_run)
    return m


def journal_study(tmp_path):
    return optuna.load_study(study_name="calibration_optuna", storage=hpc.make_storage("journal", tmp_path / "run"))


def test_thermal_driver_with_journal_and_scratch(monkeypatch, tmp_path):
    m = thermal_driver(monkeypatch, tmp_path, n_trials=3)
    m.main()

    for number in range(3):
        folder = tmp_path / "run" / f"optuna_trial_{number:06d}"
        assert (folder / "run.log").exists() and (folder / "config.yaml").exists() and (folder / "metrics.txt").exists()
        assert not (folder / "results" / "xb_files").exists()  # never copied back
        assert not (folder / "results" / "results.nc").exists()  # KEEP_RESULTS_NC is False
    assert list((tmp_path / "scratch").iterdir()) == []  # scratch is empty again
    assert (tmp_path / "run" / "optuna_journal.log").exists() and not (tmp_path / "run" / "optuna_study.db").exists()

    study = journal_study(tmp_path)
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.COMPLETE] * 3
    assert all(Path(t.user_attrs["sim_dir"]).parent == tmp_path / "run" for t in study.trials)  # the RUN_ROOT path, not the scratch one
    assert (tmp_path / "run" / "calibration_prints.txt").read_text().count("completed.") == 3


def test_failed_trial_is_copied_back_with_its_run_log(monkeypatch, tmp_path):
    m = thermal_driver(monkeypatch, tmp_path, n_trials=1)

    def failing_run(sim_dir):
        (sim_dir / "run.log").write_text("crashed\n")
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "run_simulation", failing_run)
    with pytest.raises(RuntimeError):
        m.objective(optuna.create_study().ask())
    assert (tmp_path / "run" / "optuna_trial_000000" / "run.log").read_text() == "crashed\n"
    assert list((tmp_path / "scratch").iterdir()) == []


# --- (c) trials left RUNNING by a killed job -------------------------------------------------------------------------
def test_running_trial_is_marked_fail_and_the_study_continues(monkeypatch, tmp_path, capsys):
    m = thermal_driver(monkeypatch, tmp_path, n_trials=2)
    (tmp_path / "run").mkdir()
    killed = hpc.open_study("calibration_optuna", tmp_path / "run", "journal", optuna.samplers.TPESampler(seed=1))
    killed.ask()  # started, never finished: the state a killed job leaves
    assert killed.trials[0].state == optuna.trial.TrialState.RUNNING

    m.main()
    assert "Marked 1 trials left RUNNING" in capsys.readouterr().out
    states = [t.state.name for t in journal_study(tmp_path).trials]
    assert states == ["FAIL", "COMPLETE", "COMPLETE"]  # N_TRIALS counts completed trials only


def test_sqlite_study_also_marks_running_trials_as_fail(tmp_path):
    for _ in range(2):  # the second open is the restart after the kill
        study = hpc.open_study("s", tmp_path, "sqlite", optuna.samplers.TPESampler(seed=1))
        if not study.trials:
            study.ask()
    assert [t.state.name for t in study.trials] == ["FAIL"]


# --- the coupled driver ------------------------------------------------------------------------------------------------
def test_coupled_trials_with_journal_and_scratch(monkeypatch, tmp_path, observations):
    be, sl = observations
    template = tmp_path / "template"
    template.mkdir()
    (template / "config.yaml").write_text(yaml.safe_dump(dict(model={}, xbeach={}, wrapper={})))
    for var, value in dict(AXB_BLUFF_EDGES_CSV=be, AXB_SHORELINES_CSV=sl, AXB_RUN_ROOT=tmp_path / "run", AXB_SCRATCH=tmp_path / "scratch",
                           AXB_STORAGE="journal").items():
        monkeypatch.setenv(var, str(value))
    (tmp_path / "scratch").mkdir()
    m = load_driver("auto_calibrate_coupled_model")
    monkeypatch.setattr(m, "run_simulation", fake_run)
    monkeypatch.setattr(m, "TEMPLATE", template)
    monkeypatch.setattr(m, "SPINUP_STATE", None)

    study = hpc.open_study("calibration_optuna", m.RUN_ROOT_DIR, m.STORAGE, optuna.samplers.TPESampler(seed=1, constant_liar=True))
    study.optimize(m.objective, n_trials=2, n_jobs=2)

    assert [t.state.name for t in journal_study(tmp_path).trials] == ["COMPLETE", "COMPLETE"]
    for number in range(2):
        folder = tmp_path / "run" / f"optuna_trial_{number:06d}"
        assert (folder / "run.log").exists() and not (folder / "results" / "xb_files").exists()
    assert list((tmp_path / "scratch").iterdir()) == []


# --- finish_trial_dir without scratch ----------------------------------------------------------------------------------
@pytest.mark.parametrize("keep_xb_files", [True, False])
def test_xb_files_are_deleted_only_when_not_kept(tmp_path, keep_xb_files):
    folder = tmp_path / "optuna_trial_000000"
    fake_run_dir = folder
    (fake_run_dir / "results").mkdir(parents=True)
    fake_run(fake_run_dir)
    hpc.finish_trial_dir(folder, folder, keep_results_nc=False, keep_xb_files=keep_xb_files)
    assert (folder / "results" / "xb_files").exists() == keep_xb_files
    assert (folder / "run.log").exists()


# --- export_history.py -------------------------------------------------------------------------------------------------
def test_export_history_reads_a_journal_study(monkeypatch, tmp_path):
    study = hpc.open_study("calibration_optuna", tmp_path, "journal", optuna.samplers.TPESampler(seed=1))
    for value in (2.0, float("inf")):  # the drivers store a failed run as +inf
        trial = study.ask()
        trial.suggest_float("nb_min", 0.3, 0.5)
        study.tell(trial, value)
    monkeypatch.setenv("AXB_STORAGE", "journal")
    monkeypatch.setenv("AXB_RUN_ROOT", str(tmp_path))
    m = load_driver("export_history")
    assert [Path(s[1]).name for s in m.STUDIES] == ["optuna_journal.log"]
    df = m.read_study(Path(m.STUDIES[0][1]), m.STUDY_ID)
    assert df["state"].tolist() == ["COMPLETE", "COMPLETE"]
    assert df["value"].isna().tolist() == [False, True]
    assert "nb_min" in df.columns and df["duration_s"].notna().all()
