"""Options of the coupled calibration driver added for the third study: AXB_PARAM_RANGES, AXB_TIMEOUT_H, seeding from an earlier study.

No model is run: run_simulation is replaced by a stub, as in test_calibration_hpc.py (same fixtures, same way of importing the driver).
"""
import importlib.util
import json
import sys
from pathlib import Path

import optuna
import pandas as pd
import pytest
import yaml
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

CALIB_DIR = Path(__file__).resolve().parents[1] / "calibration"
AXB_VARIABLES = [
    "AXB_BASE_SIM_DIR", "AXB_RUN_ROOT", "AXB_BLUFF_EDGES_CSV", "AXB_SHORELINES_CSV", "AXB_THERMAL_CONFIG_YAML",
    "AXB_FORCING_CSV", "AXB_STORM_CSV", "AXB_XBEACH_EXE", "AXB_XBEACH_LAUNCHER", "AXB_N_TRIALS", "AXB_N_PARALLEL", "AXB_STUDY_NAME",
    "AXB_THERMAL_DT", "AXB_STORAGE", "AXB_SCRATCH", "AXB_KEEP_XB_FILES", "AXB_CONSTANT_LIAR", "AXB_MAX_FAILED_XBEACH_CALLS", "AXB_TIME_END",
    "AXB_PARAM_RANGES", "AXB_TIMEOUT_H", "AXB_SEED_FROM_RUN_ROOT", "AXB_SEED_FROM_STUDY", "AXB_SEED_FROM_STORAGE",
]
NARROWED = {"xb_threshold": [1.75, 2.25], "dryslp": [0.3, 1.0]}

sys.path.insert(0, str(CALIB_DIR))  # the driver imports optuna_hpc from its own folder when run as a script
import optuna_hpc as hpc  # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_driver(name="auto_calibrate_coupled_model"):
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


def set_env(monkeypatch, observations, **env):
    be, sl = observations
    for var, value in dict(AXB_BLUFF_EDGES_CSV=be, AXB_SHORELINES_CSV=sl, **env).items():
        monkeypatch.setenv(var, str(value))


# --- AXB_PARAM_RANGES -------------------------------------------------------------------------------------------------------
def test_param_ranges_default_is_the_published_set(monkeypatch, observations):
    set_env(monkeypatch, observations)
    m = load_driver()
    assert m.PARAM_RANGES["xb_threshold"] == [2.0, 2.5] and m.PARAM_RANGES["dryslp"] == [0.5, 1.0]
    assert (m.TIMEOUT_H, m.SEED_FROM_RUN_ROOT, m.SEED_FROM_STUDY, m.SEED_FROM_STORAGE) == (None, None, None, m.STORAGE)


def test_param_ranges_override_replaces_only_the_named_entries(monkeypatch, observations):
    set_env(monkeypatch, observations)
    published = load_driver().PARAM_RANGES
    monkeypatch.setenv("AXB_PARAM_RANGES", json.dumps(NARROWED))
    m = load_driver()
    assert m.PARAM_RANGES["xb_threshold"] == [1.75, 2.25] and m.PARAM_RANGES["dryslp"] == [0.3, 1.0]
    assert {k: v for k, v in m.PARAM_RANGES.items() if k not in NARROWED} == {k: v for k, v in published.items() if k not in NARROWED}


def test_param_ranges_override_keeps_the_sampling_type(monkeypatch, observations):
    """[2, 3] for a float parameter must not turn into suggest_int; an int parameter takes int bounds."""
    set_env(monkeypatch, observations, AXB_PARAM_RANGES=json.dumps({"xb_threshold": [2, 3], "N_thaw_threshold": [2, 4]}))
    m = load_driver()
    assert all(isinstance(x, float) for x in m.PARAM_RANGES["xb_threshold"]) and not m.is_int_range(m.PARAM_RANGES["xb_threshold"])
    assert m.PARAM_RANGES["N_thaw_threshold"] == [2, 4] and m.is_int_range(m.PARAM_RANGES["N_thaw_threshold"])


@pytest.mark.parametrize("text, fragment", [
    ('{"nonsense": [0, 1]}', "unknown parameter 'nonsense'"),
    ('{"xb_threshold": [2.5, 2.0]}', "low > high"),
    ('{"xb_threshold": [2.0]}', "two finite numbers"),
    ('{"xb_threshold": [1.0, 2.0, 3.0]}', "two finite numbers"),
    ('{"xb_threshold": "2.0"}', "two finite numbers"),
    ('{"xb_threshold": [1.0, NaN]}', "two finite numbers"),
    ('{"N_thaw_threshold": [2.5, 4]}', "integer parameter"),
    ('[1, 2]', "JSON object"),
    ('{"xb_threshold": [2.0, ', "not valid JSON"),
])
def test_param_ranges_override_rejects_bad_input(monkeypatch, observations, text, fragment):
    set_env(monkeypatch, observations, AXB_PARAM_RANGES=text)
    with pytest.raises(SystemExit) as exc:
        load_driver()
    assert "AXB_PARAM_RANGES" in str(exc.value) and fragment in str(exc.value)


# --- seeding ---------------------------------------------------------------------------------------------------------------
GOOD = dict(facua=0.1, bedfriccoef=0.02, wetslp=0.1, dryslp=0.6, xb_threshold=2.0, lsgrad=0.0, N_thaw_threshold=3)


def add_source_trial(study, state=TrialState.COMPLETE, value=3.0, drop=(), **changes):
    """Trial of an earlier study; its distributions are wide so that any value can be stored."""
    params = {k: v for k, v in {**GOOD, **changes}.items() if k not in drop}
    dists = {k: IntDistribution(0, 10) if isinstance(v, int) else FloatDistribution(-10.0, 10.0) for k, v in params.items()}
    study.add_trial(create_trial(state=state, params=params, distributions=dists, value=None if state != TrialState.COMPLETE else value,
                                 user_attrs={"rmse": value, "sim_dir": f"/old/{len(study.trials)}"}))


def make_source(tmp_path, kind):
    """Source study with 8 trials, of which only numbers 0 and 6 can be seeded with the NARROWED ranges."""
    root = tmp_path / "src"
    root.mkdir()
    study = optuna.create_study(study_name="old", direction="minimize", storage=hpc.make_storage(kind, root))
    add_source_trial(study, value=3.0)                                    # 0 in range
    add_source_trial(study, value=2.0, xb_threshold=2.4)                  # 1 above the narrowed xb_threshold range
    add_source_trial(study, state=TrialState.FAIL)                        # 2 failed
    add_source_trial(study, value=float("inf"))                           # 3 not scored
    add_source_trial(study, value=1.0, drop=("lsgrad",))                  # 4 lacks a parameter of PARAM_RANGES
    add_source_trial(study, value=1.0, lsgrad=0.5)                        # 5 fixed parameter differs from its bound
    add_source_trial(study, value=2.5, xb_threshold=2.25, other=1.0)      # 6 on the upper bound, with a parameter that is not sampled now
    add_source_trial(study, value=1.0, N_thaw_threshold=4)                # 7 fixed int parameter differs
    return root


@pytest.fixture()
def seeded_driver(monkeypatch, tmp_path, observations):
    """Driver with NARROWED ranges and an empty target study."""
    set_env(monkeypatch, observations, AXB_RUN_ROOT=tmp_path / "run", AXB_PARAM_RANGES=json.dumps(NARROWED))
    (tmp_path / "run").mkdir()
    return load_driver()


@pytest.mark.parametrize("kind", ["journal", "sqlite"])
def test_seed_study_copies_only_usable_trials(seeded_driver, tmp_path, kind):
    m = seeded_driver
    root = make_source(tmp_path, kind)
    src_file = root / ("optuna_journal.log" if kind == "journal" else "optuna_study.db")
    before = src_file.read_bytes()
    target = optuna.create_study(study_name="new", direction="minimize", storage=hpc.make_storage(kind, tmp_path / "run"))

    assert m.seed_study(target, root, "old", kind) == 2

    trials = target.get_trials()
    assert [t.state for t in trials] == [TrialState.COMPLETE] * 2
    assert [t.value for t in trials] == [3.0, 2.5]
    assert [t.user_attrs["seeded_from"] for t in trials] == ["old#0", "old#6"]
    assert [t.user_attrs["rmse"] for t in trials] == [3.0, 2.5] and trials[0].user_attrs["sim_dir"] == "/old/0"  # the source attributes are kept
    for t in trials:
        assert set(t.params) == set(m.PARAM_RANGES)  # "other" is not copied
        assert t.distributions == {k: m.param_distribution(v) for k, v in m.PARAM_RANGES.items()}
    assert trials[1].params["xb_threshold"] == 2.25
    assert len(optuna.load_study(study_name="old", storage=hpc.make_storage(kind, root)).trials) == 8  # source unchanged
    if kind == "journal":
        assert src_file.read_bytes() == before


def test_seed_study_leaves_a_study_with_trials_alone(seeded_driver, tmp_path, capsys):
    m = seeded_driver
    root = make_source(tmp_path, "journal")
    target = optuna.create_study(study_name="new", direction="minimize", storage=hpc.make_storage("journal", tmp_path / "run"))
    assert m.seed_study(target, root, "old", "journal") == 2
    assert m.seed_study(target, root, "old", "journal") == 0  # a rerun
    assert len(target.trials) == 2
    assert "already has trials" in capsys.readouterr().out
    target2 = optuna.create_study(study_name="other", direction="minimize", storage=hpc.make_storage("journal", tmp_path / "run"))
    target2.enqueue_trial(GOOD)  # a WAITING trial (START_PARAMS) also counts as "has trials"
    assert m.seed_study(target2, root, "old", "journal") == 0


def test_seed_study_with_nothing_usable_is_fine_and_logged(seeded_driver, tmp_path, capsys):
    m = seeded_driver
    root = tmp_path / "src"
    root.mkdir()
    study = optuna.create_study(study_name="old", direction="minimize", storage=hpc.make_storage("journal", root))
    add_source_trial(study, xb_threshold=2.4)
    target = optuna.create_study(study_name="new", direction="minimize", storage=hpc.make_storage("journal", tmp_path / "run"))
    assert m.seed_study(target, root, "old", "journal") == 0
    assert f"seeded 0 of 1 trials from old ({root})" in capsys.readouterr().out
    assert "seeded 0 of 1 trials from old" in (tmp_path / "run" / "calibration_prints.txt").read_text()


def test_seed_study_stops_on_a_missing_source_and_does_not_create_it(seeded_driver, tmp_path):
    m = seeded_driver
    target = optuna.create_study(study_name="new", direction="minimize", storage=hpc.make_storage("journal", tmp_path / "run"))
    (tmp_path / "nowhere").mkdir()
    with pytest.raises(SystemExit, match="not found"):
        m.seed_study(target, tmp_path / "nowhere", "old", "journal")
    assert list((tmp_path / "nowhere").iterdir()) == []  # a journal was not created in the source folder
    root = make_source(tmp_path, "journal")
    with pytest.raises(SystemExit, match="no study named 'typo'"):
        m.seed_study(target, root, "typo", "journal")


def test_seed_study_stops_when_the_direction_differs(seeded_driver, tmp_path):
    m = seeded_driver
    root = tmp_path / "src"
    root.mkdir()
    optuna.create_study(study_name="old", direction="maximize", storage=hpc.make_storage("journal", root))
    target = optuna.create_study(study_name="new", direction="minimize", storage=hpc.make_storage("journal", tmp_path / "run"))
    with pytest.raises(SystemExit, match="optimises MAXIMIZE"):
        m.seed_study(target, root, "old", "journal")


def stub_run(monkeypatch, m, tmp_path):
    """Stubs for objective(): a template folder and a run_simulation that scores 5.0 without a model run."""
    template = tmp_path / "template"
    template.mkdir(exist_ok=True)
    (template / "config.yaml").write_text(yaml.safe_dump(dict(model={}, xbeach={}, wrapper={})))
    monkeypatch.setattr(m, "prepare_template", lambda: template)
    monkeypatch.setattr(m, "USE_SPINUP_STATE", False)
    monkeypatch.setattr(m, "MAKE_FIGURE", False)
    calls = []
    monkeypatch.setattr(m, "run_simulation", lambda sim_dir: (calls.append(sim_dir.name) or 5.0, 0))
    return template, calls


def test_seeded_study_runs_new_trials_with_the_same_distributions(monkeypatch, seeded_driver, tmp_path):
    """TPE must accept the seeded trials next to its own: the real objective() runs on a seeded study without a
    distribution-compatibility error, and its trials have the distributions of the seeded ones."""
    m = seeded_driver
    template, calls = stub_run(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "TEMPLATE", template)
    monkeypatch.setattr(m, "SPINUP_STATE", None)
    root = make_source(tmp_path, "journal")
    study = hpc.open_study("new", m.RUN_ROOT_DIR, "journal", optuna.samplers.TPESampler(seed=1, n_startup_trials=1))
    assert m.seed_study(study, root, "old", "journal") == 2

    n_complete = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    n_todo = max(4 - n_complete, 0)
    assert (n_complete, n_todo) == (2, 2)
    study.optimize(m.objective, n_trials=n_todo)

    trials = study.get_trials()
    assert [t.state for t in trials] == [TrialState.COMPLETE] * 4
    assert [t.number for t in trials] == [0, 1, 2, 3]
    assert calls == ["optuna_trial_000002", "optuna_trial_000003"]  # new trials continue after the seeded numbers
    assert trials[2].distributions == trials[0].distributions and trials[3].distributions == trials[1].distributions
    assert "seeded_from" not in trials[2].user_attrs
    for t in trials[2:]:
        assert 1.75 <= t.params["xb_threshold"] <= 2.25 and 0.3 <= t.params["dryslp"] <= 1.0


# --- main(): seeding and timeout -------------------------------------------------------------------------------------------
def coupled_main(monkeypatch, tmp_path, observations, n_trials, **env):
    set_env(monkeypatch, observations, AXB_RUN_ROOT=tmp_path / "run", AXB_STORAGE="journal", AXB_N_TRIALS=n_trials, AXB_N_PARALLEL=1,
            AXB_PARAM_RANGES=json.dumps(NARROWED), **env)
    m = load_driver()
    _, calls = stub_run(monkeypatch, m, tmp_path)
    return m, calls


def test_main_seeds_counts_seeded_trials_toward_n_trials_and_does_not_seed_twice(monkeypatch, tmp_path, observations, capsys):
    root = make_source(tmp_path, "journal")
    m, calls = coupled_main(monkeypatch, tmp_path, observations, n_trials=5, AXB_SEED_FROM_RUN_ROOT=root, AXB_SEED_FROM_STUDY="old")
    monkeypatch.setattr(m, "START_PARAMS", dict(GOOD))  # a seeded study must not enqueue it
    m.main()
    out = capsys.readouterr().out
    assert f"seeded 2 of 8 trials from old ({root})" in out
    assert "Study has 2 completed trials; running 3 more to reach 5" in out
    assert f"Parameter ranges: {m.PARAM_RANGES}" in out
    log = (tmp_path / "run" / "calibration_prints.txt").read_text()
    assert "Parameter ranges:" in log and "seeded 2 of 8 trials" in log
    assert len(calls) == 3

    m.main()  # rerun of the same job: nothing is seeded again, nothing is left to run
    assert "already has trials" in capsys.readouterr().out
    study = optuna.load_study(study_name="calibration_optuna", storage=hpc.make_storage("journal", tmp_path / "run"))
    assert len(study.trials) == 5 and len(calls) == 3
    assert sum("seeded_from" in t.user_attrs for t in study.trials) == 2


def test_main_seeding_needs_both_run_root_and_study_name(monkeypatch, tmp_path, observations, capsys):
    root = make_source(tmp_path, "journal")
    m, calls = coupled_main(monkeypatch, tmp_path, observations, n_trials=1, AXB_SEED_FROM_RUN_ROOT=root)
    m.main()
    assert "seeded" not in capsys.readouterr().out and len(calls) == 1


def test_main_seeds_from_a_sqlite_source_with_its_own_storage_kind(monkeypatch, tmp_path, observations, capsys):
    root = make_source(tmp_path, "sqlite")
    m, calls = coupled_main(monkeypatch, tmp_path, observations, n_trials=2, AXB_SEED_FROM_RUN_ROOT=root, AXB_SEED_FROM_STUDY="old",
                            AXB_SEED_FROM_STORAGE="sqlite")  # the new study is a journal
    m.main()
    assert "seeded 2 of 8 trials" in capsys.readouterr().out and calls == []


@pytest.mark.parametrize("hours, expected", [("0.5", 1800.0), (None, None)])
def test_main_passes_the_timeout_to_optimize(monkeypatch, tmp_path, observations, hours, expected):
    env = {"AXB_TIMEOUT_H": hours} if hours else {}
    m, calls = coupled_main(monkeypatch, tmp_path, observations, n_trials=1, **env)
    seen = {}
    original = optuna.Study.optimize

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(optuna.Study, "optimize", spy)
    m.main()
    assert seen["timeout"] == expected and len(calls) == 1
