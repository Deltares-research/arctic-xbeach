"""The H7 round: calibration/h7_round_design.py, the portable paths and the posix-path check of run_revision_round.py, and the scratch
copy-back of run_revision_one.py. No model is run (the child process of the scratch test is a stub that writes the files a run leaves)."""
import csv
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

CALIB_DIR = Path(__file__).resolve().parents[1] / "calibration"
CENTER = dict(facua=0.0142, bedfriccoef=0.0286, wetslp=0.1594, dryslp=0.5243, xb_threshold=1.8460, source="v3 trial 569")
OLD_NUMBERS = ("0.2254", "2.0002", "0.5239", "0.0903", "0.0343", "0.2253", "0.0342")   # C of the 7 Nov round, as the labels printed them
ENV = ["AXB_ROUND_CENTER_JSON", "AXB_REVISION_ROOT", "AXB_THERMAL_BEST_SOURCE", "AXB_SPINUP_STATES_DIR", "AXB_CENTER_OUTPUT_RES", "AXB_ROUND_DESIGN",
       "AXB_A1_DB", "AXB_A1_DRY_CONFIG", "AXB_CFL_SUMMARY", "AXB_BASE_SIM_DIR", "AXB_FORCING_CSV", "AXB_STORM_CSV", "AXB_REQUIRE_POSIX_PATHS",
       "AXB_SCRATCH"]

sys.path.insert(0, str(CALIB_DIR))


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # the round scripts set these in os.environ at import
        monkeypatch.setenv(var, "1")
    for name in ("h7_round_design", "final_round_design"):
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.fixture()
def center_json(tmp_path):
    path = tmp_path / "center.json"
    path.write_text(json.dumps(CENTER), encoding="utf-8")
    return path


def load_design(monkeypatch, center_json, **env):
    monkeypatch.setenv("AXB_ROUND_CENTER_JSON", str(center_json))
    monkeypatch.setenv("AXB_REVISION_ROOT", "/data/round")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    sys.modules.pop("h7_round_design", None)
    return importlib.import_module("h7_round_design")


def fake_run(n, label, period, change, params=None, model=None, hc_guess=None, forcing=None, wrapper=None, thermal=None, spinup=None, output=None):
    """The run factory of run_revision_round.py (same fields), with the design's prefix left out."""
    return dict(n=n, label=label, period=period, change=change, params=params or {}, model=model or {}, hc_guess=hc_guess, forcing=forcing,
                wrapper=wrapper or {}, thermal=thermal or {}, spinup=spinup, output=output or {})


# --- the centre JSON -------------------------------------------------------------------------------------------------------
def test_center_is_read_from_the_json(monkeypatch, center_json):
    d = load_design(monkeypatch, center_json)
    assert d.S0 == dict(facua=0.0142, bedfriccoef=0.0286, wetslp=0.1594, dryslp=0.5243, xb_threshold=1.8460, lsgrad=0.0, N_thaw_threshold=3)
    assert d.CENTER_SOURCE == "v3 trial 569" and d.ID_PREFIX == "h" and d.ROOT_BASE == Path("/data/round")
    assert (d.S0_CALL_XBEACH_INTER, d.S0_SEA_ICE_THRESHOLD, d.S0_HC_GUESS) == (168, 0.85, 500)


def test_center_source_is_optional(monkeypatch, tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({k: v for k, v in CENTER.items() if k != "source"}), encoding="utf-8")
    assert load_design(monkeypatch, path).CENTER_SOURCE == "c.json"


@pytest.mark.parametrize("missing", ["facua", "bedfriccoef", "wetslp", "dryslp", "xb_threshold"])
def test_missing_key_is_a_clear_error(monkeypatch, tmp_path, missing):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({k: v for k, v in CENTER.items() if k != missing}), encoding="utf-8")
    with pytest.raises(SystemExit, match=f"lacks the key.*{missing}"):
        load_design(monkeypatch, path)


def test_unset_missing_and_bad_json_are_clear_errors(monkeypatch, tmp_path):
    sys.modules.pop("h7_round_design", None)
    with pytest.raises(SystemExit, match="AXB_ROUND_CENTER_JSON is not set"):
        importlib.import_module("h7_round_design")
    with pytest.raises(SystemExit, match="does not exist"):
        load_design(monkeypatch, tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_design(monkeypatch, bad)
    text = tmp_path / "text.json"
    text.write_text(json.dumps({**CENTER, "facua": "0.1"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="must be numbers"):
        load_design(monkeypatch, text)


def test_revision_root_is_required(monkeypatch, center_json):
    monkeypatch.setenv("AXB_ROUND_CENTER_JSON", str(center_json))
    sys.modules.pop("h7_round_design", None)
    with pytest.raises(SystemExit, match="AXB_REVISION_ROOT is not set"):
        importlib.import_module("h7_round_design")


# --- the 27 runs -----------------------------------------------------------------------------------------------------------
def test_27_runs_with_the_recentred_pairs(monkeypatch, center_json):
    d = load_design(monkeypatch, center_json)
    runs = d.build_runs(fake_run)
    assert [r["n"] for r in runs] == list(range(1, 28)) and len({r["label"] for r in runs}) == 27
    by = {r["n"]: r for r in runs}
    assert by[3]["params"] == {"xb_threshold": 1.75} and by[27]["params"] == {"xb_threshold": 2.00}
    assert (by[15]["params"], by[16]["params"]) == ({"dryslp": 0.40}, {"dryslp": 0.75})
    assert (by[17]["params"], by[18]["params"]) == ({"facua": 0.05}, {"facua": 0.20})
    assert (by[19]["params"], by[20]["params"]) == ({"wetslp": 0.05}, {"wetslp": 0.25})
    assert (by[21]["params"], by[22]["params"]) == ({"bedfriccoef": 0.020}, {"bedfriccoef": 0.040})
    assert (by[8]["hc_guess"], by[9]["hc_guess"]) == (1210, 165)
    assert (by[10]["thermal"], by[11]["thermal"]) == ({"N_thaw_threshold": 1}, {"N_thaw_threshold": 5})
    assert (by[25]["wrapper"], by[26]["wrapper"]) == ({"sea_ice_threshold": 0.5}, {"sea_ice_threshold": 0.95})
    assert (by[23]["model"], by[24]["model"]) == ({"call_xbeach_inter": 72}, {"call_xbeach_inter": 336})
    assert [by[n]["forcing"] for n in (1, 4, 5, 6, 7, 12)] == ["storms_WL_plus0.2", "storms_Hs_x1.1", "storms_Hs_x0.9", "era5_T_plus2K", "era5_T_minus2K",
                                                                 "storms_WL_minus0.2"]
    assert all(v in d.PERTURBATIONS for v in (r["forcing"] for r in runs if r["forcing"]))


def test_roles_periods_and_spinup_reuse_equal_final_round(monkeypatch, center_json):
    d = load_design(monkeypatch, center_json)
    import final_round_design as f
    new, old = d.build_runs(fake_run), f.build_runs(fake_run)
    for a, b in zip(new, old):
        assert (a["n"], a["period"], a["forcing"], a["spinup"], a["hc_guess"], a["thermal"], a["wrapper"], a["model"]) == \
               (b["n"], b["period"], b["forcing"], b["spinup"], b["hc_guess"], b["thermal"], b["wrapper"], b["model"])
        assert set(a["params"]) == set(b["params"])   # the same parameter is varied, only its value moves
    assert [r["n"] for r in new if r["spinup"]] == [r["n"] for r in old if r["spinup"]] == [3, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]


def test_labels_are_built_from_c(monkeypatch, center_json, tmp_path):
    d = load_design(monkeypatch, center_json)
    text = " | ".join(r["change"] + " " + r["label"] for r in d.build_runs(fake_run))
    assert not any(old in text for old in OLD_NUMBERS)
    assert "C 0.0142" in text and "C 0.5243" in text and "C 0.1594" in text and "C 0.0286" in text and "C 1.8460" in text
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**CENTER, "facua": 0.3333}), encoding="utf-8")
    d2 = load_design(monkeypatch, other)
    assert "C 0.3333" in " ".join(r["change"] for r in d2.build_runs(fake_run))


def test_only_the_three_centre_runs_are_hourly(monkeypatch, center_json):
    d = load_design(monkeypatch, center_json)
    hourly = {r["n"] for r in d.build_runs(fake_run) if r["output"]}
    assert hourly == {2, 13, 14}
    assert {r["output"]["output_res"] for r in d.build_runs(fake_run) if r["output"]} == {1}
    d = load_design(monkeypatch, center_json, AXB_CENTER_OUTPUT_RES=6)
    assert {r["output"]["output_res"] for r in d.build_runs(fake_run) if r["output"]} == {6}


def test_thermal_and_spinup_sources_default_and_override(monkeypatch, center_json):
    d = load_design(monkeypatch, center_json)
    assert str(d.THERMAL_BEST_SOURCE).startswith("N:") and str(d.SPINUP_STATES["P2_center"]).startswith("N:")
    d = load_design(monkeypatch, center_json, AXB_THERMAL_BEST_SOURCE="/u/x/config.yaml", AXB_SPINUP_STATES_DIR="/u/spin")
    assert d.THERMAL_BEST_SOURCE == Path("/u/x/config.yaml") and d.SPINUP_STATES == {"P2_center": Path("/u/spin/P2_center.npz")}


# --- run_revision_round.py: paths and the posix check ----------------------------------------------------------------------------
def load_prep(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location("run_revision_round_under_test", CALIB_DIR / "run_revision_round.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prep_paths_default_to_windows_and_follow_the_environment(monkeypatch):
    m = load_prep(monkeypatch)
    assert all(str(p).startswith("N:") for p in (m.A1_DB, m.A1_DRY_CONFIG, m.CFL_SUMMARY, m.BASE_SIM_DIR, m.ERA5_CSV, m.STORMS_CSV, m.ROOT_BASE))
    assert m.REQUIRE_POSIX_PATHS is False
    m = load_prep(monkeypatch, AXB_A1_DB="/u/a.db", AXB_A1_DRY_CONFIG="/u/a.yaml", AXB_CFL_SUMMARY="/u/cfl.txt", AXB_BASE_SIM_DIR="/u/base",
                  AXB_FORCING_CSV="/u/era5.csv", AXB_STORM_CSV="/u/storms.csv", AXB_REVISION_ROOT="/u/round", AXB_REQUIRE_POSIX_PATHS="1")
    assert (m.A1_DB, m.A1_DRY_CONFIG, m.CFL_SUMMARY, m.BASE_SIM_DIR, m.ERA5_CSV, m.STORMS_CSV, m.ROOT_BASE) == tuple(
        Path(p) for p in ("/u/a.db", "/u/a.yaml", "/u/cfl.txt", "/u/base", "/u/era5.csv", "/u/storms.csv", "/u/round"))
    assert m.REQUIRE_POSIX_PATHS is True


def test_h7_design_replaces_the_prep_settings(monkeypatch, center_json):
    m = load_prep(monkeypatch, AXB_ROUND_DESIGN="h7_round_design", AXB_ROUND_CENTER_JSON=center_json, AXB_REVISION_ROOT="/u/round",
                  AXB_THERMAL_BEST_SOURCE="/u/thermal/config.yaml", AXB_SPINUP_STATES_DIR="/u/spin")
    assert [r["id"] for r in m.RUNS] == [f"h{i:02d}" for i in range(1, 28)]
    assert m.S0["xb_threshold"] == 1.846 and m.ROOT_BASE == Path("/u/round") and m.THERMAL_BEST_SOURCE == Path("/u/thermal/config.yaml")
    assert m.SPINUP_STATES == {"P2_center": Path("/u/spin/P2_center.npz")}
    assert {r["id"] for r in m.RUNS if r["output"]} == {"h02", "h13", "h14"}


def test_the_default_design_is_unchanged(monkeypatch):
    m = load_prep(monkeypatch)
    assert len(m.RUNS) == 16 and m.RUNS[0]["id"] == "rr01" and all(r["output"] == {} for r in m.RUNS)


def test_windows_drive_paths_are_found_and_posix_paths_pass(monkeypatch):
    m = load_prep(monkeypatch)
    assert m.windows_drive_paths({"data": {"forcing_data_path": "N:\\arctic-xbeach-runs\\database\\era5.csv"}}) == \
        ["data.forcing_data_path = N:\\arctic-xbeach-runs\\database\\era5.csv"]
    assert len(m.windows_drive_paths({"xbeach": {"version": "d:/Git/xbeach.exe"}, "model": {"x": ["P:\\a", "/b"]}})) == 2
    posix = {"data": {"forcing_data_path": "/u/nederhof/arctic-xbeach-runs/database/era5.csv", "url": "https://example.org/a"},
             "xbeach": {"version": "/u/nederhof/opt/xbeach-halloween-serial-zero/bin/xbeach", "launcher": "mpirun -np 2"}, "model": {"time_start": "01-01-2002"}}
    assert m.windows_drive_paths(posix) == []


def test_unused_output_path_is_not_counted(monkeypatch):
    m = load_prep(monkeypatch)
    cfg = {"output": {"use_default_output_path": True, "output_path": "P:/11210070-usgscoop-202324-arcticxb/runs/"}}
    assert m.windows_drive_paths(cfg) == []
    cfg["output"]["use_default_output_path"] = False
    assert len(m.windows_drive_paths(cfg)) == 1


def prep_with_configs(monkeypatch, tmp_path, flag, run_cfg):
    m = load_prep(monkeypatch, AXB_REQUIRE_POSIX_PATHS=flag)
    m.ROOT = tmp_path
    m.RUNS = [dict(id="h01", label="x")]
    for name in ("template", "h01_x"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.yaml").write_text(yaml.safe_dump(run_cfg if name == "h01_x" else {"model": {"a": 1}}), encoding="utf-8")
    return m


def test_check_posix_paths_fails_on_a_windows_path_in_a_run_config(monkeypatch, tmp_path):
    m = prep_with_configs(monkeypatch, tmp_path, "1", {"data": {"forcing_data_path": "N:\\arctic-xbeach-runs\\database\\era5.csv"}})
    with pytest.raises(SystemExit) as exc:
        m.check_posix_paths(tmp_path / "template")
    assert exc.value.code == 1
    text = (tmp_path / "FAILED").read_text()
    assert "h01_x: data.forcing_data_path = N:" in text and "Windows drive paths" in text


def test_check_posix_paths_passes_a_posix_config_and_is_off_without_the_flag(monkeypatch, tmp_path):
    posix = {"data": {"forcing_data_path": "/u/nederhof/arctic-xbeach-runs/database/era5.csv"}}
    prep_with_configs(monkeypatch, tmp_path, "1", posix).check_posix_paths(tmp_path / "template")   # no exception
    windows = tmp_path / "w"
    windows.mkdir()
    m = prep_with_configs(monkeypatch, windows, "0", {"data": {"forcing_data_path": "N:\\x\\era5.csv"}})
    m.check_posix_paths(windows / "template")   # flag off: local Windows runs are not affected


# --- run_revision_one.py: scratch ------------------------------------------------------------------------------------------------
STUB_CHILD = """\
from pathlib import Path
d = Path({sim_dir!r})
(d / "results" / "xb_files" / "0000000001").mkdir(parents=True)
(d / "results" / "xb_files" / "0000000001" / "xboutput.nc").write_text("x")
(d / "results" / "results.nc").write_text("netcdf")
(d / "results" / "xbeach_calls.csv").write_text("success\\n1\\n")
(d / "run.log").write_text("Arctic-XBeach simulation completed\\n")
(d / "spinup_state.npz").write_text("state")
"""


def load_one(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location("run_revision_one_under_test", CALIB_DIR / "run_revision_one.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_round(tmp_path):
    root = tmp_path / "round"
    run_dir = root / "h14_P2_center"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (run_dir / "job.log").write_text("pool wrote this\n", encoding="utf-8")
    (root / "READY").write_text("ok", encoding="utf-8")
    with open(root / "design.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "folder", "period"])
        w.writeheader()
        w.writerow(dict(id="h14", folder=str(run_dir), period="P2"))
    return root, run_dir


def test_scratch_run_is_copied_back_without_xb_files(monkeypatch, tmp_path):
    root, run_dir = make_round(tmp_path)
    scratch = tmp_path / "scratch"
    m = load_one(monkeypatch, AXB_REVISION_ROOT=root, AXB_SCRATCH=scratch)
    monkeypatch.setattr(m, "CHILD_SCRIPT", STUB_CHILD)
    monkeypatch.setattr(sys, "argv", ["run_revision_one.py", "h14"])
    assert m.main() == 0
    assert (run_dir / "results" / "results.nc").read_text() == "netcdf"
    assert (run_dir / "results" / "xbeach_calls.csv").exists() and (run_dir / "spinup_state.npz").exists()
    assert (run_dir / "run.log").exists() and (run_dir / "model_exit_code.txt").read_text().split()[0] == "0"
    assert not (run_dir / "results" / "xb_files").exists()
    assert (run_dir / "job.log").read_text() == "pool wrote this\n"   # the pool's log survives the merge
    assert not (scratch / "h14").exists()   # scratch copy deleted after a good copy-back


def test_without_scratch_the_model_runs_in_the_run_folder(monkeypatch, tmp_path):
    root, run_dir = make_round(tmp_path)
    m = load_one(monkeypatch, AXB_REVISION_ROOT=root)
    assert m.SCRATCH is None
    monkeypatch.setattr(m, "CHILD_SCRIPT", STUB_CHILD)
    monkeypatch.setattr(sys, "argv", ["run_revision_one.py", "h14"])
    assert m.main() == 0
    assert (run_dir / "results" / "xb_files" / "0000000001" / "xboutput.nc").exists()   # local behaviour: xb_files stay


def test_progress_files_are_synced_while_the_run_is_going(monkeypatch, tmp_path):
    root, run_dir = make_round(tmp_path)
    m = load_one(monkeypatch, AXB_REVISION_ROOT=root)
    work = tmp_path / "work"
    (work / "results").mkdir(parents=True)
    (work / "run.log").write_text("Progress 5%\n")
    (work / "results" / "xbeach_calls.csv").write_text("success\n1\n")
    m.sync_progress(work, run_dir)
    assert (run_dir / "run.log").read_text() == "Progress 5%\n" and (run_dir / "results" / "xbeach_calls.csv").exists()


def test_merge_keeps_files_that_only_the_run_folder_has(tmp_path):
    import optuna_hpc as hpc
    work, final = tmp_path / "w", tmp_path / "f"
    (work / "results" / "xb_files").mkdir(parents=True)
    (work / "results" / "results.nc").write_text("n")
    (work / "run.log").write_text("log")
    final.mkdir()
    (final / "job.log").write_text("keep")
    hpc.finish_trial_dir(work, final, keep_results_nc=True, keep_xb_files=False, merge=True)
    assert (final / "job.log").read_text() == "keep" and (final / "results" / "results.nc").exists() and not (final / "results" / "xb_files").exists()
    assert not work.exists()
