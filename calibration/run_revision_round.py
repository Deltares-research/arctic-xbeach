"""Prep job of the revision round: 16 coupled (thermal + XBeach) runs with the thermal parameters of the A1 recalibration.

Runs as a 14-core local-runs job after A1 has ended. The 16 run jobs (run_revision_one.py <rrNN>) and the collector
(collect_revision_round.py) are submitted behind it in a strict FIFO queue; the runs start only when this job has written
<ROOT>\\READY. Design: C:\\Users\\keesn\\.claude\\plans\\sprightly-munching-engelbart.md (table "The round: 16 coupled runs").

Steps (each is skipped if its output exists, so the job can be started again; all are logged to <ROOT>\\prep.log):
  a  A1 check: >= A1_MIN_COMPLETE complete trials, else <ROOT>\\FAILED. Best trial -> thermal_best\\config.yaml + thermal_best.txt
  b  A2 to A4 (thermal-only experiments of calibration/run_thermal_experiments.py, outputs on N:). A failure writes A2_A4_FAILED
     but does not stop the coupled round
  c  CFL gate: 0.9 unless the CFL check shows a nonzero exit or NaN at 0.9 (then 0.7)
  d  thermal dt of the h_c 1210 run from the wet-surface-node guard of Simulation.thermal_update (model.py); every run is checked
  e  template: prepare_template() of the coupled driver (calibration/auto_calibrate_coupled_model.py)
  f  16 run folders, perturbed forcing copies, design.csv, check of every folder (config, Simulation, forcing, guards), READY

DRY_RUN = True builds everything in <ROOT>_dryrun (A1 check and A2 to A4 skipped, no waiting for the CFL check) and starts nothing.

Another round with the same machinery: set the environment variable AXB_ROUND_DESIGN to the name of a module in calibration/ (e.g.
final_round_design). Its ROOT_BASE, S0*, PERTURBATIONS, HC_LONG_DT_RUN and the optional THERMAL_BEST_SOURCE, SKIP_A2_A4, CFL_FIXED,
SPINUP_STATES, ID_PREFIX replace those below, and its build_runs(_run) returns the RUNS. Runs may reuse a saved spin-up
(_run(..., spinup="<name of SPINUP_STATES>")); step f then proves for each such run that the state loads (see check_spinup_reuse).
Without the variable the 16-run design below is built exactly as before.
Run:  python -u calibration/run_revision_round.py     (launch_prep.bat in ROOT sets the thread variables)
"""
# === CONFIGURATION ===
import os
from pathlib import Path

DRY_RUN = False        # True: root = <ROOT>_dryrun, A1 check and A2 to A4 skipped, CFL file not waited for
ROOT_BASE = Path(os.environ.get("AXB_REVISION_ROOT", r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104"))
REPO_DIR = Path(__file__).resolve().parents[1]
CALIB_DIR = Path(__file__).resolve().parent

# A1 (thermal recalibration, true sensor depth, dt 60 s)
A1_DB = Path(r"N:\arctic-xbeach-runs\thermal_calibration_runs\20261103_desktop_truedepth_dt60\optuna_study.db")
A1_STUDY = "thermal_v11_truedepth_dt60"
A1_MIN_COMPLETE = int(os.environ.get("AXB_A1_MIN_COMPLETE", 950))  # prep starts only after A1's job has ended (FIFO queue); 950 lets a run that lost a few trials still go ahead
A1_DRY_CONFIG = Path(r"N:\arctic-xbeach-runs\v11_runs\thermal_calibration\optuna_trial_000490\config.yaml")  # thermal best of the dry run

# A2 to A4: run_thermal_experiments.py with these settings
A2_A4_N_PARALLEL = 14

# CFL gate
CFL_SUMMARY = Path(r"N:\arctic-xbeach-runs\v11_runs\cfl09_check\cfl_check_summary.txt")
CFL_DEFAULT, CFL_FALLBACK = 0.9, 0.7
CFL_WAIT_MIN, CFL_POLL_S = 0, 30        # the CFL job is ahead of prep in the FIFO queue, so a missing summary means it failed: no wait

# Optional per-design settings (set by the design module named in AXB_ROUND_DESIGN; the defaults keep the 16-run design as it was)
DESIGN_MODULE = os.environ.get("AXB_ROUND_DESIGN") or None
ID_PREFIX = "rr"                # run ids are <ID_PREFIX><NN>
THERMAL_BEST_SOURCE = None      # Path of a thermal_best\config.yaml to reuse (with thermal_best.txt next to it or one level up) instead of A1's best trial
SKIP_A2_A4 = False              # True: step b does nothing
CFL_FIXED = None                # float: step c returns it without reading CFL_SUMMARY
SPINUP_STATES = {}              # name -> source .npz; copied to <ROOT>\spinup\<name>.npz, used by _run(..., spinup=name)

# Coupled template and forcing
BASE_SIM_DIR = Path(r"N:\arctic-xbeach-runs\v11_runs\coupled_base_v11")
ERA5_CSV = Path(r"N:\arctic-xbeach-runs\database\era5.csv")
STORMS_CSV = Path(r"N:\arctic-xbeach-runs\database\storms.csv")
THERMAL_DT = 60        # [s] coupled thermal dt (h_c 1210 may need less, see step d)
TIMESTEP_S = 3600      # [s] model timestep, the thermal dt must divide it

# Periods: model.time_start, model.time_end ('DD-MM-YYYY', as the v1.0 validation configs), 10-year thermal spin-up
PERIODS = {"P1": ("01-01-1999", "01-01-2012"), "P2": ("01-01-2002", "01-01-2016"), "P3": ("01-01-2006", "01-01-2019")}
SPIN_UP_YEARS = 10

# Centre set S0 (xbeach / wrapper parameters, as the coupled driver's update_config_yaml takes them)
S0 = dict(bedfriccoef=0.02, facua=0.02, wetslp=0.2, dryslp=0.75, xb_threshold=2.0, lsgrad=0.0, N_thaw_threshold=3)
S0_CALL_XBEACH_INTER = 168   # [h]
S0_SEA_ICE_THRESHOLD = 0.85
S0_HC_GUESS = 500            # [W/m2/K] thermal.hc_guess

# Perturbed forcing copies (whole series, spin-up included); written once to <ROOT>\forcing\
PERTURBATIONS = {
    "era5_T_plus2K": dict(src=ERA5_CSV, column="2m_temperature", op="add", amount=2.0, target="era5"),
    "era5_T_minus2K": dict(src=ERA5_CSV, column="2m_temperature", op="add", amount=-2.0, target="era5"),
    "storms_Hs_x1.1": dict(src=STORMS_CSV, column="Hs(m)", op="mult", amount=1.1, target="storms"),
    "storms_WL_plus0.2": dict(src=STORMS_CSV, column="WL(m)", op="add", amount=0.2, target="storms"),
}


def _run(n, label, period, change, params=None, model=None, hc_guess=None, forcing=None, wrapper=None, thermal=None, spinup=None):
    """One row of the design: S0 plus `change`. params -> xbeach/wrapper, model -> model section, forcing -> key of PERTURBATIONS.
    wrapper / thermal: values of those sections that replace S0's (e.g. sea_ice_threshold, N_thaw_threshold); spinup: key of SPINUP_STATES,
    the run restarts from that saved state (model.spinup_state_in)."""
    return dict(id=f"{ID_PREFIX}{n:02d}", label=label, period=period, change=change, params=params or {}, model=model or {},
                hc_guess=hc_guess, forcing=forcing, wrapper=wrapper or {}, thermal=thermal or {}, spinup=spinup)


# In queue order (the order of the plan table)
RUNS = [
    _run(1, "P2_centre", "P2", "none (S0); saves spinup_state.npz", model=dict(spinup_state_out="spinup_state.npz")),
    _run(2, "P2_wl_p020", "P2", "WL(m) + 0.2 m", forcing="storms_WL_plus0.2"),
    _run(3, "P3_centre", "P3", "none (S0)"),
    _run(4, "P2_hs_x110", "P2", "Hs(m) x 1.1", forcing="storms_Hs_x1.1"),
    _run(5, "P2_thr225", "P2", "xb_threshold 2.25", params=dict(xb_threshold=2.25)),
    _run(6, "P2_thr250", "P2", "xb_threshold 2.5", params=dict(xb_threshold=2.5)),
    _run(7, "P1_centre", "P1", "none (S0)"),
    _run(8, "P1_dryslp100", "P1", "dryslp 1.0", params=dict(dryslp=1.0)),
    _run(9, "P1_inter336", "P1", "call_xbeach_inter 336 h", model=dict(call_xbeach_inter=336)),
    _run(10, "P2_dryslp050", "P2", "dryslp 0.5", params=dict(dryslp=0.5)),
    _run(11, "P2_dryslp100", "P2", "dryslp 1.0", params=dict(dryslp=1.0)),
    _run(12, "P2_airT_p2K", "P2", "2m_temperature + 2 K", forcing="era5_T_plus2K"),
    _run(13, "P2_facua020", "P2", "facua 0.2 (facSk = facAs = facua)", params=dict(facua=0.2)),
    _run(14, "P2_hc1210", "P2", "thermal.hc_guess 1210", hc_guess=1210),
    _run(15, "P2_airT_m2K", "P2", "2m_temperature - 2 K", forcing="era5_T_minus2K"),
    _run(16, "P2_hc165", "P2", "thermal.hc_guess 165", hc_guess=165),
]
HC_LONG_DT_RUN = 1210  # h_c value whose thermal dt is derived from the guard limit; all other runs use THERMAL_DT
# === END CONFIGURATION ===

import datetime
import importlib
import json
import math
import re
import shutil
import sys
import time
import traceback

# One BLAS / OpenMP thread per process; the child processes of step b and XBeach inherit them
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

sys.path.insert(0, str(CALIB_DIR))
sys.path.insert(0, str(REPO_DIR))

if DESIGN_MODULE:   # the design module replaces the design above (it must not import this module: run as a script this is __main__)
    _design = importlib.import_module(DESIGN_MODULE)
    for _name in ("ROOT_BASE", "S0", "S0_CALL_XBEACH_INTER", "S0_SEA_ICE_THRESHOLD", "S0_HC_GUESS", "PERTURBATIONS", "HC_LONG_DT_RUN",
                  "THERMAL_BEST_SOURCE", "SKIP_A2_A4", "CFL_FIXED", "SPINUP_STATES", "ID_PREFIX"):
        if hasattr(_design, _name):
            globals()[_name] = getattr(_design, _name)
    RUNS = _design.RUNS if hasattr(_design, "RUNS") else _design.build_runs(_run)
for _spec in PERTURBATIONS.values():   # a design may leave out src: it is the base file of the target
    _spec.setdefault("src", ERA5_CSV if _spec["target"] == "era5" else STORMS_CSV)

ROOT = ROOT_BASE        # set in main() (<ROOT>_dryrun for DRY_RUN)


def folder_name(run: dict) -> str:
    return f"{run['id']}_{run['label']}"


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True)
    try:
        with open(ROOT / "prep.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def fail(reason: str) -> None:
    """The round cannot start: FAILED marker (read by run_revision_one.py) and exit code 1."""
    (ROOT / "FAILED").write_text(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{reason}\n", encoding="utf-8")
    log(f"FAILED: {reason}")
    sys.exit(1)


# --- a: A1 -----------------------------------------------------------------------------------------------------------------
def step_a_thermal_best() -> Path:
    """thermal_best\\config.yaml and thermal_best.txt from the best complete trial of A1. Returns the config path."""
    out_dir = ROOT / "thermal_best"
    cfg_out, txt_out = out_dir / "config.yaml", ROOT / "thermal_best.txt"
    if cfg_out.exists() and txt_out.exists():
        log(f"a: skipped, {cfg_out} exists")
        return cfg_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if THERMAL_BEST_SOURCE is not None:   # a thermal best chosen earlier (also in the dry run): copy it, no A1 database
        src_txt = next((c for c in (THERMAL_BEST_SOURCE.with_name("thermal_best.txt"), THERMAL_BEST_SOURCE.parent.parent / "thermal_best.txt")
                        if c.exists()), None)
        if not THERMAL_BEST_SOURCE.exists() or src_txt is None:
            fail(f"thermal best source incomplete: {THERMAL_BEST_SOURCE} or its thermal_best.txt is missing")
        shutil.copy2(THERMAL_BEST_SOURCE, cfg_out)
        txt_out.write_text(src_txt.read_text(encoding="utf-8").rstrip("\n") + f"\nreused from {THERMAL_BEST_SOURCE} (txt: {src_txt})\n", encoding="utf-8")
        log(f"a: thermal best reused from {THERMAL_BEST_SOURCE}")
        return cfg_out

    if DRY_RUN:
        shutil.copy2(A1_DRY_CONFIG, cfg_out)
        txt_out.write_text(f"DRY RUN: thermal config of {A1_DRY_CONFIG} (thermal trial 490)\n", encoding="utf-8")
        log(f"a: DRY RUN, A1 check skipped, thermal best = {A1_DRY_CONFIG}")
        return cfg_out

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=A1_STUDY, storage=f"sqlite:///{A1_DB.as_posix()}")
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    log(f"a: A1 study {A1_STUDY}: {len(complete)} complete trials (need {A1_MIN_COMPLETE})")
    if len(complete) < A1_MIN_COMPLETE:
        fail(f"A1 has only {len(complete)} complete trials (< {A1_MIN_COMPLETE}); {A1_DB}")
    scored = [t for t in complete if t.value is not None and math.isfinite(t.value)]
    best = min(scored, key=lambda t: t.value)

    trial_dir = A1_DB.parent / f"optuna_trial_{best.number:06d}"
    sim_dir = best.user_attrs.get("sim_dir")
    if sim_dir and Path(sim_dir) != trial_dir:
        log(f"a: WARNING user_attr sim_dir {sim_dir} differs from {trial_dir}; using sim_dir")
        trial_dir = Path(sim_dir)
    if not (trial_dir / "config.yaml").exists():
        fail(f"config.yaml of A1 best trial {best.number} not found in {trial_dir}")
    shutil.copy2(trial_dir / "config.yaml", cfg_out)
    txt_out.write_text(f"trial {best.number}\nvalue {best.value!r}\nsim_dir {trial_dir}\nstudy {A1_STUDY} ({A1_DB})\n"
                       f"params {json.dumps(best.params)}\n", encoding="utf-8")
    log(f"a: best A1 trial {best.number}, value {best.value:.4f}, config copied from {trial_dir}")
    return cfg_out


# --- b: A2 to A4 -----------------------------------------------------------------------------------------------------------
def step_b_thermal_experiments(best_trial_number: int | None) -> None:
    if SKIP_A2_A4:
        log("b: skipped by the design (SKIP_A2_A4)")
        return
    if DRY_RUN:
        log("b: DRY RUN, A2 to A4 skipped")
        return
    marker = ROOT / "A2_A4_FAILED"
    try:
        import run_thermal_experiments as rte
        rte.STUDY_STORAGE, rte.STUDY_NAME, rte.TRIAL_NUMBER = A1_DB, A1_STUDY, None
        rte.N_PARALLEL, rte.TAG = A2_A4_N_PARALLEL, ""
        raw, source = rte.load_raw_params()
        if best_trial_number is not None and source.get("trial_number") != best_trial_number:
            log(f"b: WARNING the experiments use trial {source.get('trial_number')}, step a took trial {best_trial_number}")
        jobs = rte.make_jobs(raw)

        def missing():
            return [f"{j.exp}/{j.name}" for j in jobs if not (rte.root_of(j.exp) / j.name / "scored.json").exists()]

        if not missing():
            log("b: skipped, all A2 to A4 runs have a scored.json")
            return
        log(f"b: A2 to A4: {len(missing())} of {len(jobs)} runs to do at {A2_A4_N_PARALLEL} in parallel (trial {source.get('trial_number')})")
        rte.main()
        left = missing()
        if left:
            raise RuntimeError(f"no scored.json for: {left}")
        marker.unlink(missing_ok=True)
        log("b: A2 to A4 done")
    except BaseException as e:   # also SystemExit; the coupled round goes on
        if isinstance(e, KeyboardInterrupt):
            raise
        marker.write_text(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{type(e).__name__}: {e}\n{traceback.format_exc()}", encoding="utf-8")
        log(f"b: A2 to A4 FAILED (coupled round continues): {type(e).__name__}: {e}")


# --- c: CFL gate ------------------------------------------------------------------------------------------------------------
def step_c_cfl() -> tuple[float, str]:
    """(CFL, reason). cfl_check.py writes one line per call: '... | exit <rc at 0.7>/<rc at 0.9> | NaN <bool>/<bool> | ...'."""
    if CFL_FIXED is not None:
        log(f"c: CFL {CFL_FIXED} fixed by design")
        return CFL_FIXED, "fixed by design"
    waited = 0.0
    while not CFL_SUMMARY.exists():
        if DRY_RUN or waited >= CFL_WAIT_MIN * 60:
            cfl = CFL_DEFAULT if DRY_RUN else CFL_FALLBACK   # live: the check did not produce evidence, so keep v1.0's 0.7
            reason = f"{CFL_SUMMARY} missing, CFL {cfl}"
            log(f"c: {reason}")
            return cfl, reason
        if waited == 0:
            log(f"c: waiting up to {CFL_WAIT_MIN} min for {CFL_SUMMARY}")
        time.sleep(CFL_POLL_S)
        waited += CFL_POLL_S
    lines = [ln for ln in CFL_SUMMARY.read_text(encoding="utf-8").splitlines() if ln.strip()]
    problems = []
    for ln in lines:
        m_exit, m_nan = re.search(r"exit (-?\d+)/(-?\d+)", ln), re.search(r"NaN (True|False)/(True|False)", ln)
        if not (m_exit and m_nan):
            problems.append(f"unparsed line: {ln[:80]}")
            continue
        if int(m_exit.group(2)) != 0:
            problems.append(f"nonzero exit at 0.9 ({m_exit.group(2)}): {ln.split(':')[0]}")
        if m_nan.group(2) == "True":
            problems.append(f"NaN at 0.9: {ln.split(':')[0]}")
    if not lines:
        problems.append("summary is empty")
    if problems:
        reason = f"CFL check shows problems at 0.9 ({'; '.join(problems)}), using CFL {CFL_FALLBACK}"
        log(f"c: {reason}")
        return CFL_FALLBACK, reason
    reason = f"CFL check clean at 0.9 ({len(lines)} calls), using CFL {CFL_DEFAULT}"
    log(f"c: {reason}")
    return CFL_DEFAULT, reason


# --- d: thermal dt and the wet-surface-node guard -----------------------------------------------------------------------------
def wet_node_limits(config: dict, hc: float) -> dict:
    """Largest thermal dt [s] allowed by the explicit stability checks of Simulation.thermal_update (model.py ~1316-1362):
    wet surface node:  4 k/C dt/dz^2 + 2 hc dt/(C dz) < 2   ->  dt < C dz^2 / (2 k + hc dz)
    Fourier number:    (k_l + k_r)/(2 C) dt/dz^2 <= 0.5     ->  dt <= C dz^2 / (2 k_max)
    k and C of the surface node are those of its phase (frozen: k_soil_frozen_min, c_soil_frozen; unfrozen; in between: means), and the
    guard is evaluated in every phase, so the limit is the minimum. hc is the coefficient after the factors the model applies
    (sensitivity.factor_convective_heat_flux, and surface_flux_factor only if thermal has a heat_flux_factors key, as model.py checks)."""
    th = config["thermal"]
    dz = th["max_depth"] / (th["grid_resolution"] - 1)
    conv = 1 if "sensitivity" not in config else config["sensitivity"]["factor_convective_heat_flux"]
    flux_factor = th["surface_flux_factor"] if "heat_flux_factors" in th else 1
    hc_eff = hc * conv * flux_factor
    cf, cu = th["c_soil_frozen"], th["c_soil_unfrozen"]
    phases = {"frozen": (th["k_soil_frozen_min"], cf, th["k_soil_frozen_max"]),
              "unfrozen": (th["k_soil_unfrozen_min"], cu, th["k_soil_unfrozen_max"]),
              "inbetween": ((th["k_soil_frozen_min"] + th["k_soil_unfrozen_min"]) / 2, (cf + cu) / 2,
                            (th["k_soil_frozen_max"] + th["k_soil_unfrozen_max"]) / 2)}
    wet = {name: c * dz**2 / (2 * k + hc_eff * dz) for name, (k, c, _) in phases.items()}
    fo = {name: c * dz**2 / (2 * kmax) for name, (_, c, kmax) in phases.items()}
    return dict(dz=dz, hc_eff=hc_eff, wet=wet, fo=fo, limit=min(min(wet.values()), min(fo.values())), wet_frozen=wet["frozen"])


def choose_dt(limit: float) -> int:
    """60 s if 60 < limit, else the largest divisor of the model timestep below the limit."""
    if THERMAL_DT < limit:
        return THERMAL_DT
    return max(d for d in range(1, TIMESTEP_S + 1) if TIMESTEP_S % d == 0 and d < limit)


def step_d_thermal_dt(thermal_cfg: Path) -> dict:
    """{'hc1210_dt': ..., 'limits': {hc: limit info}}; fails if a run would not pass at its dt."""
    from arctic_xbeach.config_loader import load_config
    cfg = load_config(thermal_cfg)
    hcs = sorted({r["hc_guess"] or S0_HC_GUESS for r in RUNS})
    info = {hc: wet_node_limits(cfg, hc) for hc in hcs}
    for hc, d in info.items():
        log(f"d: h_c {hc}: wet-node limit {d['wet_frozen']:.1f} s (frozen surface node, dz {d['dz']:.5f} m), "
            f"min over phases and Fourier check {d['limit']:.1f} s")
    dt_long = choose_dt(info[HC_LONG_DT_RUN]["limit"])
    log(f"d: thermal dt of the h_c {HC_LONG_DT_RUN} run: {dt_long} s")
    for r in RUNS:
        hc = r["hc_guess"] or S0_HC_GUESS
        dt = dt_long if hc == HC_LONG_DT_RUN else THERMAL_DT
        if not dt < info[hc]["limit"] or TIMESTEP_S % dt:
            fail(f"{folder_name(r)}: thermal dt {dt} s does not pass the guard (limit {info[hc]['limit']:.1f} s at h_c {hc})")
    return dict(hc1210_dt=dt_long, limits=info)


# --- e: template -------------------------------------------------------------------------------------------------------------
def step_e_template(thermal_cfg: Path, cfl: float):
    """Template folder and the coupled driver module (imported with its AXB_* settings pointing at this round)."""
    os.environ["AXB_RUN_ROOT"] = str(ROOT)
    os.environ["AXB_THERMAL_CONFIG_YAML"] = str(thermal_cfg)
    os.environ["AXB_BASE_SIM_DIR"] = str(BASE_SIM_DIR)
    os.environ["AXB_THERMAL_DT"] = str(THERMAL_DT)
    import auto_calibrate_coupled_model as drv   # importing reads the observations, imports optuna and matplotlib; nothing is started
    import yaml
    from arctic_xbeach.config_loader import load_config
    template = drv.prepare_template()
    cfg = load_config(template / "config.yaml")
    cfg["wrapper"]["CFL_xbeach"] = cfl
    (template / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))
    log(f"e: template {template} (thermal from {thermal_cfg}, thermal dt {THERMAL_DT}, CFL_xbeach {cfl})")
    return template, drv


# --- f: forcing, run folders, checks ------------------------------------------------------------------------------------------
def perturb_csv(src: Path, dst: Path, column: str, op: str, amount: float) -> None:
    """Copy of a forcing csv with one column changed (value + amount, or value * amount). Everything else is copied as text, so the
    index column, time format, other columns and the empty cells (NaN) are exactly those of the source."""
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, newline="", encoding="utf-8") as fin, open(tmp, "w", newline="", encoding="utf-8") as fout:
        header = fin.readline()
        names = header.rstrip("\r\n").split(",")
        idx = names.index(column)
        fout.write(header)
        for line in fin:
            body = line.rstrip("\r\n")
            parts = body.split(",")
            if len(parts) != len(names):
                raise ValueError(f"{src}: {len(parts)} fields instead of {len(names)} in line: {body[:80]}")
            if parts[idx] != "":
                v = float(parts[idx])
                if v == v:   # NaN stays as it is
                    parts[idx] = repr(v + amount if op == "add" else v * amount)
            fout.write(",".join(parts) + line[len(body):])
    tmp.replace(dst)


def check_perturbation(src: Path, dst: Path, column: str, op: str, amount: float) -> None:
    """The column differs from the source by exactly the stated amount (NaN where the source is NaN); same number of rows."""
    import numpy as np
    import pandas as pd
    a = pd.read_csv(src, usecols=[column])[column].to_numpy()
    b = pd.read_csv(dst, usecols=[column])[column].to_numpy()
    if len(a) != len(b) or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError(f"{dst.name}: rows or NaN positions differ from {src.name}")
    ok = ~np.isnan(a)
    expected = a[ok] + amount if op == "add" else a[ok] * amount
    if not np.allclose(b[ok], expected, rtol=1e-12, atol=1e-9):
        raise ValueError(f"{dst.name}: column {column} is not {op} {amount} of the source")


def build_forcing() -> dict[str, Path]:
    """Perturbed forcing copies that the design uses; returns key -> path. Each is built once."""
    out_dir = ROOT / "forcing"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key in sorted({r["forcing"] for r in RUNS if r["forcing"]}):
        spec = PERTURBATIONS[key]
        dst = out_dir / f"{key}.csv"
        if dst.exists():
            log(f"f: forcing {dst.name} exists, skipped")
        else:
            t0 = time.time()
            perturb_csv(spec["src"], dst, spec["column"], spec["op"], spec["amount"])
            check_perturbation(spec["src"], dst, spec["column"], spec["op"], spec["amount"])
            log(f"f: forcing {dst.name}: {spec['column']} {spec['op']} {spec['amount']} ({time.time() - t0:.0f} s)")
        paths[key] = dst
    return paths


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def run_params(run: dict) -> dict:
    """S0 plus the run's params. An N_thaw_threshold in run['thermal'] counts as a param: update_config_yaml then writes it to the wrapper
    and xbeach sections, and the thermal section gets the same value (the model reads it from there)."""
    params = {**S0, **run["params"]}
    if "N_thaw_threshold" in run["thermal"]:
        params["N_thaw_threshold"] = run["thermal"]["N_thaw_threshold"]
    return params


def spinup_copy(run: dict) -> Path | None:
    """The copy of the saved spin-up state in this round that the run restarts from (<ROOT>/spinup/<name>.npz), or None."""
    return ROOT / "spinup" / f"{run['spinup']}.npz" if run["spinup"] else None


def run_model(run: dict) -> dict:
    """model section values of the run: its model dict, plus spinup_state_in (absolute path) for a run that reuses a spin-up."""
    return {**run["model"], **({"spinup_state_in": str(spinup_copy(run))} if run["spinup"] else {})}


def run_settings(run: dict, dt: int, forcing_paths: dict[str, Path]) -> dict:
    """The values this run must have in its config.yaml: flat key -> value, for the build and for the check."""
    start, end = PERIODS[run["period"]]
    params = run_params(run)
    want = {"model.time_start": start, "model.time_end": end, "model.spin_up_time": SPIN_UP_YEARS,
            "model.call_xbeach_inter": S0_CALL_XBEACH_INTER, "wrapper.sea_ice_threshold": S0_SEA_ICE_THRESHOLD,
            "thermal.hc_guess": run["hc_guess"] or S0_HC_GUESS, "thermal.dt": dt,
            "thermal.N_thaw_threshold": params["N_thaw_threshold"],
            "data.forcing_data_path": str(ERA5_CSV), "data.storm_data_path": str(STORMS_CSV)}
    for k, v in params.items():
        if k == "xb_threshold":
            want["wrapper.xb_threshold"] = v
        elif k == "N_thaw_threshold":
            want["wrapper.N_thaw_threshold"] = v
            want["xbeach.N_thaw_threshold"] = v
        else:
            want[f"xbeach.{k}"] = v
    want["xbeach.facSk"] = want["xbeach.facAs"] = params["facua"]
    for k, v in run_model(run).items():
        want[f"model.{k}"] = v
    for section in ("wrapper", "thermal"):
        for k, v in run[section].items():
            want[f"{section}.{k}"] = v
    if run["forcing"]:
        target = PERTURBATIONS[run["forcing"]]["target"]
        want["data.forcing_data_path" if target == "era5" else "data.storm_data_path"] = str(forcing_paths[run["forcing"]])
    return want


def build_run_folder(run: dict, dt: int, template: Path, drv, forcing_paths: dict[str, Path]) -> Path:
    """Copy of the template (into the possibly pre-created empty run folder) with S0 plus the change of the run."""
    import yaml
    from arctic_xbeach.config_loader import load_config
    run_dir = ROOT / folder_name(run)
    shutil.copytree(template, run_dir, dirs_exist_ok=True)
    start, end = PERIODS[run["period"]]
    extra_model = {"time_start": start, "time_end": end, "spin_up_time": SPIN_UP_YEARS, "call_xbeach_inter": S0_CALL_XBEACH_INTER, **run_model(run)}
    params = run_params(run)
    drv.update_config_yaml(run_dir, params, extra_model=extra_model)   # facSk = facAs = facua inside
    cfg = load_config(run_dir / "config.yaml")
    cfg["wrapper"]["sea_ice_threshold"] = S0_SEA_ICE_THRESHOLD
    cfg["wrapper"].update(run["wrapper"])
    cfg["thermal"]["hc_guess"] = run["hc_guess"] or S0_HC_GUESS
    cfg["thermal"]["dt"] = dt
    cfg["thermal"]["N_thaw_threshold"] = params["N_thaw_threshold"]   # the model reads it from the thermal section only
    cfg["thermal"].update(run["thermal"])
    if run["forcing"]:
        key = "forcing_data_path" if PERTURBATIONS[run["forcing"]]["target"] == "era5" else "storm_data_path"
        cfg["data"][key] = str(forcing_paths[run["forcing"]])
    (run_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))
    return run_dir


def check_run_folder(run: dict, dt: int, template_cfg: dict, forcing_paths: dict[str, Path], limits: dict, forcing_seen: dict) -> list[str]:
    """Checks one built folder. Returns the keys that differ from the template. Raises on any problem.
    1  config.yaml holds the intended values and differs from the template in no other key
    2  Simulation(...) constructs, set_temporal_params (thermal dt must divide the timestep) and the forcing files load for the
       period (all model timestamps present); each distinct (period, files) combination is checked once (ERA5 loaded
       as the model does it, storms by the model's coverage check; the full initialize_hydro_forcing only for the first run)
    3  the thermal dt passes the wet-node and Fourier guards of thermal_update (computed, see wet_node_limits)
    Not done here: initialize_thermal_module / thermal_update, which write x.grd, bed.dep and ne_layer.txt into the folder (a real model
    start). For a run that restarts from a saved spin-up, check_spinup_reuse does that in a throwaway copy."""
    import pandas as pd
    from arctic_xbeach.config_loader import load_config
    from arctic_xbeach.model import Simulation
    run_dir = ROOT / folder_name(run)
    cfg = load_config(run_dir / "config.yaml")
    flat, flat_t, want = flatten(cfg), flatten(template_cfg), run_settings(run, dt, forcing_paths)
    for k, v in want.items():
        if flat.get(k) != v:
            raise ValueError(f"{run_dir.name}: config {k} = {flat.get(k)!r}, intended {v!r}")
    differs = sorted(k for k in set(flat) | set(flat_t) if flat.get(k) != flat_t.get(k))
    unexpected = [k for k in differs if k not in want]
    if unexpected:
        raise ValueError(f"{run_dir.name}: keys differ from the template that are not part of the design: {unexpected}")

    cwd, results = os.getcwd(), run_dir / "results"
    made_results = not results.exists()
    try:
        sim = Simulation(str(run_dir), proj_dir=REPO_DIR)   # chdir into the folder and creates results/
        sim.set_temporal_params(cfg["model"]["time_start"], cfg["model"]["time_end"], cfg["model"]["timestep"])
        key = (run["period"], flat["data.forcing_data_path"], flat["data.storm_data_path"])
        if key not in forcing_seen:
            t0 = time.time()
            sim.load_forcing(flat["data.forcing_data_path"])   # all model timestamps present, no duplicates
            storms = pd.read_csv(flat["data.storm_data_path"], parse_dates=["time"])
            sim._at_model_timestamps(storms, flat["data.storm_data_path"])   # the same coverage check initialize_hydro_forcing starts with
            if not forcing_seen:   # its row loop takes ~75 s (O(T^2)), so the full call is made once
                sim.initialize_hydro_forcing(flat["data.storm_data_path"])
            forcing_seen[key] = True
            log(f"f: forcing loads for {key[0]} with {Path(key[1]).name} + {Path(key[2]).name} ({time.time() - t0:.0f} s)")
    finally:
        os.chdir(cwd)
        if made_results and results.exists() and not any(results.iterdir()):
            results.rmdir()
    lim = limits[run["hc_guess"] or S0_HC_GUESS]["limit"]
    if not dt < lim:
        raise ValueError(f"{run_dir.name}: thermal dt {dt} s >= guard limit {lim:.1f} s")
    return differs


def copy_spinup_states() -> None:
    """Each source state of SPINUP_STATES once to <ROOT>/spinup/<name>.npz (the source is only read)."""
    out_dir = ROOT / "spinup"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in SPINUP_STATES.items():
        dst = out_dir / f"{name}.npz"
        if dst.exists():
            log(f"f: spin-up state {dst.name} exists, skipped")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        log(f"f: spin-up state {name} copied from {src} ({dst.stat().st_size / 1e6:.1f} MB)")


def check_spinup_reuse(run: dict, spinup_seen: dict) -> None:
    """Proves that a run with model.spinup_state_in would load its state: the start-up sequence of main.py up to its call of
    sim.load_spinup_state (forcing, hydro forcing, grid, XBeach and thermal module, spin-up steps) in a throwaway copy of the built
    folder, <ROOT>/_spinup_check/<folder> (left in place: the start-up writes x.grd, bed.dep, ne_layer.txt and results/). A ValueError
    of load_spinup_state (thermal config, run context, forcing, storm table, grid or spin-up length differ) propagates and fails the prep.
    Not repeated for a run whose config is identical to an earlier checked one except for the xbeach / wrapper sections and
    model.call_xbeach_inter, none of which enters the load check (the thermal config, run context, forcing, storm table and grid)."""
    import numpy as np
    from arctic_xbeach.bathymetry import generate_schematized_bathymetry
    from arctic_xbeach.config_loader import load_config
    from arctic_xbeach.miscellaneous import resolve_path
    from arctic_xbeach.model import Simulation
    run_dir = ROOT / folder_name(run)
    cfg = load_config(run_dir / "config.yaml")
    state = Path(cfg["model"]["spinup_state_in"])
    relevant = {k: v for k, v in flatten(cfg).items() if not k.startswith(("xbeach.", "wrapper.")) and k != "model.call_xbeach_inter"}
    key = json.dumps(relevant, sort_keys=True, default=str)
    if key in spinup_seen:
        log(f"f: {run_dir.name}: spin-up state loads (same thermal config, context, forcing and grid as {spinup_seen[key]})")
        return
    check_dir = ROOT / "_spinup_check" / run_dir.name
    shutil.copytree(run_dir, check_dir, dirs_exist_ok=True)
    cwd, t0 = os.getcwd(), time.time()
    try:
        sim = Simulation(str(check_dir), proj_dir=REPO_DIR)
        c = sim.config
        sim.set_temporal_params(c.model.time_start, c.model.time_end, c.model.timestep)
        sim.load_forcing(resolve_path(c.data.forcing_data_path, sim.proj_dir))
        sim.initialize_hydro_forcing(resolve_path(c.data.storm_data_path, sim.proj_dir))
        sim.timesteps_with_xbeach_active()
        b = c.bathymetry
        if b.with_schematized_bathymetry:   # as main.py: writes x.grd and bed.dep, which generate_initial_grid then reads
            xs, zs = generate_schematized_bathymetry(
                bluff_flat_length=b.bluff_flat_length, bluff_height=b.bluff_height, bluff_slope=b.bluff_slope, beach_width=b.beach_width,
                beach_slope=b.beach_slope, nearshore_max_depth=b.nearshore_max_depth, nearshore_slope=b.nearshore_slope,
                offshore_max_depth=b.offshore_max_depth, offshore_slope=b.offshore_slope, continental_flat_width=b.continental_flat_width,
                with_artificial=b.with_artificial, artificial_max_depth=b.artificial_max_depth, artificial_slope=b.artificial_slope,
                N=b.N, artificial_flat=b.artificial_flat)
            np.savetxt("x.grd", xs)
            np.savetxt("bed.dep", zs)
        xgr, zgr, _ = sim.generate_initial_grid(nx=c.bathymetry.nx if "nx" in c.bathymetry.keys() else None,
                                                bathy_path=c.bathymetry.depfile, bathy_grid_path=c.bathymetry.xfile)
        np.savetxt("x.grd", xgr)
        np.savetxt("bed.dep", zgr)
        sim.initialize_xbeach_module()
        sim.xbeach_times[0] = sim.check_xbeach(0) if c.xbeach.with_xbeach else 0
        sim.initialize_thermal_module()
        n_spin = sim.count_spinup_steps(getattr(c.model, "spin_up_time", 0) * 365.25 * 24 * 3600)
        start_id = sim.load_spinup_state(state, n_spin)
    finally:
        os.chdir(cwd)
    spinup_seen[key] = run_dir.name
    log(f"f: {run_dir.name}: spin-up state {state.name} loads, main loop would continue at timestep {start_id} of {len(sim.T)} "
        f"({sim.timestamps[start_id]}) ({time.time() - t0:.0f} s)")


def step_f_runs(template: Path, drv, cfl: float, dt_info: dict) -> None:
    from arctic_xbeach.config_loader import load_config
    forcing_paths = build_forcing()
    copy_spinup_states()
    template_cfg = load_config(template / "config.yaml")
    rows, diffs, forcing_seen, spinup_seen = [], [], {}, {}
    for run in RUNS:
        hc = run["hc_guess"] or S0_HC_GUESS
        dt = dt_info["hc1210_dt"] if hc == HC_LONG_DT_RUN else THERMAL_DT
        run_dir = ROOT / folder_name(run)
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "run.log").exists() or (run_dir / "results").exists():
            log(f"f: {run_dir.name} has started (run.log or results exists), left untouched")
            continue
        build_run_folder(run, dt, template, drv, forcing_paths)
        differs = check_run_folder(run, dt, template_cfg, forcing_paths, dt_info["limits"], forcing_seen)
        if run["spinup"]:
            check_spinup_reuse(run, spinup_seen)
        diffs.append(f"{run_dir.name}: {', '.join(differs)}")
        log(f"f: {run_dir.name} built and checked ({run['period']}, {run['change']}, thermal dt {dt} s)")
    for run in RUNS:
        hc = run["hc_guess"] or S0_HC_GUESS
        rows.append(dict(id=run["id"], label=run["label"], folder=str(ROOT / folder_name(run)), period=run["period"], time_start=PERIODS[run["period"]][0],
                         time_end=PERIODS[run["period"]][1], changes=run["change"],
                         forcing_files=str(forcing_paths[run["forcing"]]) if run["forcing"] else "base (era5.csv, storms.csv)",
                         thermal_dt=dt_info["hc1210_dt"] if hc == HC_LONG_DT_RUN else THERMAL_DT, hc_guess=hc, cfl_xbeach=cfl))
        if DESIGN_MODULE:
            rows[-1]["spinup"] = f"{run['spinup']}.npz" if run["spinup"] else "own"
    import csv
    with open(ROOT / "design.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "config_diff_vs_template.txt").write_text("keys of config.yaml that differ from the template, per run\n" + "\n".join(diffs) + "\n", encoding="utf-8")
    log(f"f: design.csv written ({len(rows)} runs)")


def main() -> None:
    global ROOT
    ROOT = ROOT_BASE.with_name(ROOT_BASE.name + "_dryrun") if DRY_RUN else ROOT_BASE
    ROOT.mkdir(parents=True, exist_ok=True)
    for marker in ("FAILED", "READY"):   # a rerun rebuilds and rewrites them
        (ROOT / marker).unlink(missing_ok=True)
    log(f"=== revision round prep started{' (DRY RUN)' if DRY_RUN else ''}, root {ROOT}, branch of {REPO_DIR} ===")
    try:
        thermal_cfg = step_a_thermal_best()
        best_trial = None
        txt = (ROOT / "thermal_best.txt").read_text(encoding="utf-8").splitlines()
        m = re.match(r"trial (\d+)", txt[0]) if txt else None
        best_trial = int(m.group(1)) if m else None
        step_b_thermal_experiments(best_trial)
        cfl, cfl_reason = step_c_cfl()
        dt_info = step_d_thermal_dt(thermal_cfg)
        template, drv = step_e_template(thermal_cfg, cfl)
        step_f_runs(template, drv, cfl, dt_info)
    except SystemExit:
        raise
    except Exception as e:
        log(traceback.format_exc())
        fail(f"{type(e).__name__}: {e}")
    (ROOT / "READY").write_text(
        f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\nCFL_xbeach {cfl} ({cfl_reason})\nthermal best: {txt[0] if txt else '?'}"
        f"{', ' + txt[1] if len(txt) > 1 else ''}\nthermal dt {THERMAL_DT} s, h_c {HC_LONG_DT_RUN} dt {dt_info['hc1210_dt']} s\n"
        f"A2_A4_FAILED: {(ROOT / 'A2_A4_FAILED').exists()}\n", encoding="utf-8")
    log("=== READY written ===")


if __name__ == "__main__":
    main()
