"""Design of the H7 round: the 27 runs of final_round_design.py, re-centred on a new centre set C that is read from a JSON file.

Used by calibration/run_revision_round.py through   AXB_ROUND_DESIGN=h7_round_design   (hpc/h7/coupled_round.sbatch sets it). Same runs, roles and
order as final_round_design.py (fr01 to fr27 are h01 to h27: same periods, forcing copies and spin-up reuse), with these differences:
  - C is not hard-coded: AXB_ROUND_CENTER_JSON points to a JSON file {"facua": ..., "bedfriccoef": ..., "wetslp": ..., "dryslp": ...,
    "xb_threshold": ..., "source": "v3 trial 569"}; the five numbers are required, "source" is an optional label for the logs.
  - the morphology pairs are placed around the new C (xb_threshold 1.75 / 2.00, dryslp 0.40 / 0.75, facua 0.05 / 0.20, wetslp 0.05 / 0.25,
    bedfriccoef 0.020 / 0.040); all other pairs are those of final_round_design.py;
  - the labels of the changes are built from C, so no number of an older C is in them;
  - the three centre runs (h02 P3, h13 P1, h14 P2) write hourly thermal output (output.output_res = AXB_CENTER_OUTPUT_RES, default 1); every
    other run uses AXB_OUTPUT_RES of the coupled driver (24);
  - THERMAL_BEST_SOURCE and the spin-up state come from AXB_THERMAL_BEST_SOURCE and AXB_SPINUP_STATES_DIR (defaults: the Windows values of the
    7 Nov round).
This module must not import run_revision_round (run as a script, that is __main__); the prep calls build_runs(_run) with its own run factory.
"""
import json
import os
from pathlib import Path

# === CONFIGURATION ===
ID_PREFIX = "h"
SKIP_A2_A4 = True
CFL_FIXED = 0.9

# Thermal section of the round (trial 935 of the A1 recalibration, dt 60 s). thermal_best.txt is next to it or one level up (optional on H7)
THERMAL_BEST_SOURCE = Path(os.environ.get("AXB_THERMAL_BEST_SOURCE", r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104\thermal_best\config.yaml"))

# Saved spin-up state(s) that the P2 runs restart from. AXB_SPINUP_STATES_DIR: folder with <name>.npz for every name (here P2_center.npz)
SPINUP_STATES = {"P2_center": Path(r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104\rr01_P2_centre\spinup_state.npz")}
if os.environ.get("AXB_SPINUP_STATES_DIR"):
    SPINUP_STATES = {name: Path(os.environ["AXB_SPINUP_STATES_DIR"]) / f"{name}.npz" for name in SPINUP_STATES}

CENTER_KEYS = ("facua", "bedfriccoef", "wetslp", "dryslp", "xb_threshold")   # the parameters the JSON must give
S0_LSGRAD, S0_N_THAW = 0.0, 3                                                # not part of the JSON
CENTER_OUTPUT_RES = int(os.environ.get("AXB_CENTER_OUTPUT_RES", 1))   # [h] thermal record spacing of the P1, P2 and P3 centre runs (hourly: heat_interp_error.py, fig08)
S0_CALL_XBEACH_INTER = 168   # [h]
S0_SEA_ICE_THRESHOLD = 0.85
S0_HC_GUESS = 500            # [W/m2/K] thermal.hc_guess
HC_LONG_DT_RUN = 1210        # h_c value whose thermal dt is derived from the guard limit

# Perturbed forcing copies (whole series, spin-up included); src is the base file of the target (era5.csv or storms.csv)
PERTURBATIONS = {
    "era5_T_plus2K": dict(column="2m_temperature", op="add", amount=2.0, target="era5"),
    "era5_T_minus2K": dict(column="2m_temperature", op="add", amount=-2.0, target="era5"),
    "storms_Hs_x1.1": dict(column="Hs(m)", op="mult", amount=1.1, target="storms"),
    "storms_Hs_x0.9": dict(column="Hs(m)", op="mult", amount=0.9, target="storms"),
    "storms_WL_plus0.2": dict(column="WL(m)", op="add", amount=0.2, target="storms"),
    "storms_WL_minus0.2": dict(column="WL(m)", op="add", amount=-0.2, target="storms"),
}
# === END CONFIGURATION ===


def load_center(path_text: str | None) -> tuple[dict, str]:
    """(S0, source label) from the JSON file at path_text. Stops with a clear message if the path is unset, the file is missing or
    not a JSON object, or one of CENTER_KEYS is missing or not a number."""
    if not path_text:
        raise SystemExit("AXB_ROUND_CENTER_JSON is not set: give the JSON file with the centre parameter set C "
                         f"({', '.join(CENTER_KEYS)}, optional source)")
    path = Path(path_text)
    if not path.is_file():
        raise SystemExit(f"AXB_ROUND_CENTER_JSON: {path} does not exist")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"AXB_ROUND_CENTER_JSON: {path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"AXB_ROUND_CENTER_JSON: {path} must hold a JSON object, got {type(data).__name__}")
    missing = [k for k in CENTER_KEYS if k not in data]
    if missing:
        raise SystemExit(f"AXB_ROUND_CENTER_JSON: {path} lacks the key(s) {', '.join(missing)}")
    bad = [k for k in CENTER_KEYS if isinstance(data[k], bool) or not isinstance(data[k], (int, float))]
    if bad:
        raise SystemExit(f"AXB_ROUND_CENTER_JSON: {path}: {', '.join(bad)} must be numbers")
    s0 = {k: float(data[k]) for k in CENTER_KEYS}
    s0.update(lsgrad=S0_LSGRAD, N_thaw_threshold=S0_N_THAW)
    return s0, str(data.get("source", f"{path.name}"))


S0, CENTER_SOURCE = load_center(os.environ.get("AXB_ROUND_CENTER_JSON"))
if not os.environ.get("AXB_REVISION_ROOT"):
    raise SystemExit("AXB_REVISION_ROOT is not set: give the folder of this round (it is created by the prep)")
ROOT_BASE = Path(os.environ["AXB_REVISION_ROOT"])


def build_runs(_run) -> list[dict]:
    """The 27 runs, longest expected first. `_run` is the run factory of run_revision_round.py."""
    P2 = "P2_center"   # key of SPINUP_STATES that the P2 runs which do not change the thermal section, forcing or period restart from
    HOURLY = dict(output_res=CENTER_OUTPUT_RES)   # the three centre runs write hourly thermal output; all others use AXB_OUTPUT_RES (24)
    c = S0
    return [
        _run(1, "P2_wl_p020", "P2", "WL(m) + 0.2 m (C otherwise); own spin-up", forcing="storms_WL_plus0.2"),
        _run(2, "P3_center", "P3", "none (C), period P3; own spin-up, saves spinup_state.npz", model=dict(spinup_state_out="spinup_state.npz"), output=HOURLY),
        _run(3, "P2_thr175", "P2", f"xb_threshold 1.75 (C {c['xb_threshold']:.4f}); spin-up from P2_center", params=dict(xb_threshold=1.75), spinup=P2),
        _run(4, "P2_hs_x110", "P2", "Hs(m) x 1.1; own spin-up", forcing="storms_Hs_x1.1"),
        _run(5, "P2_hs_x090", "P2", "Hs(m) x 0.9; own spin-up", forcing="storms_Hs_x0.9"),
        _run(6, "P2_airT_p2K", "P2", "2m_temperature + 2 K; own spin-up", forcing="era5_T_plus2K"),
        _run(7, "P2_airT_m2K", "P2", "2m_temperature - 2 K; own spin-up", forcing="era5_T_minus2K"),
        _run(8, "P2_hc1210", "P2", f"thermal.hc_guess 1210 (C {S0_HC_GUESS}); own spin-up", hc_guess=1210),
        _run(9, "P2_hc165", "P2", f"thermal.hc_guess 165 (C {S0_HC_GUESS}); own spin-up", hc_guess=165),
        _run(10, "P2_nthaw1", "P2", f"N_thaw_threshold 1 (C {S0_N_THAW}); own spin-up", thermal=dict(N_thaw_threshold=1)),
        _run(11, "P2_nthaw5", "P2", f"N_thaw_threshold 5 (C {S0_N_THAW}); own spin-up", thermal=dict(N_thaw_threshold=5)),
        _run(12, "P2_wl_m020", "P2", "WL(m) - 0.2 m; own spin-up", forcing="storms_WL_minus0.2"),
        _run(13, "P1_center", "P1", "none (C), period P1; own spin-up, saves spinup_state.npz", model=dict(spinup_state_out="spinup_state.npz"), output=HOURLY),
        _run(14, "P2_center", "P2", "none (C); spin-up from P2_center", spinup=P2, output=HOURLY),
        _run(15, "P2_dryslp040", "P2", f"dryslp 0.40 (C {c['dryslp']:.4f}); spin-up from P2_center", params=dict(dryslp=0.40), spinup=P2),
        _run(16, "P2_dryslp075", "P2", f"dryslp 0.75 (C {c['dryslp']:.4f}); spin-up from P2_center", params=dict(dryslp=0.75), spinup=P2),
        _run(17, "P2_facua005", "P2", f"facua 0.05 (C {c['facua']:.4f}; facSk = facAs = facua); spin-up from P2_center", params=dict(facua=0.05), spinup=P2),
        _run(18, "P2_facua020", "P2", f"facua 0.20 (C {c['facua']:.4f}; facSk = facAs = facua); spin-up from P2_center", params=dict(facua=0.20), spinup=P2),
        _run(19, "P2_wetslp005", "P2", f"wetslp 0.05 (C {c['wetslp']:.4f}); spin-up from P2_center", params=dict(wetslp=0.05), spinup=P2),
        _run(20, "P2_wetslp025", "P2", f"wetslp 0.25 (C {c['wetslp']:.4f}); spin-up from P2_center", params=dict(wetslp=0.25), spinup=P2),
        _run(21, "P2_bedfric0020", "P2", f"bedfriccoef 0.020 (C {c['bedfriccoef']:.4f}); spin-up from P2_center", params=dict(bedfriccoef=0.020), spinup=P2),
        _run(22, "P2_bedfric0040", "P2", f"bedfriccoef 0.040 (C {c['bedfriccoef']:.4f}); spin-up from P2_center", params=dict(bedfriccoef=0.040), spinup=P2),
        _run(23, "P2_inter072", "P2", f"call_xbeach_inter 72 h (C {S0_CALL_XBEACH_INTER} h); spin-up from P2_center", model=dict(call_xbeach_inter=72), spinup=P2),
        _run(24, "P2_inter336", "P2", f"call_xbeach_inter 336 h (C {S0_CALL_XBEACH_INTER} h); spin-up from P2_center", model=dict(call_xbeach_inter=336), spinup=P2),
        _run(25, "P2_seaice050", "P2", f"sea_ice_threshold 0.50 (C {S0_SEA_ICE_THRESHOLD}); spin-up from P2_center", wrapper=dict(sea_ice_threshold=0.5), spinup=P2),
        _run(26, "P2_seaice095", "P2", f"sea_ice_threshold 0.95 (C {S0_SEA_ICE_THRESHOLD}); spin-up from P2_center", wrapper=dict(sea_ice_threshold=0.95), spinup=P2),
        _run(27, "P2_thr200", "P2", f"xb_threshold 2.00 (C {c['xb_threshold']:.4f}); spin-up from P2_center", params=dict(xb_threshold=2.00), spinup=P2),
    ]
