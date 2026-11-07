"""Design of the final round: 27 coupled runs around the centre set C = trial 159 of the H7 study coupled_A1t935_dt60_v2 (job 288545).

Used by calibration/run_revision_round.py through   set AXB_ROUND_DESIGN=final_round_design   (launch_round.bat in the round root sets it
together with AXB_REVISION_ROOT). It replaces the design of the 4 Nov round: same machinery, other numbers. This module must not import
run_revision_round (run as a script, that is __main__); the prep calls build_runs(_run) with its own run factory.

The thermal section is that of the 4 Nov round (THERMAL_BEST_SOURCE); the CFL is fixed. Runs marked "reuse" restart from the spin-up of
the P2 centre run of the 4 Nov round (SPINUP_STATES) and so skip the 10-year thermal spin-up; the prep proves for each of them that the
state loads (thermal config, start time, spin-up length, forcing and storm table equal). Every run whose thermal section, forcing or
period differs from that centre run does its own spin-up; the P1 and P3 centres also save theirs (spinup_state.npz in the run folder).
"""
import os
from pathlib import Path

ROOT_BASE = Path(os.environ.get("AXB_REVISION_ROOT", r"N:\arctic-xbeach-runs\v11_runs\final_round_20261107"))
ID_PREFIX = "fr"

THERMAL_BEST_SOURCE = Path(r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104\thermal_best\config.yaml")   # thermal_best.txt is one level up
SKIP_A2_A4 = True
CFL_FIXED = 0.9
SPINUP_STATES = {"P2_center": Path(r"N:\arctic-xbeach-runs\v11_runs\revision_round_20261104\rr01_P2_centre\spinup_state.npz")}

# Centre set C: H7 trial 159
S0 = dict(bedfriccoef=0.034329355581258554, facua=0.22537681823148023, wetslp=0.09033870575216449, dryslp=0.523905496146058,
          xb_threshold=2.000213207207768, lsgrad=0.0, N_thaw_threshold=3)
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


def build_runs(_run) -> list[dict]:
    """The 27 runs, longest expected first. `_run` is the run factory of run_revision_round.py."""
    P2 = "P2_center"   # key of SPINUP_STATES that the P2 runs which do not change the thermal section, forcing or period restart from
    return [
        _run(1, "P2_wl_p020", "P2", "WL(m) + 0.2 m (C otherwise); own spin-up", forcing="storms_WL_plus0.2"),
        _run(2, "P3_center", "P3", "none (C), period P3; own spin-up, saves spinup_state.npz", model=dict(spinup_state_out="spinup_state.npz")),
        _run(3, "P2_thr175", "P2", "xb_threshold 1.75 (C 2.0002); spin-up from P2_center", params=dict(xb_threshold=1.75), spinup=P2),
        _run(4, "P2_hs_x110", "P2", "Hs(m) x 1.1; own spin-up", forcing="storms_Hs_x1.1"),
        _run(5, "P2_hs_x090", "P2", "Hs(m) x 0.9; own spin-up", forcing="storms_Hs_x0.9"),
        _run(6, "P2_airT_p2K", "P2", "2m_temperature + 2 K; own spin-up", forcing="era5_T_plus2K"),
        _run(7, "P2_airT_m2K", "P2", "2m_temperature - 2 K; own spin-up", forcing="era5_T_minus2K"),
        _run(8, "P2_hc1210", "P2", "thermal.hc_guess 1210 (C 500); own spin-up", hc_guess=1210),
        _run(9, "P2_hc165", "P2", "thermal.hc_guess 165 (C 500); own spin-up", hc_guess=165),
        _run(10, "P2_nthaw1", "P2", "N_thaw_threshold 1 (C 3); own spin-up", thermal=dict(N_thaw_threshold=1)),
        _run(11, "P2_nthaw5", "P2", "N_thaw_threshold 5 (C 3); own spin-up", thermal=dict(N_thaw_threshold=5)),
        _run(12, "P2_wl_m020", "P2", "WL(m) - 0.2 m; own spin-up", forcing="storms_WL_minus0.2"),
        _run(13, "P1_center", "P1", "none (C), period P1; own spin-up, saves spinup_state.npz", model=dict(spinup_state_out="spinup_state.npz")),
        _run(14, "P2_center", "P2", "none (C); spin-up from P2_center", spinup=P2),
        _run(15, "P2_dryslp040", "P2", "dryslp 0.40 (C 0.5239); spin-up from P2_center", params=dict(dryslp=0.40), spinup=P2),
        _run(16, "P2_dryslp075", "P2", "dryslp 0.75 (C 0.5239); spin-up from P2_center", params=dict(dryslp=0.75), spinup=P2),
        _run(17, "P2_facua010", "P2", "facua 0.10 (C 0.2254; facSk = facAs = facua); spin-up from P2_center", params=dict(facua=0.10), spinup=P2),
        _run(18, "P2_facua035", "P2", "facua 0.35 (C 0.2254; facSk = facAs = facua); spin-up from P2_center", params=dict(facua=0.35), spinup=P2),
        _run(19, "P2_wetslp005", "P2", "wetslp 0.05 (C 0.0903); spin-up from P2_center", params=dict(wetslp=0.05), spinup=P2),
        _run(20, "P2_wetslp020", "P2", "wetslp 0.20 (C 0.0903); spin-up from P2_center", params=dict(wetslp=0.20), spinup=P2),
        _run(21, "P2_bedfric0020", "P2", "bedfriccoef 0.020 (C 0.0343); spin-up from P2_center", params=dict(bedfriccoef=0.020), spinup=P2),
        _run(22, "P2_bedfric0050", "P2", "bedfriccoef 0.050 (C 0.0343); spin-up from P2_center", params=dict(bedfriccoef=0.050), spinup=P2),
        _run(23, "P2_inter072", "P2", "call_xbeach_inter 72 h (C 168 h); spin-up from P2_center", model=dict(call_xbeach_inter=72), spinup=P2),
        _run(24, "P2_inter336", "P2", "call_xbeach_inter 336 h (C 168 h); spin-up from P2_center", model=dict(call_xbeach_inter=336), spinup=P2),
        _run(25, "P2_seaice050", "P2", "sea_ice_threshold 0.50 (C 0.85); spin-up from P2_center", wrapper=dict(sea_ice_threshold=0.5), spinup=P2),
        _run(26, "P2_seaice095", "P2", "sea_ice_threshold 0.95 (C 0.85); spin-up from P2_center", wrapper=dict(sea_ice_threshold=0.95), spinup=P2),
        _run(27, "P2_thr225", "P2", "xb_threshold 2.25 (C 2.0002); spin-up from P2_center", params=dict(xb_threshold=2.25), spinup=P2),
    ]
