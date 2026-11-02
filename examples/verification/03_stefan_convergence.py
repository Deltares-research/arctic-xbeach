"""Verification 3: grid and time-step convergence of the flux + volumetric options on the Stefan case.

Options (conduction_form = flux, latent_heat_form = volumetric) against the Neumann solution:
  A. spatial refinement N = 50 ... 800 at a fixed Fourier number (dt ~ dz^2), error vs dz
  B. time-step refinement at fixed dz (N = 200), error vs dt, plus self-convergence against the smallest dt
     (isolates the temporal error from the spatial error floor)
Errors: thaw-front position at t_end and temperature RMSE over the whole column at t_end.
The observed order is p = log(e_i / e_{i+1}) / log(h_i / h_{i+1}) between successive refinements, and the
least-squares slope of log e vs log h.

Output (results/): stefan_convergence.csv, stefan_convergence_summary.txt
"""
# === CONFIGURATION ===
N_LIST = [50, 100, 200, 400, 800]      # spatial refinement over the same column depth (800 added as an extra level)
FOURIER_A = 0.25                         # fixed Fourier number for series A
N_FIXED_DZ = 200                         # grid for series B
FOURIER_B = [0.4, 0.2, 0.1, 0.05, 0.025] # Fourier numbers (i.e. dt halving) for series B
OPTIONS = dict(conduction_form="flux", latent_heat_form="volumetric")
RESULTS_DIR_NAME = "results"
# === END CONFIGURATION ===

from pathlib import Path

import numpy as np
import pandas as pd

from verification_common import STEFAN, run_stefan, stable_dt, stefan_exact

out_dir = Path(__file__).resolve().parent / RESULTS_DIR_NAME
out_dir.mkdir(exist_ok=True)
p = STEFAN
exact = stefan_exact(p)
s_exact = float(exact.s(p["t_end"]))


def run_case(N, dt):
    t, front, z, T, _ = run_stefan(N, dt, OPTIONS, p, n_record=2)
    T_exact = exact.T(z[1:], p["t_end"])  # skip the Dirichlet node at z = 0
    return dict(N=N, dz_m=p["depth"] / (N - 1), dt_s=dt, front_m=front[-1], front_error_m=front[-1] - s_exact,
                temp_rmse_K=float(np.sqrt(np.mean((T[1:] - T_exact) ** 2)))), T


def orders(h, e):
    h, e = np.asarray(h, float), np.abs(np.asarray(e, float))
    pairwise = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
    slope = np.polyfit(np.log(h), np.log(e), 1)[0]
    return pairwise, slope


rows_A = []
for N in N_LIST:
    row, _ = run_case(N, stable_dt(N, FOURIER_A, p))
    row["series"] = "A: dt ~ dz^2 (Fo fixed)"
    rows_A.append(row)
    print(f"A N={N:4d} dz={row['dz_m']:.4f} dt={row['dt_s']:8.1f} front err={row['front_error_m']:+.5f} m  T RMSE={row['temp_rmse_K']:.4f} K")

rows_B, T_B = [], []
for Fo in FOURIER_B:
    dt = stable_dt(N_FIXED_DZ, Fo, p)
    row, T = run_case(N_FIXED_DZ, dt)
    row["series"] = "B: fixed dz, dt varied"
    rows_B.append(row)
    T_B.append(T)
    print(f"B Fo={Fo:5.3f} dt={dt:8.1f} front err={row['front_error_m']:+.5f} m  T RMSE={row['temp_rmse_K']:.4f} K")
# self-convergence of series B against its smallest dt
for row, T in zip(rows_B, T_B):
    row["front_diff_vs_smallest_dt_m"] = row["front_m"] - rows_B[-1]["front_m"]
    row["temp_rms_diff_vs_smallest_dt_K"] = float(np.sqrt(np.mean((T[1:] - T_B[-1][1:]) ** 2)))

df = pd.DataFrame(rows_A + rows_B)
cols = ["series", "N", "dz_m", "dt_s", "front_m", "front_error_m", "temp_rmse_K", "front_diff_vs_smallest_dt_m", "temp_rms_diff_vs_smallest_dt_K"]
df[cols].to_csv(out_dir / "stefan_convergence.csv", index=False, float_format="%.6g")

A, B = pd.DataFrame(rows_A), pd.DataFrame(rows_B)
pA_front, sA_front = orders(A["dz_m"], A["front_error_m"])
pA_T, sA_T = orders(A["dz_m"], A["temp_rmse_K"])
pB_front, sB_front = orders(B["dt_s"], B["front_error_m"])
pB_T, sB_T = orders(B["dt_s"], B["temp_rmse_K"])
sel = B.iloc[:-1]  # self-convergence excludes the reference itself
pS_front, sS_front = orders(sel["dt_s"], sel["front_diff_vs_smallest_dt_m"])
pS_T, sS_T = orders(sel["dt_s"], sel["temp_rms_diff_vs_smallest_dt_K"])

fmt = lambda a: ", ".join(f"{x:.2f}" for x in a)
lines = [
    "Stefan convergence, flux + volumetric options",
    f"Analytical s(t_end = {p['t_end'] / 86400:.0f} d) = {s_exact:.5f} m (lambda = {exact.lam:.6f})",
    "",
    df[cols].to_string(index=False, float_format=lambda x: f"{x:.5g}"),
    "",
    "Observed order of accuracy (successive pairs; least-squares slope)",
    f"A  front error vs dz        : {fmt(pA_front)}; slope {sA_front:.2f}",
    f"A  temperature RMSE vs dz   : {fmt(pA_T)}; slope {sA_T:.2f}",
    f"B  front error vs dt        : {fmt(pB_front)}; slope {sB_front:.2f}   (spatial error floor at N = {N_FIXED_DZ})",
    f"B  temperature RMSE vs dt   : {fmt(pB_T)}; slope {sB_T:.2f}",
    f"B  front self-convergence   : {fmt(pS_front)}; slope {sS_front:.2f}   (vs smallest dt)",
    f"B  T-RMS self-convergence   : {fmt(pS_T)}; slope {sS_T:.2f}",
]
(out_dir / "stefan_convergence_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Wrote {out_dir / 'stefan_convergence.csv'}\nWrote {out_dir / 'stefan_convergence_summary.txt'}")
