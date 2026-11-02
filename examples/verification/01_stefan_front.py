"""Verification 1: two-phase Stefan (Neumann) melting problem.

A frozen half-space (T0 < T_melt) has its surface stepped to Ts > T_melt. The thaw-front position from the
production thermal solver is compared with the analytical Neumann solution, for the legacy v1.0 options
(nodal conduction + per-mass latent heat) and the flux + volumetric options (flux conduction + volumetric latent heat).
The two mixed combinations are also run to separate the effect of each option.

The soil density is rho_soil = nb*rho_w + (1-nb)*rho_p = 1990 kg m-3 (not 1000), so the two latent-heat forms differ:
the per-mass form implies a volumetric latent heat rho_soil/rho_w = 1.99 times that of the volumetric form.

Output (results/): stefan_front.csv, stefan_summary.txt
"""
# === CONFIGURATION ===
N = 200                      # number of grid nodes over the column depth (see verification_common.STEFAN)
FOURIER = 0.25               # Fourier number alpha_max * dt / dz^2 (explicit stability limit is 0.5)
N_RECORD = 120               # number of front positions stored over the run
VARIANTS = {                 # label -> thermal options
    "legacy": dict(conduction_form="nodal", latent_heat_form="per_mass"),
    "flux_only": dict(conduction_form="flux", latent_heat_form="per_mass"),
    "volumetric_only": dict(conduction_form="nodal", latent_heat_form="volumetric"),
    "flux_volumetric": dict(conduction_form="flux", latent_heat_form="volumetric"),
}
RESULTS_DIR_NAME = "results"
# === END CONFIGURATION ===

from pathlib import Path

import numpy as np
import pandas as pd

from verification_common import STEFAN, neumann_solution, run_stefan, stable_dt, stefan_exact

out_dir = Path(__file__).resolve().parent / RESULTS_DIR_NAME
out_dir.mkdir(exist_ok=True)

p = STEFAN
exact = stefan_exact(p)
rho_soil = p["nb"] * p["rho_water"] + (1 - p["nb"]) * p["rho_particle"]
# the problem the per-mass latent heat actually solves (volumetric latent heat rho_soil*L*nb)
exact_permass = neumann_solution(
    k_u=p["k_unfrozen"], C_u=p["c_unfrozen"], k_f=p["k_frozen"], C_f=p["c_frozen"],
    L_vol=rho_soil * p["L_water_ice"] * p["nb"], T_surface=p["T_surface"], T_init=p["T_init"], T_melt=p["T_melt"],
)

dt = stable_dt(N, FOURIER, p)
print(f"N = {N}, dz = {p['depth'] / (N - 1):.4f} m, dt = {dt:.1f} s, Fo = {FOURIER}, lambda = {exact.lam:.6f}")

table = {}
for label, options in VARIANTS.items():
    t, front, *_ = run_stefan(N, dt, options, p, n_record=N_RECORD)
    table["time_s"] = t
    table[f"{label}_m"] = front
    print(f"  {label:16s} front at t_end = {front[-1]:.4f} m")

df = pd.DataFrame(table)
df.insert(1, "time_days", df["time_s"] / 86400)
df["analytical_m"] = exact.s(df["time_s"])
df["analytical_if_latent_per_mass_m"] = exact_permass.s(df["time_s"])
df.to_csv(out_dir / "stefan_front.csv", index=False, float_format="%.6g")

lines = [
    "Stefan (two-phase Neumann) verification",
    f"N = {N} nodes over {p['depth']} m (dz = {p['depth'] / (N - 1):.4f} m), dt = {dt:.1f} s (Fo = {FOURIER}), t_end = {p['t_end'] / 86400:.0f} d",
    f"k_f = {p['k_frozen']}, k_u = {p['k_unfrozen']} W/m/K; C_f = {p['c_frozen']:.2e}, C_u = {p['c_unfrozen']:.2e} J/m3/K",
    f"T_init = {p['T_init'] - p['T_melt']:+.1f} C, T_surface = {p['T_surface'] - p['T_melt']:+.1f} C; nb = {p['nb']}, rho_soil = {rho_soil:.0f}, L = {p['L_water_ice']:.0f} J/kg",
    f"Analytical lambda = {exact.lam:.6f}; s(t_end) = {exact.s(p['t_end']):.4f} m",
    f"Analytical s(t_end) if latent heat were rho_soil*L*nb (what per_mass solves): {exact_permass.s(p['t_end']):.4f} m",
    "",
    f"{'variant':18s}{'s(t_end) [m]':>14s}{'diff [m]':>12s}{'diff [%]':>11s}{'max |diff|/s (t>=t_end/10) [%]':>32s}",
]
late = df["time_s"] >= p["t_end"] / 10
for label in VARIANTS:
    s_end = df[f"{label}_m"].iloc[-1]
    diff = s_end - df["analytical_m"].iloc[-1]
    rel = ((df[f"{label}_m"] - df["analytical_m"]).abs() / df["analytical_m"])[late].max() * 100
    lines.append(f"{label:18s}{s_end:14.4f}{diff:12.4f}{100 * diff / df['analytical_m'].iloc[-1]:11.2f}{rel:32.2f}")
(out_dir / "stefan_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Wrote {out_dir / 'stefan_front.csv'}\nWrote {out_dir / 'stefan_summary.txt'}")
