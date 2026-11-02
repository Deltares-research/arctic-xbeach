"""Verification 2: layered column with depth-varying conductivity (k contrast x3).

A frozen column of three layers (k = 1.0, 3.0, 1.5 W m-1 K-1, uniform heat capacity) with fixed end temperatures.
The production solver is run with nodal (legacy v1.0) and flux (conservative) conduction and compared with
  (a) the analytical steady piecewise-linear profile, and
  (b) an independent fine-grid finite-volume reference (matrix exponential in time) for a transient snapshot.
Layer interfaces are placed halfway between nodes, where the harmonic-mean face conductivity is exact.

Output (results/): layered_profiles.csv, layered_summary.txt
"""
# === CONFIGURATION ===
DEPTH = 4.0                              # [m]
N = 81                                   # nodes (dz = 0.05 m)
INTERFACES = [1.025, 2.525]              # [m] layer interfaces, halfway between nodes 20|21 and 50|51
K_LAYERS = [1.0, 3.0, 1.5]               # [W m-1 K-1] frozen conductivity of each layer (x3 contrast)
C_VOL = 1.0e6                            # [J m-3 K-1] volumetric heat capacity (uniform)
T_TOP, T_BOTTOM, T_INIT = 268.15, 258.15, 263.15   # [K] all below melting, so no phase change
DT = 300.0                               # [s] time step (Fo_max = 0.36)
T_SNAPSHOT_DAYS = 3.0                    # [days] transient snapshot compared with the fine-grid reference
T_STEADY_DAYS = 300.0                    # [days] run time for the steady state (> 20 e-folding times)
VARIANTS = {"nodal": dict(conduction_form="nodal", latent_heat_form="per_mass"),
            "flux": dict(conduction_form="flux", latent_heat_form="volumetric")}
RESULTS_DIR_NAME = "results"
# === END CONFIGURATION ===

from pathlib import Path

import numpy as np
import pandas as pd

from verification_common import ColumnSim, layer_index, layered_reference, layered_steady

out_dir = Path(__file__).resolve().parent / RESULTS_DIR_NAME
out_dir.mkdir(exist_ok=True)

z = np.linspace(0, DEPTH, N)
k_nodes = np.asarray(K_LAYERS)[layer_index(z, INTERFACES)]
n_snap = int(round(T_SNAPSHOT_DAYS * 86400 / DT))
n_steady = int(round(T_STEADY_DAYS * 86400 / DT))

T_exact_steady = layered_steady(z, INTERFACES, K_LAYERS, T_TOP, T_BOTTOM)
T_ref_snap = layered_reference(z, INTERFACES, K_LAYERS, C_VOL, T_TOP, T_BOTTOM, T_INIT, T_SNAPSHOT_DAYS * 86400, DEPTH)

table = {"depth_m": z, "k_W_m_K": k_nodes, "steady_exact_K": T_exact_steady, "snapshot_reference_K": T_ref_snap}
rows = []
for label, options in VARIANTS.items():
    sim = ColumnSim(depth=DEPTH, N=N, dt=DT, T_init=T_INIT, k_frozen=k_nodes, k_unfrozen=k_nodes,
                    c_frozen=C_VOL, c_unfrozen=C_VOL, T_melt=273.15, **options)
    sim.advance(n_snap, T_top=T_TOP, T_bottom=T_BOTTOM)
    T_snap = sim.temperature()[0]
    sim.advance(n_steady - n_snap, T_top=T_TOP, T_bottom=T_BOTTOM)
    T_steady = sim.temperature()[0]
    table[f"{label}_snapshot_K"] = T_snap
    table[f"{label}_steady_K"] = T_steady
    for case, T_num, T_ref in (("transient snapshot", T_snap, T_ref_snap), ("steady state", T_steady, T_exact_steady)):
        err = T_num - T_ref
        rows.append((label, case, np.sqrt(np.mean(err**2)), np.abs(err).max()))

pd.DataFrame(table).to_csv(out_dir / "layered_profiles.csv", index=False, float_format="%.8g")

lines = [
    "Layered-column verification (fixed end temperatures)",
    f"{len(K_LAYERS)} layers k = {K_LAYERS} W/m/K, interfaces at {INTERFACES} m, N = {N} (dz = {DEPTH / (N - 1):.3f} m), dt = {DT:.0f} s",
    f"transient snapshot at {T_SNAPSHOT_DAYS:g} d vs fine-grid reference ({1600} cells, matrix exponential); steady state after {T_STEADY_DAYS:g} d vs analytical",
    "",
    f"{'scheme':8s}{'comparison':22s}{'RMSE [K]':>10s}{'max |diff| [K]':>15s}",
]
lines += [f"{lab:8s}{case:22s}{rmse:10.4f}{mx:15.4f}" for lab, case, rmse, mx in rows]
(out_dir / "layered_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Wrote {out_dir / 'layered_profiles.csv'}\nWrote {out_dir / 'layered_summary.txt'}")
