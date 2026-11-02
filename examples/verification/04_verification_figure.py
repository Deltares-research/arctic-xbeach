"""Verification 4: combined 3-panel figure from the results of scripts 01-03.

(a) Stefan thaw front: legacy options, flux + volumetric options, analytical Neumann solution
(b) Convergence of the flux + volumetric options (front error and temperature RMSE vs dz, fixed Fourier number)
(c) Layered column: steady-state temperature, nodal vs flux conduction vs analytical

Run 01_stefan_front.py, 02_layered_conduction.py and 03_stefan_convergence.py first.
Output (results/): verification_summary.png
"""
# === CONFIGURATION ===
RESULTS_DIR_NAME = "results"
FIGURE_NAME = "verification_summary.png"
DPI = 300
INTERFACES = [1.025, 2.525]       # [m] layer interfaces of the layered case (02_layered_conduction.py)
K_LAYERS = [1.0, 3.0, 1.5]        # [W m-1 K-1]
COLORS = {"analytical": "k", "legacy": "#d55e00", "flux": "#0072b2"}  # colour-blind safe
# === END CONFIGURATION ===

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

res = Path(__file__).resolve().parent / RESULTS_DIR_NAME
stefan = pd.read_csv(res / "stefan_front.csv")
conv = pd.read_csv(res / "stefan_convergence.csv")
lay = pd.read_csv(res / "layered_profiles.csv")

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11, "axes.linewidth": 1.0,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

# (a) Stefan front
ax = axs[0]
ax.plot(stefan["time_days"], stefan["analytical_m"], color=COLORS["analytical"], lw=2.2, label="Analytical (Neumann)")
ax.plot(stefan["time_days"], stefan["legacy_m"], color=COLORS["legacy"], lw=1.8, ls="--", label="Legacy (nodal, per-mass latent)")
ax.plot(stefan["time_days"], stefan["flux_volumetric_m"], color=COLORS["flux"], lw=1.8, label="Flux, volumetric latent")
ax.set_xlim(0, stefan["time_days"].max()); ax.set_ylim(0, None)
ax.set_xlabel("Time (days)"); ax.set_ylabel("Thaw depth (m)")
ax.legend(loc="lower right", frameon=False)

# (b) convergence, series A
A = conv[conv["series"].str.startswith("A")]
ax = axs[1]
ax.loglog(A["dz_m"], A["front_error_m"].abs(), "o-", color=COLORS["flux"], label="Front position error (m)")
ax.loglog(A["dz_m"], A["temp_rmse_K"], "s-", color="#009e73", label="Temperature RMSE (K)")
h = np.array([A["dz_m"].min(), A["dz_m"].max()])
ref = 0.4 * A["front_error_m"].abs().iloc[0] * (h / h[0])  # first-order reference line
ax.loglog(h, ref, "k:", lw=1.2)
ax.text(h[1], ref[1] * 0.62, "first order", ha="right", va="top", fontsize=10)
ax.set_xlabel("Grid spacing Δz (m), Δt ∝ Δz²"); ax.set_ylabel("Error at t = 30 d (m or K)")
ax.legend(loc="lower right", frameon=False)
ax.grid(True, which="both", lw=0.3, alpha=0.5)

# (c) layered column, steady state
ax = axs[2]
z = lay["depth_m"]
edges = [0.0] + INTERFACES + [z.max()]
for i, k in enumerate(K_LAYERS):
    ax.axhspan(edges[i], edges[i + 1], color="0.93" if i % 2 == 0 else "0.82", zorder=0)
    ax.text((0.14 if i == 0 else 0.02) if i < 2 else 0.98, edges[i] + 0.22, f"k = {k:g} W m$^{{-1}}$ K$^{{-1}}$", transform=ax.get_yaxis_transform(),
            ha="left" if i < 2 else "right", va="center", fontsize=10)
C = 273.15
ax.plot(lay["steady_exact_K"] - C, z, color=COLORS["analytical"], lw=2.2, label="Analytical")
ax.plot(lay["nodal_steady_K"] - C, z, color=COLORS["legacy"], lw=1.8, ls="--", label="Nodal k ∂²T/∂z² (legacy)")
ax.plot(lay["flux_steady_K"] - C, z, "o", color=COLORS["flux"], ms=3.5, markevery=2, label="Flux form ∂/∂z(k ∂T/∂z)")
ax.set_ylim(z.max(), 0); ax.set_xlim(lay["steady_exact_K"].min() - C - 0.3, lay["steady_exact_K"].max() - C + 0.3)
ax.set_xlabel("Steady-state temperature (°C)"); ax.set_ylabel("Depth (m)")
ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")

for ax, lab in zip(axs, "abc"):
    ax.text(0.02, 0.97, f"({lab})", transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

out = res / FIGURE_NAME
fig.savefig(out, dpi=DPI)
print(f"Wrote {out}")
