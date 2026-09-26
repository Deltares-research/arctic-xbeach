"""Run the Arctic-XBeach v1.0 analytical thermal tests with the legacy and the corrected thermal
numerics, and score both against the analytical solutions.

Cases (conceptual mode: constant air temperature, no radiation, XBeach off, 1 year, dt = 120 s):
    01_dirichlet_warming   frozen, 250 K -> 270 K surface (no phase change)
    02_dirichlet_cooling   thawed, 300 K -> 280 K surface (no phase change)
    03_neumann_constant    frozen, 5 W m-2 into the surface (no phase change)
    04_dirichlet_warming_latent_heat   250 K -> 300 K surface, with thawing (Stefan problem);
                           config read from the main checkout (not in this worktree)

Variants:
    legacy   thermal.conduction_form = nodal, latent_heat_legacy = true (v1.0 as published)
    new      thermal.conduction_form = flux,  latent_heat_legacy = false (defaults)

The "Dirichlet" cases are driven by v1.0's sensible heat flux h (T_air - T_s) with ERA5 wind
(h ~ 16 U^0.8 W m-2 K-1), so they are near-Dirichlet; the analytical reference is the same one
used for the v1.0 paper figures (thermo-morphological-model/scripts_kees/compare_analytical_v3.py
and compare_latent_heat.py): a finite slab with zero bottom gradient (01, 02), the semi-infinite
constant-flux solution (03) and the two-phase Neumann/Stefan solution with L_vol = rho_w L nb (04).

Outputs (examples/2dv/output/compare_1d/): analytical_v10_summary.txt, analytical_v10.png.
Case copies and model results: examples/2dv/compare_1d/analytical_runs/<case>_<variant>/.
"""
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from scipy.optimize import brentq
from scipy.special import erf, erfc

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parents[2]                       # worktree root (contains main.py)
MAIN_CHECKOUT = Path("D:/Git/arctic-xbeach")           # read-only source of case 04 and the full forcing files
ERA5_PATH = MAIN_CHECKOUT / "examples/case_studies/barter_island/database/era5.csv"      # full ERA5 (wind)
STORM_PATH = MAIN_CHECKOUT / "examples/case_studies/barter_island/database/storms.csv"   # full water levels
CASES = {
    "01_dirichlet_warming": PROJ_DIR / "examples/analytical/01_dirichlet_warming",
    "02_dirichlet_cooling": PROJ_DIR / "examples/analytical/02_dirichlet_cooling",
    "03_neumann_constant": PROJ_DIR / "examples/analytical/03_neumann_constant",
    "04_dirichlet_warming_latent_heat": MAIN_CHECKOUT / "examples/analytical/04_dirichlet_warming_latent_heat",
}
VARIANTS = {
    "legacy": {"conduction_form": "nodal", "latent_heat_legacy": True},
    "new": {"conduction_form": "flux", "latent_heat_legacy": False},
}
RUN_DIR = SCRIPT_DIR / "analytical_runs"
OUT_DIR = PROJ_DIR / "examples/2dv/output/compare_1d"
RUN_MODELS = True          # False: only re-score existing results
X_INDEX = 1                # middle of the 3 x-points
EVAL_DAYS = [30, 90, 365]
DPI = 300


# ------------------------------------------------------------------ analytical solutions
def dirichlet_slab(z, t, T_i, T_s, alpha, L, n_terms=400):
    """Slab 0 <= z <= L, T(0,t) = T_s, dT/dz(L) = 0, T(z,0) = T_i (eigenfunction series)."""
    T = np.full_like(z, T_s, dtype=float)
    for n in range(1, n_terms + 1):
        lam = (2 * n - 1) * np.pi / (2 * L)
        T += (2 / L) * (T_i - T_s) / lam * np.sin(lam * z) * np.exp(-alpha * lam**2 * t)
    return T


def neumann_semi_infinite(z, t, q, T_i, k, alpha):
    """Semi-infinite medium, constant flux q into the surface (Carslaw & Jaeger 2.9)."""
    s = np.sqrt(alpha * t)
    return T_i + 2 * q / k * s / np.sqrt(np.pi) * np.exp(-(z / (2 * s)) ** 2) - q * z / k * erfc(z / (2 * s))


def stefan(T_i, T_s, T_m, c_vol, k, L_vol):
    """Two-phase Neumann solution with equal properties; returns (T(z, t), s(t))."""
    alpha = k / c_vol
    St_s, St_i = c_vol * (T_s - T_m) / L_vol, c_vol * (T_m - T_i) / L_vol
    lam = brentq(lambda x: x * np.sqrt(np.pi) * np.exp(x**2) - St_s / erf(x) + St_i / erfc(x), 1e-4, 5.0)

    def T(z, t):
        eta = z / (2 * np.sqrt(alpha * t))
        return np.where(z < 2 * lam * np.sqrt(alpha * t),
                        T_s + (T_m - T_s) * erf(eta) / erf(lam),
                        T_i + (T_m - T_i) * erfc(eta) / erfc(lam))

    return T, (lambda t: 2 * lam * np.sqrt(alpha * t))


def reference(cfg):
    """Analytical T(z, t) (and thaw depth s(t) or None) for a case config."""
    th, data = cfg["thermal"], cfg["data"]
    T_i, T_air, L = float(data["conceptual_ini"]), float(data["conceptual_temp"]), float(th["max_depth"])
    frozen = T_i < th["T_melt"]
    k = th["k_soil_frozen_max"] if frozen else th["k_soil_unfrozen_max"]
    c = th["c_soil_frozen"] if frozen else th["c_soil_unfrozen"]
    q = data.get("conceptual_flux")
    if q not in (None, []):
        return (lambda z, t: neumann_semi_infinite(z, t, float(q), T_i, k, k / c)), None
    if (T_i - th["T_melt"]) * (T_air - th["T_melt"]) < 0:     # crosses the melting point: Stefan
        L_vol = th["rho_water"] * th["L_water_ice"] * th["nb_max"]
        return stefan(T_i, T_air, th["T_melt"], c, k, L_vol)
    return (lambda z, t: dirichlet_slab(z, t, T_i, T_air, k / c, L)), None


# ------------------------------------------------------------------ model runs
def prepare_case(name, src, variant):
    dst = RUN_DIR / f"{name}_{variant}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ["x.grd", "bed.dep", "ne_layer.txt"]:
        shutil.copy2(src / f, dst / f)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    cfg["data"]["forcing_data_path"] = ERA5_PATH.as_posix()
    cfg["data"]["storm_data_path"] = STORM_PATH.as_posix()
    cfg["thermal"].update(VARIANTS[variant])
    (dst / "config.yaml").write_text(
        f"# generated by {Path(__file__).name} from {src.as_posix()}/config.yaml (variant '{variant}')\n"
        + yaml.safe_dump(cfg, sort_keys=False))
    return dst


def run_case(case_dir):
    """Run one case in a fresh interpreter (v1.0 main() chdir's and owns a module-level logger)."""
    code = ("import sys; sys.path.insert(0, r'%s'); from arctic_xbeach.model import Simulation; "
            "from main import main; main(Simulation(r'%s', proj_dir=r'%s'), print_to_screen=False)"
            % (PROJ_DIR, case_dir, PROJ_DIR))
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return case_dir.name, r.returncode, time.perf_counter() - t0, r.stderr[-2000:]


# ------------------------------------------------------------------ scoring
def zero_crossing_depth(prof, z, T_m):
    """Depth of the first crossing below T_m from the surface (linear); 0 if the surface is frozen."""
    if prof[0] <= T_m:
        return 0.0
    j = int(np.argmax(prof <= T_m))
    if prof[j] > T_m:
        return z[-1]
    return z[j - 1] + (prof[j - 1] - T_m) / (prof[j - 1] - prof[j]) * (z[j] - z[j - 1])


def score(name, variant):
    case_dir = RUN_DIR / f"{name}_{variant}"
    cfg = yaml.safe_load((case_dir / "config.yaml").read_text())
    th = cfg["thermal"]
    T_ref, s_ref = reference(cfg)
    ds = xr.open_dataset(case_dir / "results" / "results.nc")
    T = ds["ground_temperature_distribution"].values[:, X_INDEX, :].astype(float)
    ds.close()
    z = np.linspace(0, th["max_depth"], th["grid_resolution"])
    t = (np.arange(T.shape[0]) + 1) * 3600.0           # output i is the state at the end of hour i
    rows = []
    for d in EVAL_DAYS:
        i = min(int(round(d * 24)) - 1, T.shape[0] - 1)
        Ta = T_ref(z, t[i])
        err = T[i] - Ta
        row = {"day": d, "max_abs": np.abs(err).max(), "rmse": np.sqrt(np.mean(err**2)), "T": T[i], "Ta": Ta}
        if s_ref is not None:
            row["thaw"] = zero_crossing_depth(T[i], z, th["T_melt"])
            row["thaw_ref"] = s_ref(t[i])
        rows.append(row)
    return z, rows


def main():
    RUN_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [prepare_case(n, src, v) for n, src in CASES.items() for v in VARIANTS]
    if RUN_MODELS:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            for name, rc, wall, err in ex.map(run_case, jobs):
                print(f"{name}: return code {rc}, {wall:.0f} s wall")
                if rc != 0:
                    print(err)

    lines = ["Arctic-XBeach v1.0 analytical tests: legacy (nodal conduction, L_water_ice * nb) vs new "
             "(flux-form conduction, rho_w L nb latent). Errors = model - analytical at the x-midpoint.", ""]
    fig, axes = plt.subplots(1, len(CASES), figsize=(4 * len(CASES), 5.5), constrained_layout=True)
    for ax, name in zip(axes, CASES):
        for variant, style in zip(VARIANTS, [dict(marker="o", ls="none", ms=3, mfc="none"), dict(ls="-", lw=1.2)]):
            z, rows = score(name, variant)
            for k, r in enumerate(rows):
                msg = (f"{name:34s} {variant:6s} day {r['day']:3d}: max|err| {r['max_abs']:7.4f} K, "
                       f"RMSE {r['rmse']:7.4f} K")
                if "thaw" in r:
                    msg += f", thaw depth {r['thaw']:.3f} m (analytical {r['thaw_ref']:.3f} m)"
                lines.append(msg)
                col = plt.cm.viridis(k / (len(rows) - 1))
                ax.plot(r["T"], z, color=col, **style,
                        label=f"{variant}, {r['day']} d" if variant == "new" or k == 0 else None)
                if variant == "new":
                    ax.plot(r["Ta"], z, color="k", lw=0.8, ls="--")
        ax.plot([], [], "k--", lw=0.8, label="analytical")
        ax.set(title=name, xlabel="T (K)", ylim=(min(6.0, z[-1]), 0))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("depth (m)")
    fig.savefig(OUT_DIR / "analytical_v10.png", dpi=DPI)
    summary = "\n".join(lines)
    (OUT_DIR / "analytical_v10_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
