"""Regenerate tests/data/legacy_reference.npz with the ORIGINAL v1.0 solver (git commit c6ab74b, arctic_xbeach/model.py).

Not needed for normal test runs; kept for provenance. The v1.0 file is extracted with `git show` and imported as a
separate module, then driven through the same scenario as tests/test_thermal_solver.py::test_defaults_reproduce_v10.
"""
# === CONFIGURATION ===
LEGACY_COMMIT = "c6ab74b"
# === END CONFIGURATION ===
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import arctic_xbeach.miscellaneous as um  # noqa: E402
from legacy_scenario import SCENARIO as S, initial_temperature  # noqa: E402


class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self


src = subprocess.run(["git", "show", f"{LEGACY_COMMIT}:arctic_xbeach/model.py"], cwd=ROOT, capture_output=True, check=True).stdout
with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / "legacy_model.py"
    f.write_bytes(src)
    spec = importlib.util.spec_from_file_location("legacy_model", f)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)


class LegacyColumn(legacy.Simulation):
    def _get_ghost_node_boundary_condition(self, timestep_id, subgrid_timestep_id):
        self.heat_flux = np.full(1, S["surface_flux"])
        return self.temp_matrix[:, 1] + 2 * self.dz * self.heat_flux / self.k_matrix[:, 0]


N = S["N"]
sim = LegacyColumn.__new__(LegacyColumn)
sim.config = AttrDict(thermal=AttrDict(
    L_water_ice=S["L_water_ice"], rho_water=S["rho_water"], rho_particle=S["rho_particle"], rho_soil="None",
    T_melt=S["T_melt"], dt=S["dt"], max_depth=S["depth"], grid_resolution=N,
    c_soil_frozen=S["c_frozen"], c_soil_unfrozen=S["c_unfrozen"],
    k_soil_frozen_min=1.2, k_soil_frozen_max=2.4, k_soil_unfrozen_min=0.5, k_soil_unfrozen_max=1.0,
    depth_constant_k=1.0, nb_max=S["nb_max"], nb_min=S["nb_min"], nb_max_depth=0.5, nb_min_depth=1.5,
    geothermal_gradient=S["geothermal_gradient"]))
sim.xgr = np.zeros(1)
sim.dz = S["depth"] / (N - 1)
sim.thermal_zgr = np.linspace(0, S["depth"], N)
sim.define_soil_property_matrices(1, define_nb=True)
sim.k_frozen_matrix = S["k_frozen"].reshape(1, -1).copy()
sim.k_unfrozen_matrix = S["k_unfrozen"].reshape(1, -1).copy()
sim.A_matrix = um.get_A_matrix(N)
T = initial_temperature(N).reshape(1, -1)
sim.temp_matrix = T.copy()
frozen = T <= S["T_melt"]
sim.enthalpy_matrix = frozen * sim.Cs_matrix * T + (1 - frozen) * (
    sim.Cl_matrix * T + (sim.Cs_matrix - sim.Cl_matrix) * S["T_melt"] + S["L_water_ice"] * sim.nb_matrix)
for _ in range(S["n_steps"]):
    sim.thermal_update(0, 0)
out = Path(__file__).parent / "data" / "legacy_reference.npz"
np.savez(out, enthalpy=sim.enthalpy_matrix[0], temperature=sim.temp_matrix[0])
print(f"Wrote {out} (legacy commit {LEGACY_COMMIT})")
