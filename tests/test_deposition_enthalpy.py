"""Deposited submerged sediment must get the temperature of the sea water, also when thermal.rho_soil is a fixed number.

The fill enthalpy is interpreted by the solver with soil_density_matrix, so it has to be built with the same density.
Driven like tests/test_diagnostics.py: a fake XBeach output file, no XBeach needed.
"""
import os

import numpy as np
import pytest

from run_helpers import build_run
from test_diagnostics import TID, fake_xbeach_output, make_sim

SEA_TEMP = 280.0  # K, above T_melt so the latent heat term is part of the fill enthalpy
SEDERO = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.20])  # m, deposition on the two wet nodes only (the new bed stays below the water level)


@pytest.mark.parametrize("rho_soil", ["None", 2000])
def test_deposited_sediment_has_sea_temperature(tmp_path, rho_soil):
    cwd = os.getcwd()
    try:
        sim = make_sim(build_run(tmp_path / "run", thermal=dict(rho_soil=rho_soil)), with_diag=False)
        fake_xbeach_output(sim, SEDERO)
        old_x, old_z = sim.abs_xgr.copy(), sim.abs_zgr.copy()
        sim.current_sea_temp = SEA_TEMP
        sim.update_grid(TID, "xboutput.nc")

        # new nodes above the top of the old column they are interpolated from (nearest top node) and below the water are filled
        src = np.abs(sim.abs_xgr[:, :1] - old_x[:, 0][None, :]).argmin(axis=1)
        filled = (sim.abs_zgr > old_z.max(axis=1)[src][:, None] + 1e-9) & (sim.abs_zgr <= sim.water_level)
        assert filled[5:].any() and not filled[:5].any()
        temp = sim._temperature_from_enthalpy(*sim._get_phase_masks())
        assert temp[filled] == pytest.approx(SEA_TEMP, abs=1e-6)
    finally:
        os.chdir(cwd)
