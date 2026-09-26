# 2DV water–soil–ice thermal model: demonstrations

First, standalone version of `arctic_xbeach/thermal2dv` (branch `2DV-redesign`). The design and
the literature behind it are in `docs/2dv_design.md`. The model is not yet coupled to XBeach or to
`main.py`; the 1D v1.0 model is untouched.

Run from the repository root with the project environment:

```bash
python -m pytest tests/test_thermal2dv.py        # verification, 12 tests, ~25 s
python examples/2dv/run_analytical_suite.py      # analytical figure, ~1 min
python examples/2dv/run_flume_block.py           # ~45 s
python examples/2dv/run_bluff_niche.py           # ~1.5 min
python examples/2dv/run_barter_transect.py       # ~20 min for one year
```

Each script has a `# === CONFIGURATION ===` block at the top and writes a NetCDF file plus figures
to `examples/2dv/output/<case>/` (`*.nc` is git-ignored; the PNGs are committed).

## What the model does

- **One grid for sea, beach, bluff and tundra.** Every cell carries a mineral fraction `phi_s`, a
  total-H2O fraction `phi_h2o` (liquid + ice) and air. The state is volumetric enthalpy, so sea
  water, sea ice, pore ice and ice-rich permafrost share one equation.
- **Salt as a second conserved variable.** Salt is excluded from ice, so it sits in the liquid, and
  below the liquidus T_f = −0.054 S the liquid content follows θ_w = m·salt/(−T). This one curve
  gives mushy sea ice with brine and partly frozen saline permafrost. Fresh water keeps a sharp
  plateau at 0 °C. Salt diffuses (turbulently in water, molecularly in pores and brine), relaxes to
  an ocean reference profile, and drains from sea ice into the water beneath it.
- **Implicit time stepping.** Backward Euler with a Newton iteration on the piecewise H–T
  relation, solved with `scipy.sparse.linalg.splu`. Typically 3–5 iterations per step, with no
  Fourier-number limit: the Barter year uses 3 h steps.
- **Water.** Liquid cells mix turbulently (surf-zone and offshore eddy diffusivities). They are
  relaxed towards an ocean reference profile in an offshore sponge and by alongshore flushing, which
  is how a 2DV slice sees the large-scale circulation. Shortwave penetrates the water column in two
  bands, and whatever reaches the bed heats the bed.
- **Surface energy balance.** One balance for water, ice and land faces: absorbed shortwave,
  longwave in/out and bulk sensible heat. The half-cell conduction to the cell centre is included
  through the surface temperature.
- **Boundaries.** Geothermal flux at the base (60 mW m⁻²); zero flux on the landward and offshore
  soil boundaries.
- **Bluff melt.** Faces between water and soil or ice carry a boundary-layer resistance 1/h
  (h = 500 W m⁻² K⁻¹, as in v1.0). A melting face is clamped at its melting point, so it receives
  h (T_w − T_f) even while the cell centre is still colder. Without the clamp, a cell-centred scheme
  underestimates the melt of cold ice-rich bluffs by roughly a factor of two at 10 cm resolution.
- **Erosion as fraction change.**
  - Thawed, ice-rich soil (φ_h2o ≥ 0.5) that touches water becomes water. Any heat absorbed after
    it fully thawed is passed on to the solid behind it.
  - An overhang whose niche is deeper than `d_crit` (1.5 m) drops into the niche, keeping its
    enthalpy.
  - Sediment that is not ice-rich is left alone; that is XBeach's job in the coupled model.
- **Bookkeeping.** Energy is tracked for every step and every fraction change. Across all demos the
  budget closes to better than 1e-8 of the net exchange.

## Verification (`tests/test_thermal2dv.py`, 12 tests, all passing)

`run_analytical_suite.py` draws eight of them against their analytic solution in one figure
(`output/analytical/analytical_suite.png`, numbers in `analytical_suite.txt`).

| Test | Check | Result |
|---|---|---|
| Conduction | erfc solution, surface step, 10 days | max error 0.006 K |
| Stefan thaw | two-phase Neumann front, saturated silt, 30 days | +0.5% |
| Ice block | faces retreat at h ΔT / L_v | melted area within 1.3% at 12 h |
| Cold ice-rich face | −8 °C, 65% ice under 5 °C water | 0.85 m/day against 0.93 idealised; the remaining 8% is meltwater cooling the water at the face |
| Energy budget | domain change against boundary + logged exchanges | < 1e-6 relative |
| Undercut detection | niche geometry | exact |
| H–T–salt closure | round trip, liquidus θ_w = m·salt/(−T), dT/dH, continuity at T_f | exact to 1e-9 |
| Salt diffusion | erf profile, open water (turbulent) and pore water (molecular × tortuosity, 1 year) | < 1% of the step |
| Salt relaxation | decay towards S_ref against the backward-Euler series; logged flux | exact |
| Fresh ice growth | one-phase Neumann, −10 °C surface, 5–10 days | −1.5% at dz = 1 cm (first order in dz) |
| Saline freezing | liquid fraction of saline pore water during freezing, S = 5, 15, 35 | on the liquidus to 1e-15 |
| Brine rejection | sea water freezing for 10 days | brine on the liquidus; consolidated ice 8–26 psu; water below enriched to 32.1 psu at its own freezing point; salt budget 1e-10, energy 4e-11 |

## Demo 1: ice block in a wave flume (`run_flume_block.py`)

This is a laboratory-scale analogue of a bluff, after the cold-room flume experiments with frozen
blocks (Costard et al. 2003; Dupeyrat et al. 2011).

- **Setup.** A 2 m × 1 m block of pure ice at −5 °C stands against the landward wall of an 8 m
  flume, on 5 cm cells (the grid is drawn on the figures).
- **Water.** Fresh water 0.4 m deep is held at 5 °C by recirculation: a sponge at the wave-maker
  end plus flushing. The waves are not resolved; they act only through the well-mixed water and
  the face heat transfer coefficient h = 500 W m⁻² K⁻¹.
- **Niche.** The submerged face melts and cuts a niche at the waterline. Before the first
  failure it retreats at 0.61 m/day against the quasi-steady h ΔT / (L_v + C_i ΔT₀) = 0.63 m/day
  (−2.4%).
- **Failure.** The ice above the water becomes an overhang and drops into the niche once the
  niche is 0.3 m deep (`Settings.collapse_ice`, at 12, 21, 33 and 46 h). Each fallen block has to
  melt before the face is exposed again, so the crest retreats 0.68 m in 48 h.
- **Budget.** Energy closes to 1e-9.

Setting `PHI_S_BLOCK = 0.35` gives an ice-rich silt block instead (thaw, then removal of the
thawed matrix). The free ice-block case, a block melting on three faces, is kept as a
verification test and as panel (c) of the analytical suite.
Figures: `flume_block_snapshots.png` (phase and temperature) and `flume_block_retreat.png`.

## Demo 2: bluff niche and block failure (`run_bluff_niche.py`)

A 5 m ice-rich bluff at −8 °C is exposed to a six-day, +1 m storm with 5 °C water:

- Before the first failure the niche cuts in at close to the idealised 0.93 m/day.
- The niche reaches d_crit = 1.5 m after 1.7 days. The overhang drops in, is wetted again and the
  cycle repeats every ~1.5 days.
- After the storm (water at +0.1 m) a thin notch keeps working at the new waterline.
- After ten days the crest has retreated 1.55 m (`bluff_niche_snapshots.png`,
  `bluff_niche_retreat.png`).

## Demo 3: Barter Island transect, one synthetic year (`run_barter_transect.py`)

The repo's Barter Island profile (−5 m offshore to a 13 m ice-rich bluff) runs for one year with
idealised forcing. The forcing has a Barter-like air-temperature cycle, polar night, a sea at its
freezing point in winter and warm-over-cold stratification in summer, a 0.1 m tide and one 1.6 m
surge in mid-August.

The grid has 276 × 103 cells (17.5k active): 0.5 m × 0.25 m around the bluff, stretching to 10 m
offshore and 3 m at the 30 m deep base. It runs 2920 steps of 3 h in about 23 minutes on a desktop
(0.48 s per step with salt, 3.7 iterations on average). The energy budget closes to 9e-10 of the
net exchange and the salt budget to 8e-12 of the salt content.

- **Tundra.** Thaw starts in mid-June, the active layer reaches 0.85 m, and a zero curtain holds
  at 0.5–1 m depth until mid-November. Without snow and an organic layer this is deeper than
  observed (roughly 0.4–0.6 m) and the winter ground is too cold. These are expected biases of this
  first version.
- **Sea ice.** It forms by itself from the water cells: 1.2–1.4 m of landfast ice, grounded in the
  shallows as bottom-fast ice. The seabed under it cools strongly, which is visible in the March
  panel. The nearshore is ice-free from day 179 to day 293.
- **Water.** It warms to about 6.6 °C at 50 m offshore and 5.7 °C at 250 m, following the ocean
  reference profile through the sponge and flushing. Shortwave warms the seabed sediment beneath.
- **Salinity.**
  - New ice forms at the salinity of the water and drains towards about 5 psu. Each new layer
    shows as a sawtooth in the probe series.
  - Brine keeps the water under the ice a few tenths above 30 psu; flushing limits the enrichment.
  - Grounded ice cannot drain and keeps about 9 psu.
  - At break-up in late June the meltwater freshens the surface to about 8 psu for a few days.
  - In summer the water follows the fresher surface layer of the ocean reference, with a lag of a
    few days.
- **Subsurface.** A seaward-dipping permafrost wedge sits under the shoreline, with unfrozen saline
  sediment offshore. Freeze-up in November leaves the summer heat pulse trapped in the seabed.
- **Bluff.** The August surge (+1.6 m) brings 5–6 °C water to the toe of the ice-rich bluff. It cuts
  a notch, the overhang fails once, and 3.2 m³ per m of sediment is released. Barter's long-term
  mean is roughly 20 m³ per m per year across all storms.

Figures: `barter_snapshots.png` (temperature and phase at six dates), `barter_full_domain.png`
(whole domain at the surge peak) and `barter_timeseries.png` (forcing, tundra ground temperature,
thaw depth, nearshore water temperature and ice thickness) and `barter_salinity.png` (bulk
salinity in March and at the surge, and salinity of the water and sea ice through the year).

## Comparison with the v1.0 1D model (`compare_1d/`)

The same flat, dry silt column at Barter Island (φ_s = 0.6, φ_H₂O = 0.4, 10 m deep) is run for
2016 by both models. Forcing, soil properties, initial profile and geothermal gradient are the
same, and so is v1.0's surface formulation:
- sensible heat with h = 16.12 U^0.8;
- ERA5 net SW + net LW + latent flux, prescribed hourly.

The 2DV model takes these through `Forcing.q_surface` and `Forcing.h_air`, so only the numerics
differ.

```bash
python examples/2dv/compare_1d/run_v10_column.py                  # v1.0, ~5 min (needs the full ERA5 file)
python examples/2dv/compare_1d/run_v10_column_barter_latent.py    # v1.0 with the Barter-config latent heat
python examples/2dv/compare_1d/run_compare_1d.py                  # 2DV column, v1.0 scheme replicas, figure
```

| | v1.0 | 2DV | v1.0 scheme, conservative flux form |
|---|---|---|---|
| Maximum thaw depth | 1.16 m | 0.94 m | 0.90 m |
| Refreeze of the thawed layer | day 330 | day 312 | day 311 |
| Energy budget error after one year | +136 MJ m⁻² | 1e-9 MJ m⁻² | — |
| Wall time (thermal part) | 179 s (dt = 120 s) | 60 s (dt = 1 h) | — |

- **Why they differ.** v1.0 updates each node with its own k times the discrete Laplacian,
  k ∂²T/∂z², instead of the flux form ∂/∂z(k ∂T/∂z).
  - Where k jumps (frozen 2.6, thawed 1.5 W m⁻¹ K⁻¹), heat that leaves one node is not the heat
    that enters the next.
  - The column gains a spurious 136 MJ m⁻² in 2016, about 4 W m⁻² on average and as much as
    the latent heat of the whole thawed layer.
  - A re-implementation of v1.0's scheme reproduces its output to 0.03 K. Changing only the
    conduction term to the flux form gives the 2DV result (mean difference 0.001 K). The
    implicit time step and the mushy-zone treatment do not matter here; conservation does.
- **Barter-config latent heat.** The Barter Island case-study config sets `L_water_ice:
  334000000` with `rho_soil: None`. v1.0 stores enthalpy per unit mass and adds
  `L_water_ice * nb` to it, so the effective latent heat is ρ_soil · L · nb ≈ 2215 × 3.34e8 ×
  0.263 ≈ 1.9e11 J m⁻³, about 2200 times ρ_w L nb. Run with that setting, the column never thaws
  in 2016: the surface sits at 0 °C all summer (green line).

Figure: `output/compare_1d/compare_1d.png`. It shows ground temperature at five depths, thaw
depth, the 2DV − v1.0 temperature difference and the energy budget error.

## Limitations of this first version

- **Not yet coupled to XBeach.** Non-ice-rich sediment never moves, and there is no wave-dependent
  h: a constant h is used everywhere.
- **No snow, latent heat flux or organic layer on land.** Winter ground is too cold, and the active
  layer is set only by the mineral/ice mix.
- **Salt is simplified.** The liquidus is linear. Brine drainage is a relaxation, not a
  mushy-layer convection model. Density does not feed back on mixing: brine sinks only through the
  drainage rule, and fresh meltwater does not stratify. Eroded ice-rich soil becomes sea water at
  S_sea.
- **No flow.** No cross-shore advection; water heat moves by eddy diffusion and relaxation only.
- **Staircase geometry.** Bed and bluff are resolved as whole cells. Flooding ignores connectivity,
  so a tide rising above an ice lid floods the air cells above it.
- **Block failure is a geometric rule** (niche depth > d_crit), not a stability calculation.
- **Forcing is synthetic.** Nothing here is calibrated or validated against Barter observations yet.
