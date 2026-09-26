# 2DV water–soil–ice thermal model: demonstrations

First, standalone version of `arctic_xbeach/thermal2dv` (branch `2DV-redesign`). The design and
the literature behind it are in `docs/2dv_design.md`. The model is not yet coupled to XBeach or to
`main.py`; the 1D v1.0 model is untouched.

Run from the repository root with the project environment:

```bash
python -m pytest tests/test_thermal2dv.py        # verification, ~10 s
python examples/2dv/run_ice_block.py             # ~10 s
python examples/2dv/run_bluff_niche.py           # ~1.5 min
python examples/2dv/run_barter_transect.py       # ~20 min for one year
```

Each script has a `# === CONFIGURATION ===` block at the top and writes a NetCDF file plus figures
to `examples/2dv/output/<case>/` (`*.nc` is git-ignored; the PNGs are committed).

## What the model does

- **One grid for sea, beach, bluff and tundra.** Every cell carries a mineral fraction `phi_s`, a
  total-H2O fraction `phi_h2o` (liquid + ice) and air. The state is volumetric enthalpy, so sea
  water, sea ice, pore ice and ice-rich permafrost share one equation. Static salinity sets the
  freezing point: seawater at S = 30 freezes at −1.64 °C, fresh pore water at 0 °C.
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

## Verification (`tests/test_thermal2dv.py`, 6 tests, all passing)

| Test | Check | Result |
|---|---|---|
| Conduction | erfc solution, surface step, 10 days | max error < 0.05 K |
| Stefan | two-phase Neumann thaw front, saturated silt, 30 days | within 3% |
| Ice block | faces retreat at h ΔT / L_v | melted volume within 10% (≈ 4% at 7 h) |
| Cold ice-rich face | −8 °C, 65% ice under 5 °C water | 0.85 m/day against 0.93 idealised; the remaining 8% is meltwater cooling the water at the face |
| Energy budget | domain change against boundary + logged exchanges | < 1e-6 relative |
| Undercut detection | niche geometry | exact |

## Demo 1: ice block (`run_ice_block.py`)

A 1 m × 1 m ice block at 0 °C sits in a tank of well-mixed 5 °C water. The top and both side faces
retreat at v = h ΔT / L_v = 0.65 m/day, and the model follows the analytic area (W − 2vt)(H − vt)
until the block is gone (`ice_block_area.png`). Energy closes to 1e-8.

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
offshore and 3 m at the 30 m deep base. It runs 2920 steps of 3 h in about 19 minutes on a desktop
(0.39 s per step, 3.8 iterations on average). The energy budget closes to 5e-9 of the net exchange.

- **Tundra.** Thaw starts in mid-June, the active layer reaches 0.85 m, and a zero curtain holds
  at 0.5–1 m depth until mid-November. Without snow and an organic layer this is deeper than
  observed (roughly 0.4–0.6 m) and the winter ground is too cold. These are expected biases of this
  first version.
- **Sea ice.** It forms by itself from the water cells: 1.3 m of landfast ice, grounded in the
  shallows as bottom-fast ice. The seabed under it cools strongly, which is visible in the March
  panel. The nearshore is ice-free from day 179 to day 293.
- **Water.** It warms to about 6.6 °C at 50 m offshore and 5.7 °C at 250 m, following the ocean
  reference profile through the sponge and flushing. Shortwave warms the seabed sediment beneath.
- **Subsurface.** A seaward-dipping permafrost wedge sits under the shoreline, with unfrozen saline
  sediment offshore. Freeze-up in November leaves the summer heat pulse trapped in the seabed.
- **Bluff.** The August surge (+1.6 m) brings 5–6 °C water to the toe of the ice-rich bluff. It cuts
  a notch, the overhang fails once, and 3.2 m³ per m of sediment is released. Barter's long-term
  mean is roughly 20 m³ per m per year across all storms.

Figures: `barter_snapshots.png` (temperature and phase at six dates), `barter_full_domain.png`
(whole domain at the surge peak) and `barter_timeseries.png` (forcing, tundra ground temperature,
thaw depth, nearshore water temperature and ice thickness).

## Limitations of this first version

- **Not yet coupled to XBeach.** Non-ice-rich sediment never moves, and there is no wave-dependent
  h: a constant h is used everywhere.
- **No snow, latent heat flux or organic layer on land.** Winter ground is too cold, and the active
  layer is set only by the mineral/ice mix.
- **Static salinity.** No brine rejection or salt transport. Meltwater from ice is converted to
  seawater salinity.
- **No flow.** No cross-shore advection; water heat moves by eddy diffusion and relaxation only.
- **Staircase geometry.** Bed and bluff are resolved as whole cells. Flooding ignores connectivity,
  so a tide rising above an ice lid floods the air cells above it.
- **Block failure is a geometric rule** (niche depth > d_crit), not a stability calculation.
- **Forcing is synthetic.** Nothing here is calibrated or validated against Barter observations yet.
