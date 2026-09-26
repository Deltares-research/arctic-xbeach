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
| Ice block | faces retreat at h ΔT / L_v | melted area within 4% at 12 h |
| Cold ice-rich face | −8 °C, 65% ice under 5 °C water | 0.85 m/day against 0.93 idealised; the remaining 8% is meltwater cooling the water at the face |
| Energy budget | domain change against boundary + logged exchanges | < 1e-6 relative |
| Undercut detection | niche geometry | exact |
| H–T–salt closure | round trip, liquidus θ_w = m·salt/(−T), dT/dH, continuity at T_f | exact to 1e-9 |
| Salt diffusion | erf profile, open water (turbulent) and pore water (molecular × tortuosity, 1 year) | < 1% of the step |
| Salt relaxation | decay towards S_ref against the backward-Euler series; logged flux | exact |
| Fresh ice growth | one-phase Neumann, −10 °C surface, 5–10 days | −1.5% at dz = 1 cm (first order in dz) |
| Saline freezing | liquid fraction of saline pore water during freezing, S = 5, 15, 35 | on the liquidus to 1e-15 |
| Brine rejection | sea water freezing for 10 days | brine on the liquidus; consolidated ice 8–26 psu; water below enriched to 32.1 psu at its own freezing point; salt budget 1e-10, energy 4e-11 |

## Demo 1: frozen block in a wave flume (`run_flume_block.py`)

This is a laboratory-scale analogue of a bluff, after the cold-room flume experiments with frozen
sediment blocks (Costard et al. 2003; Dupeyrat et al. 2011).

- **Setup.** A 2 m × 1 m block of ice-rich frozen silt (65% ice, −5 °C) stands against the
  landward wall of an 8 m flume.
- **Water.** Fresh water 0.4 m deep is held at 5 °C by recirculation: a sponge at the wave-maker
  end plus flushing. The waves are not resolved; they act only through the well-mixed water and
  the face heat transfer coefficient h = 500 W m⁻² K⁻¹.
- **Niche.** The submerged face melts and cuts a niche at the waterline. Before the first
  failure it retreats at 0.92 m/day against the quasi-steady h ΔT / (L_v + C_f ΔT₀) = 0.95 m/day
  (−4%).
- **Failure.** The frozen part above the water becomes an overhang and drops into the niche once
  the niche is 0.3 m deep (at 8, 15, 24 and 32 h).
- **Armouring.** Each fallen block must melt before the face is exposed again, so after 36 h the
  crest has retreated 0.61 m, about half the undisturbed rate.
- **Budget.** Energy closes to 9e-9.

Setting `PHI_S_BLOCK = 0` gives a pure ice block at 0 °C with the same geometry: it melts without
failures at h ΔT / L_v. The unconfined pure-ice case (the three melting faces of a free block) is
kept as a verification test and as panel (c) of the analytical suite.
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
