# Arctic-XBeach v2: design for a 2DV water–soil–ice thermal model

*Deep-research synthesis, 25 September 2026. It proposes a replacement for the 1D enthalpy module in `arctic_xbeach/model.py`. Branch `2DV-redesign` implements a first, standalone version: see `examples/2dv/README.md`.*

Evidence tags: **[V]** means read in a primary source during this run (adversarially checked, or read directly by a research agent). **[U]** means from standard references and not re-fetched, so check it before it goes into a manuscript. **[S]** means my own synthesis or design judgement.

---

## 1. Bottom line

No existing code does what you describe. That is: one cross-shore grid in which every cell carries soil, water and ice fractions; ocean and permafrost solved in a single enthalpy equation; a relaxation boundary to the large-scale ocean; one surface energy balance heating both water and land; and a bluff that melts because warm water touches it. The nearest relatives each cover part of it:

- **ACE** (Sandia) is 2DV permafrost plus mechanics. It does not resolve water temperature, it prescribes the face temperature as a Dirichlet condition, and it uses explicit time stepping [V].
- **thermalFOAM** is enthalpy–porosity with ablation, but only at niche and flume scale [V].
- **CryoGrid** has the right per-cell fraction and enthalpy state, but it is 1D [V].
- **Frederick & Buffett** is a 2D temperature–salinity transect, but at shelf scale [per landscape report].

This would therefore be new, and a defensible GMD v2 or proposal deliverable, not a reinvention.

The literature points to one design, and it is buildable in Python (numpy and scipy) at a cost comparable to one XBeach storm per simulated year:

- **Structured finite-volume x–z grid, fixed in space.** Each cell stores volume fractions (mineral, organic, total H₂O, air) and **volumetric enthalpy** as the prognostic variable, in CryoGrid's form. Liquid and ice are derived from H through a selectable freezing curve: free water, smoothed, or saline salt-exclusion.
- **Water is just H₂O-filled cells** in the same equation, with turbulent eddy diffusivity in place of molecular conductivity. Landfast ice and bottom-fast ice then emerge from the physics at no extra cost.
- **Implicit backward Euler** with Voller–Swaminathan enthalpy iteration, solved with `scipy.sparse.linalg.splu`. The time step becomes the wrapper step (1 h), not the 120 s FTCS substep.
- **The water–soil interface is a series resistance.** The resistance of each face adds a boundary-layer term 1/h, with h wave-dependent after Kobayashi (1999) and ArcticBeach. Bluff melt then falls out of the face flux; nothing is special-cased.
- **Erosion changes fractions, not the mesh.** Thawed matrix removed by XBeach, or by a block-failure rule, becomes water. XBeach still sees a single-valued bed and a non-erodible layer, both derived from the 2DV state.

The main risks are scientific, not numerical:

- the field-scale value of h (it controls niche growth, and the plausible range spans more than an order of magnitude);
- how a 2DV slice represents alongshore-driven nearshore water temperature;
- overhangs, which XBeach's single-valued bed cannot represent.

---

## 2. Where v1.0 stands (code audit)

| v1.0 component (`model.py`) | What it does | Fate in 2DV |
|---|---|---|
| `generate_perpendicular_grids`, `update_grid` (l. 1623) | Independent 1D columns normal to the bed at each XBeach x-point. After each XBeach call they are rebuilt and the field is re-interpolated. | **Replaced** by a fixed x–z grid. Erosion becomes a fraction change, so there is no re-interpolation error and energy is conserved. |
| `thermal_update` (l. 1170) | Explicit FTCS on specific enthalpy with a Fo ≤ 0.5 guard. dt = 120 s, 200 nodes over 10 m, 30 substeps per hour. | **Replaced** by implicit backward Euler, one step per wrapper hour (sub-stepped only if the front moves more than one cell). |
| `_get_phase_masks`, `_temperature_from_enthalpy` | Sharp melting at T_melt; k in the mushy zone set to the average of frozen and thawed. | **Generalised** to volumetric H with a freezing-curve closure; k from a fraction mixing rule. |
| `_get_ghost_node_boundary_condition` (l. 1259) | Flux boundary on the surface node: sensible heat from air (Man 2023), SW/LW/latent from ERA5 on **dry** points only, `hc_guess·(SST − T)` on wet points. | **Split.** Land-surface SEB kept, with the Man formula as an option. New water-surface SEB with SW penetration. The wet-point Robin condition becomes the interface resistance inside the domain. |
| `_calculate_sensible_heat_flux_water` (l. 1561) | Kobayashi 1999 h. Currently dead code: `raise Exception` at l. 1297. | **Revived** as the default h (§5.3). |
| `_get_bottom_boundary_temperature` (l. 1480) | Geothermal Neumann at 10 m, along the slanted column. | **Kept** as a Neumann flux, but moved to 30–50 m depth (§5.5). |
| `find_thaw_depth`, `write_ne_layer` | Thaw depth along each column, interpolated to XBeach x. | **Replaced** by a vertical scan of the 2DV frozen mask (§6). |
| Solar factor calculator, forcing loaders, `NCAppender`, the `check_xbeach` event logic | — | **Reused** unchanged. |

v1.0 has no water column, no lateral conduction and no landward boundary. Radiation never reaches wet points, and the sea enters only as `hc_guess = 500` times (SST − T_surface).

---

## 3. Governing equations

### 3.1 State and closures

Per cell: fixed-in-time fractions θ_m (mineral) and θ_o (organic); θ_H₂O, the total water (liquid plus ice); θ_a = 1 − θ_m − θ_o − θ_H₂O (air, for unsaturated soil); optional salinity S; and volumetric enthalpy H (J m⁻³). Following CryoGrid [V; Westermann et al. 2023, GMD 16:2607]:

  H(T) = c(T)·T − L_v (θ_H₂O − θ_w(T)),  with L_v = ρ_w L_f = 3.34 × 10⁸ J m⁻³, T in °C
  c(T) = θ_m c_m + θ_o c_o + θ_w(T) c_w + (θ_H₂O − θ_w(T)) c_i + θ_a c_a

Open water is the case θ_m = θ_o = 0, θ_H₂O = 1, so seawater, sea ice, pore ice and ice wedges all share the same equation. An ice-wedge cell has θ_H₂O ≈ 1 and starts frozen.

**Freezing curve θ_w(T)**, selectable per material:

| Option | Form | Use |
|---|---|---|
| `free` | step at T_f: θ_w = θ_H₂O above, 0 below; latent plateau at T = T_f | v1.0-equivalent; ice wedges and fresh water |
| `smooth` | Painter–Karra, or a logistic curve with 0.1–0.5 K width [V: Painter & Karra 2014, VZJ 13(4); ACE uses a generalised logistic] | silt with unfrozen water; improves Newton convergence |
| `saline` | salt exclusion: θ_w = θ_H₂O·min(1, S / S_brine(T)), where T_f(S_brine) = T; T_f(S) = −0.0575 S + 1.710523e-3 S^1.5 − 2.154996e-4 S² [U: UNESCO 1983] | seawater and sea ice (a mushy layer), saline pore water at Drew Point (porewater S rising to 36.9 below +3 m; sediment partly unfrozen at −8 °C [V: Bristol et al. 2021]) |

Salt exclusion gives about 25–30% unfrozen water at −8 °C for S₀ ≈ 37 [S], which is consistent with Bristol et al. A single curve therefore covers both sea ice and saline permafrost.

**Effective conductivity** of soil cells (selectable):
- geometric mean k = Π k_j^θ_j (Farouki 1981), the default;
- Cosenza quadratic parallel mean k = (Σ θ_j √k_j)², as in CryoGrid [V];
- Johansen / Côté–Konrad for unsaturated soil.

For saturated ice-rich silt (S_r ≈ 1), Johansen and Côté–Konrad both reduce to the saturated geometric mean [V/U: Johansen 1975; Côté & Konrad 2005]. SUTRA-ICE uses a plain arithmetic mean, which is an upper bound [V]. **Open-water cells** use k_mol + c_w K_t, with anisotropic turbulent diffusivity K_t = (K_x, K_z) from §5.2.

### 3.2 Conservation equation

For each active cell P, with volume V_P and faces f:

  V_P ∂H_P/∂t = Σ_f G_f (T_N − T_P)                          conduction and turbulent diffusion
       + V_P q_sw(z)                                        shortwave absorbed in the water column and at the bed
       + V_P γ_P c_w θ_open (T_ref(z,t) − T_P)               relaxation: offshore sponge and alongshore flushing
       + A_top Q_surf(T_P)                                  surface energy balance (top cells)
       + A_bot q_geo                                        geothermal flux (bottom cells)
       [ − ∇·(c_w u T) ]                                     cross-shore advection, off in version 1

Here γ_P is the relaxation rate: 1/τ in the offshore sponge, 1/τ_flush elsewhere in water, 0 in soil.

**Face conductance** as a series resistance. This one expression replaces v1.0's separate "wet boundary condition":

  G_f = A_f / ( d_P/k_P + d_N/k_N + R_bl ),  R_bl = 1/h if the face separates open water from soil or ice, else 0

For water cells k is turbulent and huge, of order c_w K_t ≈ 4 × 10⁴ W m⁻¹ K⁻¹. The face flux is then limited by 1/h and by soil conduction, which is exactly the Kobayashi-type thermal-abrasion flux h·(T_w − T_face). It is applied automatically on the seabed, the beach, the bluff face and the niche roof. It switches off when the neighbouring cell becomes air, and swaps to the ice–water value when the neighbour is sea ice. A_f is the open-area fraction of partially filled cells, in the FAVOR style [U: Hirt & Sicilian 1985].

---

## 4. Numerics

### 4.1 Grid

- **Structured and non-uniform.**
  - x runs from an offshore boundary at about 5–8 m depth, roughly 1–2 km out at Barter, to 100–200 m landward of the bluff edge.
  - dx: 0.25–0.5 m across the bluff face and beach, stretching to 5–10 m offshore and 2–5 m inland.
  - z: 30–50 m below MSL up to about 2 m above the bluff top.
  - dz: 0.05 m over the upper 2 m, stretching to 1–2 m at depth.
- **Size.** About 600 × 250 cells, with about 8 × 10⁴ active after the atmosphere cells above the land surface and water level are masked out.
- **Cut cells.** The bed and the water surface are cut cells with fractional fill. The sub-cell bed elevation is z_b = z_cell,bottom + φ_s Δz, which is what XBeach receives. For classifying faces as water–soil, a cell counts as "open" when φ_open ≥ 0.5 [S]. Heat capacity and latent content always use the true fractions.
- **Wet/dry switching.** Cells between the bed and the instantaneous water level (tide, surge, the v1.0 R2 runup band) are water. Cells above are air and inactive. When a water cell becomes air, its enthalpy is dropped from the budget and logged. When an air cell is flooded, it is initialised at the local water temperature.

### 4.2 Time integration: implicit enthalpy [V/U]

Backward Euler with the Voller & Swaminathan (1991) source-update iteration. Write H = c T − L_v f_i with f_i = θ_H₂O − θ_w, and iterate k = 0, 1, …:

  1. Solve (V c/Δt + K) T^{k+1} = V c/Δt · T^n + V L_v/Δt · (f_i^{k} − f_i^n) + b.
  2. Update the ice fraction from the local imbalance: f_i^{k+1} = clip₀^{θ_H₂O}( f_i^k − a_P (T_P^{k+1} − T_f)/(L_v V/Δt) ).
  3. Cells with 0 < f_i < θ_H₂O are set to T = T_f.
  4. Stop when max |Δf| < 10⁻⁶ and the relative energy residual is below 10⁻⁸. This usually takes 2–5 iterations. [V: Voller & Swaminathan 1991, Num. Heat Transfer B 19:175; U: iteration counts]

For the `smooth` and `saline` curves, use Newton on T with Jacobian c + L_v dθ_w/dT, a line search and |ΔT| ≤ 1 K per iteration. This is the ATS/PFLOTRAN practice [U].

- **Time step.** Unconditionally stable. For accuracy, keep the front from moving more than about one cell per step. Default 3600 s, with adaptive halving when any cell crosses its plateau by more than one cell per step.
- **Linear solve.** Sparse 5-point, symmetric positive definite without advection. On this desktop, `splu` factorises 10⁵ unknowns in 0.7 s and back-solves in 0.04 s [V: timed during this run].
- **Factor reuse.** Lag k and c by one step, so each step costs one factorisation plus 2–5 back-solves, about 0.9 s per hour. That is about 2 h of wall time per simulated year, or minutes if the factorisation is reused over several steps while the phase pattern is quasi-steady.
- **At 10⁶ unknowns** switch to CG with pyamg, or to pypardiso (neither was benchmarked). JAX is not worth it: its sparse direct solve is GPU-only [U].
- **Advection (version 2).** Explicit TVD substeps (van Leer, CFL ≤ 1), Strang-split around the implicit diffusion and phase step. The matrix then stays symmetric.

### 4.3 Energy bookkeeping

Log a global budget every step: ΔH_domain versus the sum of surface, geothermal, relaxation, advective and cell-removal fluxes. This catches remapping and wet/dry leaks, which v1.0's re-interpolation could not detect. It is also a strong GMD selling point.

---

## 5. Physics modules and parameter values

### 5.1 Surface energy balance, shared by water and land

Forcing is the same ERA5 set v1.0 already reads.

- **Shortwave.** ERA5 ssrd times (1 − albedo). Albedo: water 0.06–0.10, tundra 0.15–0.20, snow and ice 0.6–0.85 [U].
  - In water, use a two-band split. A fraction β ≈ 0.55 (the NIR band) is absorbed in the top cell. The remainder decays with Kd (Beer–Lambert) and **whatever reaches the bed goes into the top sediment cell** [S]. In 1–2 m of water with Kd ≈ 1 m⁻¹, 14–37% of the shortwave reaches the bed [V calculation].
  - Kd(PAR) defaults: calm 0.7 m⁻¹ (range 0.43–1.34, Stefansson Sound), storm 3 m⁻¹ (2–3 measured, above 3 unverified). Sources: [V: Dunton et al., cANIMIDA OCS Study MMS 2009-040]; Canadian Beaufort shelf 0.06–0.24 [V: Para et al. 2013]. The Jerlov-type two-band table in ROMS (Coastal 1–7: R ≈ 0.55–0.57, ζ₂ = 1.5–5.1 m) is an alternative [V: ROMS mod_scalars.F; Paulson & Simpson 1977].
  - On land, apply shortwave as a surface flux, as in v1.0, with the solar-angle factor kept.
- **Longwave.** ERA5 strd downward minus εσT_s⁴ with the **model's own surface temperature**, ε = 0.97 for water and 0.95–0.97 for soil [V: COARE 3.6]. Linearise implicitly: 4εσT³ goes on the diagonal.
- **Sensible and latent heat over water.** Recompute from T2m, dewpoint, U10 and the model SST with bulk coefficients c_H = c_E = 1.3 × 10⁻³ [V: Delft3D-FLOW heat model 5 default; D-Flow FM]. **Do not use ERA5 sshf/slhf over water**: they are computed with ERA5's own 0.25° SST, and coastal cells blend land and ice [V reasoning in agent report; S]. Over land, keep the v1.0 options (Man 2023 sensible heat, ERA5 latent heat) so v1.0 calibration carries over.
- **Sea ice.** In the resolved domain it emerges on its own (§3.1). The top water cells freeze, albedo switches on the ice fraction, and in water shallower than the ice thickness the ice grounds and couples the seabed to the atmosphere. That is the bottom-fast ice effect observed at Elson Lagoon [V: Overduin et al. 2012; Angelopoulos et al. 2019]. ERA5 sea-ice cover is still used for the XBeach gate and optionally for weighting offshore fluxes, Q = (1 − A)Q_ow + A Q_io [U].
- **Snow** is deferred, but needed. It is the prime suspect for the v1.0 cold bias in RC1 point 2 (see landscape report §0). A single-layer ERA5 snow-depth insulation, the CryoGrid simple scheme, drops into the same top-flux slot.

### 5.2 Water-column mixing and the large-scale ocean in a 2DV slice

"Large-scale circulation" in a cross-shore slice mostly means alongshore renewal of water, which a 2DV model cannot advect. It enters through two relaxation terms [S]:

1. **Offshore sponge.** Over the outer 10–20 cells, relax toward T_ref(z,t) (and S_ref) with τ_in ≈ 3 h (range 1–6 h). This is Marchesiello et al. (2001) scaled down from a 150 km, 1-day regional setting [V source; S scaling]. Delft3D's Thatcher–Harleman return-time boundary is the equivalent if flow is added later [V].
2. **Alongshore flushing** throughout the water domain: γ = 1/τ_flush, with τ_flush ≈ L_y/|v_alongshore|. For example, 1 km / 0.1 m s⁻¹ ≈ 3 h, or days in lagoons. It is a calibration knob. Turning it off gives a closed "bathtub", which is useful as a limiting case.

**Sources for T_ref.** CMEMS Arctic reanalysis products do not resolve water shallower than about 10 m:
- ARCTIC_MULTIYEAR_PHY_002_003 (TOPAZ4b), 12.5 km, from 1991;
- ARCTIC_ANALYSISFORECAST_PHY_002_001 (TOPAZ5), 6.25 km, from July 2021;
- GLORYS12 at 1/12° [V].

Build T_ref(z,t) as follows:
- the surface layer from satellite L4 SST (DMI, 0.05°) or ERA5 SST, as v1.0 does;
- the bottom layer from reanalysis at the nearest offshore point;
- calibrate against the **USGS 2011 Barter Island CTD** [V: Erikson et al. 2020, doi:10.5066/P9LGYO2Q].

That record shows **cold, saline bottom water** at 4.7 m off Arey Island: monthly means −1.1, 0.4 and 1.4 °C for July to September, with a maximum of 3.7 °C. The lagoon at 1.2 m is warm, with means of 5.4–5.8 °C and a maximum of 11.4 °C. A depth-uniform T_ref would be wrong [V]. ACE used GOFS 3.1 top-2 m T/S, clipped at ≥ 0 °C [V].

**Eddy diffusivity** (parameterised; no flow solve in version 1):

| Zone | K_x (m² s⁻¹) | K_z (m² s⁻¹) | Source |
|---|---|---|---|
| Surf zone | 1 (0.5–2.5) | 10⁻² (10⁻³–10⁻¹) | [V: Clark, Feddersen & Guza 2010; Spydell et al. 2007/2009] |
| Surf zone, from XBeach | h·(D_r/ρ)^{1/3} (nuhfac = 1) | same, or well mixed | [V: XBeach flow_timestep.F90; U: Battjes 1975] |
| Seaward, stratified | 0.1–1 | 10⁻⁴ (3 × 10⁻⁵–10⁻³) | [V: Hally-Rosendahl et al. 2014 lower bound; S] |
| Under ice | small | 10⁻⁵–10⁻⁴ | [U] |

Link K to the event state. Outside XBeach events, derive a cheap surf zone from ERA5 or storm Hs using the same Stockdon/Battjes logic `check_xbeach` already uses. During events, use XBeach D_r and H.

### 5.3 Bluff-face and seabed heat transfer (thermal abrasion)

- **Default.** Kobayashi et al. (1999) as used in ArcticBeach: h = α f_w C_w U_b / (1 + F√(0.5 f_w)), with α = 0.5, f_w from roughness 2.5 D₅₀ and U_b the near-bed velocity [V: Rolph et al. 2022]. The F regime expressions (Kobayashi et al. 1999, JCR 15(2), Eqs. 10–11) were **not** retrieved. Use the form already coded in v1.0 (l. 1596: E = 0.52 Re*^0.45 Pr^0.8) and check it against the paper.
  - U_b comes from XBeach u_rms plus infragravity swash during events, and from linear theory with ERA5 or storm Hs otherwise.
  - The flux is applied only on wet faces.
- **Magnitude check.** f_w = 0.01–0.05 with U_b = 0.3–1 m/s gives h of order 10³–10⁴ W m⁻² K⁻¹ [S]. The ice–ocean analogue ρc_p α_h u* with α_h ≈ 0.005–0.006 gives 250–1200 [V: McPhee 2008]. The v1.0 value of 500 is therefore at the low end.
  - Sanity check: 500 × 5 K = 2.5 kW m⁻², which melts ice-rich soil (0.7 ρ_i L ≈ 2.1 × 10⁸ J m⁻³) at about 1 m day⁻¹ [S]. At these values the face is effectively Dirichlet, as in ACE, and the retreat rate is limited by **how long water is in contact** and by removal of the thawed matrix, not by h.
- **Evidence that waves matter.** In flume tests, niche erosion rose by more than 100% from low to high waves, and 2 cm waves added more than 50% over still water. Higher ice content reduced niche deepening by 38% [V: Omonigbehin et al. 2025, JGR-ES, doi:10.1029/2025JF008528]. Barnhart et al. (2014) found that the White formulation matched Drew Point block melt, and that Kobayashi melted the block about 5 days early [V via summary].
- **Seabed.** The same h applies to seabed faces. Under grounded ice the face is ice–sediment conduction (R_bl = 0).

### 5.4 Removing thawed material, niches and block failure

Heat alone will not erode the bluff [V: Ravens et al. 2012; Barnhart et al. 2014]. Version 1 uses three rules [S]:

1. **Mechanical removal of thawed wet face cells between XBeach events.**
   - The trigger is a wet face cell with f_i = 0 (thawed) and wave exposure.
   - That cell's sediment is converted to water; the sediment volume goes to a debris budget, and the enthalpy removed is logged.
   - The debris is handed to XBeach at the next event as a bed change at the toe.
   - This is thermalFOAM's thaw-gated ablation [V] in fraction form rather than mesh deletion.
2. **Niche and block failure.**
   - A niche is the horizontal depth of open water cells under frozen overhang.
   - When it exceeds d_crit, collapse the column above. Options for d_crit:
     - a calibrated value, of the order of the niche depth observed before collapse at Drew Point;
     - Hoque & Pollard stability;
     - a tensile threshold of about 100 kPa [U; landscape report].
   - Move the block's cells down into the niche and the toe with **their enthalpy conserved**. The fallen block then thaws in place and is removed by rules 1 and 3, which is how Barnhart's "block degradation" phase arises.
3. **During XBeach events,** erosion is whatever XBeach removes above the non-erodible layer (§6).

### 5.5 Soil boundaries (as requested)

- **Base.** Neumann geothermal flux q_geo = 60 mW m⁻² (North Slope 66 ± 14 [V: Hornbach et al. 2020]; Prudhoe about 54 [V: Lachenbruch et al. 1982]). This equals about 0.02–0.035 K m⁻¹ for k = 1.7–3.4.
  - Put it at **30–50 m**, not 10 m. The annual skin depth in frozen silt is about 3.3 m; at 10 m about 5% of the annual amplitude remains; a 30-yr signal has a skin depth of about 18 m [S calculation]. The deeper base costs little on a stretched grid.
  - Do not reuse ACE's −0.23 K/m: it is a near-surface summer gradient, not a geothermal one [V caveat].
- **Landward.** Zero gradient, i.e. homogeneous Neumann, as in ACE [V]. Place it at least 100 m beyond the bluff edge so it does not constrain the bluff.
- **Offshore, below the seabed.** Zero flux. The subsea permafrost there is slow relict permafrost; initialise it from a spin-up.

### 5.6 Parameter table (defaults for the Barter Island config)

| Quantity | Default | Range | Source |
|---|---|---|---|
| L_f; ρ_i; ρ_w | 3.34e5 J/kg; 917; 1000 (sea 1025) kg/m³ | — | [V] preprint Table 1 |
| k_ice | 2.2 (2.07 at 0 °C, T-dependent: 9.828 exp(−0.0057 T[K])) | — | [V] Yen 1981 |
| k_water; k_seawater | 0.57; 0.56 | — | [U]; Angelopoulos 2019 uses 0.57 [V] |
| k_mineral (silt); k_quartz; k_organic; k_air | 2.9; 7.7; 0.25; 0.025 | 2.0–3.0 | [U] Johansen 1975; CryoGrid |
| C: water; ice; mineral; organic | 4.2e6; 1.9e6; 2.0e6; 2.5e6 J m⁻³ K⁻¹ | — | [U] CryoGrid constants |
| Ground ice, Beaufort bluffs | total volumetric ice about 77% in the upper 3 m; wedge ice about 11% (3–50%) | wedges up to 4 m deep, more than 5 m wide | [V] Kanevskiy et al. 2013 |
| Drew Point porewater S | < 3.5 above +3 m, rising to 36.9 at depth | — | [V] Bristol et al. 2021 |
| q_geo | 60 mW/m² | 52–80 | [V] Hornbach 2020; Lachenbruch 1982 |
| h (water → face) | Kobayashi formula; fallback 500 | 250–10⁴ | §5.3 |
| c_H = c_E over water | 1.3e-3 | 1.1–1.5e-3 | [V] Delft3D |
| ε_water | 0.97 | 0.97–0.985 | [V] COARE 3.6; Delft3D |
| Kd(PAR), calm / storm | 0.7 / 3 m⁻¹ | 0.4–1.4 / 2–10 | [V/U] cANIMIDA 2009 |
| NIR fraction absorbed in top cell | 0.55 | 0.4–0.6 | [V] ROMS Jerlov table |
| Surf-zone K_x / K_z | 1 / 1e-2 m²/s | see §5.2 | [V] |
| Sponge τ_in / width | 3 h / 15 cells | 1–6 h / 10–20 | [V source, S scaled] |
| Seawater T_f | −1.8 °C at 30–35 psu | — | [U] UNESCO 1983 |

---

## 6. Coupling with XBeach surfbeat

The event logic in `check_xbeach` (sea ice, runup threshold, weekly inter-storm calls, thaw present) is kept unchanged. Only the two handoffs change:

- **2DV → XBeach, before an event.**
  - `zb(x)` is the envelope of the sediment surface: the highest solid cut-cell top. Overhangs are invisible to XBeach, and rule 5.4.2 handles them.
  - `ne_layer(x)` is zb(x) minus the elevation of the first cell below the surface with f_i/θ_H₂O > 0.5. This is a plain vertical scan and replaces the column interpolation in `find_thaw_depth`.
  - On a steep face the vertical thaw depth is overestimated relative to the face-normal thaw depth, which v1.0 also did. XBeach's avalanching (`wetslp`/`dryslp`) sees the geometry, so keep the vertical definition and document it.
  - Water temperature near the bed is also exported as a diagnostic; XBeach does not use it.
- **XBeach → 2DV, after an event.**
  - Where Δzb(x) < 0, remove sediment fraction top-down from each column. Cells fully emptied become water or air; the enthalpy removed goes into the budget.
  - Where Δzb > 0, add sediment fraction at the local water temperature, thawed. This mirrors v1.0's `fill_value_top_water`.
  - During the event, K_t and h are taken from XBeach (H, u_rms, D_r).
- **XBeach on Linux.** v1.0 launches XBeach through `run.bat` with `shell=True`. Replace that with `subprocess.run([exe, "params.txt"], cwd=...)` so SLURM ensembles work. The landscape report lists this for v1.1 anyway.

---

## 7. Verification and validation ladder

Each step gates the next.

1. **Regression against v1.0.** A 1-column 2DV run (nx = 1, no water) should reproduce `examples/analytical/01–04` and the v1.0 Barter borehole series.
2. **Phase change.**
   - Neumann two-phase Stefan solution, including ice-content scaling of L.
   - Lunardini three-zone solution for the smooth curve. SUTRA-ICE reached 0.01–0.1 °C error on a 1 cm grid [V: McKenzie et al. 2007].
3. **2D.**
   - Rathjen & Jiji (1971) corner freezing similarity solution [V].
   - InterFrost TH2 (frozen inclusion) and TH3 (talik) in a conduction-only variant. Grenier et al. 2018, AWR 114:196 [V]; inter-code spread is large, so do not use tight tolerances [V].
4. **Water column.**
   - Energy closure under pure Beer–Lambert heating.
   - Relaxation-zone step response.
   - Comparison with the 2011 USGS Barter CTD (bottom water at 4.7 m, lagoon at 1.2 m).
5. **Bluff melt.**
   - Lumped check against ArcticBeach: retreat ≈ h(T_w − T_m)/(L_c·height) when conduction into the block is small [V formula: Rolph et al. 2022].
   - Omonigbehin et al. (2025) flume niche series (scale down the grid). Doing this also answers RC1 point 7.
6. **Field.**
   - Barter Island borehole temperatures and 2–3 yr shoreline retreat, run against v1.0.
   - Drew Point intercomparison with ACE (niche and block cycle, Bayat et al. 2026 calibration case on Zenodo [V]).

---

## 8. Implementation plan

Keep v1.0 intact for the GMD revision. Add a switch `thermal.model: "1d" | "2dv"`, so the `Simulation` API and `main.py` loop stay the same.

```
arctic_xbeach/
  thermal2dv/
    grid.py        # stretched x–z mesh, masks, cut-cell fractions, face areas/distances
    materials.py   # constituent constants, mixing rules, freezing curves, H<->T, dθw/dT
    operator.py    # face conductances (incl. R_bl=1/h), sparse assembly, BC terms
    forcing.py     # SEB (land + water), SW two-band penetration, T_ref(z,t) builder, K_t zones, h(U_b)
    solver.py      # backward Euler + Voller–Swaminathan / Newton, splu reuse, adaptive dt, energy budget
    erosion.py     # thawed-cell removal, niche detection, block failure, debris budget
    coupling.py    # zb/ne_layer export, sedero -> fractions, XBeach-derived K_t/h
    output.py      # xarray/NetCDF (x, z, time): T, H, θ_i, θ_w, masks, fluxes, budget
  tests/           # pytest: every benchmark in §7 as a test with a tolerance
```

| Phase | Content | Exit criterion | Effort |
|---|---|---|---|
| 0 | Grid, materials, implicit conduction without phase change; 1-column regression | matches v1.0 analytical examples 01–03 | 1 wk |
| 1 | Enthalpy iteration (free and smooth curves); geothermal and landward boundaries; stretched 50 m base | Stefan, Lunardini, Rathjen–Jiji pass; energy residual below 10⁻⁸ | 2 wk |
| 2 | Water cells: K_t zones, sponge and flushing, water SEB with SW penetration, wet/dry switching | energy closure; 2011 CTD summer bottom and lagoon T within about 1 °C | 2 wk |
| 3 | Interface R_bl with Kobayashi h; thawed-cell removal; niche and block failure | ArcticBeach lumped rate reproduced; flume niche trend | 2 wk |
| 4 | XBeach coupling (zb, ne_layer, sedero into fractions); cross-platform launch; Barter run | Barter borehole RMSE ≤ v1.0; plausible retreat | 2–3 wk |
| 5 (later) | Saline curve and salt transport with brine rejection; snow; advection from XBeach velocities; pyamg; ensembles | — | — |

**What to simplify in the first version.**
- No flow solve: turbulent diffusion plus relaxation only.
- No salt transport: static S field, saline curve optional.
- No snow.
- Fixed grid, no adaptive refinement.
- Block failure by a calibrated critical niche depth.
- One Kd per event state.
- Land SEB carried over from v1.0 for continuity of calibration.

---

## 9. Key risks

1. **Field-scale h controls everything at the face.** Plausible values span 250 to more than 10⁴ W m⁻² K⁻¹, and lab-to-field transfer is untested. Above a few hundred, retreat is limited by contact time and matrix removal, so put effort into the wet/dry timing and removal rules as much as into h. Make h the first sensitivity parameter.
2. **Alongshore physics in a 2DV slice.** Nearshore water temperature at Barter is set by lagoon outflow, alongshore currents and stratification. τ_flush and T_ref absorb all of it. Mitigate by forcing the calm-weather T_ref from observations (CTD, SST) and reporting τ_flush sensitivity. Note in the paper that this is a slice model.
3. **Overhang versus a single-valued XBeach bed.** Niche geometry exists only in the thermal model, so block failure is a parameterised hand-off, not emergent. This is the same limitation as ACE and Barnhart, but say so.
4. **No nearshore ocean data shallower than 10 m in the reanalyses.** T_ref construction becomes a site-specific calibration, which works against transferability (RC1 point 5).
5. **Energy leaks at wet/dry switching and cell removal.** The global budget in §4.3 is the mitigation. Make it a test.
6. **Solver cost is not the risk:** about 0.5–2 h per simulated year serial, trivially parallel over SLURM ensemble members. The risk is Python-side overhead in assembly. Vectorise assembly from the start; use numba only if profiling says so.
7. **Scope versus the GMD revision.** Do not block the v1.0 revision on this. The snow layer, the revived Kobayashi h and the saline freezing point (landscape report §2d) help both versions and should come first.

---

## 10. Sources (verified during this run unless marked)

- Westermann et al. 2023, CryoGrid community model v1.0, GMD 16:2607, https://gmd.copernicus.org/articles/16/2607/2023/
- McKenzie, Voss & Siegel 2007, SUTRA-ICE, Adv. Water Resour. 30:966
- Jan, Coon & Painter 2020, ATS v0.88, GMD 13:2259
- Painter & Karra 2014, Vadose Zone J. 13(4), doi:10.2136/vzj2013.04.0071
- Grenier et al. 2018, InterFrost, Adv. Water Resour. 114:196, doi:10.1016/j.advwatres.2018.02.001
- Barnhart et al. 2014, JGR-ES, doi:10.1002/2013JF002845
- Ravens et al. 2012, JWPCOE (USGS pub record)
- Rolph et al. 2022, ArcticBeach v1.0, Front. Earth Sci., doi:10.3389/feart.2022.962208
- Bayat et al. 2026 (ACE), JAMES, doi:10.1029/2025MS005045; Frederick et al. 2021
- Omonigbehin et al. 2025, JGR-ES, doi:10.1029/2025JF008528; thermalFOAM v1.0, SSRN doi:10.2139/ssrn.6642471 (preprint)
- Voller & Swaminathan 1991, Num. Heat Transfer B 19:175, doi:10.1080/10407799108944962
- Rathjen & Jiji 1971, J. Heat Transfer 93(1):101
- Hornbach et al. 2020, G³, doi:10.1029/2020GC008933; Lachenbruch et al. 1982, JGR doi:10.1029/JB087iB11p09301 and USGS OFR 82-1039
- Kanevskiy et al. 2013, CRST 85:56, doi:10.1016/j.coldregions.2012.08.002
- Bristol et al. 2021, Front. Earth Sci., doi:10.3389/feart.2020.598933
- Marchesiello, McWilliams & Shchepetkin 2001, Ocean Modelling 3:1, doi:10.1016/S1463-5003(00)00013-5
- Delft3D-FLOW and D-Flow FM user manuals (heat-flux model 5, Secchi depth, Thatcher–Harleman boundary, ice module)
- Paulson & Simpson 1977, JPO 7:952; ROMS mod_scalars.F (Jerlov table)
- Dunton et al. 2009, cANIMIDA, OCS Study MMS 2009-040; Para et al. 2013, Biogeosciences 10:2761
- Fairall et al. 2003 and Edson et al. 2013 (COARE 3.x)
- Clark, Feddersen & Guza 2010, JGR doi:10.1029/2009JC005683; Spydell et al. 2007, JPO 37:2920; Hally-Rosendahl et al. 2014, doi:10.1002/2013JC009722
- Erikson et al. 2020, USGS data release doi:10.5066/P9LGYO2Q (Barter Island 2011 CTD)
- Overduin et al. 2012, doi:10.1029/2011JF002088; Angelopoulos et al. 2019, JGR-ES doi:10.1029/2018JF004823
- McPhee 2008, Air–Ice–Ocean Interaction, Springer
- [U] Kobayashi et al. 1999, JCR 15(2):332; Johansen 1975; Farouki 1981; Côté & Konrad 2005; UNESCO 1983; Hirt & Sicilian 1985; Battjes 1975; Anderson & Tice 1972

**Refuted during verification (do not cite):**
- that CryoGrid's documented setup is a constant geothermal flux plus a full SEB, as worded;
- that the InterFrost codes agreed tightly on TH2/TH3;
- that thermalFOAM does not resolve the water column.

**Coverage caveat.** The first research pass verified the soil and bluff side at 3-0 votes. The ocean-side and numerics values come from two targeted follow-up agents that read primary texts but were not put through three-vote verification. Treat §5.1–5.2 numbers as [V]-by-one-reader.
