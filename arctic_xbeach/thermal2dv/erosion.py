"""Erosion rules for the 2DV thermal model: fraction changes on a fixed grid.

1. Thawed, ice-rich soil touching water loses its matrix: the cell becomes sea water. This
   is thaw-gated ablation (thermalFOAM) written as a fraction change instead of mesh
   deletion. Non-ice-rich sediment (beach, seabed) is left to the morphodynamic model.
2. Once the niche under a frozen overhang is deeper than d_crit, the overhang drops into
   the niche as a block, keeping its enthalpy (Barnhart et al. 2014 block topple; the
   fallen block then degrades through rule 1).
"""

import numpy as np

from . import materials as mat


def remove_thawed_face_cells(model):
    """Convert fully thawed, ice-rich soil cells with a water neighbour into water cells."""
    s, g = model.s, model.grid
    p = model.state()
    fl = p.fluid
    wet_neighbour = np.zeros(g.shape, dtype=bool)
    wet_neighbour[:, 1:] |= fl[:, :-1]
    wet_neighbour[:, :-1] |= fl[:, 1:]
    wet_neighbour[1:, :] |= fl[:-1, :]
    wet_neighbour[:-1, :] |= fl[1:, :]

    remove = p.soil & wet_neighbour & (model.H >= p.H_liq) & (model.phi_h2o >= s.erodible_ice_threshold)
    if not np.any(remove):
        return 0

    V = g.volume
    e_before = model.total_energy()
    salt_before = model.total_salt()
    T_melt = p.Tf[remove]
    # heat absorbed after the cell had fully thawed (it thaws part-way through a time step) is
    # passed on to the solid behind the new face, as a sub-step front tracking would do
    excess = V[remove] * (model.H[remove] - p.H_liq[remove])
    model.eroded_sediment += float(np.sum(V[remove] * model.phi_s[remove]))
    model.phi_s[remove] = 0.0
    model.phi_h2o[remove] = 1.0
    model.salt[remove] = s.S_sea  # the cell fills with sea water (phi_h2o = 1)
    Tf_sea = mat.freezing_point(s.S_sea)
    model.H[remove] = mat.C_WATER * np.maximum(T_melt, Tf_sea)  # new water cell at the soil's melting point

    keep_soil = p.soil & ~remove
    wet = fl & ~remove
    for (j, i), e in zip(zip(*np.nonzero(remove)), excess):
        nbs = [(jj, ii) for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1))
               if 0 <= jj < g.nz and 0 <= ii < g.nx]
        targets = [t for t in nbs if keep_soil[t]] or [(j, i)]
        w = np.array([V[t] for t in targets])
        for t, wt in zip(targets, w / w.sum()):
            model.H[t] += e * wt / V[t]
        # the melt water is stirred into the adjacent turbulent water at once (mixing over a
        # cell takes seconds). Leaving it at the melting point would make the next face
        # non-melting for one step and slow the retreat by ~15%.
        mix = [(j, i)] + [t for t in nbs if wet[t]]
        vm = np.array([V[t] for t in mix])
        for arr in (model.H, model.salt):
            mean = np.sum(vm * np.array([arr[t] for t in mix])) / vm.sum()
            for t in mix:
                arr[t] = mean

    model.energy["erosion"] += model.total_energy() - e_before
    model.salt_exchange["erosion"] += model.total_salt() - salt_before
    return int(remove.sum())


def mix_meltwater(model, fluid_before):
    """Hand on the heat of ice cells that melted during the last step, then stir the melt water
    into the water it touches.

    1. A clamped face delivers h (T_w - T_f) for the whole step, so an ice cell that melts
       part-way through the step ends up warmer than its melting point. That surplus belongs
       to the ice behind it (a sub-step front tracking would have moved the face on), so it is
       passed to the adjacent cells that are still solid. Without this the surplus returned to
       the water and pure-ice faces retreated ~8% too slowly.
    2. The melt water, now at its melting point, is averaged (heat and salt, by volume) with its
       fluid neighbours: mixing over one cell takes seconds in turbulent water, and water left at
       the melting point would leave the next face unclamped for a step.
    Returns the number of melted cells."""
    g = model.grid
    p = model.state()
    new = p.fluid & ~fluid_before
    if not np.any(new):
        return 0
    V = g.volume
    solid = p.active & ~p.fluid
    for j, i in zip(*np.nonzero(new)):
        nbs = [(jj, ii) for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1))
               if 0 <= jj < g.nz and 0 <= ii < g.nx]
        behind = [t for t in nbs if solid[t]]
        surplus = V[j, i] * (model.H[j, i] - p.H_liq[j, i])
        if behind and surplus > 0.0:
            w = np.array([V[t] for t in behind])
            for t, wt in zip(behind, w / w.sum()):
                model.H[t] += surplus * wt / V[t]
            model.H[j, i] = p.H_liq[j, i]
        mix = [(j, i)] + [t for t in nbs if fluid_before[t]]
        if len(mix) == 1:
            continue
        vm = np.array([V[t] for t in mix])
        for arr in (model.H, model.salt):
            mean = np.sum(vm * np.array([arr[t] for t in mix])) / vm.sum()
            for t in mix:
                arr[t] = mean
    return int(new.sum())


def solid_mask(model):
    """Cells that can form an overhang: soil, plus ice if Settings.collapse_ice is on. Ice is
    opt-in because floating sea ice over water would otherwise count as one long overhang."""
    solid = model.phi_s >= model.s.open_threshold
    if model.s.collapse_ice:
        p = model.state()
        solid |= p.open & ~p.fluid
    return solid


def undercut_columns(model):
    """Columns with an open gap (water or air; ice too unless it is itself solid) under a solid
    overhang, and the largest niche depth among the connected niches (see undercut_components).

    Returns a list of (i, j_gap_bottom, j_gap_top) for all undercut columns and that depth."""
    comps = undercut_components(model)
    found = [c for cols, _ in comps for c in cols]
    return found, max((d for _, d in comps), default=0.0)


def undercut_components(model):
    """Connected niches: runs of neighbouring undercut columns whose gaps overlap in z. Each
    niche is returned as (columns, depth), with columns a list of (i, j_gap_bottom, j_gap_top)
    and depth the summed width of its columns. Separate notches are separate niches, so two
    subcritical notches never add up to a failure."""
    g = model.grid
    soil = solid_mask(model)
    found = []
    for i in range(g.nx):
        js = np.nonzero(soil[:, i])[0]
        if len(js) == 0:
            continue
        j_top = js.max()
        gaps = np.nonzero(~soil[:j_top, i])[0]
        if len(gaps) == 0:
            continue
        j_gap_top = gaps.max()
        j_gap_bot = j_gap_top
        while j_gap_bot > 0 and not soil[j_gap_bot - 1, i]:
            j_gap_bot -= 1
        found.append((i, j_gap_bot, j_gap_top))
    comps = []
    for c in found:
        if comps:
            i0, jb0, jt0 = comps[-1][-1]
            if c[0] == i0 + 1 and c[1] <= jt0 and jb0 <= c[2]:  # adjacent column, gaps overlap
                comps[-1].append(c)
                continue
        comps.append([c])
    return [(cols, float(sum(g.dx[i] for i, _, _ in cols))) for cols in comps]


def _drop_column(model, i, jb, jt, j_top):
    """Lower the solid stack jt+1..j_top of column i by the gap height onto ze[jb].

    Extensive contents (V phi_s, V phi_h2o, V salt, V H) are remapped by geometric overlap,
    so mineral, H2O, salt and energy are conserved on a stretched grid; the stack's top cell
    may end up partly filled (the rest air). The gap's own contents (niche water or air) are
    displaced and returned as (energy, salt) leaving the domain."""
    g = model.grid
    ze = g.ze
    V = g.volume[:, i]
    arrays = (model.phi_s, model.phi_h2o, model.salt, model.H)
    gap = slice(jb, jt + 1)
    displaced = (float(np.sum(V[gap] * model.H[gap, i])), float(np.sum(V[gap] * model.salt[gap, i])))
    G = ze[jt + 1] - ze[jb]
    rows = np.arange(jb, j_top + 1)
    new = [np.zeros(len(rows)) for _ in arrays]
    for k in range(jt + 1, j_top + 1):
        a, b = ze[k] - G, ze[k + 1] - G
        ov = np.maximum(np.minimum(b, ze[rows + 1]) - np.maximum(a, ze[rows]), 0.0) / (b - a)
        for n, arr in zip(new, arrays):
            n += ov * V[k] * arr[k, i]
    for n, arr in zip(new, arrays):
        arr[rows, i] = n / V[rows]
    tiny = (model.phi_s[rows, i] + model.phi_h2o[rows, i]) < 1e-9
    for arr in arrays:
        arr[rows[tiny], i] = 0.0
    return displaced


def collapse_overhang(model):
    """Drop each overhang into its niche once that niche is deeper than d_crit. The block
    keeps its mineral, H2O, salt and enthalpy; the niche water it displaces leaves the domain
    and is the only exchange logged under "collapse"."""
    s = model.s
    comps = [(cols, d) for cols, d in undercut_components(model) if d >= s.d_crit]
    if not comps:
        return False
    solid = solid_mask(model)
    for cols, depth in comps:
        e_out = salt_out = 0.0
        for i, jb, jt in cols:
            j_top = np.nonzero(solid[:, i])[0].max()
            e, sl = _drop_column(model, i, jb, jt, j_top)
            e_out += e
            salt_out += sl
        model.energy["collapse"] -= e_out
        model.salt_exchange["collapse"] -= salt_out
        model.collapse_events.append(dict(time=model.time, niche_depth=depth, columns=[c[0] for c in cols]))
    return True
