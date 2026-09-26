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

    remove = p.soil & wet_neighbour & (model.H >= p.lv) & (model.phi_h2o >= s.erodible_ice_threshold)
    if not np.any(remove):
        return 0

    V = g.volume
    e_before = model.total_energy()
    T_melt = p.Tf[remove]
    # heat absorbed after the cell had fully thawed (it thaws part-way through a time step) is
    # passed on to the solid behind the new face, as a sub-step front tracking would do
    excess = V[remove] * (model.H[remove] - p.lv[remove])
    model.eroded_sediment += float(np.sum(V[remove] * model.phi_s[remove]))
    model.phi_s[remove] = 0.0
    model.phi_h2o[remove] = 1.0
    model.S[remove] = s.S_sea
    Tf_sea = mat.freezing_point(s.S_sea)
    _, cu, lv = mat.heat_capacities(0.0, 1.0)
    model.H[remove] = lv + cu * (np.maximum(T_melt, Tf_sea) - Tf_sea)  # new water cell at the soil's melting point

    keep_soil = p.soil & ~remove
    for (j, i), e in zip(zip(*np.nonzero(remove)), excess):
        nbs = [(jj, ii) for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1))
               if 0 <= jj < g.nz and 0 <= ii < g.nx and keep_soil[jj, ii]]
        targets = nbs or [(j, i)]
        w = np.array([V[t] for t in targets])
        for t, wt in zip(targets, w / w.sum()):
            model.H[t] += e * wt / V[t]

    model.energy["erosion"] += model.total_energy() - e_before
    return int(remove.sum())


def undercut_columns(model):
    """Columns with an open gap (water, ice or air) under soil, and the extent of that gap.

    Returns a list of (i, j_gap_bottom, j_gap_top) and the summed width of those columns,
    i.e. the niche depth."""
    s, g = model.s, model.grid
    soil = model.phi_s >= s.open_threshold
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
    depth = float(sum(g.dx[i] for i, _, _ in found))
    return found, depth


def collapse_overhang(model):
    """Drop every overhang into its niche once the niche depth exceeds d_crit."""
    s, g = model.s, model.grid
    cols, depth = undercut_columns(model)
    if depth < s.d_crit or not cols:
        return False

    V = g.volume
    e_before = model.total_energy()
    for i, jb, jt in cols:
        n = jt - jb + 1
        j_top = np.nonzero(model.phi_s[:, i] >= s.open_threshold)[0].max()
        for arr in (model.phi_s, model.phi_h2o, model.S, model.H):
            arr[jb:j_top + 1 - n, i] = arr[jt + 1:j_top + 1, i]
            arr[j_top + 1 - n:j_top + 1, i] = 0.0  # the space vacated at the top becomes air
    model.energy["collapse"] += model.total_energy() - e_before
    model.collapse_events.append(dict(time=model.time, niche_depth=depth, columns=[c[0] for c in cols]))
    return True
