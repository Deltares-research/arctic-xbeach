"""Structured, stretched cross-shore (x) by vertical (z) finite-volume grid.

Arrays on the grid are shaped (nz, nx): row j = 0 is the bottom of the domain, column
i = 0 the offshore boundary. Cell volumes are per metre alongshore.
"""

from dataclasses import dataclass

import numpy as np


def stretched_edges(start, end, fine_start, fine_end, d_fine, d_max, growth=1.1):
    """Cell edges with uniform spacing d_fine inside [fine_start, fine_end], growing
    geometrically (ratio `growth`, capped at d_max) towards start and end."""
    fine_start = max(fine_start, start)
    fine_end = min(fine_end, end)
    n_fine = max(1, int(round((fine_end - fine_start) / d_fine)))
    edges = list(np.linspace(fine_start, fine_start + n_fine * d_fine, n_fine + 1))

    # grow towards `end`
    d = d_fine
    while edges[-1] < end - 1e-9:
        d = min(d * growth, d_max)
        edges.append(min(edges[-1] + d, end))
    if len(edges) > 2 and edges[-1] - edges[-2] < 0.3 * (edges[-2] - edges[-3]):
        edges.pop(-2)  # merge a sliver cell into its neighbour

    # grow towards `start`
    d = d_fine
    while edges[0] > start + 1e-9:
        d = min(d * growth, d_max)
        edges.insert(0, max(edges[0] - d, start))
    if len(edges) > 2 and edges[1] - edges[0] < 0.3 * (edges[2] - edges[1]):
        edges.pop(1)

    return np.array(edges)


@dataclass
class Grid:
    xe: np.ndarray  # cell edges in x [m], length nx + 1
    ze: np.ndarray  # cell edges in z [m], length nz + 1

    @property
    def xc(self):
        return 0.5 * (self.xe[1:] + self.xe[:-1])

    @property
    def zc(self):
        return 0.5 * (self.ze[1:] + self.ze[:-1])

    @property
    def dx(self):
        return np.diff(self.xe)

    @property
    def dz(self):
        return np.diff(self.ze)

    @property
    def nx(self):
        return len(self.xe) - 1

    @property
    def nz(self):
        return len(self.ze) - 1

    @property
    def shape(self):
        return (self.nz, self.nx)

    @property
    def volume(self):
        return self.dz[:, None] * self.dx[None, :]

    def meshgrid(self):
        """Cell-centre coordinates as (nz, nx) arrays."""
        return np.meshgrid(self.xc, self.zc)

    def below_profile(self, x_prof, z_prof):
        """Mask of cells whose centre lies below a bed profile z_prof(x_prof) (staircase bed)."""
        zb = np.interp(self.xc, x_prof, z_prof)
        return self.zc[:, None] < zb[None, :]
