"""Plot helpers for 2DV snapshots (house style: no titles, larger fonts, fixed colour limits)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from .output import CLASS_NAMES

PHASE_COLORS = ["#ffffff", "#2b83ba", "#bfe7f7", "#5b3a1e", "#c49a6c", "#e8d3a9"]
PHASE_CMAP = ListedColormap(PHASE_COLORS)
PHASE_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), PHASE_CMAP.N)


def apply_style():
    mpl.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.bbox": "tight",
    })


def panel_label(ax, text):
    ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))


def plot_phase(ax, ds, it, rasterized=True):
    m = ax.pcolormesh(ds.x_edge, ds.z_edge, ds.phase.isel(time=it), cmap=PHASE_CMAP, norm=PHASE_NORM,
                      rasterized=rasterized)
    return m


def phase_legend(ax, loc="lower left", classes=range(6), ncol=3):
    handles = [Patch(facecolor=PHASE_COLORS[c], edgecolor="0.5", label=CLASS_NAMES[c]) for c in classes]
    ax.legend(handles=handles, loc=loc, ncol=ncol, frameon=True, framealpha=0.9)


def plot_temperature(ax, ds, it, vlim=10.0, rasterized=True, contour=True):
    T = ds.T.isel(time=it)
    m = ax.pcolormesh(ds.x_edge, ds.z_edge, T, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -vlim, vlim),
                      rasterized=rasterized)
    if contour and np.nanmin(T) < 0 < np.nanmax(T):
        ax.contour(ds.x, ds.z, T, levels=[0.0], colors="k", linewidths=0.8)
    return m


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
