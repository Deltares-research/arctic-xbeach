"""2DV (cross-shore x vertical) water-soil-ice thermal model for Arctic-XBeach.

First, standalone version on branch 2DV-redesign: it is not yet coupled to XBeach or to
main.py. See docs/2dv_design.md for the design and examples/2dv/ for the demonstrations.
"""

from .forcing import Forcing, constant_forcing, synthetic_arctic_forcing
from .grid import Grid, stretched_edges
from .output import Recorder, phase_class
from .solver import Model2DV, Settings


def run(model, forcing_fn, t_end, dt, recorder=None, progress_every=None):
    """Step `model` to t_end [s] with a fixed outer time step dt [s]."""
    if recorder is not None:
        recorder(force=True)
    n = 0
    while model.time < t_end - 1e-6:
        model.step(min(dt, t_end - model.time), forcing_fn)
        n += 1
        if recorder is not None:
            recorder()
        if progress_every and n % progress_every == 0:
            print(f"  t = {model.time / 86400:7.2f} d, iterations = {model.last_step.get('iterations')}")
    if recorder is not None:
        recorder(force=True)
    return model


__all__ = ["Forcing", "constant_forcing", "synthetic_arctic_forcing", "Grid", "stretched_edges", "Recorder",
           "phase_class", "Model2DV", "Settings", "run"]
