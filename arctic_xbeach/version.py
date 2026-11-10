"""
Version of Arctic-XBeach and the v1.1 solver options in effect for a run, for the run.log header and results.nc.

The three solver options are set in the config and default to their legacy (v1.0) values, so the version number alone
does not say which solver ran; run_header() reports each option with its value.
"""
import subprocess
from pathlib import Path

__version__ = "1.1.0"

# (config section, option, legacy value = default, v1.1 value, what the option does)
V11_OPTIONS = (
    ("thermal", "conduction_form",       "nodal",    "flux",       "conservative flux-form conduction"),
    ("thermal", "latent_heat_form",      "per_mass", "volumetric", "latent heat rho_w L nb per unit volume"),
    ("data",    "era5_flux_shift_hours", 0,          1,            "ERA5 end-of-hour fluxes shifted to the start of the averaging hour"),
)


def git_revision():
    """Short hash and branch of the checkout that is imported, with '+uncommitted' when the working tree has changes."""
    repo = Path(__file__).resolve().parents[1]
    try:
        run = lambda *a: subprocess.check_output(["git", *a], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = "+uncommitted" if run("status", "--porcelain", "--untracked-files=no") else ""
        return f"git {run('rev-parse', '--short', 'HEAD')}{dirty} ({run('rev-parse', '--abbrev-ref', 'HEAD')})"
    except Exception:
        return "git revision unknown"


def option_states(config):
    """[(option, value, is_v11, description)] for the v1.1 options, read from the run config."""
    states = []
    for section, option, legacy, v11, what in V11_OPTIONS:
        value = getattr(config, section).get(option, legacy)
        states.append((option, value, value == v11, what))
    return states


def options_summary(config):
    """One line, e.g. 'all v1.1 options on' or 'legacy: conduction_form=nodal'."""
    legacy = [f"{o}={v}" for o, v, on, _ in option_states(config) if not on]
    return "all v1.1 options on" if not legacy else "legacy: " + ", ".join(legacy)


def run_header(config):
    """Lines for the top of run.log."""
    lines = [f"Arctic-XBeach {__version__} | {git_revision()}", f"Solver: {options_summary(config)}"]
    for option, value, on, what in option_states(config):
        lines.append(f"  {option:<22} = {str(value):<10} {'v1.1' if on else 'legacy'}: {what}")
    return lines
