"""A repeated key in config.yaml is an error: PyYAML would silently keep the last one."""
import subprocess
from pathlib import Path

import pytest

from arctic_xbeach.config_loader import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_normal_config_loads():
    cfg = load_config("model:\n  timestep: 1\nthermal:\n  dt: 60\n  grid_resolution: 200\n")
    assert cfg == {"model": {"timestep": 1}, "thermal": {"dt": 60, "grid_resolution": 200}}


def test_same_key_in_different_sections_is_fine():
    cfg = load_config("a:\n  dt: 1\nb:\n  dt: 2\n")
    assert cfg["a"]["dt"] == 1 and cfg["b"]["dt"] == 2


def test_duplicate_top_level_key_raises():
    text = "thermal:\n  dt: 60\nmodel:\n  timestep: 1\nthermal:\n  dt: 360\n"
    with pytest.raises(ValueError, match=r"Duplicate key 'thermal'.*line 5.*line 1"):
        load_config(text)


def test_duplicate_nested_key_raises():
    text = "thermal:\n  dt: 60\n  conduction_form: flux\n  dt: 360\n"
    with pytest.raises(ValueError, match=r"Duplicate key 'dt'.*line 4"):
        load_config(text)


def test_load_config_reads_a_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\na: 2\n")
    with pytest.raises(ValueError, match="Duplicate key 'a'"):
        load_config(path)


def _tracked_example_configs() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "examples/**/config.yaml"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [ROOT / line for line in out.splitlines() if line]


def test_tracked_example_configs_have_no_duplicate_keys():
    configs = _tracked_example_configs()
    if not configs:
        pytest.skip("git or tracked example configs not available")
    for path in configs:
        assert isinstance(load_config(path), dict), path
