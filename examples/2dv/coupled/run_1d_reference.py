"""The v1.0 (fixed) 1D thermal module on the same case and window as run_coupled_test.py, as
the reference for the coupled 2DV run (bed change and thaw depth)."""
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parents[2]
CASE_DIR = SCRIPT_DIR / "barter_1d_test"
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from arctic_xbeach.model import Simulation  # noqa: E402
from main import main  # noqa: E402

sim = Simulation(CASE_DIR.relative_to(PROJ_DIR).as_posix(), proj_dir=PROJ_DIR)
t0 = time.perf_counter()
main(sim, print_to_screen=False)
print(f"main(): {time.perf_counter() - t0:.0f} s wall")
