"""Simple example of running Arctic-XBeach simulation."""
from pathlib import Path
import sys

# Add project root to path
proj_dir = Path(__file__).parent.parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))

from arctic_xbeach.model import Simulation
from main import main

# Configure and run simulation
case_study_path = Path(__file__).parent / "analytical" / "01_dirichlet_warming"
sim = Simulation(case_study_path, proj_dir=proj_dir)
main(sim)

