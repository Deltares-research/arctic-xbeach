"""Simple example of running Arctic-XBeach simulation."""
" => should finish within half a minute up to several minutes depending on CPU"

# Add project root to path
from pathlib import Path
import sys
proj_dir = Path(__file__).parent.parent.resolve()
if str(proj_dir) not in sys.path:
    sys.path.insert(0, str(proj_dir))
from arctic_xbeach.model import Simulation
from main import main

# Configure and run simulation
case_study_path = Path(__file__).parent / "simple"
sim = Simulation(case_study_path, proj_dir=proj_dir)

# More log information
import logging
logging.getLogger("thermo_model").setLevel(logging.DEBUG)

# Run simulation
main(sim)