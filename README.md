# Arctic-XBeach
Coupled thermo-morphodynamic model for Arctic permafrost coastlines, combining XBeach hydrodynamics with an enthalpy-based thermal erosion module

## Overview
Arctic-XBeach couples the XBeach morphodynamic model with an enthalpy-based thermal erosion module to simulate coastal retreat processes in permafrost environments. The model uses an innovative event-driven approach that executes erosion calculations only during storms when thawed sediment is available, improving computational efficiency while maintaining physical realism.

## Key Features
- **Coupled approach**: Integrates XBeach hydrodynamics with permafrost thaw processes
- **Enthalpy-based thermal module**: Robust handling of phase change in frozen sediments
- **Event-driven methodology**: Storm-focused calculations optimize computational performance
- **Arctic-specific**: Designed for permafrost coastlines experiencing rapid climate change

## Status
🚧 **In Development** - This model is currently under active development. A manuscript describing the model formulation and validation is in preparation for submission to Geoscientific Model Development (GMD).

## Case Study
The model has been developed and tested for Barter Island, Alaska, leveraging well-documented permafrost characteristics and observational datasets for validation.

## Installation

### Step 1: Create environment and clone repository

```bash
# Create conda environment with Python and git
conda create -n arctic-xbeach python=3.12 git -y
conda activate arctic-xbeach

# Enable long paths (Windows only, run once)
git config --global core.longpaths true

# Clone repository
git clone https://github.com/deltares-research/arctic-xbeach.git
cd arctic-xbeach
```

### Step 2: Install Python dependencies

```bash
# Install package in editable mode
pip install -e .

# Install all dependencies
pip install -r requirements.txt

```

### Step 3: Download XBeach executable
XBeach must be installed separately. Download from the official repository: https://download.deltares.nl/xbeach


**Installation:**
1. Download the appropriate version for your system
2. Extract to a location of your choice
3. Note the path to the executable:
4. Update your run configuration (`config.yaml`) with this path:
   ```yaml
   xbeach:
     version: "C:/software/XBeach/xbeach.exe"  # Use forward slashes
   ```

### Step 4: Download case study data
```bash
# From the arctic-xbeach directory
python download_data.py
```

This downloads the Barter Island forcing data (~ERA5 and storm datasets) to:
```
examples/case_studies/barter_island/database/
├── era5.csv      # ERA5 reanalysis forcing data
└── storms.csv    # Storm/wave conditions
```

**Manual download** (if script fails):
- [era5.csv](https://deltares-usa-software.s3.us-east-1.amazonaws.com/arctic_xbeach/database_barter_island/era5.csv)
- [storms.csv](https://deltares-usa-software.s3.us-east-1.amazonaws.com/arctic_xbeach/database_barter_island/storms.csv)

Place files in: `examples/case_studies/barter_island/database/`

### Step 5: Verify installation
```bash
python -c "from arctic_xbeach.model import Simulation; print('✓ Arctic-XBeach installed successfully!')"
```

---

### Simple Example (Recommended First Run)
You can quickly test the full model coupling and functionality using the provided `simple` example. This runs a one-day simulation and exercises all major features of the code. It should finish within a few minutes (depending on your CPU) and is a good starting point to confirm that the installation and coupling work properly.

Run from Python:
```python
from arctic_xbeach.model import Simulation
from main import main
from pathlib import Path

# Set up project directory
proj_dir = Path(__file__).parent.parent.resolve()
case_study_path = Path(__file__).parent / "simple"
sim = Simulation(case_study_path, proj_dir=proj_dir)

# (Optional) Enable debug logging for more information
import logging
logging.getLogger("thermo_model").setLevel(logging.DEBUG)

main(sim)
```

Or from command line:
```bash
python main.py examples/simple
```

---

You can also run the main Barter Island case study:
```python
from arctic_xbeach.model import Simulation

# Initialize simulation
sim = Simulation("examples/case_studies/barter_island")

# Run model
from main import main
main(sim)
```

Or from command line:
```bash
python main.py examples/case_studies/barter_island
```

---

## Repository Structure
```
arctic-xbeach/
├── arctic_xbeach/          # Main package
│   ├── model.py            # Simulation class
│   ├── bathymetry.py       # Grid generation
│   └── miscellaneous.py    # Utility functions
├── examples/
│   ├── analytical/         # Validation test cases
│   └── case_studies/
│       └── barter_island/  # Primary case study
│           ├── config.yaml
│           └── database/   # Forcing data (downloaded)
├── main.py                 # CLI entry point
├── download_data.py        # Data download script
├── requirements.txt
└── README.md
```

## Citation
If you use Arctic-XBeach in your research, please cite:
> Nederhoff, K., de Bruijn, K., Seyfert, C., et al. (in preparation). Arctic-XBeach: A Python-Based Thermo-Morphodynamic Model for Arctic Permafrost Coastal Erosion. *Geoscientific Model Development*.

---
## License
Arctic-XBeach is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or collaboration inquiries:
- **Kees Nederhoff** - Deltares USA
- **Kevin de Bruijn** - Delft University of Technology  
- **Carola Seyfert** - Stichting Deltares Netherlands

---

*This is a [Deltares Research](https://www.deltares.nl) project*
