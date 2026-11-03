#!/bin/bash -l
#SBATCH --job-name=axb_coupled
#SBATCH --account=amo
#SBATCH --qos=normal
#SBATCH --partition=44vcpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --output=%x_%j.out
# Coupled (thermal + XBeach) Optuna calibration (calibration/auto_calibrate_coupled_model.py) on one 44vcpu node, one trial per core.
# Submit from the folder that should hold the job log:   sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/coupled_calibration.sbatch
# Override any setting at submission:   sbatch --export=ALL,AXB_N_TRIALS=2,AXB_N_PARALLEL=2 .../coupled_calibration.sbatch
# Submitting again with the same AXB_RUN_ROOT continues the same study (the spin-up state is reused).
set -euo pipefail

REPO_DIR=${AXB_REPO_DIR:-/u/nederhof/git/arctic-xbeach}
ENV_DIR=${AXB_ENV_DIR:-/u/nederhof/envs/arctic-xbeach}
DATA=${AXB_DATA_DIR:-/u/nederhof/arctic-xbeach-runs}   # see hpc/h7/README_H7.md for what goes where

set +u
module load miniconda/py312_24.7.1-0
module load xbeach/xbeach-Halloween_BETA_gcc_12.2.0_openmpi_4.1.5_HEAD   # XBeach 1.24.6057 Halloween, the release of the desktop runs
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
set -u

# settings of the driver (calibration/README.md); an AXB_* variable given with --export wins
export AXB_BASE_SIM_DIR=${AXB_BASE_SIM_DIR:-$DATA/coupled_base}
export AXB_RUN_ROOT=${AXB_RUN_ROOT:-$DATA/coupled_calibration}
export AXB_THERMAL_CONFIG_YAML=${AXB_THERMAL_CONFIG_YAML:-$DATA/thermal_best/config.yaml}
export AXB_BLUFF_EDGES_CSV=${AXB_BLUFF_EDGES_CSV:-$DATA/observations/bluff_edges_without_geometry.csv}
export AXB_SHORELINES_CSV=${AXB_SHORELINES_CSV:-$DATA/observations/shore_lines_without_geometry.csv}
export AXB_FORCING_CSV=${AXB_FORCING_CSV:-$DATA/database/era5.csv}
export AXB_STORM_CSV=${AXB_STORM_CSV:-$DATA/database/storms.csv}
export AXB_XBEACH_EXE=${AXB_XBEACH_EXE:-$(command -v xbeach)}
[ -x "$AXB_XBEACH_EXE" ] || { echo "XBeach executable not found: ${AXB_XBEACH_EXE:-<empty>}"; exit 1; }
export AXB_N_TRIALS=${AXB_N_TRIALS:-200}
export AXB_N_PARALLEL=${AXB_N_PARALLEL:-$((SLURM_CPUS_PER_TASK - 2))}   # two cores left for the driver and the file system
export AXB_STORAGE=${AXB_STORAGE:-journal}            # SQLite locking is not reliable on NFS
export AXB_CONSTANT_LIAR=${AXB_CONSTANT_LIAR:-1}
export AXB_KEEP_XB_FILES=${AXB_KEEP_XB_FILES:-0}      # only matters without scratch
export AXB_SCRATCH=${AXB_SCRATCH-${TMPDIR:-}}         # job TMPDIR; AXB_SCRATCH= (empty) switches scratch off

export HDF5_USE_FILE_LOCKING=FALSE
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "host:   $(hostname)"
echo "date:   $(date)"
echo "TMPDIR: ${TMPDIR:-<unset>}"
df -h "${TMPDIR:-/tmp}"
findmnt -T "${TMPDIR:-/tmp}" || stat -f "${TMPDIR:-/tmp}" || true
free -g
python --version
python -c "import optuna; print('optuna', optuna.__version__)"
echo "commit: $(git -C "$REPO_DIR" rev-parse HEAD)"
echo "xbeach: $AXB_XBEACH_EXE"
ldd "$AXB_XBEACH_EXE" | grep 'not found' || true
env | grep '^AXB_' | sort

cd "$REPO_DIR"
python -u calibration/auto_calibrate_coupled_model.py
