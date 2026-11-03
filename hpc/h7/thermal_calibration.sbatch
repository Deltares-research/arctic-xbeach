#!/bin/bash -l
#SBATCH --job-name=axb_thermal
#SBATCH --account=amo
#SBATCH --qos=normal
#SBATCH --partition=44vcpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
# Thermal-only Optuna calibration (calibration/run_thermal_calibration.py) on one 44vcpu node, one trial per core.
# Submit from the folder that should hold the job log:   sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/thermal_calibration.sbatch
# Override any setting at submission:   sbatch --export=ALL,AXB_N_TRIALS=1,AXB_N_PARALLEL=1,AXB_THERMAL_DT=90 .../thermal_calibration.sbatch
# Submitting again with the same AXB_RUN_ROOT continues the same study.
set -euo pipefail

REPO_DIR=${AXB_REPO_DIR:-/u/nederhof/git/arctic-xbeach}
ENV_DIR=${AXB_ENV_DIR:-/u/nederhof/envs/arctic-xbeach}
DATA=${AXB_DATA_DIR:-/u/nederhof/arctic-xbeach-runs}   # see hpc/h7/README_H7.md for what goes where

set +u
module load miniconda/py312_24.7.1-0
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
set -u

# settings of the driver (calibration/README.md); an AXB_* variable given with --export wins
export AXB_BASE_SIM_DIR=${AXB_BASE_SIM_DIR:-$DATA/thermal_base}
export AXB_RUN_ROOT=${AXB_RUN_ROOT:-$DATA/thermal_calibration}
export AXB_OBS_CSV=${AXB_OBS_CSV:-$DATA/observations/BI_T-1_processed.csv}
export AXB_FORCING_CSV=${AXB_FORCING_CSV:-$DATA/database/era5.csv}
export AXB_STORM_CSV=${AXB_STORM_CSV:-$DATA/database/storms.csv}
export AXB_N_TRIALS=${AXB_N_TRIALS:-1000}
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
env | grep '^AXB_' | sort

cd "$REPO_DIR"
python -u calibration/run_thermal_calibration.py
