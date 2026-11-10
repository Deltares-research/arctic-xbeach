#!/bin/bash -l
#SBATCH --job-name=axb_coupled
#SBATCH --account=amo
#SBATCH --qos=normal
#SBATCH --partition=44vcpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --dependency=singleton
#SBATCH --time=14-00:00:00
#SBATCH --output=%x_%j.out
# Coupled (thermal + XBeach) Optuna calibration (calibration/auto_calibrate_coupled_model.py) on one 44vcpu node, one trial per core.
# XBeach is MPI-only on H7: every call is started as `mpirun -np 2 xbeach` (AXB_XBEACH_LAUNCHER), with the trunk module. A preflight run of a
# known-good XBeach case (AXB_XBEACH_PREFLIGHT_DIR) must pass before the driver starts, and the job exits nonzero if no trial got a score,
# so `sbatch --dependency=afterok:<jobid>` holds follow-up jobs. See hpc/h7/README_H7.md.
# Submit from the folder that should hold the job log:   sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/coupled_calibration.sbatch
# Override any setting at submission:   sbatch --export=ALL,AXB_N_TRIALS=2,AXB_N_PARALLEL=2 .../coupled_calibration.sbatch
# Optional driver settings (calibration/README.md), also set with --export:  AXB_TIMEOUT_H (stop starting trials after this many hours, below --time),
# AXB_SEED_FROM_RUN_ROOT + AXB_SEED_FROM_STUDY (+ AXB_SEED_FROM_STORAGE) to start from an earlier study, AXB_PARAM_RANGES (JSON of narrowed ranges).
# A JSON value contains commas, which --export cannot carry: export it in the shell before sbatch (--export=ALL, the default, passes it on):
#   export AXB_PARAM_RANGES='{"xb_threshold": [1.75, 2.25], "dryslp": [0.3, 1.0]}'; sbatch .../coupled_calibration.sbatch
# Submitting again with the same AXB_RUN_ROOT continues the same study (the spin-up state is reused).
set -euo pipefail

REPO_DIR=${AXB_REPO_DIR:-/u/nederhof/git/arctic-xbeach}
ENV_DIR=${AXB_ENV_DIR:-/u/nederhof/envs/arctic-xbeach}
DATA=${AXB_DATA_DIR:-/u/nederhof/arctic-xbeach-runs}   # see hpc/h7/README_H7.md for what goes where

set +u
module load miniconda/py312_24.7.1-0
# Halloween 1.24.6057 (the desktop release) on H7 is MPI-only and its gfortran build crashes writing hotstart files (ncoutput.F90:2122);
# trunk (built 2026-10-14) runs with 2 MPI processes. See hpc/h7/README_H7.md.
module load xbeach/xbeach-trunk_gcc_12.2.0_openmpi_4.1.5_HEAD
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
# XBeach is started with AXB_XBEACH_NP MPI processes; mpirun is not on PATH after the module load. --oversubscribe: the job has --ntasks=1,
# so OpenMPI under SLURM sees one slot.
export AXB_MPIRUN=${AXB_MPIRUN:-/opt/apps/openmpi/4.1.5_gcc12.2.0/bin/mpirun}
export AXB_XBEACH_NP=${AXB_XBEACH_NP:-2}
export AXB_XBEACH_LAUNCHER=${AXB_XBEACH_LAUNCHER-"$AXB_MPIRUN -np $AXB_XBEACH_NP --oversubscribe --bind-to none"}   # AXB_XBEACH_LAUNCHER= (empty) starts xbeach directly
# 43 on a 44vcpu node: the second MPI rank of each XBeach call is expected to be mostly idle. To give every rank its own core use
# AXB_N_PARALLEL=$(( (SLURM_CPUS_PER_TASK - 1) / AXB_XBEACH_NP )) (21). Both are being tested.
export AXB_N_PARALLEL=${AXB_N_PARALLEL:-$((SLURM_CPUS_PER_TASK - 1))}   # one core left for the driver and the file system
export AXB_STORAGE=${AXB_STORAGE:-journal}            # SQLite locking is not reliable on NFS
export AXB_CONSTANT_LIAR=${AXB_CONSTANT_LIAR:-1}
export AXB_KEEP_XB_FILES=${AXB_KEEP_XB_FILES:-0}      # only matters without scratch
export AXB_SCRATCH=${AXB_SCRATCH-${TMPDIR:-}}         # job TMPDIR; AXB_SCRATCH= (empty) switches scratch off

# when the job ends, also when set -e aborts it, list what is left in the scratch folder (the job epilog deletes it)
trap 'if [ -n "${AXB_SCRATCH:-}" ]; then echo "left in AXB_SCRATCH=$AXB_SCRATCH:"; ls -la "$AXB_SCRATCH" || true; fi' EXIT

export HDF5_USE_FILE_LOCKING=FALSE
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# XBeach is MPI-linked and started through mpirun by main.py. Without this every OpenMPI process binds itself to a core near
# core 0 of the node, and the ~43 parallel XBeach runs would share a few cores (the launcher also passes --bind-to none).
export OMPI_MCA_hwloc_base_binding_policy=none

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
echo "launcher: ${AXB_XBEACH_LAUNCHER:-<none, xbeach is started directly>}"
echo "mpirun: $("$AXB_MPIRUN" --version 2>&1 | head -1 || true)"
ldd "$AXB_XBEACH_EXE" | grep 'not found' || true
env | grep '^AXB_' | sort

cd "$REPO_DIR"

# file locks must work in AXB_RUN_ROOT (the run lock of arctic_xbeach); NFSv3 without a lock daemon would fail every run
mkdir -p "$AXB_RUN_ROOT/_lockprobe"
python -c 'import sys
from arctic_xbeach.run_lock import acquire_run_lock, release_run_lock
try:
    release_run_lock(acquire_run_lock(sys.argv[1]))
except Exception as e:
    print(f"File lock probe FAILED in {sys.argv[1]}: {e}")
    sys.exit(1)
print(f"File lock probe OK in {sys.argv[1]}")' "$AXB_RUN_ROOT/_lockprobe"

# XBeach preflight: one known-good case run exactly as the trials start XBeach (launcher, module, executable). A wrong module, a launcher
# that does not work under SLURM or a crash at hotstart writing then stops the job in minutes instead of after hundreds of failed calls.
# Copy a small XBeach case that has run before to AXB_XBEACH_PREFLIGHT_DIR (params.txt with writehotstart = 1 checks hotstart writing too).
AXB_XBEACH_PREFLIGHT_DIR=${AXB_XBEACH_PREFLIGHT_DIR-$DATA/xbeach_preflight}   # AXB_XBEACH_PREFLIGHT_DIR= (empty) skips the preflight
if [ -n "$AXB_XBEACH_PREFLIGHT_DIR" ]; then
    if [ ! -d "$AXB_XBEACH_PREFLIGHT_DIR" ]; then
        echo "XBeach preflight case not found: $AXB_XBEACH_PREFLIGHT_DIR"
        echo "Copy a known-good XBeach case (params.txt and its input files) there, or set AXB_XBEACH_PREFLIGHT_DIR= (empty) to skip the preflight."
        exit 1
    fi
    PREFLIGHT_RUN=$(mktemp -d "${TMPDIR:-/tmp}/xbeach_preflight.XXXXXX")
    cp -r "$AXB_XBEACH_PREFLIGHT_DIR"/. "$PREFLIGHT_RUN"/
    echo "XBeach preflight: $AXB_XBEACH_LAUNCHER $AXB_XBEACH_EXE in $PREFLIGHT_RUN (case $AXB_XBEACH_PREFLIGHT_DIR)"
    preflight_start=$SECONDS
    rc=0
    (cd "$PREFLIGHT_RUN" && timeout 900 $AXB_XBEACH_LAUNCHER "$AXB_XBEACH_EXE" < /dev/null > preflight.log 2>&1) || rc=$?   # launcher unquoted: it must word-split
    problem=""
    if [ "$rc" -ne 0 ]; then
        problem="XBeach exited with code $rc (124 = longer than 900 s)"
    elif [ ! -f "$PREFLIGHT_RUN/xboutput.nc" ]; then
        problem="XBeach exited with 0 but wrote no xboutput.nc"
    elif grep -Eiq '^[[:space:]]*writehotstart[[:space:]]*=[[:space:]]*1([[:space:]]|$)' "$PREFLIGHT_RUN/params.txt" \
            && ! compgen -G "$PREFLIGHT_RUN/hotstart_*.dat" > /dev/null; then
        problem="params.txt has writehotstart = 1 but XBeach wrote no hotstart_*.dat"
    fi
    if [ -n "$problem" ]; then
        echo "XBeach preflight FAILED: $problem"
        echo "--- tail of $PREFLIGHT_RUN/preflight.log"; tail -n 30 "$PREFLIGHT_RUN/preflight.log" || true
        if [ -f "$PREFLIGHT_RUN/XBerror.txt" ]; then echo "--- XBerror.txt"; tail -n 30 "$PREFLIGHT_RUN/XBerror.txt" || true; fi
        echo "(run folder kept for inspection: $PREFLIGHT_RUN)"
        exit 1
    fi
    echo "XBeach preflight OK in $((SECONDS - preflight_start)) s"
    rm -rf "$PREFLIGHT_RUN"
fi

python -u calibration/auto_calibrate_coupled_model.py
