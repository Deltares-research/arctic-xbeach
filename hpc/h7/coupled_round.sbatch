#!/bin/bash -l
#SBATCH --job-name=axb_round_v3
#SBATCH --account=amo
#SBATCH --qos=normal
#SBATCH --partition=44vcpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --dependency=singleton
#SBATCH --time=5-00:00:00
#SBATCH --output=/u/nederhof/arctic-xbeach-runs/logs/%x_%j.log
# The 27-run revision round (calibration/h7_round_design.py: centre C from a JSON, sensitivity pairs around it) on ONE 44vcpu node, all 27 coupled
# (thermal + XBeach) runs in parallel, one core each. Steps: prep (calibration/run_revision_round.py: run folders, perturbed forcing copies, checks,
# READY), pool (calibration/run_revision_pool.py: the 27 runs + an hourly collector pass), final collector pass (collect_revision_round.py).
# XBeach is the SERIAL "zero" build of Halloween 1.24.6057 started directly (no mpirun): the desktop release, so the numbers are comparable.
# The job exits nonzero if the prep fails or any run has a nonzero model exit code. See hpc/h7/README_H7.md, section "Revision round".
#
# Submit from any folder (the log goes to /u/nederhof/arctic-xbeach-runs/logs):
#   export AXB_REVISION_ROOT=/u/nederhof/arctic-xbeach-runs/round_v3_t569
#   export AXB_ROUND_CENTER_JSON=/u/nederhof/arctic-xbeach-runs/round_v3_t569_center.json
#   sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/coupled_round.sbatch        (--export=ALL is the default: the exports above are passed on)
# Other time or name:   sbatch --time=3-00:00:00 -J axb_round_smoke .../coupled_round.sbatch
# The name defaults to axb_round_v3; --dependency=singleton lets only one job of a name run at a time (two on one AXB_REVISION_ROOT would corrupt it).
# Submitting again with the same AXB_REVISION_ROOT redoes the prep for runs that have not started and skips runs that have model_exit_code.txt;
# runs that were interrupted by a time limit start again from their spin-up (nothing of a running model survives the node's scratch).
#
# REQUIRED (the job stops at once without them):
#   AXB_REVISION_ROOT        round folder on /u (created): run folders h01_... to h27_..., forcing/, spinup/, design.csv, READY, logs, summary/
#   AXB_ROUND_CENTER_JSON    JSON with the centre set C: {"facua":..,"bedfriccoef":..,"wetslp":..,"dryslp":..,"xb_threshold":..,"source":"v3 trial 569"}
# DATA (defaults under AXB_DATA_DIR = /u/nederhof/arctic-xbeach-runs; each must exist, else the job stops):
#   AXB_BASE_SIM_DIR         template run folder with config.yaml, x.grd, bed.dep, ne_layer.txt   ($DATA/coupled_base_v11)
#   AXB_FORCING_CSV          era5.csv                                                             ($DATA/database/era5.csv)
#   AXB_STORM_CSV            storms.csv                                                           ($DATA/database/storms.csv)
#   AXB_BLUFF_EDGES_CSV, AXB_SHORELINES_CSV   observed bluff edge and shoreline, used by the scorer ($DATA/observations/..._without_geometry.csv)
#   AXB_THERMAL_BEST_SOURCE  thermal config of trial 935 (dt 60 s); thermal_best.txt next to it or one level up is optional ($DATA/thermal_best_A1_t935/config.yaml)
#   AXB_SPINUP_STATES_DIR    folder with P2_center.npz, the spin-up state that 15 runs restart from; the prep proves that it loads ($DATA/spinup_states)
# XBeach:
#   AXB_XBEACH_EXE           default /u/nederhof/opt/xbeach-halloween-serial-zero/bin/xbeach
#   AXB_XBEACH_LAUNCHER      default empty = xbeach is started directly (set a command prefix such as "mpirun -np 2" only for an MPI-only build)
#   AXB_XBEACH_MODULE        optional module to load for the executable's libraries (default none); ldd must show no "not found"
#   AXB_XBEACH_PREFLIGHT_DIR optional small known-good XBeach case that is run before the prep (default none = skipped)
# Output and scratch:
#   AXB_OUTPUT_RES           [h] thermal record spacing of results.nc of every run except the centre runs (24)
#   AXB_CENTER_OUTPUT_RES    [h] the same for the three centre runs h02, h13, h14 (1 = hourly; their results.nc are ~8-13 GB each)
#   AXB_SCRATCH              folder on the node's disk where each run works ($TMPDIR/axb_round, <AXB_SCRATCH>/<run id>); copied back to
#                            AXB_REVISION_ROOT without results/xb_files when the run ends. AXB_SCRATCH= (empty) runs on /u (slow, ~26,000 files per run)
# Pool:
#   AXB_MAX_PARALLEL         runs at the same time (27)
# SMOKE TEST ONLY (leave unset for the round):
#   AXB_SMOKE_DAYS           > 0: every run ends this many days after its period start (prep writes that time_end; scores are not comparable)
#   AXB_ONLY_RUNS            comma list of run ids the pool starts (e.g. h14); the exit check looks at these only
# Fixed by this script: AXB_ROUND_DESIGN=h7_round_design, AXB_REQUIRE_POSIX_PATHS=1 (the prep fails if a config.yaml holds a Windows drive path),
#   AXB_REPO_DIR (/u/nederhof/git/arctic-xbeach, branch h7-round), AXB_ENV_DIR (/u/nederhof/envs/arctic-xbeach), one BLAS/OpenMP thread per process.
set -euo pipefail

REPO_DIR=${AXB_REPO_DIR:-/u/nederhof/git/arctic-xbeach}
ENV_DIR=${AXB_ENV_DIR:-/u/nederhof/envs/arctic-xbeach}
DATA=${AXB_DATA_DIR:-/u/nederhof/arctic-xbeach-runs}

: "${AXB_REVISION_ROOT:?AXB_REVISION_ROOT is not set: the round folder on /u}"
: "${AXB_ROUND_CENTER_JSON:?AXB_ROUND_CENTER_JSON is not set: the JSON file with the centre parameter set C}"
[ -f "$AXB_ROUND_CENTER_JSON" ] || { echo "AXB_ROUND_CENTER_JSON does not exist: $AXB_ROUND_CENTER_JSON"; exit 1; }

set +u
module load miniconda/py312_24.7.1-0
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
if [ -n "${AXB_XBEACH_MODULE:-}" ]; then module load "$AXB_XBEACH_MODULE"; fi
set -u

export AXB_ROUND_DESIGN=h7_round_design
export AXB_REQUIRE_POSIX_PATHS=1
export AXB_BASE_SIM_DIR=${AXB_BASE_SIM_DIR:-$DATA/coupled_base_v11}
export AXB_FORCING_CSV=${AXB_FORCING_CSV:-$DATA/database/era5.csv}
export AXB_STORM_CSV=${AXB_STORM_CSV:-$DATA/database/storms.csv}
export AXB_BLUFF_EDGES_CSV=${AXB_BLUFF_EDGES_CSV:-$DATA/observations/bluff_edges_without_geometry.csv}
export AXB_SHORELINES_CSV=${AXB_SHORELINES_CSV:-$DATA/observations/shore_lines_without_geometry.csv}
export AXB_THERMAL_BEST_SOURCE=${AXB_THERMAL_BEST_SOURCE:-$DATA/thermal_best_A1_t935/config.yaml}
export AXB_SPINUP_STATES_DIR=${AXB_SPINUP_STATES_DIR:-$DATA/spinup_states}
export AXB_XBEACH_EXE=${AXB_XBEACH_EXE:-/u/nederhof/opt/xbeach-halloween-serial-zero/bin/xbeach}
export AXB_XBEACH_LAUNCHER=${AXB_XBEACH_LAUNCHER-}          # empty: no launcher; the variable must exist (empty) so that the prep drops a launcher of the base config
export AXB_OUTPUT_RES=${AXB_OUTPUT_RES:-24}
export AXB_CENTER_OUTPUT_RES=${AXB_CENTER_OUTPUT_RES:-1}
export AXB_MAX_PARALLEL=${AXB_MAX_PARALLEL:-27}
export AXB_XBEACH_PREFLIGHT_DIR=${AXB_XBEACH_PREFLIGHT_DIR-}
if [ -z "${AXB_SCRATCH+x}" ]; then   # unset: the job's TMPDIR; set but empty: no scratch
    [ -n "${TMPDIR:-}" ] || { echo "TMPDIR is not set and AXB_SCRATCH was not given"; exit 1; }
    export AXB_SCRATCH=$TMPDIR/axb_round
else
    export AXB_SCRATCH
fi

for f in "$AXB_BASE_SIM_DIR/config.yaml" "$AXB_BASE_SIM_DIR/x.grd" "$AXB_BASE_SIM_DIR/bed.dep" "$AXB_BASE_SIM_DIR/ne_layer.txt" "$AXB_FORCING_CSV" \
         "$AXB_STORM_CSV" "$AXB_BLUFF_EDGES_CSV" "$AXB_SHORELINES_CSV" "$AXB_THERMAL_BEST_SOURCE" "$AXB_SPINUP_STATES_DIR/P2_center.npz"; do
    [ -f "$f" ] || { echo "Required file not found: $f"; exit 1; }
done
[ -x "$AXB_XBEACH_EXE" ] || { echo "XBeach executable not found or not executable: $AXB_XBEACH_EXE"; exit 1; }

# when the job ends, also when set -e aborts it, list what is left in the scratch folder (the job epilog deletes it)
trap 'if [ -n "${AXB_SCRATCH:-}" ]; then echo "left in AXB_SCRATCH=$AXB_SCRATCH:"; ls -la "$AXB_SCRATCH" || true; fi' EXIT
[ -z "$AXB_SCRATCH" ] || mkdir -p "$AXB_SCRATCH"

export HDF5_USE_FILE_LOCKING=FALSE
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "host:   $(hostname)"
echo "date:   $(date)"
echo "TMPDIR: ${TMPDIR:-<unset>}"
df -h "${AXB_SCRATCH:-${TMPDIR:-/tmp}}"
findmnt -T "${AXB_SCRATCH:-${TMPDIR:-/tmp}}" || stat -f "${AXB_SCRATCH:-${TMPDIR:-/tmp}}" || true
free -g
python --version
echo "commit: $(git -C "$REPO_DIR" rev-parse HEAD) ($(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD))"
echo "xbeach: $AXB_XBEACH_EXE"
echo "launcher: ${AXB_XBEACH_LAUNCHER:-<none, xbeach is started directly>}"
if ldd "$AXB_XBEACH_EXE" | grep 'not found'; then echo "XBeach has unresolved libraries (see above): set AXB_XBEACH_MODULE"; exit 1; fi
env | grep '^AXB_' | sort

cd "$REPO_DIR"

# file locks must work in AXB_REVISION_ROOT (the run lock of arctic_xbeach; the pool's run script waits forever on a lock error)
mkdir -p "$AXB_REVISION_ROOT/_lockprobe"
python -c 'import sys
from arctic_xbeach.run_lock import acquire_run_lock, release_run_lock
try:
    release_run_lock(acquire_run_lock(sys.argv[1]))
except Exception as e:
    print(f"File lock probe FAILED in {sys.argv[1]}: {e}")
    sys.exit(1)
print(f"File lock probe OK in {sys.argv[1]}")' "$AXB_REVISION_ROOT/_lockprobe"

# Optional XBeach preflight: a small known-good case run exactly as the model starts XBeach (launcher, module, executable)
if [ -n "$AXB_XBEACH_PREFLIGHT_DIR" ]; then
    [ -d "$AXB_XBEACH_PREFLIGHT_DIR" ] || { echo "XBeach preflight case not found: $AXB_XBEACH_PREFLIGHT_DIR"; exit 1; }
    PREFLIGHT_RUN=$(mktemp -d "${TMPDIR:-/tmp}/xbeach_preflight.XXXXXX")
    cp -r "$AXB_XBEACH_PREFLIGHT_DIR"/. "$PREFLIGHT_RUN"/
    rc=0
    (cd "$PREFLIGHT_RUN" && timeout 900 $AXB_XBEACH_LAUNCHER "$AXB_XBEACH_EXE" < /dev/null > preflight.log 2>&1) || rc=$?   # launcher unquoted: it must word-split
    if [ "$rc" -ne 0 ] || [ ! -f "$PREFLIGHT_RUN/xboutput.nc" ]; then
        echo "XBeach preflight FAILED (exit code $rc, 124 = longer than 900 s, or no xboutput.nc)"
        tail -n 30 "$PREFLIGHT_RUN/preflight.log" || true
        exit 1
    fi
    echo "XBeach preflight OK"
    rm -rf "$PREFLIGHT_RUN"
fi

echo "=== prep $(date) ==="
python -u calibration/run_revision_round.py
echo "=== pool $(date): $AXB_MAX_PARALLEL runs at a time ==="
pool_rc=0
python -u calibration/run_revision_pool.py || pool_rc=$?
echo "=== collector $(date) ==="
AXB_COLLECT_ONCE=1 python -u calibration/collect_revision_round.py || true

# nonzero exit if a run did not finish: model_exit_code.txt of every run folder (written by run_revision_one.py)
n_bad=0
for code in "$AXB_REVISION_ROOT"/h[0-9][0-9]_*/model_exit_code.txt; do
    run_id=$(basename "$(dirname "$code")"); run_id=${run_id%%_*}
    if [ -n "${AXB_ONLY_RUNS:-}" ] && [[ ",${AXB_ONLY_RUNS// /}," != *",$run_id,"* ]]; then continue; fi   # smoke test: only the runs it ran
    [ -f "$code" ] || { echo "no model exit code file: $code"; n_bad=$((n_bad + 1)); continue; }
    [ "$(head -n 1 "$code")" = "0" ] || { echo "run failed: $code ($(head -n 1 "$code"))"; n_bad=$((n_bad + 1)); }
done
echo "=== done $(date): pool exit $pool_rc, $n_bad runs with a nonzero model exit code; summary in $AXB_REVISION_ROOT/summary ==="
[ "$pool_rc" -eq 0 ] && [ "$n_bad" -eq 0 ]
