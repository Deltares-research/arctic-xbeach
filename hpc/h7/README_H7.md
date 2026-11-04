# Arctic-XBeach calibration on the Deltares H7 cluster

The two Optuna drivers in `calibration/` run on one 44-core node of the `44vcpu` partition, about 42 trials at a time (coupled: 43 or 21, see "Open checks for the first job", item 2). The same scripts
run on the Windows desktop; on H7 they are configured with `AXB_*` environment variables that the sbatch scripts set (list and meaning:
`calibration/README.md`, section "Running on Linux / HPC").

## Files

| File | Purpose |
|---|---|
| `setup_env.sh` | (run with `bash -l`) creates (or updates) the conda environment `/u/nederhof/envs/arctic-xbeach` from `environment.yml`, installs the repo and runs the tests |
| `thermal_calibration.sbatch` | step 1, thermal-only calibration, `run_thermal_calibration.py`, 3 days |
| `coupled_calibration.sbatch` | step 2, coupled calibration, `auto_calibrate_coupled_model.py`, 14 days; loads the XBeach trunk module, starts XBeach through `mpirun`, runs an XBeach preflight and passes the executable |

Both sbatch files request account `amo`, QOS `normal`, partition `44vcpu`, one node and 44 cores (which takes the whole 44vcpu node, so there
is no `--exclusive`). They use the study in `AXB_RUN_ROOT` with journal storage, TPE `constant_liar`, `AXB_N_PARALLEL = cpus - 2` (coupled: `cpus - 1`), and the
job's `$TMPDIR` as scratch. They write the host, the `$TMPDIR` size and mount, memory, package versions, the git commit and the `AXB_*`
settings to the top of the job log, then check that file locks work in `AXB_RUN_ROOT` ("File lock probe OK", otherwise the job stops). When
the job ends, for any reason, the log lists what is left in `AXB_SCRATCH`.

`#SBATCH --dependency=singleton` lets only one job of a given name (and user) run at a time, because only one driver per study is supported:
a second submission of `axb_thermal` or `axb_coupled` waits until the first has ended. A smoke job needs its own name (`-J`, below), or it
waits for the production job, or the other way round.

## One-time setup

```bash
mkdir -p /u/nederhof/git /u/nederhof/arctic-xbeach-runs/logs
git clone -b linux-deployment https://github.com/deltares-research/arctic-xbeach.git /u/nederhof/git/arctic-xbeach
bash -l /u/nederhof/git/arctic-xbeach/hpc/h7/setup_env.sh
```

The last line builds the environment (a few minutes), installs the repo with the `calibration` extra and runs the test suite; it should
end with all tests passed. Run it again after a `git pull` that changes `environment.yml`.

## Staging the data

Everything lives under `/u/nederhof/arctic-xbeach-runs/` (4.9 TB free, NFS). The sbatch scripts expect these names; change them with
`AXB_DATA_DIR` or the individual `AXB_*` variables.

| H7 target | Desktop source | Needed by |
|---|---|---|
| `database/era5.csv` (220 MB), `database/storms.csv` (41 MB) | `N:\arctic-xbeach-runs\database\` | both |
| `thermal_base/` (`config.yaml`, `x.grd`, `bed.dep`, `ne_layer.txt`) | `N:\arctic-xbeach-runs\thermal_calibration_runs\20260201_calibration_runs\base\` | thermal |
| `observations/BI_T-1_processed.csv` | `D:\Git\thermo-morphological-model\database\raw_datasets\erikson2\Temp_arrays USGS-UCSC_Oberle\BI_T-1_processed.csv` | thermal |
| `coupled_base/` (same four files) | `N:\arctic-xbeach-runs\coupled_calibration_runs\20260214_calibration_runs\base\` | coupled |
| `thermal_best/config.yaml` | `N:\arctic-xbeach-runs\v11_runs\thermal_calibration\optuna_trial_000490\config.yaml` (or the best config of the H7 thermal study) | coupled |
| `observations/bluff_edges_without_geometry.csv`, `observations/shore_lines_without_geometry.csv` | `D:\Git\thermo-morphological-model\database\ts_datasets\` | coupled |

The forcing is public. Download it on H7 with the repo script into a separate folder (it would otherwise overwrite the small example files
that are tracked in git), then move it:

Python commands on the login node need the environment first (the sbatch files do this themselves):

```bash
module load miniconda/py312_24.7.1-0; eval "$(conda shell.bash hook)"; conda activate /u/nederhof/envs/arctic-xbeach
```

Then:

```bash
cd /u/nederhof/git/arctic-xbeach
python download_data.py --target /u/nederhof/arctic-xbeach-runs/_download
mkdir -p /u/nederhof/arctic-xbeach-runs/database
mv /u/nederhof/arctic-xbeach-runs/_download/examples/case_studies/barter_island/database/*.csv /u/nederhof/arctic-xbeach-runs/database/
md5sum /u/nederhof/arctic-xbeach-runs/database/*.csv
```

The sizes of the S3 files equal those of the desktop files. The desktop checksums are `67ea136ddca8fd580c0c8c509ae88b6c` (`era5.csv`) and
`e276a064b73a1efa2d7064ee4998bc16` (`storms.csv`); if the downloaded files differ, copy the desktop files instead. The observation CSVs
and the base folders are not public. They go from the desktop through the Syncthing exchange folder (`D:\exchange_Desktop-Kees_VM-Deltares`)
to the VM and from there to `/u`. The `config.yaml` files contain Windows paths for the forcing, `xbeach.version` and `output.output_path`;
the drivers overwrite the forcing paths (and, for the coupled driver, the XBeach path) in every trial config, and `output_path` is unused
(`use_default_output_path: True`). The thermal base config keeps a Windows `xbeach.version`, which is not used with `with_xbeach: False`.

## Smoke test (thermal)

One trial on a small partition, under its own job name (`-J`) so that it does not wait for, or block, the production job. A new study enqueues `START_PARAMS` of the driver as trial 0, which is the parameter set of thermal trial 869
of the published study (not trial 490). With dt 90 s and sensor depth offset 0.1 m the desktop gives the objective **1.36571** K
(`N:\arctic-xbeach-runs\v11_runs\h7_desktop_reference`, 14.5 min on one core).

```bash
cd /u/nederhof/arctic-xbeach-runs/logs
sbatch -J axb_thermal_smoke -p 16vcpu --cpus-per-task=4 --time=02:00:00 \
  --export=ALL,AXB_RUN_ROOT=/u/nederhof/arctic-xbeach-runs/thermal_smoke,AXB_N_TRIALS=1,AXB_N_PARALLEL=1,AXB_THERMAL_DT=90,AXB_SENSOR_DEPTH_OFFSET=0.1 \
  /u/nederhof/git/arctic-xbeach/hpc/h7/thermal_calibration.sbatch
```

Check in `axb_thermal_smoke_<jobid>.out` and `thermal_smoke/`:

* "Best RMSE" equals the desktop value to about 5 decimals. A difference in the last digit is floating point; more than that is a problem.
* `thermal_smoke/optuna_trial_000000/` has `run.log`, `config.yaml`, `rmse_bias_layers.txt`, `temperature_layers.png` and no `results/results.nc`.
* `thermal_smoke/optuna_journal.log` exists. The job log shows "File lock probe OK", and the listing of `AXB_SCRATCH` at the end is empty.
  A trial folder left there, or "ERROR copying" lines, would mean the copy-back failed (`run.log` and `config.yaml` are then still copied).

## Smoke test (coupled)

The coupled run starts many XBeach processes at once, so test it with several trials in parallel (`AXB_N_PARALLEL` of 8 or more) on a short
window. `AXB_TIME_END` (`DD-MM-YYYY`) replaces `model.time_end` of the template; the spin-up is still the full 10 years, so the window must
end after it and should contain open water, or no XBeach call is made (`xbeach_calls.csv` is then missing and the trial gets `inf`). The
published base starts on 01-01-2002 with a spin-up of 10 years, so `01-10-2012` covers the 2012 open-water season; check `time_start` and
`spin_up_time` in `coupled_base/config.yaml`. The job needs `AXB_N_PARALLEL + 1` cores. With `AXB_TIME_END` set a trial is scored only up to that date and can be `inf` without having failed, so the job
exits nonzero only if a trial failed (child exit code, failed XBeach calls beyond the limit); without it the job exits nonzero when no trial has a finite value.

```bash
cd /u/nederhof/arctic-xbeach-runs/logs
sbatch -J axb_coupled_smoke -p 16vcpu --cpus-per-task=10 --time=06:00:00 \
  --export=ALL,AXB_RUN_ROOT=/u/nederhof/arctic-xbeach-runs/coupled_smoke,AXB_N_TRIALS=8,AXB_N_PARALLEL=8,AXB_TIME_END=01-10-2012 \
  /u/nederhof/git/arctic-xbeach/hpc/h7/coupled_calibration.sbatch
```

Check:

* The job log shows "File lock probe OK", `ldd` without `not found`, the launcher, the `mpirun` version and "XBeach preflight OK in N s"; the first
  XBeach call in a `run.log` does not hang or fail.
* Failed XBeach calls: `grep -c "Failed to run XBeach" /u/nederhof/arctic-xbeach-runs/coupled_smoke/optuna_trial_*/run.log` is 0 in every
  file. The driver also counts them (trial attribute `n_failed_xbeach_calls`) and gives `inf` to a trial above `AXB_MAX_FAILED_XBEACH_CALLS` (0).
* Core placement, while the trials run: `srun --jobid=<jobid> --overlap bash -c 'ps -eo psr,comm | grep xbeach'`. The first column is the
  core of each XBeach process (two per call); they should be spread over the cores, not piled on a few.
* `coupled_smoke/spinup_state.npz` exists. It can be copied into the production `AXB_RUN_ROOT` (`mkdir -p coupled_calibration; cp
  coupled_smoke/spinup_state.npz coupled_calibration/`) to save the spin-up there. A trial refuses a state that was made with a different
  thermal config (thermal hash), so a wrong copy cannot be used silently.

## Full run

```bash
cd /u/nederhof/arctic-xbeach-runs/logs
sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/thermal_calibration.sbatch      # 1000 trials, 42 at a time
sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/coupled_calibration.sbatch      # 200 trials, after the thermal study and its best config in thermal_best/
```

The coupled job exits nonzero when the XBeach preflight fails or no trial gets a finite score, so a follow-up job submitted with
`sbatch --dependency=afterok:<jobid> ...` does not start after a study that scored nothing.

The studies are in `thermal_calibration/` and `coupled_calibration/`. Settings such as `AXB_N_TRIALS` are changed with
`sbatch --export=ALL,AXB_N_TRIALS=100 ...`. Do not start two jobs on the same `AXB_RUN_ROOT`: only one driver per study is supported (`--dependency=singleton` makes a second job of the same name wait).

## Monitoring

* Queue and start estimate: `squeue -u $USER`, `squeue -u $USER --start`.
* Driver output: `tail -f /u/nederhof/arctic-xbeach-runs/logs/axb_thermal_<jobid>.out`; completed trials: `<RUN_ROOT>/calibration_prints.txt`.
* A running trial lives in the job's `$TMPDIR` on the compute node, not on `/u`, so its `run.log` is not visible from the login node until
  it has finished and been copied back. On the node: `srun --jobid=<jobid> --overlap bash -c 'tail -n 3 $TMPDIR/optuna_trial_000003/run.log'`
  (single quotes, so that `$TMPDIR` is the node's; not tested).
* History of the study (CSV, best-so-far figure, summary), without touching the running job (after the `conda activate` shown under
  "Staging the data"):

```bash
AXB_RUN_ROOT=/u/nederhof/arctic-xbeach-runs/thermal_calibration AXB_STORAGE=journal \
AXB_EXPORT_DIR=/u/nederhof/arctic-xbeach-runs/thermal_calibration/history AXB_OBJECTIVE_LABEL="Ground temperature RMSE (K)" \
python /u/nederhof/git/arctic-xbeach/calibration/export_history.py
```

## Resuming

After a time limit, a node failure or `scancel`, submit the same script again with the same `AXB_RUN_ROOT`. The driver continues the study,
marks the trials that were running as FAIL (journal storage; the number is in the job log) and runs the trials still missing to reach `AXB_N_TRIALS`.
The previous job must have ended first (`--dependency=singleton`).
The trials that were running are lost, because their scratch folders are gone with the job.

## Open checks for the first job

1. **`$TMPDIR` size and type.** TmpDisk is 0 on H7 and `$TMPDIR` is `/var/lib/containers/storage-rootless/tmp/<jobid>`. The job log shows
   `df -h` and `findmnt` of it. A coupled trial needs 0.4 to 0.7 GB, 43 trials about 25 GB. If it is small or on NFS, submit with
   `--export=ALL,AXB_SCRATCH=` (no scratch): the trials then run on `/u` and `results/xb_files` is deleted after scoring.
2. **XBeach under SLURM (MPI-only, trunk module, `mpirun -np 2`).** The XBeach modules on H7 are MPI-only builds: started as one process they halt
   ("Number of MPI processes must be 2 or greater"). The Halloween module (1.24.6057, the release of the desktop runs) also crashes writing hotstart files
   under gfortran (`ncoutput.F90:2122`). `coupled_calibration.sbatch` therefore loads `xbeach/xbeach-trunk_gcc_12.2.0_openmpi_4.1.5_HEAD` (built 2026-10-14)
   and `main.py` starts every call through the launcher `xbeach.launcher`, set from `AXB_XBEACH_LAUNCHER`:
   `/opt/apps/openmpi/4.1.5_gcc12.2.0/bin/mpirun -np 2 --oversubscribe --bind-to none` (mpirun is not on `PATH` after the module load; `AXB_MPIRUN` and
   `AXB_XBEACH_NP` change the pieces, `AXB_XBEACH_LAUNCHER=` empty starts `xbeach` directly). `--oversubscribe` is needed because the job has `--ntasks=1`, so
   OpenMPI under SLURM sees one slot; `--bind-to none` and `OMPI_MCA_hwloc_base_binding_policy=none` keep the parallel `mpirun` processes from binding to the same cores.
   Two choices of `AXB_N_PARALLEL` are being tested: 43 (the default, `cpus - 1`; the second MPI rank of each call is expected to be mostly idle) and 21
   (`(cpus - 1) / 2`, one core per rank). Compare the wall time per trial and the core load (`ps -eo psr,comm | grep xbeach`, first column).
   **Preflight:** before the driver starts, the job copies the case in `AXB_XBEACH_PREFLIGHT_DIR` (default `$DATA/xbeach_preflight`, i.e.
   `/u/nederhof/arctic-xbeach-runs/xbeach_preflight`) to `$TMPDIR` and runs it with the launcher, with a limit of 900 s. It passes if XBeach exits 0 and writes
   `xboutput.nc` (and a `hotstart_*.dat` if `params.txt` has `writehotstart = 1`); otherwise the log tail and `XBerror.txt` are printed and the job exits 1.
   Copy a small XBeach case that has run before to that folder (a case with `writehotstart = 1` also tests the hotstart crash), or submit with
   `AXB_XBEACH_PREFLIGHT_DIR=` (empty) to skip it. **Fail-fast:** every trial config gets `xbeach.max_failed_calls = AXB_MAX_FAILED_XBEACH_CALLS` (0), so a
   trial stops at the first failed XBeach call (exit code 3, scored `inf`) instead of running on for hours.
   **Not yet checked:** trunk is not the Halloween release of the desktop runs, so results may differ slightly; an equivalence check (the same case on
   the desktop and on H7) is pending.
3. **`module` and `conda` in the batch shell.** The sbatch scripts and `setup_env.sh` use `#!/bin/bash -l` (start `setup_env.sh` with `bash -l`) so that the module command is defined.
4. **The XBeach module.** `ldd` in the job log must show no `not found`.
5. **Journal on `/u`.** The file `optuna_journal.log` should grow during the run; lock errors in the job log would mean that the NFS lock
   method of Optuna does not work there. The sbatch scripts also probe the run lock of `arctic_xbeach` in `AXB_RUN_ROOT` before the driver starts.
