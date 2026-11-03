# Arctic-XBeach calibration on the Deltares H7 cluster

The two Optuna drivers in `calibration/` run on one 44-core node of the `44vcpu` partition, about 42 trials at a time. The same scripts
run on the Windows desktop; on H7 they are configured with `AXB_*` environment variables that the sbatch scripts set (list and meaning:
`calibration/README.md`, section "Running on Linux / HPC").

## Files

| File | Purpose |
|---|---|
| `setup_env.sh` | creates (or updates) the conda environment `/u/nederhof/envs/arctic-xbeach` from `environment.yml`, installs the repo and runs the tests |
| `thermal_calibration.sbatch` | step 1, thermal-only calibration, `run_thermal_calibration.py`, 3 days |
| `coupled_calibration.sbatch` | step 2, coupled calibration, `auto_calibrate_coupled_model.py`, 14 days; loads the XBeach module and passes the executable |

Both sbatch files request account `amo`, QOS `normal`, partition `44vcpu`, one node, 44 cores, `--exclusive`. They use the study in
`AXB_RUN_ROOT` with journal storage, TPE `constant_liar`, `AXB_N_PARALLEL = cpus - 2`, and the job's `$TMPDIR` as scratch. They write the host,
the `$TMPDIR` size and mount, memory, package versions, the git commit and the `AXB_*` settings to the top of the job log.

## One-time setup

```bash
mkdir -p /u/nederhof/git /u/nederhof/arctic-xbeach-runs/logs
git clone -b linux-deployment https://github.com/deltares-research/arctic-xbeach.git /u/nederhof/git/arctic-xbeach
bash /u/nederhof/git/arctic-xbeach/hpc/h7/setup_env.sh
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

One trial on a small partition. A new study enqueues `START_PARAMS` of the driver as trial 0, which is the parameter set of thermal trial 869
of the published study (not trial 490). With dt 90 s and sensor depth offset 0.1 m the desktop gives the objective **1.36571** K
(`N:\arctic-xbeach-runs\v11_runs\h7_desktop_reference`, 14.5 min on one core).

```bash
cd /u/nederhof/arctic-xbeach-runs/logs
sbatch -p 16vcpu --cpus-per-task=4 --time=02:00:00 \
  --export=ALL,AXB_RUN_ROOT=/u/nederhof/arctic-xbeach-runs/thermal_smoke,AXB_N_TRIALS=1,AXB_N_PARALLEL=1,AXB_THERMAL_DT=90,AXB_SENSOR_DEPTH_OFFSET=0.1 \
  /u/nederhof/git/arctic-xbeach/hpc/h7/thermal_calibration.sbatch
```

Check in `axb_thermal_<jobid>.out` and `thermal_smoke/`:

* "Best RMSE" equals the desktop value to about 5 decimals. A difference in the last digit is floating point; more than that is a problem.
* `thermal_smoke/optuna_trial_000000/` has `run.log`, `config.yaml`, `rmse_bias_layers.txt`, `temperature_layers.png` and no `results/results.nc`.
* `thermal_smoke/optuna_journal.log` exists. A trial folder left in `$TMPDIR` after the job would mean the copy-back failed.

## Full run

```bash
cd /u/nederhof/arctic-xbeach-runs/logs
sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/thermal_calibration.sbatch      # 1000 trials, 42 at a time
sbatch /u/nederhof/git/arctic-xbeach/hpc/h7/coupled_calibration.sbatch      # 200 trials, after the thermal study and its best config in thermal_best/
```

The studies are in `thermal_calibration/` and `coupled_calibration/`. Settings such as `AXB_N_TRIALS` are changed with
`sbatch --export=ALL,AXB_N_TRIALS=100 ...`. Do not start two jobs on the same `AXB_RUN_ROOT`: only one driver per study is supported.

## Monitoring

* Queue and start estimate: `squeue -u $USER`, `squeue -u $USER --start`.
* Driver output: `tail -f /u/nederhof/arctic-xbeach-runs/logs/axb_thermal_<jobid>.out`; completed trials: `<RUN_ROOT>/calibration_prints.txt`.
* A running trial lives in the job's `$TMPDIR` on the compute node, not on `/u`, so its `run.log` is not visible from the login node until
  it has finished and been copied back. On the node: `srun --jobid=<jobid> --overlap tail -n 3 $TMPDIR/optuna_trial_000003/run.log` (not tested).
* History of the study (CSV, best-so-far figure, summary), without touching the running job:

```bash
AXB_RUN_ROOT=/u/nederhof/arctic-xbeach-runs/thermal_calibration AXB_STORAGE=journal \
AXB_EXPORT_DIR=/u/nederhof/arctic-xbeach-runs/thermal_calibration/history AXB_OBJECTIVE_LABEL="Ground temperature RMSE (K)" \
python /u/nederhof/git/arctic-xbeach/calibration/export_history.py
```

## Resuming

After a time limit, a node failure or `scancel`, submit the same script again with the same `AXB_RUN_ROOT`. The driver continues the study,
marks the trials that were running as FAIL (the number is in the job log) and runs the trials still missing to reach `AXB_N_TRIALS`.
The trials that were running are lost, because their scratch folders are gone with the job.

## Open checks for the first job

1. **`$TMPDIR` size and type.** TmpDisk is 0 on H7 and `$TMPDIR` is `/var/lib/containers/storage-rootless/tmp/<jobid>`. The job log shows
   `df -h` and `findmnt` of it. A coupled trial needs 0.4 to 0.7 GB, 42 trials about 25 GB. If it is small or on NFS, submit with
   `--export=ALL,AXB_SCRATCH=` (no scratch): the trials then run on `/u` and `results/xb_files` is deleted after scoring.
2. **XBeach as one process under SLURM.** The module binary is MPI-linked; starting it without `mpirun` is untested, and OpenMPI may pick up
   the SLURM environment in each of the 42 launches. Test with one coupled trial (`AXB_N_TRIALS=1,AXB_N_PARALLEL=1`) on a small partition and
   look at the first XBeach call in `run.log`. If it hangs or fails, the launcher has to remove the `SLURM_*` and `PMI*` variables from
   XBeach's environment or call `mpirun -np 1`.
3. **`module` and `conda` in the batch shell.** The sbatch scripts use `#!/bin/bash -l` so that the module command is defined.
4. **The XBeach module.** `ldd` in the job log must show no `not found`.
5. **Journal on `/u`.** The file `optuna_journal.log` should grow during the run; lock errors in the job log would mean that the NFS lock
   method of Optuna does not work there.
