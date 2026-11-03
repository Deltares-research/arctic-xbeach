# Calibration drivers

Two Optuna drivers calibrate the model in two steps, as in the published studies. Each trial runs `main.py` in a child process of the Python
interpreter that started the driver (activate the environment first, no `.bat` needed), so the drivers run on Windows and Linux.
Paths, number of trials, parallel workers, sampler seed and the starting parameter set are variables in the `# === CONFIGURATION ===`
block at the top of each script. Nothing is launched by importing or by reading these files; start a calibration with
`python calibration/<script>.py`.

## Step 1: thermal-only, `run_thermal_calibration.py`

Calibrates the ground thermal parameters against the borehole temperatures (BI_T-1, 17 depths between 0 and 4.25 m) with XBeach switched off.

* Sampled parameters: porosity (`nb_min`, `nb_increase`), transition depths (`transition_top`, `transition_bottom`), frozen and unfrozen
  conductivity (`k_soil_frozen_min`, `k_soil_unfrozen_min`, `k_compaction_factor`), heat capacity (`c_soil_frozen`, `c_ratio`) and
  `geothermal_gradient`. Derived values (`nb_max`, `*_max` conductivities, `c_soil_unfrozen`, `depth_constant_k`) follow from them.
* Objective: mean over the 17 depths of the RMSE between modelled and measured temperature (degC); x-index 1 of the thermal grid. If
  the surface temperature never exceeds 0.01 degC the RMSE is multiplied by 10. A failed run returns `inf`.
* Template: `BASE_SIM_DIR` holds `config.yaml`, `x.grd`, `bed.dep`, `ne_layer.txt` (the published `base/` folder works). Each trial copies it.
* Output in `RUN_ROOT_DIR`: `optuna_study.db` (or `optuna_journal.log`), one `optuna_trial_NNNNNN/` per trial (`config.yaml`, `rmse_bias_layers.txt`,
  `temperature_layers.png`), `calibration_prints.txt`, `calibration_results.csv`, `convergence.png`.
* Result of this step: the `config.yaml` of the best trial. Its `thermal:` section is the input of step 2.

## Step 2: coupled, `auto_calibrate_coupled_model.py`

Calibrates the XBeach and wrapper parameters (`facua` = `facSk` = `facAs`, `bedfriccoef`, `wetslp`, `dryslp`, `xb_threshold`) against
the observed bluff edge and shoreline positions with the thermal parameters fixed.

* Set `THERMAL_CONFIG_YAML` to the best `config.yaml` of step 1 (or leave `None` to keep the thermal section of the template).
* Objective: combined RMSE `sqrt((RMSE_bluff^2 + RMSE_shoreline^2) / 2)` (m) over 2012-01-01 to 2016-01-01; model and observed positions
  are both normalised at 2012-07-11. Bluff edge: first crossing of z = 13.2 m; shoreline: first crossing of z = +2.0 m; 1-day smoothing.
* The 10-year thermal spin-up is computed once. The driver builds `RUN_ROOT_DIR/template`, runs it without XBeach to the end of the
  spin-up, and saves `RUN_ROOT_DIR/spinup_state.npz` (`model.spinup_state_out`). Every trial loads it (`model.spinup_state_in`), so a trial
  only runs the XBeach period. The state file stores a hash of the thermal config, the start time, spin-up length, forcing during the spin-up,
  the water levels and waves (storm table) during the spin-up and the initial grid; a trial refuses to start if any differ. Delete `spinup_state.npz` after changing thermal parameters, solver options,
  forcing or bathymetry. `results.nc` of a trial then contains the post spin-up period only.
* Output in `RUN_ROOT_DIR`: `template/`, `_spinup/`, `spinup_state.npz`, `optuna_study.db` (or `optuna_journal.log`), trial folders (`metrics.txt`,
  `calibration_timeseries.png`, `xbeach_calls.csv`, `conservation_log.csv` in `results/`), `calibration_prints.txt`, `calibration_results.csv`, `convergence.png`.

## v1.1 settings in both drivers

* `SOLVER_OPTIONS = dict(conduction_form="flux", latent_heat_form="volumetric")` and `ERA5_FLUX_SHIFT_HOURS = 1` are written into every
  trial config. Set them to `nodal` / `per_mass` / `0` for the legacy (v1.0) solver. Use the same options in step 1 and step 2.
* `SEED` seeds `optuna.samplers.TPESampler`. With `N_PARALLEL > 1` the trials finish in a varying order, so the sequence of suggestions is
  not exactly reproducible; `N_PARALLEL = 1` is.
* `START_PARAMS` enqueues one parameter set as trial 0 of a new study (ignored when the study already has trials). The published best
  sets are given as comments in the scripts. For step 1 that set is the published thermal trial 869 config; with the flux and volumetric
  options it is a starting point, not the optimum; `c_ratio` of that set is close to the upper bound (2.0) of its range.
* Resuming: run the script again with the same `RUN_ROOT_DIR`; `load_if_exists` continues the study and the driver runs the trials still
  missing to reach `N_TRIALS` completed ones. With journal storage (the cluster), trials that were `RUNNING` when a job was killed are marked
  `FAIL` at the start of the next run (the number is printed); with sqlite (the desktop default) they are left as they are. This assumes that
  **only one driver process per study** runs at a time: a second driver on the same study would have its live trials marked as failed
  (the sbatch files use `--dependency=singleton`).

## Running on Linux / HPC

Both drivers read the variables below from the environment; the value in the `# === CONFIGURATION ===` block is the default, so on the
desktop nothing needs to be set. `hpc/h7/*.sbatch` sets them for the Deltares H7 cluster (see `hpc/h7/README_H7.md`).

| Variable | Driver | Meaning (default = desktop value) |
|---|---|---|
| `AXB_BASE_SIM_DIR` | both | template run folder (`config.yaml`, `x.grd`, `bed.dep`, `ne_layer.txt`) |
| `AXB_RUN_ROOT` | both | output folder: study, trial folders, logs |
| `AXB_OBS_CSV` | thermal | borehole record `BI_T-1_processed.csv` |
| `AXB_BLUFF_EDGES_CSV`, `AXB_SHORELINES_CSV` | coupled | observed bluff edge and shoreline positions (read when the script is imported) |
| `AXB_THERMAL_CONFIG_YAML` | coupled | `config.yaml` of the best thermal trial |
| `AXB_FORCING_CSV`, `AXB_STORM_CSV` | both | ERA5 forcing and storm table, written into every trial config |
| `AXB_XBEACH_EXE` | coupled | path of the XBeach executable written into every config (empty = keep the base config's) |
| `AXB_N_TRIALS`, `AXB_N_PARALLEL` | both | completed trials to reach, trials at the same time |
| `AXB_MAX_FAILED_XBEACH_CALLS` | coupled | a trial with more failed XBeach calls (rows with `success` 0 in `results/xbeach_calls.csv`) than this gets `inf` (0); the count is the trial attribute `n_failed_xbeach_calls` |
| `AXB_TIME_END` | coupled | `DD-MM-YYYY`, replaces `model.time_end` of the template (short smoke test; unset = keep it) |
| `AXB_STUDY_NAME` | both | Optuna study name (`calibration_optuna`) |
| `AXB_THERMAL_DT` | both | thermal time step [s] (thermal 360, coupled 90) |
| `AXB_SENSOR_DEPTH_OFFSET` | thermal | [m] added to the sensor depths (0.0) |
| `AXB_STORAGE` | both, `export_history.py` | `sqlite` (default, `optuna_study.db`) or `journal` (`optuna_journal.log`) |
| `AXB_SCRATCH` | both | empty = trials run in `AXB_RUN_ROOT`. A folder = each trial runs in `<scratch>/optuna_trial_NNNNNN` and is copied back |
| `AXB_KEEP_XB_FILES` | both | `1` (default) keeps `results/xb_files` of a trial; `0` deletes it after scoring (without scratch) |
| `AXB_CONSTANT_LIAR` | both | `1` = TPE `constant_liar`, `0` (default) = off |

* **Journal storage.** SQLite file locking is not reliable on NFS (`/u` on H7). `AXB_STORAGE=journal` keeps the study in one append-only
  file, `RUN_ROOT/optuna_journal.log`, opened with Optuna's `JournalFileOpenLock`, which the Optuna documentation gives as safe on NFS.
  `export_history.py` reads such a study with `AXB_RUN_ROOT` (the run folder), `AXB_STORAGE=journal` and optionally `AXB_STUDY_NAME`,
  `AXB_EXPORT_DIR` and `AXB_OBJECTIVE_LABEL`.
* **Scratch.** A coupled trial writes about 26,000 small files (the XBeach call folders), which is slow and hard on NFS. With `AXB_SCRATCH`
  set to a local disk the trial runs there; afterwards its folder is copied to `RUN_ROOT` (also after a failed run, because `run.log` is
  needed) without `results/xb_files` and, unless it is kept, `results/results.nc`, and the scratch copy is deleted. The trial attribute
  `sim_dir` is the `RUN_ROOT` path. The coupled template and `spinup_state.npz` stay in `RUN_ROOT`.
* **Constant liar.** With about 40 trials running at once, TPE would otherwise propose near-duplicates of the trials that have no value yet.
* **Threads.** The drivers set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless they are already set; the trial
  processes inherit them.
* **Log.** Appends to `calibration_prints.txt` go through one lock (the trials are threads of the driver). The helpers shared by both
  drivers are in `optuna_hpc.py`.

## `export_history.py`

Reads one or more `optuna_study.db` (copied to a temporary folder first, opened read-only with `sqlite3`) or a journal study (see above) and writes, per study, a trials
CSV (value, times, parameters, best-so-far), a best-so-far figure and a text summary (number of trials, best value and trial, trial and number of
completed trials at which the best-so-far came within 1 % and 0.1 % of the final best). The outputs for the two published studies are in
`published_history/`.

## What differs from the drivers used for the published studies (`_scripts/` on the run drive)

The objective, parameter ranges, penalty and metrics are unchanged in definition, but the code now behaves differently in three ways, so the
published reference values (thermal trial 869: 1.0589 K, trial 490: 0.8203 K; coupled trial 88: 4.1222 m) are not reproduced exactly:
output time labels are at the end of the model step (thermal objective: each observation is paired with the state one step earlier than before;
trial 490 re-scored 0.82024 K to 0.82011 K); the coupled series include the XBeach records of storm calls on an inter-storm step
(`xbeach_times` 2) and no longer include failed calls; and a nonzero exit code or a results.nc that stops early is not scored (`inf`). Removed: the unused `sample_params`, `apply_parameter_constraints`
and `worker` functions; the `.bat`/conda launch (replaced by `sys.executable`); the hard-coded `d:\` and `e:\` paths; the geopandas reading of two
small CSV files (plain pandas, same columns and casts). Added: seed, starting set, spin-up restart, v1.1 options, error messages for failed scoring.
