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
* Output in `RUN_ROOT_DIR`: `optuna_study.db`, one `optuna_trial_NNNNNN/` per trial (`config.yaml`, `rmse_bias_layers.txt`,
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
* Output in `RUN_ROOT_DIR`: `template/`, `_spinup/`, `spinup_state.npz`, `optuna_study.db`, trial folders (`metrics.txt`,
  `calibration_timeseries.png`, `xbeach_calls.csv`, `conservation_log.csv` in `results/`), `calibration_prints.txt`, `calibration_results.csv`, `convergence.png`.

## v1.1 settings in both drivers

* `SOLVER_OPTIONS = dict(conduction_form="flux", latent_heat_form="volumetric")` and `ERA5_FLUX_SHIFT_HOURS = 1` are written into every
  trial config. Set them to `nodal` / `per_mass` / `0` for the legacy (v1.0) solver. Use the same options in step 1 and step 2.
* `SEED` seeds `optuna.samplers.TPESampler`. With `N_PARALLEL > 1` the trials finish in a varying order, so the sequence of suggestions is
  not exactly reproducible; `N_PARALLEL = 1` is.
* `START_PARAMS` enqueues one parameter set as trial 0 of a new study (ignored when the study already has trials). The published best
  sets are given as comments in the scripts. For step 1 that set is the published thermal trial 869 config; with the flux and volumetric
  options it is a starting point, not the optimum; `c_ratio` of that set is close to the upper bound (2.0) of its range.
* Resuming: run the script again with the same `RUN_ROOT_DIR`; `load_if_exists` continues the study and `N_TRIALS` more trials are run.
  Trials that were running when a study was stopped stay as `RUNNING` in the database.

## `export_history.py`

Reads one or more `optuna_study.db` (copied to a temporary folder first, opened read-only with `sqlite3`) and writes, per study, a trials
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
