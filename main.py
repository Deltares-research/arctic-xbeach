#!/usr/bin/env python3
"""
==============================================================================
Arctic-XBeach: Coupled Thermo-Morphological Model
==============================================================================

Main simulation driver for Arctic coastal evolution modeling.

This module provides the main execution framework for the Arctic-XBeach model,
which couples thermal permafrost dynamics with coastal morphological evolution
through the XBeach nearshore model. The model simulates:

- Permafrost thermal dynamics with phase change
- Wave-driven coastal erosion and sediment transport  
- Coupling between thaw depth and sediment erodibility
- Climate forcing from ERA5 reanalysis data
- Storm-driven morphological change events

Usage:
    python main.py <run_id> [--no-screen-log]
    where <run_id> is the name of the run configuration folder in ./runs/

License: GPL 3.0 
Repository: https://github.com/deltares-research/arctic-xbeach
==============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import argparse
import logging
import logging.handlers
import matplotlib.pyplot as plt
import gc

# Arctic-XBeach modules
from arctic_xbeach.model import Simulation
from arctic_xbeach.bathymetry import generate_schematized_bathymetry
from arctic_xbeach.miscellaneous import textbox, datetime_from_timestamp

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger("thermo_model")
logger.setLevel(logging.INFO)

def setup_logger(sim, print_to_screen=True):
    """Initialize logging for the simulation.
    
    Args:
        sim: Simulation instance
        print_to_screen: Whether to print log messages to screen
    """
    if logger.handlers:
        return
    
    log_file = os.path.join(sim.cwd, "run.log")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler
    if print_to_screen:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)


def _setup_logging(level: str, run_id: str, log_file: Path | None):
    """Alternative logging setup function (currently unused)."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================
def main(sim, print_to_screen=True):
    """Main simulation driver function.
    
    Orchestrates the complete Arctic-XBeach simulation including initialization,
    time loop with coupled thermal-morphological updates, and finalization.

    Args:
        sim (Simulation): Simulation instance with loaded configuration
        print_to_screen (bool): Whether to display log messages on screen
        
    Returns:
        tuple: Final cross-shore coordinates (xgr, zgr)
    """
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    setup_logger(sim, print_to_screen)
    t_start = time.time()
    logger.info('Initializing Arctic-XBeach')

    config = sim.config
    logger.debug("Successfully read configuration")

    # -------------------------------------------------------------------------
    # Temporal Parameters
    # -------------------------------------------------------------------------
    sim.set_temporal_params(
        config.model.time_start,
        config.model.time_end,
        config.model.timestep
    )
    logger.debug("Successfully set temporal parameters")

    # -------------------------------------------------------------------------
    # Load Forcing Data
    # -------------------------------------------------------------------------
    sim.load_forcing(
        os.path.join(sim.proj_dir, sim.config.data.forcing_data_path)
    )
    logger.debug("Successfully loaded forcing")
    
    sim.initialize_hydro_forcing(
        os.path.join(sim.proj_dir, sim.config.data.storm_data_path),
    )
    logger.debug("Successfully loaded hydrodynamic forcing")
    
    xb_times = sim.timesteps_with_xbeach_active()
    logger.debug("Successfully generated times that we are going to run XBeach")

    # -------------------------------------------------------------------------
    # Grid and Bathymetry Setup
    # -------------------------------------------------------------------------
    if sim.config.bathymetry.with_schematized_bathymetry:
        xgr, zgr = generate_schematized_bathymetry(
            bluff_flat_length=sim.config.bathymetry.bluff_flat_length,
        
            bluff_height=sim.config.bathymetry.bluff_height, 
            bluff_slope=sim.config.bathymetry.bluff_slope,
            
            beach_width=sim.config.bathymetry.beach_width, 
            beach_slope=sim.config.bathymetry.beach_slope,
            
            nearshore_max_depth=sim.config.bathymetry.nearshore_max_depth, 
            nearshore_slope=sim.config.bathymetry.nearshore_slope,
            
            offshore_max_depth=sim.config.bathymetry.offshore_max_depth, 
            offshore_slope=sim.config.bathymetry.offshore_slope,
            
            contintental_flat_width=sim.config.bathymetry.continental_flat_width,
            
            with_artificial=sim.config.bathymetry.with_artificial,
            artificial_max_depth=sim.config.bathymetry.artificial_max_depth,
            artificial_slope=sim.config.bathymetry.artificial_slope,
            
            N=sim.config.bathymetry.N,
            artificial_flat=sim.config.bathymetry.artificial_flat
        )
        np.savetxt("x.grd", xgr)
        np.savetxt("bed.dep", zgr)
        logger.info("Successfully generated schematized bathymetry")
    
    
    # generate initial grid files and save them
    xgr, zgr, ne_layer = sim.generate_initial_grid(
        nx=sim.config.bathymetry.nx if 'nx' in sim.config.bathymetry.keys() else None,
        bathy_path=sim.config.bathymetry.depfile,
        bathy_grid_path=sim.config.bathymetry.xfile
        )
    np.savetxt("x.grd", xgr)
    np.savetxt("bed.dep", zgr)
    logger.debug("Successfully generated grid")
    
    # initialize xbeach module
    sim.initialize_xbeach_module()
    logger.debug("Successfully initialized XBeach module")
    
    # initialize first xbeach timestep
    if sim.config.xbeach.with_xbeach:
        sim.xbeach_times[0] = sim.check_xbeach(0)
    else:
        sim.xbeach_times[0] = 0
    
    # Check R2% criteria
    try:
        return_code, r2_percentage_with_ice, r2_values_wanted1, r2_values_wanted2 = sim.check_r2_criteria()
        if return_code == 1:
            logger.info(f"You are expected to run ~{r2_percentage_with_ice * 100:.2f}% of XBeach simulations")
            logger.info(f"Suggest reducing the threshold to {r2_values_wanted1:.2f} or {r2_values_wanted2:.2f}")
    except Exception as e:
        logger.error(f"Failed to check R2% criteria: {e}")
        return_code = 2

    # initialize thermal model
    sim.initialize_thermal_module()
    logger.debug("Successfully initialized thermal module")
    
    # initialize solar flux calculator
    if sim.config.thermal.with_solar_flux_calculator:
        sim.initialize_solar_flux_calculator(
            sim.config.model.time_zone_diff,
            angle_min=sim.config.thermal.angle_min,
            angle_max=sim.config.thermal.angle_max,
            delta_angle=sim.config.thermal.delta_angle,
            t_start=sim.config.thermal.t_start,
            t_end=sim.config.thermal.t_end,
            )
    logger.debug("Successfully initialized solar flux calculator")
    
    # show CFL values (they have already been checked to be below 0.5)
    logger.debug(f"Current maximum CFL {np.max(sim.cfl_matrix):.4f}")

    # Get spin-up time in years (default 0)
    spinup_years    = getattr(sim.config.model, 'spin_up_time', 0)
    spinup_seconds  = spinup_years * 365.25 * 24 * 3600
    logger.info("Starting Arctic-XBeach")

    # =========================================================================
    # MAIN TIME LOOP
    # =========================================================================
    last_progress_info = -1
    for timestep_id in np.arange(len(sim.T)):
        
        # ---------------------------------------------------------------------
        # Progress Tracking and Logging
        # ---------------------------------------------------------------------
        logger.debug(f"Timestep {timestep_id+1}/{len(sim.T)}")

        if timestep_id > 0:
            elapsed = time.time() - t_start
            avg_step_time = elapsed / timestep_id
            remaining_steps = len(sim.T) - (timestep_id + 1)
            eta_seconds = avg_step_time * remaining_steps
            progress_pct = int(((timestep_id + 1) / len(sim.T)) * 100)
            
            if progress_pct % 1 == 0 and progress_pct != last_progress_info:
                eta_hours = eta_seconds / 3600
                if eta_hours < 1:
                    logger.info(f"Progress {progress_pct}% | avg_step={avg_step_time:.1f}s | "
                              f"{sim.timestamps[timestep_id]} | ETA ~ {eta_hours * 60:.2f}min")
                else:
                    logger.info(f"Progress {progress_pct}% | avg_step={avg_step_time:.1f}s | "
                              f"{sim.timestamps[timestep_id]} | ETA ~ {eta_hours:.2f}h")
                last_progress_info = progress_pct
            
        # ---------------------------------------------------------------------
        # XBeach Decision and Execution
        # ---------------------------------------------------------------------
        elapsed_sim_seconds = (sim.timestamps[timestep_id] - sim.timestamps[0]) / pd.Timedelta("1s")

        # Check spin-up period
        if elapsed_sim_seconds < spinup_seconds:
            sim.xbeach_times[timestep_id] = 0
            logger.debug(f"Spin-up period for {sim.timestamps[timestep_id]} - skipping XBeach")
        else:
            if sim.config.xbeach.with_xbeach:
                sim.xbeach_times[timestep_id] = sim.check_xbeach(timestep_id)
            else:
                sim.xbeach_times[timestep_id] = 0

        # Execute XBeach if conditions are met
        if sim.xbeach_times[timestep_id] and sim.config.xbeach.with_xbeach:
            sim.write_ne_layer()
            sim.xbeach_setup(timestep_id)
            logger.debug(f"Starting XBeach for timestep {sim.timestamps[timestep_id]}")
            
            run_successful = sim.start_xbeach(
                os.path.join(sim.proj_dir, Path(sim.config.xbeach.version)),
                sim.cwd,
                timestep_id=timestep_id
            )
            
            try:
                if run_successful:
                    logger.debug(f"Successfully ran XBeach for timestep {sim.timestamps[timestep_id]} "
                               f"to {sim.timestamps[timestep_id+1]}")
                else:
                    logger.error(f"Failed to run XBeach for timestep {sim.timestamps[timestep_id]} "
                               f"to {sim.timestamps[timestep_id+1]}")
            except IndexError:
                logger.info(f"XBeach ran successfully for final timestep ({sim.timestamps[timestep_id]})")
            
            # Prepare for next timestep
            if timestep_id + 1 < len(sim.T):
                if sim.config.xbeach.with_xbeach:
                    sim.xbeach_times[timestep_id + 1] = sim.check_xbeach(timestep_id + 1)
                else:
                    sim.xbeach_times[timestep_id + 1] = 0
            
            # Update grid with new morphology
            sim.update_grid(timestep_id, fp_xbeach_output="xboutput.nc")

        # ---------------------------------------------------------------------
        # Thermal Model Updates
        # ---------------------------------------------------------------------
        for subgrid_timestep_id in np.arange(0, config.model.timestep * 3600, config.thermal.dt):
            sim.thermal_update(timestep_id, subgrid_timestep_id)
        sim.find_thaw_depth()

        # ---------------------------------------------------------------------
        # Output Generation
        # ---------------------------------------------------------------------
        if timestep_id in sim.temp_output_ids:
            sim.write_output(timestep_id, t_start)
            logger.debug("Successfully generated output")

        # Optional ground temperature validation output
        if 'save_ground_temp_layers' in sim.config.output.keys():
            sim.save_ground_temp_layers_in_memory(
                timestep_id, 
                layers=sim.config.output.save_ground_temp_layers,
                heat_fluxes=sim.config.output.heat_fluxes,
                write=(timestep_id == np.arange(len(sim.T))[-1]),
            )

        # Free memory
        #if timestep_id % 10 == 0:
        #    gc.collect()

    # =========================================================================
    # SIMULATION FINALIZATION
    # =========================================================================
    logger.info('Arctic-XBeach simulation completed!')
    logger.info(f"Simulation started at: {datetime_from_timestamp(t_start)}")
    logger.info(f"Simulation finished at: {datetime_from_timestamp(time.time())}")
    logger.info(f"Total simulation time: {(time.time() - t_start) / 1:.1f} seconds")
    logger.info(f"Total simulation time: {(time.time() - t_start) / 60:.1f} minutes")
    logger.info(f"Total simulation time: {(time.time() - t_start) / 3600:.1f} hours")
    
    # Cleanup and finalization
    if sim.nc_writer is not None:
        sim.nc_writer.close()

    sim.cleanup()

    return sim.xgr, sim.zgr


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
if __name__ == '__main__':
    """
    Command line execution entry point.
    
    Usage:
        python main.py <run_id> [--no-screen-log]
        
    Example:
        python main.py validation_run_001
    """
    
    # -------------------------------------------------------------------------
    # Argument Parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Run the Arctic-XBeach coupled thermo-morphological simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
            Examples:
                python main.py validation_run_001
                python main.py "d:\Git\thermo-morphological-model\runs\20250922_coupled_runs\run001_val_per2_2_short_v2"
                python main.py storm_scenario_2023 --no-screen-log
                    """)
    parser.add_argument("runid", help="Run ID (folder name in ./runs/ directory)")
    parser.add_argument("--no-screen-log", action="store_true", 
                       help="Disable logging to console (file logging only)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------------
    # Reduce IPython cache if available (memory optimization)
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            ipython.Completer.cache_size = 5
    except ImportError:
        pass  # IPython not available

    # Ensure project root is in Python path
    proj_dir = Path(__file__).parent.resolve()
    if str(proj_dir) not in sys.path:
        sys.path.insert(0, str(proj_dir))

    # -------------------------------------------------------------------------
    # Simulation Execution
    # -------------------------------------------------------------------------
    try:
        sim = Simulation(args.runid, proj_dir=proj_dir)
        main(sim, print_to_screen=not args.no_screen_log)
    except Exception as e:
        print(f"ERROR: Simulation failed with exception: {e}")
        sys.exit(1)