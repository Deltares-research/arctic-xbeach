#!/usr/bin/env python3
"""
==============================================================================
Arctic-XBeach Model Core: Simulation Class Implementation
==============================================================================

Core implementation of the Arctic-XBeach coupled thermo-morphological model.

This module contains the Simulation class which orchestrates the coupling 
between thermal permafrost dynamics and coastal morphological evolution. 
The model integrates:

Classes:
    Simulation: Main model orchestrator
    NCAppender: NetCDF output handler with incremental writing

License: GPL 3.0 
Repository: https://github.com/deltares-research/arctic-xbeach

==============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
from pathlib import Path
import shutil
import time
import yaml
from datetime import datetime
import math
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from arctic_xbeach.version import __version__, git_revision, options_summary
from scipy.interpolate import interp1d
import xarray as xr
from netCDF4 import Dataset
import arctic_xbeach.miscellaneous as um
import arctic_xbeach.diagnostics as dg
import hashlib
import json
import subprocess
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)  # module-scoped logger

# XBeach modules
import xbTools
from xbTools.grid.creation import xgrid
from xbTools.xbeachtools import XBeachModelSetup

# =============================================================================
# SIMULATION CLASS
# =============================================================================
class Simulation():
    """
    Main Arctic-XBeach simulation orchestrator.
    
    This class manages the coupled thermo-morphological simulation by coordinating
    between the thermal permafrost model and XBeach morphological model. It handles
    initialization, time stepping, data I/O, and model coupling.
    
    Attributes:
        runid (str): Unique identifier for the simulation run
        config_file (str): Path to YAML configuration file
        proj_dir (str): Project root directory path
        cwd (str): Current working directory for the run
        result_dir (str): Output directory for results
        
    Methods:
        Main workflow:
            - __init__(): Initialize simulation instance
            - set_temporal_params(): Configure time parameters
            - load_forcing(): Load climate forcing data
            - initialize_*_module(): Setup model components
            - thermal_update(): Advance thermal solution
            - check_xbeach(): Determine if morphological update needed
            - write_output(): Save results to NetCDF
    """
    
    # Run diagnostics collector (arctic_xbeach.diagnostics.RunDiagnostics), set by main(); None = no diagnostics.
    diag = None

    # =========================================================================
    # INITIALIZATION AND SETUP
    # =========================================================================
    def __init__(self, runid, config_file="config.yaml", proj_dir=None):
        """
        Initialize the Arctic-XBeach simulation.
        
        Args:
            runid (str): Run identifier (folder name in runs directory)
            config_file (str): Configuration file name (default: "config.yaml")
            proj_dir (str): Project root directory (auto-detected if None)
            
        Note:
            The runid should correspond to a folder containing the configuration
            file and input data for the simulation.
        """
        # Store basic parameters
        self.runid = runid
        self.config_file = config_file
        self.proj_dir = str(Path(proj_dir).resolve()) if proj_dir else str(Path(__file__).resolve().parents[1])

        # Initialize simulation
        self.read_config(config_file)
        self._set_directory()

        # Initialize output handling
        self.nc_writer = None
        self._t0_seconds = None  # Base for "time" coordinate

        # Log initialization
        logger.info("Initialized Simulation run_id=%s proj_dir=%s", self.runid, self.proj_dir)

    def __repr__(self) -> str:
        """Return string representation of the simulation."""
        description = (
            f"RUNID: {self.runid}\n"
            f"PROJECT DIRECTORY: {self.proj_dir}\n"
            f"CURRENT WORKING DIRECTORY: {self.cwd}\n"
            f"TIMESERIES DIRECTORY: {self.ts_dir}\n"
            f"RESULTS DIRECTORY: {self.result_dir}\n"
        )
        return description
    
    def _set_directory(self):
        """
        Set up directory structure for the simulation.
        
        Creates working directory, timeseries directory, and results directory
        based on configuration settings. Changes working directory to run folder.
        """
        # Set up directory paths
        self.cwd = os.path.join(self.proj_dir, self.runid)
        self.ts_dir = os.path.join(self.proj_dir, "database/ts_datasets/")
        
        # Change to run directory
        os.chdir(self.cwd)    

        # Configure output location
        if self.config.output.use_default_output_path:
            self.result_dir = os.path.join(self.cwd, "results/")
        else:
            self.result_dir = os.path.join(Path(self.config.output.output_path), self.runid)
            
        # Create output directory if needed
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
    
    def read_config(self, config_file):
        """
        Load configuration from YAML file.
        
        Args:
            config_file (str): Name of YAML configuration file
            
        Returns:
            AttrDict: Configuration dictionary with attribute access
        """
        class AttrDict(dict):
            """Dictionary subclass that allows attribute-style access."""
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
                    
        cwd = os.path.join(self.proj_dir, self.runid)
                    
        with open(os.path.join(cwd, config_file)) as f:
            cfg = yaml.safe_load(f)
            
        self.config = AttrDict(cfg)
                
        for key in cfg:
            self.config[key] = AttrDict(cfg[key])
               
        return self.config
        
    def set_temporal_params(self, t_start, t_end, dt):
        """
        Configure temporal parameters for the simulation.
        
        Args:
            t_start (str): Start time (e.g., "2010-01-01")
            t_end (str): End time (e.g., "2010-12-31") 
            dt (int): Time step in hours
            
        Note:
            Creates timestamps array and output timing configuration.
        """
        # Store time parameters
        self.dt = dt
        self.t_start = pd.to_datetime(t_start, dayfirst=True)
        self.t_end = pd.to_datetime(t_end, dayfirst=True)
        
        # Create timestamp array
        unrepeated_timestamps = pd.date_range(
            start=self.t_start, 
            end=self.t_end, 
            freq=f'{self.dt}h', 
            inclusive='left'
        )
        
        self.timestamps = unrepeated_timestamps
        
        # Create time indexing array for numerical model
        self.T = np.arange(0, len(self.timestamps), 1) 
        
        # Define output timing. Optional output.output_start / output.output_end (dates, as model.time_start) limit the
        # output to a window, e.g. the observation period of a thermal calibration; the default is the whole run.
        self.temp_output_ids = np.arange(0, len(self.timestamps), self.config.output.output_res)
        out_start, out_end = self.config.output.get('output_start'), self.config.output.get('output_end')
        if out_start is not None or out_end is not None:
            t_out = self.timestamps[self.temp_output_ids]
            keep = np.ones(len(t_out), dtype=bool)
            if out_start is not None:
                keep &= t_out >= pd.to_datetime(out_start, dayfirst=True)
            if out_end is not None:
                keep &= t_out < pd.to_datetime(out_end, dayfirst=True)
            self.temp_output_ids = self.temp_output_ids[keep]

    # -------------------------------------------------------------------------
    # Grid and Bathymetry Setup
    # -------------------------------------------------------------------------
    def generate_initial_grid(self, nx=None, len_x=None, bathy_path=None, bathy_grid_path=None):
        """
        Generate initial computational grid and load bathymetry.

        Args:
            nx (int, optional): Number of grid points in x-direction
            len_x (float, optional): Total length of grid in x-direction 
            bathy_path (str, optional): Path to bathymetry file (uses bed.dep if None)
            bathy_grid_path (str, optional): Path to grid file (uses x.grd if None)
            
        Returns:
            tuple: (xgr, zgr, thaw_depth) - grid coordinates, bathymetry, thaw depth
            
        Note:
            Currently implements 1D grid only. Sets up grid coordinates and
            interpolates bathymetry to grid points.
        """
        # Load initial bathymetry and the accompanying grid
        if bathy_path:
            self._load_bathy(
                os.path.join(self.cwd, bathy_path)
            )
        else:
            self._load_bathy(os.path.join(self.cwd, "bed.dep"))
            
        if bathy_grid_path:
            self._load_grid_bathy(
                os.path.join(self.cwd, bathy_grid_path)
            )
        else:
            self._load_grid_bathy(os.path.join(self.cwd, "x.grd"))
        
        # Check whether to use XBeach generated xgrid or generate uniform grid
        if nx and self.config.bathymetry.with_nx:
            self.xgr = np.linspace(min(self.bathy_grid), max(self.bathy_grid), nx)
        else:
            # Transform into a more suitable grid for XBeach
            self.xgr, self.zgr = xgrid(self.bathy_grid, self.bathy_initial, dxmin=2, ppwl=self.config.bathymetry.ppwl)
        
        # Interpolate bathymetry to grid
        self.zgr = np.interp(self.xgr, self.bathy_grid, self.bathy_initial)
        
        # Initialize thaw depth array (non-erodible layer)
        self.thaw_depth = np.zeros(self.xgr.shape)
        
        # Store grid origin
        self.x_ori = np.min(self.xgr)
        
        return self.xgr, self.zgr, self.thaw_depth

    
    def _load_bathy(self, fp_initial_bathy):
        """
        Load the initial bathymetry data.
        
        Args:
            fp_initial_bathy (str): Filepath to the initial bathymetry file
        """
        with open(fp_initial_bathy) as f:
            self.bathy_initial = np.loadtxt(f)
            
    def _load_grid_bathy(self, fp_bathy_grid):
        """
        Load the initial bathymetry x-grid coordinates.
        
        Args:
            fp_bathy_grid (str): Filepath to the initial bathymetry x-grid file
        """
        with open(fp_bathy_grid) as f:
            self.bathy_grid = np.loadtxt(f)

    # -------------------------------------------------------------------------
    # Forcing Data Management
    # -------------------------------------------------------------------------
            
    def load_forcing(self, fpath):
        """
        Load forcing data and make it an attribute of the simulation instance.
        
        Args:
            fpath (str): Path to forcing data file
        """
        # Read in forcing conditions
        # data.era5_flux_shift_hours: 0 = no shift (flux label applied as given), 1 = shift to the start of the ERA5 averaging hour
        self.forcing_data = self._get_timeseries(
            self.t_start, self.t_end, fpath,
            flux_shift_hours=self.config.data.get('era5_flux_shift_hours', 0)
            )
        
        # Check if conceptual mode is enabled
        use_conceptual = getattr(self.config.data, 'conceptual', False)
        
        if use_conceptual:
            # Conceptual mode: constant temperature and zero radiation/latent heat
            conceptual_temp = getattr(self.config.data, 'conceptual_temp', 270.0)
            logger.info(f"Conceptual mode enabled: using constant temperature ({conceptual_temp}K) and zero radiation fluxes")
            self.forcing_data["2m_temperature"] = conceptual_temp  # Constant air temperature
            self.forcing_data["sea_surface_temperature"] = conceptual_temp  # Constant sea temperature
            self.forcing_data["mean_surface_latent_heat_flux"] = 0.0
            self.forcing_data["mean_surface_net_long_wave_radiation_flux"] = 0.0
            self.forcing_data["mean_surface_net_short_wave_radiation_flux"] = 0.0
            self.forcing_data["sea_ice_cover"] = 1.0  # Full sea ice cover to prevent XBeach
        else:
            # add terms or factors for thermodynamic part of sensitivity analysis
            self.forcing_data["mean_surface_latent_heat_flux"] = self.forcing_data['mean_surface_latent_heat_flux'] * \
                (1 if "sensitivity" not in self.config.keys() else self.config.sensitivity.factor_latent_heat_flux)
            self.forcing_data["mean_surface_net_long_wave_radiation_flux"] = self.forcing_data['mean_surface_net_long_wave_radiation_flux'] * \
                (1 if "sensitivity" not in self.config.keys() else self.config.sensitivity.factor_longwave_heat_flux) 
            self.forcing_data["mean_surface_net_short_wave_radiation_flux"] = self.forcing_data['mean_surface_net_short_wave_radiation_flux'] * \
                (1 if "sensitivity" not in self.config.keys() else self.config.sensitivity.factor_shortwave_heat_flux)
            self.forcing_data["sea_surface_temperature"] = self.forcing_data['sea_surface_temperature'] + \
                (0 if "sensitivity" not in self.config.keys() else self.config.sensitivity.term_sea_temperature)
            self.forcing_data["2m_temperature"] = self.forcing_data['2m_temperature'] + \
                (0 if "sensitivity" not in self.config.keys() else self.config.sensitivity.term_2m_air_temperature)
        
        return None
    
    def _ensure_nc_writer(self):
        if self.nc_writer is not None:
            return
        # choose an xbeach x-grid. If none yet, use current wet part of xgr.
        xgr_xb = None
        xb_path = os.path.join(self.cwd, "xboutput.nc")
        if os.path.exists(xb_path):
            try:
                _ds = xr.load_dataset(xb_path).squeeze()
                xgr_xb = _ds.x.values
                _ds.close()
            except Exception:
                pass
        if xgr_xb is None or len(xgr_xb) == 0:
            xgr_xb = self.xgr
        depth_id = np.arange(self.config.thermal.grid_resolution, dtype=np.int32)

        out_nc = os.path.join(self.result_dir, "results.nc")
        tstart_str = str(self.timestamps[0])
        self.nc_writer = NCAppender(out_nc, self.xgr, depth_id, xgr_xb.astype("f4"), tstart=tstart_str)
        self.nc_writer.ds.setncattr("Solver", options_summary(self.config))
        self._t0_seconds = float((self.timestamps[0] - pd.Timestamp(self.timestamps[0])) / pd.Timedelta("1s"))

    
    ################################################
    ##                                            ##
    ##            # XBEACH FUNCTIONS              ##
    ##                                            ##
    ################################################
    
    def initialize_xbeach_module(self):
        """This method initializes the xbeach module. Currently only used to set values for the first output at t=0.

        Returns:
            None
        """       
        # first get the correct forcing timestep
        row = self.forcing_data.iloc[0]
        
        # with associated forcing values
        self.current_air_temp = row["2m_temperature"]  # also used in output
        self.current_sea_temp = row['sea_surface_temperature']  # also used in output
        self.current_sea_ice = row["sea_ice_cover"]  # not used in this function, but loaded in preperation for output
        
        self.wind_direction, self.wind_velocity = self._get_wind_conditions(timestep_id=0)
        
        self.water_level = 0
        self.xb_check = 0
        
        self.beta_f = np.zeros(self.T.shape)
        self.R2 = np.zeros(self.T.shape)
        
        self.storm_write_counter = 0
        
        return None
    
    def initialize_hydro_forcing(self, fp_storm):
        """This function is used to initialize hydrodynamic forcing conditions, and what the conditions are. It uses either 
        'database/ts_datasets/storms_erikson.csv' (for cmip hindcast) or 'database/ts_datasets/storms_engelstad.csv' 
        (for era5 hindcast).

        Args:
            fp_storm (Path): path to storm dataset

        Returns:
            array: array of length T that for each timestep contains hydrodynamic forcing
        """
        # Initialize conditions array
        self.conditions = np.zeros(self.T.shape, dtype=object)  # also directly read wave conditions here
        
        # Check if conceptual mode is enabled
        use_conceptual = getattr(self.config.data, 'conceptual', False)
        
        # Initialize zero conditions (minimal waves to prevent XBeach activation)
        if use_conceptual:
            self.zero_conditions = {
                        "Hso(m)": 0.01,  # Very small wave height
                        "Hs(m)": 0.01,  # Very small wave height
                        "Dp(deg)": 270,                    
                        "Tp(s)": 2,
                        "Hindcast_or_projection": 0,
                        }
        else:
            self.zero_conditions = {
                        "Hso(m)": 0.05,  # placeholder
                        "Hs(m)": 0.05,  # placeholder
                        "Dp(deg)": 270,                    
                        "Tp(s)": 2,
                        "Hindcast_or_projection": 0,
                        }
        
        # read file and mask out correct timespan
        with open(fp_storm) as f:
            df = pd.read_csv(f, parse_dates=['time'])
            mask = (df['time'] >= self.t_start) * (df['time'] <= self.t_end)
            df = df[mask]
        
        # In conceptual mode, override all wave conditions with minimal values
        if use_conceptual:
            logger.info("Conceptual mode enabled: setting minimal wave heights to prevent XBeach activation")
            df['WL(m)'] = 0.0  # Zero water level
            df['Hs(m)'] = 0.01  # Minimal wave height
            df['Tp(s)'] = 2.0  # Short period
        else:
            # add terms or factors for hydrodynamic part of sensitivity analysis
            df['WL(m)'] = df['WL(m)'] + (0 if "sensitivity" not in self.config.keys() else self.config.sensitivity.term_water_level)
            df['Hs(m)'] = df['Hs(m)'] * (1 if "sensitivity" not in self.config.keys() else self.config.sensitivity.factor_wave_height)
            df['Tp(s)'] = df['Tp(s)'] * (1 if "sensitivity" not in self.config.keys() else self.config.sensitivity.factor_wave_period)
        
        # Loop through complete data to save conditions            
        # df_dropna = df.dropna(axis=0)
        
        for i, row in df.iterrows():
            
            index = np.argwhere(self.timestamps==row.time)
                        
            if not row.isnull().values.any():
                            
                # safe storm conditions for this timestep as well            
                self.conditions[index] = {
                        "Hs(m)": row["Hs(m)"],
                        "Dp(deg)": row["Dp(deg)"],
                        "Tp(s)": row["Tp(s)"],
                        "WL(m)": row["WL(m)"],
                            }
                
            else:
                conds = self.zero_conditions
                conds['WL(m)'] = row['WL(m)']
                
                self.conditions[index] = conds
                
        self.water_levels = np.tile(df['WL(m)'].values, 1)
        
        return self.conditions
    
    def timesteps_with_xbeach_active(self):
        """This function gets the timestep ids for which xbeach should be active, without looking at 2% runup threshold yet.

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach should be ran and 0 if not.
        """
        
        # get inter-storm timestep ids
        self.xbeach_inter = self._when_xbeach_inter(self.config.model.call_xbeach_inter)
        
        # get sea-ice timestep ids
        self.xbeach_sea_ice = self._when_xbeach_no_sea_ice(self.config.wrapper.sea_ice_threshold)
        
        # initialize xbeach storms array
        self.xbeach_storms = np.zeros(self.xbeach_inter.shape)

        # initialize xbeach_times array
        self.xbeach_times = np.zeros(self.xbeach_inter.shape)

        return self.xbeach_times
    
    
    def _when_xbeach_inter(self, call_xbeach_inter):
        """This function determines the timestamps that xbeach is called regardless of sea-ice or storms.

        Args:
            call_xbeach_inter (int): Call xbeach every 'call_xbeach_inter' timestamps, regardsless of the presence of sea ice or a storm.

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach should be ran and 0 if not.
        """
        ct = np.zeros(self.T.shape)
        
        # set xbeach active at provided intervals
        ct[::call_xbeach_inter] = 1
        
        return ct
    
    def _when_xbeach_no_sea_ice(self, sea_ice_threshold):
        """This function determines when xbeach should not be ran due to sea ice, based on a threshold value)

        Args:
            sea_ice_threshold (float): _description_

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach can be ran and 0 if not (w.r.t. sea ice).
        """
        it =  (self.forcing_data.sea_ice_cover.values < sea_ice_threshold)
                
        return it
    
    def _when_xbeach_storms(self, timestep_id):
        """This function checks whether or not there is actually a storm during the upcoming each timestep, and is based on a 2% runup threshold

        Args:
            timestep_id (int): current timestep

        Returns:
            int: 1 for storm, 0 for no storm
        """
        # read hydrodynamic conditions for current timestep
        H = self.conditions[timestep_id]['Hs(m)']
        T = self.conditions[timestep_id]['Tp(s)']
        wl = self.conditions[timestep_id]['WL(m)']
        
        # assume these are the deep water conditions
        H0 = H
        L0 = 9.81 * T**2 / (2 * np.pi)
        
        # determine the +- 2*sigma envelope (for the stockdon, 2006 formulation)
        sigma = H0 / 4
        mask = np.nonzero((self.zgr > wl - 2*sigma) * (self.zgr < wl + 2*sigma))
        z_envelope = self.zgr[mask]
        x_envelope = self.xgr[mask]
            
        # if waves are too small, the envelope doesn't exist on the grid, so this method will fail
        try:
            # compute beta_f as the average slope in this envelope
            dz = z_envelope[np.argmax(x_envelope)] - z_envelope[np.argmin(x_envelope)]
            dx = x_envelope[np.argmax(x_envelope)] - x_envelope[np.argmin(x_envelope)]
            
            # if there's only one point in the envelope then dx will be 0, so go ahead with the except block as well
            if dx == 0:
                raise ValueError
        
        # and in that case, the local angle of the two grid points nearest to the water level is used
        except ValueError:
            
            dry_mask = self.zgr > wl
            dry_indices = np.nonzero(dry_mask)
            
            wet_mask = np.ones(dry_mask.shape) - dry_mask
            wet_indices = np.nonzero(wet_mask)
            
            try:
                first_dry_id = np.min(dry_indices)
                last_wet_id = np.max(wet_indices)
                
                x1, z1 = self.xgr[first_dry_id], self.zgr[first_dry_id]
                x2, z2 = self.xgr[last_wet_id], self.zgr[last_wet_id]
            
                dz = z2 - z1
                dx = x2 - x1
                
                # compute beta_f            
                self.beta_f[timestep_id] = np.abs(dz / dx)
            except ValueError:
                # if we are unable to determine beta, simply use a constant of 0.1
                self.beta_f[timestep_id] = 0.1

        # now the empirical formulation by Stockdon et al. (2006) can be used to determine R2%
        self.beta_f[timestep_id] = 0.1  # force constant slope
        self.R2[timestep_id] = 1.1 * (0.35 * self.beta_f[timestep_id] * (H0 * L0)**0.5 + (H0 * L0 * (0.563 * self.beta_f[timestep_id]**2 + 0.004))**0.5 / 2)
        
        # Estimate if we need to run
        run_xb_storm = int(self.R2[timestep_id] + wl > self.config.wrapper.xb_threshold)
        

        return run_xb_storm
    
    def check_xbeach(self, timestep_id):
        """This function checks whether XBeach should be ran for the upcoming timestep.

        Args:
            timestep_id (int): id of the current timestep

        Returns:
            int: whether or not to run XBeach. 1 if yes, 0 if no.
        """
        
        # Check the standard procedure
        self.xbeach_storms[timestep_id] = self._when_xbeach_storms(timestep_id)
        self.xbeach_times[timestep_id]  = self.xbeach_inter[timestep_id] + self.xbeach_sea_ice[timestep_id] * self.xbeach_storms[timestep_id]

        # However, only if there is any thaw depth larger than 0, other not
        if np.all(self.thaw_depth <= 0.0):
            self.xbeach_times[timestep_id] = 0.0

        # Return        
        return self.xbeach_times[timestep_id]
    
    def check_r2_criteria(self):
        
        # Make estimate of the number of times XBeach will be launched
        run_xb_storm = np.zeros(self.T.shape)
        for timestep_id in range(len(self.conditions)):
            # Update R2% using the current bathymetry
            self._when_xbeach_storms(timestep_id)
            run_xb_storm[timestep_id] = int(self.R2[timestep_id] + self.conditions[timestep_id]['WL(m)'] > self.config.wrapper.xb_threshold)
        r2_percentage_with_ice  = np.sum(run_xb_storm[self.xbeach_sea_ice]) / len(self.conditions)
        r2_values_wanted1       = np.quantile(self.R2[self.xbeach_sea_ice],0.90)
        r2_values_wanted2       = np.quantile(self.R2[self.xbeach_sea_ice],0.98)

        # Add a print and logger for R2% analysis
        return_code = 0
        if r2_percentage_with_ice > 0.1:
            return_code = 1

        # Return code
        return return_code, r2_percentage_with_ice, r2_values_wanted1, r2_values_wanted2

    def xbeach_setup(self, timestep_id):
        """
        Initialize an XBeach run by writing all input files.
        
        Args:
            timestep_id (int): Current timestep ID for the simulation
            
        Note:
            Creates destination folder, configures XBeach setup, writes model files,
            and handles hotstart functionality for storm continuation.
        """
        # Create destination folder for XBeach output
        destination_folder = os.path.join(
            self.result_dir, 
            "xb_files/", 
            f"{timestep_id:010d}/"
        )
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # Create instance of XBeachModelSetup
        self.xb_setup = XBeachModelSetup(f"Run {destination_folder}: timestep {timestep_id}")
        
        # Configure grid
        self.xb_setup.set_grid(
            self.xgr, 
            None, 
            self.zgr, 
            posdwn=-1,
            xori=0,
            yori=0,
            dtheta=self.config.xbeach.dtheta,
            thetanaut=self.config.xbeach.thetanaut,
        )
        
        # Determine wave conditions (zero or normal)
        if self.xbeach_inter[timestep_id] and not self.xbeach_storms[timestep_id] * self.xbeach_sea_ice[timestep_id]:
            conditions = self.zero_conditions
            conditions['WL(m)'] = self.water_levels[timestep_id]
        else:
            conditions = self.conditions[timestep_id]
        
        # Configure wave parameters
        self.xb_setup.set_waves('parametric', {
            "Hm0": round(conditions["Hs(m)"], 4),
            "Tp": round(conditions["Tp(s)"], 4),
            'mainang': 270,  # Default value for 1D XBeach
            "gammajsp": 3.3,  # Value recommended by Kees
            "s": 1000,  # Value recommended by Kees
            "duration": self.dt * 3600,
            "dtbc": 60,
            "fnyq": 1,
        })
        
        # Turn off wave model if there is no storm
        if not self.xbeach_storms[timestep_id]:
            self.xb_setup.wbctype = 'off'
        
        # Get wind and water level conditions
        wind_direction, wind_velocity = self._get_wind_conditions(timestep_id)
        wl = self.water_levels[timestep_id]
        tintg = self.dt * 3600
        
        # Configure XBeach parameters
        params = self._get_xbeach_params(
            timestep_id, wind_direction, wind_velocity, wl, tintg
        )
        self.xb_setup.set_params(params)
                
        # Write model setup to destination folder
        self.xb_setup.write_model(destination_folder, figure=False)

        # Handle hotstart and boundary conditions
        self._configure_xbeach_files(timestep_id, destination_folder, wl)
        
        # Copy necessary files
        self._copy_xbeach_files(destination_folder)
        
        return None
    
    def _get_xbeach_params(self, timestep_id, wind_direction, wind_velocity, wl, tintg):
        """
        Generate XBeach parameter dictionary.
        
        Args:
            timestep_id (int): Current timestep ID
            wind_direction (float): Wind direction in degrees
            wind_velocity (float): Wind velocity in m/s
            wl (float): Water level in meters
            tintg (float): Time integration step in seconds
            
        Returns:
            dict: Complete XBeach parameters configuration
        """
        return {
            # Sediment parameters
            "D50": self.config.xbeach.D50,
            "D90": self.config.xbeach.D50 * 1.5,
            "rhos": self.config.xbeach.rho_solid,
            "dryslp": self.config.xbeach.dryslp,
            "wetslp": self.config.xbeach.wetslp,
            
            # Flow boundary condition parameters
            "front": "abs_1d",
            "back": "wall",
            "left": "neumann",
            "right": "neumann",
            
            # Flow parameters
            "facSk": self.config.xbeach.get('facSk', 0.15),
            "facAs": self.config.xbeach.get('facAs', 0.20),
            "facua": self.config.xbeach.get('facua', 0.175),

            # General parameters
            "bedfriccoef": self.config.xbeach.bedfriccoef,
            
            # Model time
            "tstop": self.dt * 3600,
            "CFL": self.config.wrapper.get('CFL_xbeach', 0.95),
            
            # Morphology parameters
            "morfac": 1,
            "morstart": 0,
            "ne_layer": "ne_layer.txt",
            "lsgrad": self.config.xbeach.get('lsgrad', 0),
            
            # Physical constants
            "rho": self.config.xbeach.rho_sea_water,
            
            # Physical processes
            "avalanching": 1,
            "morphology": 1,
            "sedtrans": 1,
            "wind": 1 if self.config.xbeach.with_wind else 0,
            "struct": 1 if self.config.xbeach.with_ne_layer else 0,
            
            "flow": 1 if self.xbeach_storms[timestep_id] else 0,
            "lwave": 1 if self.xbeach_storms[timestep_id] else 0,
            "swave": 1 if self.xbeach_storms[timestep_id] else 0,
            
            # Tide boundary conditions
            "tideloc": 0,
            "zs0": round(wl, 4),

            # Wave boundary conditions
            "instat": self.config.xbeach.wbctype,
            "bcfile": self.config.xbeach.bcfile,
            "wavemodel": "surfbeat",
            
            # Wind boundary conditions
            "windth": wind_direction if self.config.xbeach.with_wind else 0,
            "windv": wind_velocity if self.config.xbeach.with_wind else 0,
            
            # Output variables
            "outputformat": "netcdf",
            "tintg": tintg,
            "tstart": 0
        }
    
    def _configure_xbeach_files(self, timestep_id, destination_folder, wl):
        """
        Configure XBeach parameter file with hotstart and boundary conditions.
        
        Args:
            timestep_id (int): Current timestep ID
            destination_folder (str): Path to destination folder
            wl (float): Water level in meters
        """
        # Read existing params.txt
        with open(os.path.join(destination_folder, 'params.txt'), 'r') as f:
            text = f.readlines()
        
        # Configure hotstart
        writehotstart = 1 if timestep_id + 1 < len(self.xbeach_times) else 0
        self.last_xbeach_hotstart = bool(self.xbeach_times[timestep_id - 1] and self.xbeach_storms[timestep_id - 1] and timestep_id != 0)  # for the call log
        hotstart_text = [
            "%% hotstart (during a storm, use the previous xbeach timestep as hotstart for current timestep)\n\n",
            f"writehotstart  = {writehotstart}\n",
            f"hotstart       = {1 if (self.xbeach_times[timestep_id - 1] and self.xbeach_storms[timestep_id - 1] and timestep_id != 0) else 0}\n",
            f"hotstartfileno = {1 if (self.xbeach_times[timestep_id - 1] and self.xbeach_storms[timestep_id - 1] and timestep_id != 0) else 0}\n",
            "\n"
        ]
        
        # Copy hotstart files from previous run if needed
        if (self.xbeach_times[timestep_id - 1] and self.xbeach_storms[timestep_id - 1] and timestep_id != 0):
            previous_folder = os.path.join(self.result_dir, "xb_files/", f"{timestep_id-1:010d}/")
            
            if os.path.exists(previous_folder):
                for filename in os.listdir(previous_folder):
                    if filename.startswith("hotstart_"):
                        src_file = os.path.join(previous_folder, filename)
                        dst_file = os.path.join(destination_folder, filename)
                        shutil.copy2(src_file, dst_file)

        # Configure boundary conditions for calm conditions
        wbc_ts1_text = ["lwave = 0\n", "swave = 0\n", "flow = 0\n"]
        bc_text = [
            f"%% t (s) eta LF(m)  E (J/m2)\n",
            f"0  {wl}  0\n",
            f"3600  {wl}  0\n",
        ]
        
        # Process params.txt
        new_input_text = []
        output_vars_added = False
        for line in text:
            if not self.xbeach_storms[timestep_id]:
                # Create empty BC file for calm conditions
                bc_dir = os.path.join(destination_folder, 'bc/')
                bc_file = os.path.join(bc_dir, 'gen.ezs')
                
                if not os.path.exists(bc_file):
                    os.makedirs(bc_dir, exist_ok=True)
                    with open(bc_file, 'w') as f:
                        f.writelines(bc_text)
                
                if "wbctype" in line:
                    line = "wbctype = ts_1\n"
                if "swave" in line:
                    line = wbc_ts1_text
            
            # Add hotstart configuration before output section
            if "%% Output variables" in line:
                new_input_text += hotstart_text
            
            # Add nglobalvar after output section header
            if "%% Output variables" in line and not output_vars_added:
                if not isinstance(line, list):
                    line = [line]
                new_input_text += line
                
                # Add nglobalvar output variables
                globalvars = ["x", "y", "zb", "zs", "H", "sedero", 
                             "E", "Sxx", "Sxy", "Syy", "thetamean", "vmag", "urms"]
                new_input_text.append("\n")
                new_input_text.append(f"nglobalvar = {len(globalvars)}\n")
                for var in globalvars:
                    new_input_text.append(f"{var}\n")
                new_input_text.append("\n")
                
                output_vars_added = True
                continue
                
            if not isinstance(line, list):
                line = [line]
            new_input_text += line
        
        # Write updated params.txt
        with open(os.path.join(destination_folder, 'params.txt'), 'w') as f:
            f.writelines(new_input_text)
    
    def _copy_xbeach_files(self, destination_folder):
        """
        Copy necessary files to XBeach destination folder.
        
        Args:
            destination_folder (str): Path to destination folder
        """
        files_to_copy = ['ne_layer.txt']
        for file in files_to_copy:
            src_path = os.path.join(self.cwd, file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, destination_folder)
    
    def start_xbeach(self, xbeach_path, params_path, batch_fname="run.bat", timestep_id=None):
        """
        Running this function starts the XBeach module as a subprocess.
        --------------------------
        xbeach_path: str
            string containing the file path to the xbeach executible from the project directory
        params_path: str
            string containing the file path to the params.txt file from the project directory
        batch_fname: str
            name used for the generated batch file
        timestep_id: int
            timestep ID to determine the destination folder
        --------------------------

        returns boolean (True if process was a sucess, False if not)
        """
        # Determine working directory (destination folder if timestep_id is provided)
        if timestep_id is not None:
            work_dir = os.path.join(self.result_dir, "xb_files/", (10 - len(str(int(timestep_id)))) * '0' + str(int(timestep_id)) + '/')
        else:
            work_dir = self.cwd
            
        with open(os.path.join(work_dir, batch_fname), "w") as f:
            f.write(f'cd "{work_dir}"\n')
            f.write(f'call "{xbeach_path}"')
        
        # Command to run XBeach
        command = [str(os.path.join(work_dir, batch_fname))]

        # # Call XBeach using subprocess
        return_code = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode

        return return_code == 0
    
    def copy_xb_output_to_result_dir(self, timestep_id, fp_xbeach_output="xboutput.nc"):
        
        destination_folder = os.path.join(self.result_dir, "xb_files/", (10 - len(str(int(timestep_id)))) * '0' + str(int(timestep_id)) + '/')
        source_file = os.path.join(destination_folder, fp_xbeach_output)
        destination_file = os.path.join(self.result_dir, f"storm{self.storm_write_counter}.nc")
        
        shutil.copy(source_file, destination_file)
        
        return None
        
    def _forcing_row(self, timestep_id):
        """Forcing of one timestep as a dict, cached: the thermal substeps of a timestep all ask for the same row, and a pandas
        row lookup per substep took about a third of the run time of a thermal-only run. Same values as forcing_data.iloc."""
        if getattr(self, '_forcing_row_id', None) != timestep_id:
            self._forcing_row_cache = self.forcing_data.iloc[timestep_id].to_dict()
            self._forcing_row_id = timestep_id
        return self._forcing_row_cache

    def _get_wind_conditions(self, timestep_id):
        """This function gets the wind conditions from the forcing dataset. Wind direction is defined in degrees
        clockwise from the north (i.e., east = 90 degrees)

        Args:
            timestep_id (int): timestep id for the current timestep for which wind dta is requested

        Returns:
            tuple: (wind_direction, wind_velocity)
        """
        row = self._forcing_row(timestep_id)
        
        u = row["10m_u_component_of_wind"]
        v = row["10m_v_component_of_wind"]
        
        direction = math.atan2(u, v) / (2*np.pi) * 360  # clockwise from the nord
        
        velocity = math.sqrt(u**2 + v**2)
        
        return direction, velocity
    
    ################################################
    ##                                            ##
    ##            # THERMAL FUNCTIONS             ##
    ##                                            ##
    ################################################
    def initialize_thermal_module(self):
        """This function initializes the thermal module of the model.

        Raises:
            ValueError: raised if CFL > config.wrapper.CFL_thermal
        """
        
        # read initial conditions
        ground_temp_distr_dry, ground_temp_distr_wet = self._generate_initial_ground_temperature_distribution(
            self.forcing_data, 
            self.t_start, 
            self.config.thermal.grid_resolution,
            self.config.thermal.max_depth
            )
        
        # save the grid resolution
        self.dz = self.config.thermal.max_depth / (self.config.thermal.grid_resolution - 1) 
               
        # save thermal grid distribution
        self.thermal_zgr = ground_temp_distr_dry[:,0]
                
        # initialize temperature matrix, which is used to keep track of temperatures through the grid
        self.temp_matrix = np.zeros((len(self.xgr), self.config.thermal.grid_resolution))
        
        # initialize the associated grid
        self.abs_xgr, self.abs_zgr = um.generate_perpendicular_grids(
            self.xgr, 
            self.zgr, 
            resolution=self.config.thermal.grid_resolution, 
            max_depth=self.config.thermal.max_depth
            )
        
        # set the above determined initial conditions for the xgr
        for i in range(len(self.temp_matrix)):
            if self.zgr[i] >= self.config.thermal.MSL:  # assume that the initial water level is at zero
                self.temp_matrix[i,:] = ground_temp_distr_dry[:,1]
            else:
                self.temp_matrix[i,:] = ground_temp_distr_wet[:,1]
        
        # set initial ghost node temperature as a copy of the surface node of the temperature matrix
        self.ghost_nodes_temperature = self.temp_matrix[:,0]
        
        # find and write the initial thaw depth
        self.find_thaw_depth()
        self.write_ne_layer()
        
        # with the temperature matrix, the initial state (frozen/unfrozen can be determined). This should be the only place where state is defined through temperature
        frozen_mask = (self.temp_matrix <= self.config.thermal.T_melt)
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask
        
        # define soil properties (k, density, nb, Cs & Cl)
        self.define_soil_property_matrices(len(self.xgr), define_nb=True)
            
        # (latent heat term depends on thermal.latent_heat_form, see define_soil_property_matrices)
        self.enthalpy_matrix = self._enthalpy_from_temperature(self.temp_matrix)

        # calculate the courant-friedlichs-lewy number matrix
        self.k_matrix = frozen_mask * self.k_frozen_matrix + unfrozen_mask * self.k_unfrozen_matrix
        self.cfl_matrix = self.k_matrix / self.soil_density_matrix * self.config.thermal.dt / self.dz**2
        
        if np.max(self.cfl_matrix >= self.config.wrapper.CFL_thermal):
            # raise ValueError(f"CFL should be smaller than {self.config.wrapper.CFL_thermal}, currently {np.max(self.cfl_matrix):.4f}")
            # print(f"CFL should be smaller than 0.5, currently {np.max(self.cfl_matrix):.4f}")
            pass
        
        # get the 'A' matrix, which is used to make the numerical scheme faster. It is based on second order central differences for internal points
        # at the border points, the grid is extended with an identical point (i.e. mirrored), in order to calculate the second derivative
        self.A_matrix = um.get_A_matrix(self.config.thermal.grid_resolution)
        
        # initialize angles
        self._update_angles()
        
        # initialize output
        self.factors = np.zeros(self.xgr.shape)
        self.sw_flux = np.zeros(self.xgr.shape)
        
        self.latent_flux = np.zeros(self.xgr.shape)  # also used in output
        self.lw_flux = np.zeros(self.xgr.shape)  # also used in output  # also used in output
        
        self.convective_flux = np.zeros(self.xgr.shape)
        self.heat_flux = np.zeros(self.xgr.shape)

        return None
    
    def define_soil_property_matrices(self, N, define_nb=True):
        """This function is ran to easily define (and redefine) matrices with soil properties. 
        It is only a function of the x-grid, since the perpendicular z-grid is does not change in size.
        
        Args:
            N (int): number of surface grid points
            define_nb (bool): whether or not to (re)define nb. Defaults to True.
            """
        
        # initialize linear distribution of k, starting at min value and ending at max value (at a depth of 1m)
        id_kmax = np.argmin(np.abs(self.thermal_zgr - self.config.thermal.depth_constant_k))  # id of the grid point at which the maximum k should be reached
        self.k_unfrozen_distr = np.append(
            np.linspace(
                self.config.thermal.k_soil_unfrozen_min, 
                self.config.thermal.k_soil_unfrozen_max,
                len(self.thermal_zgr[:id_kmax])), 
            np.ones(len(self.thermal_zgr[id_kmax:])) * self.config.thermal.k_soil_unfrozen_max)
        self.k_frozen_distr = np.append(
            np.linspace(
                self.config.thermal.k_soil_frozen_min, 
                self.config.thermal.k_soil_frozen_max,
                len(self.thermal_zgr[:id_kmax])), 
            np.ones(len(self.thermal_zgr[id_kmax:])) * self.config.thermal.k_soil_frozen_max)
        
        # initialize k-matrix
        self.k_frozen_matrix =  np.tile(self.k_frozen_distr, (N, 1))
        self.k_unfrozen_matrix = np.tile(self.k_unfrozen_distr, (N, 1))
        
        # initialize distribution of ground ice content
        if define_nb:
            self.nb_distr = Simulation._compute_nb_distr(
                nb_max=self.config.thermal.nb_max,
                nb_min=self.config.thermal.nb_min,
                nb_max_depth=self.config.thermal.nb_max_depth,
                nb_min_depth=self.config.thermal.nb_min_depth,
                N=self.config.thermal.grid_resolution,
                max_depth=self.config.thermal.max_depth
            )            
            
            self.nb_matrix = np.tile(self.nb_distr, (N, 1))
            
        # calculate / read in density
        if self.config.thermal.rho_soil == "None":
            self.soil_density_matrix = self.nb_matrix * self.config.thermal.rho_water + (1 - self.nb_matrix) * self.config.thermal.rho_particle
        else:
            self.soil_density_matrix = np.ones(self.nb_matrix.shape) * self.config.thermal.rho_soil
        
        # using the states, the initial enthalpy can be determined. The enthalpy matrix is used as the 'preserved' quantity, and is used to numerically solve the
        # heat balance equation. Enthalpy formulation from Ravens et al. (2023).
        self.Cs_matrix = self.config.thermal.c_soil_frozen / self.soil_density_matrix
        self.Cl_matrix = self.config.thermal.c_soil_unfrozen / self.soil_density_matrix

        # latent heat of fusion in the same unit as the enthalpy (J per kg of soil), see _latent_heat_matrix
        self.latent_matrix = self._latent_heat_matrix(self.nb_matrix, self.soil_density_matrix)

        return None

    def _get_option(self, name, default, allowed):
        """Read a string option from config.thermal and validate it. The default is the legacy (v1.0) form."""
        value = self.config.thermal.get(name, default)
        if value not in allowed:
            raise ValueError(f"thermal.{name} must be one of {allowed}, got {value!r}")
        return value

    def _latent_heat_matrix(self, nb, rho_soil):
        """Latent heat of fusion per kg of soil [J/kg] for (saturated) porosity nb and soil density rho_soil.

        The enthalpy is stored per unit mass of soil (Cs = C_frozen / rho_soil), so the latent term is too:
            per_mass   (v1.0, legacy): L * nb, i.e. a volumetric latent heat of rho_soil * L * nb.
            volumetric               : L * nb * rho_water / rho_soil, i.e. a volumetric latent heat of rho_water * L * nb.
        L_water_ice is in J/kg of water in both cases.
        """
        form = self._get_option('latent_heat_form', 'per_mass', ('per_mass', 'volumetric'))
        L = self.config.thermal.L_water_ice
        if form == 'volumetric':
            return L * nb * self.config.thermal.rho_water / rho_soil
        return L * nb

    def _enthalpy_from_temperature(self, temp_matrix):
        """Enthalpy [J/kg soil] of a temperature matrix, using the current Cs, Cl and latent matrices.
        T <= T_melt is frozen, T > T_melt is fully thawed (the latent heat is added in full)."""
        T_melt = self.config.thermal.T_melt
        frozen_mask = (temp_matrix <= T_melt)
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask
        return \
            frozen_mask * self.Cs_matrix * temp_matrix + \
            unfrozen_mask * (self.Cl_matrix * temp_matrix + (self.Cs_matrix - self.Cl_matrix) * T_melt + self.latent_matrix)
    
    @classmethod
    def _compute_nb_distr(self, nb_max, nb_min, nb_max_depth, nb_min_depth, N, max_depth):
        """This function returns an nb distribution, with a sigmoid type curve connecting constant values above and below the min and max depth

        Args:
            nb_max (float): maximum value of nb (used close to surface),
            nb_min (float): minimum value of nb (used at depth),
            nb_max_depth (float): depth at which the nb value starts going down,
            nb_min_depth (float): depth at which minimum nb value is reached. Below this depth, nb is constant,
            N (int): number of (evenly spaced) grid points,
            max_depth (float): maximum depth

        Returns:
            array: array containing nb values for the given grid.
        """
        nb = np.zeros(N)
        z = np.linspace(0, max_depth, N)

        mid = (nb_max_depth + nb_min_depth) / 2;
            
        nb = (1 / (1 + np.exp(-(z - mid) * 10 / (nb_min_depth - nb_max_depth)))) * (nb_min - nb_max) + nb_max;
        
        nb[np.argwhere(z<=nb_max_depth)] = nb_max
        nb[np.argwhere(z>=nb_min_depth)] = nb_min
            
        return nb
    
    def print_and_return_A_matrix(self):
        """This function prints and returns the A_matrix"""
        print(self.A_matrix)
        return self.A_matrix
    
    def _generate_initial_ground_temperature_distribution(self, df, t_start, n, max_depth):
        """This method generates an initial ground temperature distribution using soil temperature in different layers from ERA5 data.
        
        The ECMWF ERA5 dataset has a four-layer representation of soil:
        Layer 1: 0 - 7cm, 
        Layer 2: 7 - 28cm, 
        Layer 3: 28 - 100cm, 
        Layer 4: 100 - 289cm. 
        
        Temperature is linearly interpolated for the entire depth, and assumed constant below the center of Layer 4.
        Same temperature profile is used for both wet and dry points.
        
        In conceptual mode, uses constant initial temperature (conceptual_ini) instead of ERA5 data.
        """                                    
        # Check if conceptual mode is enabled
        use_conceptual = getattr(self.config.data, 'conceptual', False)
        
        if use_conceptual:
            # Conceptual mode: use constant initial temperature
            conceptual_ini = getattr(self.config.data, 'conceptual_ini', 260.0)
            logger.info(f"Conceptual mode: using constant initial temperature ({conceptual_ini}K)")
            
            # Create uniform temperature distribution
            depth_array = np.linspace(0, max_depth, n)
            ground_temp_distr_dry = np.column_stack([depth_array, np.full(n, conceptual_ini)])
            ground_temp_distr_wet = ground_temp_distr_dry.copy()
        else:
            # Use ERA5 soil temperature data directly
            era5_points = np.array([
                [0, df.soil_temperature_level_1.values[0]],
                [(0.07+0)/2, df.soil_temperature_level_1.values[0]],
                [(0.28+0.07)/2, df.soil_temperature_level_2.values[0]],
                [(1+0.28)/2, df.soil_temperature_level_3.values[0]],
                [(2.89+1)/2, df.soil_temperature_level_4.values[0]],
                [max_depth, df.soil_temperature_level_4.values[0]],
            ])
            
            # Same temperature distribution for both dry and wet points
            ground_temp_distr_dry = um.interpolate_points(era5_points[:,0], era5_points[:,1], n)
            ground_temp_distr_wet = ground_temp_distr_dry.copy()
        
        return ground_temp_distr_dry, ground_temp_distr_wet
    
    def thermal_update(self, timestep_id, subgrid_timestep_id):
        """This function is called each subgrid timestep of each timestep, and performs the thermal update of the model.
        The C-matrices are not updated as they are a function of only density.

        Args:
            timestep_id (int): id of the current timestep
            subgrid_timestep_id (int): id of the current subgrid timestep
        """
        
        # get the phase masks, based on enthalpy
        frozen_mask, inbetween_mask, unfrozen_mask = self._get_phase_masks()
        
        # get temperature matrix
        self.temp_matrix = self._temperature_from_enthalpy(frozen_mask, inbetween_mask, unfrozen_mask)
        
        # determine the actual k-matrix using the masks
        self.k_matrix = \
            frozen_mask * self.k_frozen_matrix + \
            inbetween_mask * ((self.k_frozen_matrix + self.k_unfrozen_matrix) / 2) + \
            unfrozen_mask * self.k_unfrozen_matrix
        
        # get the new boundary condition (this also sets self.heat_flux, the net surface heat flux into the ground)
        self.ghost_nodes_temperature = self._get_ghost_node_boundary_condition(timestep_id, subgrid_timestep_id)
        self.bottom_boundary_temperature = self._get_bottom_boundary_temperature()

        dt = self.config.thermal.dt
        if self.diag is not None:  # diagnostics only: net surface heat exchange integrated over the transect, per calendar year
            if subgrid_timestep_id == 0 or self.diag.current_year is None:
                self.diag.current_year = int(self.timestamps[timestep_id].year)
            self.diag.add_surface_exchange(self.heat_flux, self._surface_weights(), dt)
        conduction_form = self._get_option('conduction_form', 'nodal', ('nodal', 'flux'))

        # volumetric heat capacity per node (only used for the stability check below)
        Cvol_matrix = (
            frozen_mask * self.config.thermal.c_soil_frozen +
            inbetween_mask * 0.5 * (self.config.thermal.c_soil_frozen +
                                    self.config.thermal.c_soil_unfrozen) +
            unfrozen_mask * self.config.thermal.c_soil_unfrozen
        )

        if conduction_form == 'nodal':
            # v1.0 (legacy): nodal k times the second difference of T, i.e. k d2T/dz2. Not conservative for k = k(z, phase).
            aggregated_temp_matrix = np.concatenate((
                self.ghost_nodes_temperature.reshape(len(self.xgr), 1),
                self.temp_matrix,
                self.bottom_boundary_temperature.reshape(len(self.xgr), 1)
                ), axis=1)
            update_coeff = self.k_matrix / self.soil_density_matrix * dt / self.dz**2

            # Verify stability using the Fourier number
            Fo_matrix = (self.k_matrix / Cvol_matrix) * (dt / self.dz**2)
            Fo_max = float(np.nanmax(Fo_matrix))

            # get the new enthalpy matrix
            laplacian_T = aggregated_temp_matrix @ self.A_matrix
            self.enthalpy_matrix = self.enthalpy_matrix + update_coeff * laplacian_T

        else:
            # Conservative finite-volume form d/dz(k dT/dz) with harmonic-mean conductivity on the cell faces.
            # Node i owns the cell [z_i - dz/2, z_i + dz/2]; the surface and bottom nodes own half cells (thickness dz/2).
            # Boundary conditions are the same physical ones as in the nodal form (the ghost-node construction of
            # _get_ghost_node_boundary_condition is equivalent to these fluxes for ghost_node_order = 2):
            #   top   : net surface heat flux self.heat_flux [W/m2] into the ground
            #   bottom: geothermal heat flux k * geothermal_gradient [W/m2] from below
            T = self.temp_matrix
            k = self.k_matrix
            k_face = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])  # harmonic mean, shape (N, nz-1)
            F = k_face * (T[:, 1:] - T[:, :-1]) / self.dz  # [W/m2] heat flowing upward through face i+1/2 into node i

            rate = np.empty_like(T)  # volumetric heating rate [W/m3]
            rate[:, 1:-1] = (F[:, 1:] - F[:, :-1]) / self.dz
            rate[:, 0] = 2 * (self.heat_flux + F[:, 0]) / self.dz
            rate[:, -1] = 2 * (k[:, -1] * self.config.thermal.geothermal_gradient - F[:, -1]) / self.dz

            # stored quantity is per unit mass of soil, so divide by the node's soil density
            self.enthalpy_matrix = self.enthalpy_matrix + dt * rate / self.soil_density_matrix

            # Fourier number of the explicit scheme: sum of the two face conductivities over the node's heat capacity
            k_face_pad = np.concatenate((k_face[:, :1], k_face, k_face[:, -1:]), axis=1)
            Fo_matrix = (k_face_pad[:, :-1] + k_face_pad[:, 1:]) / (2 * Cvol_matrix) * (dt / self.dz**2)
            Fo_max = float(np.nanmax(Fo_matrix))

        if Fo_max > 0.5:
            raise ValueError(
                f"Thermal timestep too large: max(Fo) = {Fo_max:.3f} > 0.5. "
                f"Reduce dt to <= {0.5 * dt / Fo_max:.3e} s"
            )

        # A wet surface node (half cell) also gets the explicit convective flux hc * (T_sea - T_0), which adds
        # 2 hc dt / (C dz) to the conduction coefficient 2 Fo of that node. The sum must stay below 2 or the node rings.
        wet_mask = getattr(self, 'wet_mask', None)  # not set by subclasses that override the boundary condition
        if wet_mask is not None and wet_mask.any():
            wet = wet_mask.astype(bool)
            coeff_wet = (2 * self.k_matrix[wet, 0] / Cvol_matrix[wet, 0] * dt / self.dz**2
                         + 2 * self.hc_water * dt / (Cvol_matrix[wet, 0] * self.dz))
            coeff_max = float(np.nanmax(coeff_wet))
            if coeff_max >= 2:
                raise ValueError(
                    f"Thermal timestep too large for the wet surface node: 2 Fo + 2 hc dt / (C dz) = {coeff_max:.3f} >= 2. "
                    f"Reduce dt to <= {2 * dt / coeff_max:.3e} s"
                )

        return None
    
    def _get_phase_masks(self):
        "Returns phase masks from current enthalpy distribution."
        # determine state masks (which part of the domain is frozen, in between, or unfrozen (needed to later calculate temperature from enthalpy))
        frozen_mask = (self.enthalpy_matrix)  < (self.config.thermal.T_melt * self.Cs_matrix)
        
        unfrozen_mask = (self.enthalpy_matrix) > (
            self.config.thermal.T_melt * self.Cl_matrix + \
            (self.Cs_matrix - self.Cl_matrix) * self.config.thermal.T_melt + \
            self.latent_matrix
            )
        
        inbetween_mask = np.ones(frozen_mask.shape) - frozen_mask - unfrozen_mask
        
        return frozen_mask, inbetween_mask, unfrozen_mask
    
    def _temperature_from_enthalpy(self, frozen_mask, inbetween_mask, unfrozen_mask):
        """Returns the temperature matrix for given phase masks."""
        temp_matrix = \
            frozen_mask * \
                (self.enthalpy_matrix / self.Cs_matrix) + \
            inbetween_mask * \
                (self.config.thermal.T_melt) + \
            unfrozen_mask * \
                (self.enthalpy_matrix - \
                (self.Cs_matrix - self.Cl_matrix) * self.config.thermal.T_melt - \
                self.latent_matrix) / \
                    (self.Cl_matrix)
                    
        return temp_matrix
     
    def _get_ghost_node_boundary_condition(self, timestep_id, subgrid_timestep_id):
        """This function uses the forcing at a specific timestep to return an array containing the ghost node temperature.

        Args:
            timestep_id (int): index of the current timestep.
            subgrid_timestep_id (int): id of the current subgrid timestep

        Returns:
            array: temperature values for the ghost nodes.
        """
        # first get the correct forcing timestep
        row = self._forcing_row(timestep_id)
        
        # with associated forcing values
        self.current_air_temp = row["2m_temperature"]  # also used in output
        self.current_sea_temp = row['sea_surface_temperature']  # also used in output
        self.current_sea_ice = row["sea_ice_cover"]  # not used in this function, but loaded in preperation for output
        
        # update the water level
        self.water_level = (self._update_water_level(timestep_id, subgrid_timestep_id=subgrid_timestep_id))
        
        dry_mask = (self.zgr >= self.water_level + self.R2[timestep_id])
        wet_mask = (self.zgr < self.water_level + self.R2[timestep_id])
        
        # determine convective transport from air (formulation from Man, 2023)
        self.wind_direction, self.wind_velocity = self._get_wind_conditions(timestep_id=timestep_id)  # also used in output
        convective_transport_air = Simulation._calculate_sensible_heat_flux_air(
            self.wind_velocity, 
            self.temp_matrix[:,0], 
            self.current_air_temp, 
            )
        
        # use either an educated guess from Kobayashi et al (1999), or their entire formulation
        # the entire formulation is a bit slower because of the dispersion relation that needs to be solved
        if self.config.thermal.with_convective_transport_water_guess:
            hc = self.config.thermal.hc_guess
        else:
            
            raise Exception("This piece of code is outdated")
            
            # determine hydraulic parameters for convective heat transfer computation
            
            if self.xb_times[timestep_id] and self.config.xbeach.with_xbeach:
                
                data_path = os.path.join(self.cwd, "xboutput.nc")
                
                ds = xr.load_dataset(data_path)
                
                H = ds.H.values.flatten()

                wl = self._update_water_level(timestep_id)
                
                d = np.maximum(wl - ds.zb.values.flatten(), np.zeros(H.shape))
                T = self.conditions[timestep_id]["Tp(s)"]
                
                ds.close()
                                
            else:
                H = self.conditions[timestep_id]['Hs(m)']
                
                wl = self._update_water_level(timestep_id)
                
                d = np.maximum(wl - self.zgr, 0)
                T = self.conditions[timestep_id]['Tp(s)']
                
            # determine convective transport from water (formulation from Kobayashi, 1999)
            hc = Simulation._calculate_sensible_heat_flux_water(
                H, T, d, self.config.xbeach.rho_sea_water,
                CW=3989, alpha=0.5, nu=1.848*10**-6, ks=2.5*1.90*10**-3, Pr=13.4
            )
        
        # scale hc with temperature difference
        convective_transport_water = hc * (self.current_sea_temp - self.temp_matrix[:,0])
            
        # compute total convective transport
        self.convective_flux = dry_mask * convective_transport_air + wet_mask * convective_transport_water  # also used in output
        
        # multiply convective heat flux with factor for sensitivity analysis
        convective_factor = 1 if "sensitivity" not in self.config.keys() else self.config.sensitivity.factor_convective_heat_flux
        self.convective_flux = self.convective_flux * convective_factor

        # kept for the explicit stability check of the wet surface node in thermal_update
        self.wet_mask = wet_mask
        self.hc_water = hc * convective_factor

        if subgrid_timestep_id == 0:  # determine radiation fluxes only during first subgrid timestep, as they are constant for each subgrid timestep
        
            # Check if conceptual mode is enabled - if so, disable all radiation
            use_conceptual = getattr(self.config.data, 'conceptual', False)
            
            if use_conceptual:
                # Conceptual mode: no radiation fluxes
                self.latent_flux = np.zeros(self.xgr.shape)
                self.lw_flux = np.zeros(self.xgr.shape)
                self.sw_flux = np.zeros(self.xgr.shape)
            else:
                # determine radiation, assuming radiation only influences the dry domain
                self.latent_flux = dry_mask * (row["mean_surface_latent_heat_flux"] if self.config.thermal.with_latent else 0)  # also used in output
                self.lw_flux = dry_mask * (row["mean_surface_net_long_wave_radiation_flux"] if self.config.thermal.with_longwave else 0)  # also used in output
                
                if self.config.thermal.with_solar:  # also used in output
                    I0 = row["mean_surface_net_short_wave_radiation_flux"]  # float value
                    
                    if self.config.thermal.with_solar_flux_calculator:
                        self.sw_flux = dry_mask * self._get_solar_flux(I0, timestep_id)  # sw_flux is now an array instead of a float
                    else:
                        self.sw_flux = dry_mask * I0
                else:
                    self.sw_flux = np.zeros(self.xgr.shape)
            
            # save this constant flux
            self.constant_flux = self.latent_flux + self.lw_flux + self.sw_flux
        
        # add all heat fluxes  together (also used in output)
        self.heat_flux = self.convective_flux + self.constant_flux
        
        # compute heat flux factors
        if 'heat_flux_factors' in self.config.thermal.keys():
            self.heat_flux_factors = (np.abs(self.angles / (2 * np.pi) * 360) < self.config.thermal.surface_flux_angle) * self.config.thermal.surface_flux_factor
        else:
            self.heat_flux_factors = np.ones(self.xgr.shape)
        
        # Check if conceptual mode with constant flux is enabled
        use_conceptual = getattr(self.config.data, 'conceptual', False)
        conceptual_flux = getattr(self.config.data, 'conceptual_flux', None)
        
        if use_conceptual and conceptual_flux is not None and conceptual_flux != []:
            # Conceptual mode: use constant heat flux
            self.heat_flux = np.full(self.xgr.shape, conceptual_flux)
        else:
            # Normal mode: multiply with heat flux factor
            self.heat_flux = self.heat_flux * self.heat_flux_factors
        
        # Determine temperature of the ghost nodes using configurable order
        ghost_node_order = self.config.thermal.get('ghost_node_order', 2)  # default to second order

        # Different methods included
        if ghost_node_order == 1:
            # First-order ghost node (forward difference)
            # Uses T[0] (surface node)
            ghost_node_top_boundary = (
                self.temp_matrix[:,0] + 
                self.heat_flux * self.dz / self.k_matrix[:,0]
            )
            
        elif ghost_node_order == 2:
            # Second-order ghost node (central difference)
            # CRITICAL: Uses T[1], NOT T[0] (because surface is at T[0])
            ghost_node_top_boundary = (
                self.temp_matrix[:,1] + 
                2 * self.heat_flux * self.dz / self.k_matrix[:,0]
            )
            
        elif ghost_node_order == 3:
            # Third-order ghost node (four-point formula)
            # Uses T[0] (surface), T[1], T[2]
            T0 = self.temp_matrix[:,0]
            T1 = self.temp_matrix[:,1]
            T2 = self.temp_matrix[:,2]
            k0 = self.k_matrix[:,0]
            
            ghost_node_top_boundary = (
                (-3*T0 + 6*T1 - T2)/2 + 
                (3*self.dz/k0) * self.heat_flux
            )
            
        elif ghost_node_order == 4:
            # Fourth-order ghost node (five-point formula)
            # Uses T[0] (surface), T[1], T[2], T[3]
            T0 = self.temp_matrix[:,0]
            T1 = self.temp_matrix[:,1]
            T2 = self.temp_matrix[:,2]
            T3 = self.temp_matrix[:,3]
            k0 = self.k_matrix[:,0]
            
            ghost_node_top_boundary = (
                (-10*T0 + 18*T1 - 6*T2 + T3)/3 + 
                (4*self.dz/k0) * self.heat_flux
            )
            
        else:
            logger.warning(f"Invalid ghost_node_order {ghost_node_order}. Using second-order (default).")
            ghost_node_top_boundary = (
                self.temp_matrix[:,1] + 
                2 * self.heat_flux * self.dz / self.k_matrix[:,1]
            )

        # Return temperature at the top
        return ghost_node_top_boundary
    
    def _update_water_level(self, timestep_id, subgrid_timestep_id=0):


        # get the water level
        if self.xbeach_times[timestep_id-1] and self.config.xbeach.with_xbeach:
            
            # construct path to xbeach output in destination folder
            destination_folder = os.path.join(self.result_dir, "xb_files/", (10 - len(str(int(timestep_id-1)))) * '0' + str(int(timestep_id-1)) + '/')
            xbeach_output_path = os.path.join(destination_folder, "xboutput.nc")
            
            try:
                # load dataset
                ds = xr.load_dataset(xbeach_output_path).squeeze()  # get xbeach data
                
                # select only the final timestep
                ds = ds.sel(globaltime=np.max(ds.globaltime.values))
                
                # load zs values
                zs_values = ds.zs.values.flatten()
                
                # remove nan values and get maximum
                self.water_level = np.max(zs_values[~np.isnan(zs_values)])
                
                # close dataset
                ds.close()
            except Exception as e:
                logger.warning(f"Failed to load XBeach water level data for timestep {timestep_id-1} ({self.timestamps[timestep_id-1]}). "
                              f"Error: {str(e)}. Using static water level instead.")
                self.water_level = self.water_levels[timestep_id]

            
        else:
            self.water_level = self.water_levels[timestep_id]
            
        return self.water_level
    
    def _get_bottom_boundary_temperature(self):
        """This function returns a bottom boundary condition temperature, based on the geothermal gradient. It accounts for the
        angle that each 1D thermal grid makes with the horizontal.

        Returns:
            array: bottom temperature
        """
                
        # Configure bottom boundary order (same as ghost node order for consistency)
        bottom_boundary_order = self.config.thermal.get('ghost_node_order', 2)  # default to second order

        # The geothermal gradient creates a flux at bottom boundary
        # q_bottom = -k * geothermal_gradient (negative because heat flows upward)
        vertical_dist = self.dz
        gradient_term = vertical_dist * self.config.thermal.geothermal_gradient

        if bottom_boundary_order == 1:
            # First-order bottom boundary (backward difference)
            # Uses T[-1] (bottom boundary node)
            ghost_node_bottom_boundary = self.temp_matrix[:,-1] + gradient_term
            
        elif bottom_boundary_order == 2:
            # Second-order bottom boundary (central difference)
            # CRITICAL: Uses T[-2], NOT T[-1] (because bottom boundary is at T[-1])
            ghost_node_bottom_boundary = self.temp_matrix[:,-2] + 2 * gradient_term
            
        elif bottom_boundary_order == 3:
            # Third-order bottom boundary (four-point formula)
            # Uses T[-1] (bottom), T[-2], T[-3]
            T_last = self.temp_matrix[:,-1]
            T_second_last = self.temp_matrix[:,-2]
            T_third_last = self.temp_matrix[:,-3]
            
            ghost_node_bottom_boundary = (
                (-3*T_last + 6*T_second_last - T_third_last)/2 + 
                (3 * gradient_term)
            )
            
        elif bottom_boundary_order == 4:
            # Fourth-order bottom boundary (five-point formula)
            # Uses T[-1] (bottom), T[-2], T[-3], T[-4]
            T_last = self.temp_matrix[:,-1]
            T_second_last = self.temp_matrix[:,-2]
            T_third_last = self.temp_matrix[:,-3]
            T_fourth_last = self.temp_matrix[:,-4]
            
            ghost_node_bottom_boundary = (
                (-10*T_last + 18*T_second_last - 6*T_third_last + T_fourth_last)/3 + 
                (4 * gradient_term)
            )
            
        else:
            logger.warning(f"Invalid bottom boundary order {bottom_boundary_order}. Using second-order (default).")
            ghost_node_bottom_boundary = self.temp_matrix[:,-2] + 2 * gradient_term

        # Return bottom temperature
        return ghost_node_bottom_boundary
    
    @classmethod
    def _calculate_sensible_heat_flux_air(self, v_w, T_soil_surface, T_air, L_e=0.003, nu_air=1.33*10**-5, Pr=0.71, k_air=0.024):
        """This function computes the sensible heat flux Qs [W/m2] at the soil surface, using a formulation described by Man (2023).
            A positive flux means that the flux is directed into the soil.

        Args:
            v_w (float): wind speed at 10-meter height [m/s],
            T_soil_surface (array): an array containing the soil surface temperature [K],
            T_air (float): air temperature [K],
            L_e (float): convective length scale [m]. Defaults to 0.003.
            nu_air (float, optional): air kinematic viscosity. Defaults to 1.33*10**-5.
            Pr (float, optional): Prandtl number. Defaults to 0.71.
            k_air (float, optional): thermal conductivity of air. Defaults to 0.024.

        Returns:
            array: array containing the sensible heat flux for each point for which a soil surface temperature was provided.
        """
       
        Qs =  0.0296 * (v_w * L_e / nu_air)**(4/5) * Pr**(1/3) * k_air / L_e * (T_air - T_soil_surface)
       
        return Qs
    
    @classmethod
    def _calculate_sensible_heat_flux_water(
        H, T, d, rho_seawater, CW=3989, alpha=0.5, nu=1.848*10**-6, ks=2.5*1.90*10**-3, Pr=13.4
        ):
        """This function calculates the sensible heat flux between (sea) water and soil. The computation is based on Kobayashi 
        et al (1999) for the general formulation of the heat flux, Kobayashi & Aktan (1986) for specific formulations of 
        parameters, and Jonsson (1966) for specific parameter values. Note: the formulation by Kobayashi et al (1999) is meant
        specifically for breaking waves. A positive flux means that the flux is directed into the soil."""
        # check for nan values in H array
        mask = np.nonzero(1 - np.isnan(H))
        
        # calculate k based on linear wave theory
        kr = []
        for dr in d[mask]:
            
            kr.append(dispersion(2 * np.pi / T, dr))
        
        # convert to array
        kr = np.array(kr)
        
        # compute volumetric heat capacity
        cw = CW * rho_seawater
        
        # compute representative fluid velocity immediately outside the boundary layer
        u_b = np.pi * H[mask] / (T * np.sinh(kr * dr))
        
        # fw is set to 0.05
        fw = Simulation._get_fw()
        
        # calculate u-star, which is necessary for determining the final coefficient
        u_star = np.sqrt(0.5 * fw) * u_b
        
        # the roughness should be checked to be above 70, otherwise another formulation should be used for the coefficient E
        roughness = u_star * ks / nu
        
        # parameter depending on whether the turbulent boundary layer flow is hydraulically smooth or fully rough
        E = 0.52 * roughness**0.45 * Pr**0.8
        
        # sensible heat flux factor [W/m2/K]
        hc = alpha * fw * cw * u_b / (1 + np.sqrt(0.5 * fw) * E)
        
        return hc
        
    @classmethod
    def _get_fw():
        """this computation is required to get values from the graph providing fw in Jonsson (1966)"""
        # Reynolds number
        # Re = np.max(d[mask]) * u_b / nu
        
        # maximum surface elevation, and particle amplitude.
        # a = H[mask] / 2
        
        # z0 = - d[mask] + h_bed
        # amx = a * np.cosh(kr * (d[mask] + z0)) / np.sinh(kr * d[mask])
        # amz = a * np.sinh(kr * (d[mask] + z0)) / np.sinh(kr * d[mask])
        
        # with the Reynolds number and the maximum particle displacement divided by k, a value for fw can be read.
        # this value has a high uncertainty. 
        # a value of 0.05 is used here, which was found to be somewhat representible
        
        return 0.05

    
    def update_grid(self, timestep_id, fp_xbeach_output="sedero.txt"):
        """This function updates the current grid, calculates the angles of the new grid with the horizontal, generates a new thermal grid 
        (perpendicular to the existing grid), and fits the previous temperature and enthalpy distributions to the new grid."""
        
        # construct full path to xbeach output in destination folder
        destination_folder = os.path.join(self.result_dir, "xb_files/", (10 - len(str(int(timestep_id)))) * '0' + str(int(timestep_id)) + '/')
        full_xbeach_output_path = os.path.join(destination_folder, fp_xbeach_output)
        
        # update the current bathymetry
        cum_sedero = self._get_cum_sedero(fp_xbeach_output=full_xbeach_output_path)  # placeholder
        
        # update bed level
        bathy_current = self.zgr + cum_sedero

        # diagnostics only: remember the thermal state before the regrid (nothing below depends on it)
        log_conservation = self.diag is not None and self.diag.log_conservation
        snapshot = self._snapshot_for_conservation() if (log_conservation and not all(cum_sedero == 0)) else None

        # only update the grid of there actually was a change in bed level
        if not all(cum_sedero == 0):
                        
            # generate a new xgrid and zgrid (but only if the next timestep does not require a hotstart, which requires the same xgrid)
            if timestep_id + 1 < len(self.xbeach_times) and not self.check_xbeach(timestep_id + 1):
                                
                self.xgr_new, self.zgr_new = xgrid(self.xgr, bathy_current, dxmin=2, ppwl=self.config.bathymetry.ppwl)
                self.zgr_new = np.interp(self.xgr_new, self.xgr, bathy_current)
                
                # ensure that the grid doesn't extend further offshore than the original grid (this is a bug in the xbeach python toolbox)
                while self.xgr_new[0] < self.x_ori:
                    self.xgr_new = self.xgr_new[1:]
                    self.zgr_new = self.zgr_new[1:]
                
            else:
                self.xgr_new = self.xgr
                self.zgr_new = bathy_current
            
            self.xgr_new = self.xgr
            self.zgr_new = bathy_current

            # generate perpendicular grids for next timestep (to cast temperature and enthalpy)
            self.abs_xgr_new, self.abs_zgr_new = um.generate_perpendicular_grids(
                self.xgr_new, 
                self.zgr_new, 
                resolution=self.config.thermal.grid_resolution, 
                max_depth=self.config.thermal.max_depth
            )
            
            # cast temperature matrix
            if self.config.thermal.grid_interpolation == "linear_interp_with_nearest":
                self.temp_matrix = um.linear_interp_with_nearest(self.abs_xgr, self.abs_zgr, self.temp_matrix, self.abs_xgr_new, self.abs_zgr_new)
                self.enthalpy_matrix = um.linear_interp_with_nearest(self.abs_xgr, self.abs_zgr, self.enthalpy_matrix, self.abs_xgr_new, self.abs_zgr_new)
            
            elif self.config.thermal.grid_interpolation == "linear_interp_z":
                
                # Compute density
                rho_value = self.config.thermal.nb_max * self.config.thermal.rho_water + (1 - self.config.thermal.nb_max) * self.config.thermal.rho_particle
                
                # From denisty, compute specific heat
                Cs_value = self.config.thermal.c_soil_frozen / rho_value
                Cl_value = self.config.thermal.c_soil_unfrozen / rho_value
                
                # Compute top fill value for submerged sediment
                if (self.current_sea_temp <= self.config.thermal.T_melt):
                    enthalpy_submerged_sediment = self.current_sea_temp * Cs_value
                else:
                    enthalpy_submerged_sediment = self.current_sea_temp * Cl_value + (Cs_value - Cl_value) * self.config.thermal.T_melt + \
                        self._latent_heat_matrix(self.config.thermal.nb_max, rho_value)
                
                # get current water level
                self.water_level = (self._update_water_level(timestep_id))
                
                # Interpolate enthalpy to new grid             
                self.enthalpy_matrix = um.linear_interp_z(
                    self.abs_xgr, 
                    self.abs_zgr, 
                    self.enthalpy_matrix, 
                    self.abs_xgr_new, 
                    self.abs_zgr_new,
                    water_level=self.water_level,
                    fill_value_top_water=enthalpy_submerged_sediment,
                    fill_value_top_air='nearest',
                    )
                
                # interpolate nb to new grid
                self.nb_matrix = um.linear_interp_z(
                    self.abs_xgr,
                    self.abs_zgr,
                    self.nb_matrix,
                    self.abs_xgr_new,
                    self.abs_zgr_new,
                    water_level=self.water_level,
                    fill_value_top_water=self.config.thermal.nb_max,
                    fill_value_top_air=self.config.thermal.nb_max,
                )
                
                # redefine matrices with soil properties
                self.define_soil_property_matrices(len(self.xgr_new), define_nb=False)
                
            else:
                raise ValueError("Invalid value for grid_interpolation")
            
            # set the grid to be equal to this new grid
            self.xgr = self.xgr_new
            self.zgr = self.zgr_new
            
            self.abs_xgr = self.abs_xgr_new
            self.abs_zgr = self.abs_zgr_new
            
            # update the angles
            self._update_angles()

        if log_conservation:
            try:
                self._log_regrid_conservation(timestep_id, snapshot, cum_sedero)
            except Exception as e:  # diagnostics must never stop a run
                logger.warning(f"Conservation diagnostics failed at timestep {timestep_id}: {e}")

        return None

    def _surface_weights(self):
        """Surface arc length owned by each node [m] (cached per grid), used to integrate over the transect."""
        cache = getattr(self, "_ds_cache", None)
        if cache is None or cache[0] is not self.zgr or cache[1] is not self.xgr:
            cache = (self.zgr, self.xgr, dg.arc_length_weights(self.xgr, self.zgr))
            self._ds_cache = cache
        return cache[2]

    def _snapshot_for_conservation(self):
        """Copy of everything the heat budget of a regrid needs from the state before the regrid."""
        return dict(
            abs_xgr=self.abs_xgr.copy(), abs_zgr=self.abs_zgr.copy(), H=self.enthalpy_matrix.copy(),
            rho=self.soil_density_matrix.copy(), Cs=self.Cs_matrix.copy(),
        )

    def _log_regrid_conservation(self, timestep_id, snap, cum_sedero):
        """Diagnostics only: stored heat (split into material / geometry / interpolation) and sediment budget of one
        grid update. Writes one row of conservation_log.csv. Does not modify the model state."""
        def trapz(y, x):
            y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
            return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) if len(x) > 1 else 0.0

        regridded = snap is not None
        row = dict(call_id=self.diag.call_id, timestep_id=int(timestep_id), time=str(self.timestamps[timestep_id]),
                   regridded=int(regridded))

        # --- sediment budget [m3 per m alongshore]: XBeach raw output vs what the thermal grid was given
        raw = getattr(self, "_xb_raw", None)
        if raw is not None:
            row["vol_xb_net"] = trapz(raw["sedero"], raw["x"])
            row["vol_xb_eroded"] = trapz(np.maximum(-raw["sedero"], 0), raw["x"])
            row["vol_xb_deposited"] = trapz(np.maximum(raw["sedero"], 0), raw["x"])
            row["vol_xb_zbdiff"] = trapz(raw["zb1"] - raw["zb0"], raw["x"]) if raw["zb0"] is not None else np.nan
        # (cum_sedero is on the old thermal x grid, which does not change in a regrid)
        row["vol_grid_net"] = trapz(cum_sedero, self.xgr)
        row["vol_grid_eroded"] = trapz(np.maximum(-cum_sedero, 0), self.xgr)
        row["vol_grid_deposited"] = trapz(np.maximum(cum_sedero, 0), self.xgr)
        row["vol_grid_minus_xb"] = row["vol_grid_net"] - row.get("vol_xb_net", np.nan)

        # --- stored heat of the thermal columns [J per m alongshore], relative to soil frozen at T_melt
        if regridded:
            T_melt, dz = self.config.thermal.T_melt, self.dz
            wx = dg.x_weights(self.xgr)  # the x grid does not change in a regrid
            cols_before = dg.column_energy(snap["H"], snap["rho"], snap["Cs"], T_melt, snap["abs_zgr"])
            cols_after = dg.column_energy(self.enthalpy_matrix, self.soil_density_matrix, self.Cs_matrix, T_melt, self.abs_zgr)
            row["E_before"] = float((cols_before * wx).sum())
            row["E_after"] = float((cols_after * wx).sum())
            row["dE"] = row["E_after"] - row["E_before"]
            if self.config.thermal.grid_interpolation == "linear_interp_z":
                m = dg.regrid_node_masks(snap["abs_xgr"], snap["abs_zgr"], self.abs_xgr, self.abs_zgr)
                e_old = snap["rho"] * (snap["H"] - snap["Cs"] * T_melt)
                e_new = self.soil_density_matrix * (self.enthalpy_matrix - self.Cs_matrix * T_melt)
                mat = dg.material_heat(snap["abs_zgr"], e_old, self.abs_zgr, e_new)
                for key, name in (("E_added_top", "added_top"), ("E_added_bottom", "added_bottom"),
                                  ("E_removed_top", "removed_top"), ("E_removed_bottom", "removed_bottom")):
                    row[key] = float((mat[name] * wx).sum())
                row["dE_material"] = row["E_added_top"] + row["E_added_bottom"] - row["E_removed_top"] - row["E_removed_bottom"]
                row["dE_interp"] = row["dE"] - row["dE_material"]
                row["interp_pct_of_E_before"] = 100 * row["dE_interp"] / abs(row["E_before"]) if row["E_before"] else np.nan
                row["n_filled_top_nodes"] = int(m["filled_top"].sum())
                row["n_filled_bottom_nodes"] = int(m["filled_bottom"].sum())
                row["n_removed_top_nodes"] = int(m["removed_top"].sum())
                row["n_removed_bottom_nodes"] = int(m["removed_bottom"].sum())
                # discrete thickness of the filled / removed cover: nodes * vertical node spacing * width of the node
                dzv_new, dzv_old = self.abs_zgr[:, 0] - self.abs_zgr[:, 1], snap["abs_zgr"][:, 0] - snap["abs_zgr"][:, 1]
                row["vol_nodes_fill_top"] = float((m["filled_top"].sum(axis=1) * dzv_new * wx).sum())
                row["vol_nodes_removed_top"] = float((m["removed_top"].sum(axis=1) * dzv_old * wx).sum())
        else:
            for key in ("dE", "dE_material", "dE_interp"):
                row[key] = 0.0
        self.diag.record_conservation(row)

    def _update_angles(self):
        """This function geneartes an array of local angles (in radians) for the grid, based on the central differences method.
        """
        gradients = np.gradient(self.zgr, self.xgr)
        
        self.angles = np.arctan(gradients)
        
        return self.angles
    
    def _get_cum_sedero(self, fp_xbeach_output):
        """This method updates the current bed given the xbeach output.
        ---------
        fp_xbeach_output: string
            filepath to the xbeach sedero (sedimentation-erosion) output relative to the current working directory."""
            
        # Check if output file exists with absolute path
        abs_path = os.path.abspath(fp_xbeach_output)
        if not os.path.exists(abs_path):
            logger.error(f"XBeach output file not found at: {abs_path}")
            logger.error(f"Original path provided: {fp_xbeach_output}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.warning("Returning zero cumulative sedimentation/erosion")
            return np.zeros(self.xgr.shape)
            
        # Read output file
        self._xb_raw = None
        try:
            ds = xr.load_dataset(abs_path)
            zb_first = zb_last = None
            if self.diag is not None:  # diagnostics only: bed at first and last output time for the sediment budget
                try:
                    zb_first = ds.zb.isel(globaltime=0).squeeze().values.astype(float)
                    zb_last = ds.zb.isel(globaltime=-1).squeeze().values.astype(float)
                except Exception:
                    zb_first = zb_last = None
            ds = ds.sel(globaltime = np.max(ds.globaltime.values)).squeeze()

            cum_sedero = ds.sedero.values
            xgr = ds.x.values
            if self.diag is not None:
                self._xb_raw = dict(x=np.asarray(xgr, dtype=float).copy(), sedero=np.asarray(cum_sedero, dtype=float).copy(),
                                    zb0=zb_first, zb1=zb_last)

            ds.close()
        except FileNotFoundError as e:
            logger.error(f"File disappeared during read operation: {abs_path}")
            logger.warning("Returning zero cumulative sedimentation/erosion")
            return np.zeros(self.xgr.shape)
        except Exception as e:
            logger.error(f"Failed to read XBeach output file {abs_path}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.warning("Returning zero cumulative sedimentation/erosion")
            return np.zeros(self.xgr.shape)

        # Create an interpolation function
        interpolation_function = interp1d(xgr, cum_sedero, kind='linear', fill_value='extrapolate')
        
        # interpolate values to the used grid
        interpolated_cum_sedero = interpolation_function(self.xgr)
        
        return interpolated_cum_sedero
    
    def _get_solar_flux(self, I0, timestep_id):
        """This function is used to obtain an array of incoming solar radiation for some timestep_id, with values for each grid point in the computational domain.

        Args:
            I0 (float): incoming radiation (flat surface)
            timestep_id (int): index of the current timestep

        Returns:
            array: incoming solar radiation for each grid point in the computational domain
        """        
        # get current timestamp
        timestamp = self.timestamps[timestep_id]
        
        # get id of current timestamp w.r.t. solar_flux_map (-1 because: 'minimum id is 0' and 'minimum day of year is 1')
        id_t = timestamp.dayofyear - 1  # factors are associated with day of year only (as it is based on maximum angle per day)
        
        # get correct row in solar flux map (so row corresponding to current day of the year)
        row = self.solar_flux_map[id_t, :]
        
        # transform angles to ids (first convert to degrees)        
        ids_angle = np.int32((self.angles / (2 * np.pi) * 360 - self.config.thermal.angle_min) / self.config.thermal.delta_angle)
        
        # use ids to get correct factors in correct order from the row of solar fluxes
        self.factors = row[ids_angle]
        
        solar_flux = I0 * self.factors
        
        return solar_flux
        
    def initialize_solar_flux_calculator(self, timezone_diff, angle_min=-89, angle_max=89, delta_angle=1, t_start='2000-01-01', t_end='2001-01-01'):
        """This function initializes a mapping variable for the solar flux calculator. This is required because the enhancement factor is calculated using the
        maximum insolance per day, making it impossible to calculate each hour seperately. Specific values for enhancement factor are indexed using an angle,
        followed by the number of the current day of the year minus 1.

        Args:
            timezone_diff (int): describes the difference in timezone between UTC and the area of interest
            angle_min (int, optional): minimum angle in the mapping variable. Defaults to -89.
            angle_max (int, optional): maximum angle in the mapping. Defaults to 89.
            delta_angle (int, optional): diffence between mapped angles. Defaults to 1.
            t_start (str, optional): start of the daterange used to calculate the enhancement factor. It is recommended to use a full leap year. Defaults to '2000-01-01'.
            t_end (str, optional): end of the daterange used to calculate the enhancement factor. It is recommended to use a full leap year.. Defaults to '2001-01-01'.

        Returns:
            dictionary: the mapping variable used to quickly obtain solar flux enhancement factor values.
        """        
        self.solar_flux_angles = np.arange(angle_min, angle_max+1, delta_angle)
        
        t_start_datetime = pd.to_datetime(t_start)
        t_end_datetime = pd.to_datetime(t_end)
        self.solar_flux_times = pd.date_range(t_start_datetime, t_end_datetime, freq='1h', inclusive='left')
                
        self.solar_flux_map = np.zeros((np.int32(len(self.solar_flux_times)/24), len(self.solar_flux_angles)))
        
        for angle in self.solar_flux_angles:
            
            angle_id = np.nonzero(angle==self.solar_flux_angles)
            
            # for each integer angle in the angle range, an array of enhancement factors is saved, indexable by N (i.e., the N-th day of the year)
            self.solar_flux_map[:, angle_id] = self._calculate_solar_flux_factors(self.solar_flux_times, angle, timezone_diff).reshape((-1, 1, 1))
            
        # can't have negative factors (which may occur in winter when the angle between light rays and a flat surface is negative but between light rays and inclined surface (facing southward) is positive)
        self.solar_flux_map[np.nonzero(self.solar_flux_map < 0)] = 0
            
        # dont believe there is a need for this    
        #np.savetxt(os.path.join(self.result_dir, 'solar_flux_map.txt'), self.solar_flux_map)
        
        return self.solar_flux_map
    
    def _calculate_solar_flux_factors(self, daterange, angle, timezone_diff):
        """
        This function calculates the effective solar radiation flux on a sloped surface. The method from Buffo (1972) is used, 
        assuming that the radiaton on the surface already includes the atmospheric transmission coefficient. Using the radiation data for a flat surface 
        and the angle of the incoming rays with the flat sruface, the intensity of the incoming rays can be estimated, which can then be projected on an inclined
        surface.

        Args:
            daterange (daterange): the dates for which to calculate the solar flux.
            angle (float): incline of the surface for which to calculate the enhancement factors.
            timezone_diff (float): difference in hours for the timezone which is modelled relative to UTC.

        Returns:
            array: enhancement factors for radiation for the given angle for each day in the daterange
        """       
        
        # 1) latitude and orientation
        phi = self.config.model.latitude / 360 * 2 * np.pi
        beta = (90 - self.config.bathymetry.grid_orientation) / 360 * 2 * np.pi  # clockwise from the north
        
        # 2) local angles
        alpha = angle / 360 * 2 * np.pi
        
        # 3) declination, Sarbu (2017)
        delta = 23.45 * np.sin(
            (360/365 * (284 + daterange.dayofyear.values)) / 360 * 2 * np.pi
            ) / 360 * 2 * np.pi
        
        # 4) hour angle (for Alaska timezone difference w.r.t. UTC is -8h)
        local_hour_of_day = daterange.hour.values + timezone_diff
            # convert to hour angle
        h = (((local_hour_of_day - 12) % 24)/24) * 2 * np.pi
            # convert angles to range [-pi, pi]
        mask = np.nonzero(h>=np.pi)
        h[mask] = -((2 * np.pi) - h[mask])
        
        # 5) calculate altitude angle off of the horizontal that the suns rays strike a horizontal surface
        A = np.arcsin(np.cos(phi) * np.cos(delta) * np.cos(h) + np.sin(phi) * np.sin(delta))
        
        # 6) calculate the (unmodified) azimuth
        AZ_no_mod = np.arcsin((np.cos(delta) * (np.sin(h)) / np.cos(A)))
        
        AZ_mod = np.copy(AZ_no_mod)  # create a copy of the unmodified azimuth, which will be modified
        
        # correct azimuth for when close to solstices (“Central Beaufort Sea Wave and Hydrodynamic Modeling Study Report 1: Field Measurements and Model Development,” n.d.)
        ew_AM_mask = np.nonzero(
            (A > 0) * (np.cos(h) <= np.tan(delta) / np.tan(phi)) * (local_hour_of_day <= 12)
            ) # east-west AM mask
        ew_PM_mask = np.nonzero(
            (A > 0) * (np.cos(h) <= np.tan(delta) / np.tan(phi)) * (local_hour_of_day > 12)
            ) # east-west PM mask
        
        # modify azimuth
        AZ_mod[ew_AM_mask] = -np.pi + np.abs(AZ_no_mod[ew_AM_mask])
        AZ_mod[ew_PM_mask] = np.pi - AZ_no_mod[ew_PM_mask]
        
        # convert from (clockwise from south) to (clockwise from east)
        Z = AZ_mod + 1/2 * np.pi

        # 7) calculate multiplication factor for computational domain
        sin_theta = np.sin(A) * np.cos(alpha) - np.cos(A) * np.sin(alpha) * np.sin(Z - beta)
        theta = np.arcsin(sin_theta)
        
        # 8) calculate multiplication factor for flat surface
        sin_0 = np.sin(A) * np.cos(0) - np.cos(A) * np.sin(0) * np.sin(Z - beta)
        
        # 9) in order to avoid very peaky scales, let us take the daily maximum and use that for scaling.
        sin_theta_2d = sin_theta.reshape((-1, 24))
        sin_theta_daily_max = np.max(sin_theta_2d, axis=1).flatten()

        sin_0_2d = sin_0.reshape((-1, 24))
        sin_0_daily_max = np.max(sin_0_2d, axis=1).flatten()

        # 10) calculate enhancement factor for each day
        factor = sin_theta_daily_max / sin_0_daily_max
        
        # 11) filter out values where it the angle theta is negative (as that means radiation hits the surface from below)
        shadow_mask = np.zeros(factor.shape)
        
        for i, row in enumerate(theta.reshape((-1, 24))):
            if all(row < 0):
                shadow_mask[i] = 1
            
        factor[np.nonzero(shadow_mask)] = 0
        
        return factor
        
    def write_ne_layer(self):
        """This function writes the thaw depth obtained from the thermal update to a file to be used by xbeach.
        """
        np.savetxt(os.path.join(self.cwd, "ne_layer.txt"), self.thaw_depth)
                
        return None
        
    def find_thaw_depth(self):
        """Finds thaw depth based on the z-values of the two nearest thaw points."""
        # initialize thaw depth array
        self.thaw_depth = np.zeros(self.xgr.shape)

        # get the points from the temperature models
        x_matrix, z_matrix = um.generate_perpendicular_grids(
            self.xgr, self.zgr, 
            resolution=self.config.thermal.grid_resolution, 
            max_depth=self.config.thermal.max_depth)
        
        # determine indices of thaw depth in perpendicular model
        indices = um.count_nonzero_until_n_zeros(
            self.temp_matrix > self.config.thermal.T_melt, 
            dN=(self.config.thermal.N_thaw_threshold if "N_thaw_threshold" in self.config.thermal.keys() else 1)
            )
        
        # find associated coordinates of these points
        x_thaw = x_matrix[np.arange(x_matrix.shape[0]), indices]
        z_thaw = z_matrix[np.arange(x_matrix.shape[0]), indices]
        
        # sort 
        sort_indices = np.argsort(x_thaw)
        x_thaw_sorted = x_thaw[sort_indices]
        z_thaw_sorted = z_thaw[sort_indices]
        
        # loop through the grid        
        for i, x, z in zip(np.arange(len(self.xgr)), self.xgr, self.zgr):
            # try to find two points between which to interpolate for the thaw depth, otherwise set thaw depth to 0
            try:
                mask1 = np.nonzero((x_thaw_sorted < x))
                x1 = x_thaw_sorted[mask1][-1]
                z1 = z_thaw_sorted[mask1][-1]
                
                mask2 = np.nonzero((x_thaw_sorted >= x))
                x2 = x_thaw_sorted[mask2][0]
                z2 = z_thaw_sorted[mask2][0]
                
                z_thaw_interpolated = z1 + (z2 - z1)/(x2 - x1) * (x - x1)
                
                self.thaw_depth[i] = z - (z_thaw_interpolated)
            except:
                self.thaw_depth[i] = 0
        
        # ensure thaw depth is larger than zero everywhere
        self.thaw_depth = np.max(np.column_stack((self.thaw_depth, np.zeros(self.thaw_depth.shape))), axis=1)
        
        return self.thaw_depth
        
    ################################################
    ##                                            ##
    ##            # OUTPUT FUNCTIONS              ##
    ##                                            ##
    ################################################
        
    ################################################
    ##                                            ##
    ##          # SPIN-UP SAVE / RESTORE          ##
    ##                                            ##
    ################################################
    # XBeach does not run during the spin-up, so the state at the end of the spin-up is the thermal state on the
    # unchanged initial bathymetry. It is saved to one .npz file (model.spinup_state_out) and can replace the spin-up in
    # later runs (model.spinup_state_in), e.g. all trials of a coupled calibration.
    SPINUP_STATE_VERSION = 1
    _SPINUP_ARRAYS = (
        "enthalpy_matrix", "temp_matrix", "k_matrix", "ghost_nodes_temperature", "bottom_boundary_temperature",
        "nb_matrix", "soil_density_matrix", "Cs_matrix", "Cl_matrix", "latent_matrix", "k_frozen_matrix", "k_unfrozen_matrix",
        "xgr", "zgr", "abs_xgr", "abs_zgr", "angles", "thaw_depth",
        "convective_flux", "heat_flux", "latent_flux", "lw_flux", "sw_flux", "constant_flux", "factors", "heat_flux_factors",
    )
    _SPINUP_SCALARS = (
        "water_level", "current_air_temp", "current_sea_temp", "current_sea_ice", "wind_direction", "wind_velocity",
    )

    def count_spinup_steps(self, spinup_seconds):
        """Number of leading timesteps that belong to the spin-up (same criterion as the main loop)."""
        elapsed = np.asarray((self.timestamps - self.timestamps[0]) / pd.Timedelta("1s"))
        return int(np.sum(elapsed < spinup_seconds))

    @staticmethod
    def _hash_json(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalised_thermal(thermal):
        """Thermal config as a plain dict with the effective value of every option that has a default."""
        thermal = dict(thermal)
        for key, default in (("conduction_form", "nodal"), ("latent_heat_form", "per_mass"), ("ghost_node_order", 2)):
            thermal.setdefault(key, default)
        return thermal

    @classmethod
    def thermal_state_hash(cls, thermal):
        """Hash of a thermal config section as stored in a spin-up state file (for checks before a run is started)."""
        return cls._hash_json(cls._normalised_thermal(thermal))

    def _spinup_fingerprint(self, n_spin):
        """What the spin-up state depends on: the thermal config (hashed, effective values of the options that have a
        default), the run context (start time, spin-up length, timestep, forcing options) and the forcing used."""
        thermal = self._normalised_thermal(self.config.thermal)
        context = dict(
            time_start=str(self.t_start), timestep_hours=self.dt, n_spinup_steps=int(n_spin),
            era5_flux_shift_hours=self.config.data.get("era5_flux_shift_hours", 0),
            conceptual={k: self.config.data.get(k) for k in ("conceptual", "conceptual_temp", "conceptual_ini", "conceptual_flux")},
            sensitivity=dict(self.config.sensitivity) if "sensitivity" in self.config else None,
        )
        forcing = np.ascontiguousarray(self.forcing_data.iloc[:n_spin].select_dtypes("number").to_numpy(dtype=float))
        return thermal, context, hashlib.sha256(forcing.tobytes()).hexdigest()

    def save_spinup_state(self, path, n_spin):
        """Save the model state at the end of the spin-up (call after the last spin-up timestep, n_spin = number of
        spin-up timesteps). Everything needed to continue bit-identically at timestep n_spin goes into one npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {n: np.asarray(getattr(self, n)) for n in self._SPINUP_ARRAYS if hasattr(self, n)}
        scalars = {n: float(getattr(self, n)) for n in self._SPINUP_SCALARS if hasattr(self, n)}
        thermal, context, forcing_hash = self._spinup_fingerprint(n_spin)
        meta = dict(
            version=self.SPINUP_STATE_VERSION, n_spinup_steps=int(n_spin),
            next_timestamp=str(self.timestamps[n_spin]) if n_spin < len(self.timestamps) else None,
            thermal=thermal, thermal_hash=self._hash_json(thermal),
            context=context, context_hash=self._hash_json(context), forcing_hash=forcing_hash,
            arrays=list(arrays), scalars=scalars,
        )
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:  # file object: np.savez must not add a suffix
            np.savez(f, __meta__=np.array(json.dumps(meta, default=str)), **arrays)
        os.replace(tmp, path)
        return path

    def load_spinup_state(self, path, n_spin):
        """Restore a state saved by save_spinup_state and return the timestep id at which the main loop continues.

        Raises ValueError (nothing is changed) if the thermal config, the run context, the forcing during the
        spin-up or the initial grid differ from those stored in the file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Spin-up state file not found: {path}")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            arrays = {k: z[k] for k in meta["arrays"]}
        if meta["version"] != self.SPINUP_STATE_VERSION:
            raise ValueError(f"Spin-up state {path} has format version {meta['version']}, expected {self.SPINUP_STATE_VERSION}")

        thermal, context, forcing_hash = self._spinup_fingerprint(n_spin)
        problems = []
        if self._hash_json(thermal) != meta["thermal_hash"]:
            old = json.loads(json.dumps(meta["thermal"], default=str))
            new = json.loads(json.dumps(thermal, default=str))
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            problems.append(f"thermal config differs from the one stored in the state (keys: {', '.join(diff)})")
        if self._hash_json(context) != meta["context_hash"]:
            old, new = meta["context"], json.loads(json.dumps(context, default=str))
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            problems.append(f"run context differs (keys: {', '.join(diff)})")
        if forcing_hash != meta["forcing_hash"]:
            problems.append("the forcing data during the spin-up differ from those used to make the state")
        if not (np.array_equal(arrays["xgr"], self.xgr) and np.array_equal(arrays["zgr"], self.zgr)):
            problems.append("the initial grid/bathymetry differs from the one stored in the state")
        if meta["n_spinup_steps"] != n_spin:
            problems.append(f"spin-up length differs ({meta['n_spinup_steps']} steps in the state, {n_spin} now)")
        if problems:
            raise ValueError(f"Cannot restart from {path}: " + "; ".join(problems))

        for name, value in arrays.items():
            setattr(self, name, value.copy())
        for name, value in meta["scalars"].items():
            setattr(self, name, value)
        # bookkeeping the main loop would have left behind: XBeach was off in every spin-up step
        self.xbeach_times[:n_spin] = 0
        self._ds_cache = None
        return int(n_spin)

    def cleanup(self):
        """
        Cleanup method to delete variables and free memory at the end of the simulation.
        """
        # Delete large attributes
        # removed xgr, zgr from this list => do we really need this as output?
        attributes_to_delete = [
            "forcing_data", "temp_matrix", "enthalpy_matrix", "k_matrix",
            "soil_density_matrix", "nb_matrix", "abs_xgr",
            "abs_zgr", "angles", "timestamps", "conditions",
            "convective_flux", "heat_flux", "lw_flux", "sw_flux", "latent_flux",
            "thaw_depth", "temperature_timeseries", "solar_flux_map"
        ]

        # Deleting them
        for attr in attributes_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)

        # Force garbage collection
        import gc
        gc.collect()
        logger.info("Simulation cleanup completed. Memory has been freed.")


    def write_output(self, timestep_id, t_start):

        """Append one time-slice to results.nc."""
        self._ensure_nc_writer()

        # time index and seconds since t0
        if "time" in self.nc_writer.v:
            i = len(self.nc_writer.v["time"])  # next index
        else:
            i = 0
        # Count how many XBeach timesteps have been written so far
        # For output: count previous XBeach runs (excluding current) to get correct index
        if "time_xbeach" in self.nc_writer.v:
            i_xb = sum(self.xbeach_times[:timestep_id] == 1.0)
        else:
            i_xb = 0

        # Determine times
        t_now = float((self.timestamps[timestep_id] - self.timestamps[0]) / pd.Timedelta("1s"))
        t_rel = t_now - self._t0_seconds

        # --- hydrodynamics on xbeach grid ---
        # Robustly get xgr_xb from nc_writer, fallback to self.xgr if missing/empty
        try:
            xgr_xb = self.nc_writer.v["xgr_xb"][:]
            if xgr_xb is None or len(xgr_xb) == 0:
                xgr_xb = self.xgr
        except Exception:
            xgr_xb = self.xgr

        # Check XBeach output
        xb_folder   = os.path.join(self.result_dir, "xb_files/", (10 - len(str(int(timestep_id)))) * '0' + str(int(timestep_id)) + '/')
        xb_ok       = os.path.exists(os.path.join(xb_folder, "xboutput.nc")) 
        if xb_ok:
            try:
                ds = xr.load_dataset(os.path.join(xb_folder, "xboutput.nc")).squeeze()
                ds = ds.sel(globaltime=np.max(ds.globaltime.values))
                # pad/truncate helper
                def fit1(a, L):
                    a = np.asarray(a).ravel().astype("f4")
                    if len(a) == L: return a
                    out = np.full((L,), np.nan, dtype="f4")
                    out[:min(L,len(a))] = a[:min(L,len(a))]
                    return out
                H    = fit1(ds.H.values,    len(xgr_xb))
                zb   = fit1(ds.zb.values,   len(xgr_xb))
                zs   = fit1(ds.zs.values,   len(xgr_xb))
                E    = fit1(ds.E.values,    len(xgr_xb))
                Sxx  = fit1(ds.Sxx.values,  len(xgr_xb))
                Sxy  = fit1(ds.Sxy.values,  len(xgr_xb))
                Syy  = fit1(ds.Syy.values,  len(xgr_xb))
                vmag = fit1(ds.vmag.values, len(xgr_xb))
                urms = fit1(ds.urms.values, len(xgr_xb))
                ds.close()
            except Exception as e:
                logger.warning(f"XBeach output file exists but could not be loaded for timestep {timestep_id} ({self.timestamps[timestep_id]}). "
                              f"Error: {str(e)}. Using zero values for XBeach variables.")
                Z = np.zeros_like(xgr_xb, dtype="f4")
                H=zb=zs=E=Sxx=Sxy=Syy=vmag=urms = Z
        else:
            Z = np.zeros_like(xgr_xb, dtype="f4")
            H=zb=zs=E=Sxx=Sxy=Syy=vmag=urms = Z

        # water level on xgr
        wl_line = np.asarray(self._update_water_level(timestep_id), dtype="f4")
        if wl_line.ndim == 0:  # if scalar returned, expand to xgr
            wl_line = np.full((len(self.xgr),), wl_line, dtype="f4")

        # pack 2D thermo fields (xgr, depth_id)
        gt  = self.temp_matrix.astype("f4")
        ge  = self.enthalpy_matrix.astype("f4")
        nb  = self.nb_matrix.astype("f4")
        k   = self.k_matrix.astype("f4")
        rho = self.soil_density_matrix.astype("f4")

        # append thermal parameters (use default "time" dimension)
        self.nc_writer.append(i, {
            "time": t_rel,
            "timestep_id":  int(timestep_id),
            "cumtime":      float(time.time() - t_start),
            "zgr": self.zgr.astype("f4"),
            "angles": self.angles.astype("f4"),
            "abs_xgr": self.abs_xgr.astype("f4"),
            "abs_zgr": self.abs_zgr.astype("f4"),
            "ground_temperature_distribution": gt,
            "ground_enthalpy_distribution":    ge,
            "nb":  nb,
            "k":   k,
            "rho": rho,
            "solar_radiation_factor":   self.factors.astype("f4"),
            "solar_radiation_flux":     self.sw_flux.astype("f4"),
            "long_wave_radiation_flux": self.lw_flux.astype("f4"),
            "latent_heat_flux":         self.latent_flux.astype("f4"),
            "convective_heat_flux":     self.convective_flux.astype("f4"),
            "total_heat_flux":          self.heat_flux.astype("f4"),
            "thaw_depth":               self.thaw_depth.astype("f4"),
            "air_temperature_2m":       float(self.current_air_temp),
            "sea_surface_temperature":  float(self.current_sea_temp),
            "sea_ice_cover":            float(self.current_sea_ice),
            "wind_velocity":            float(self.wind_velocity),
            "wind_direction":           float(self.wind_direction),
            "water_level_offshore":     self.conditions[timestep_id]['WL(m)'],
            "wave_height_offshore":     self.conditions[timestep_id]['Hs(m)'],
            "beta_f":                   float(self.beta_f[timestep_id]),
            "run_up2pct":               float(self.R2[timestep_id]),
        }, time_dim="time")
        
        # append XBeach parameters (use "time_xbeach" dimension)
        # Only write XBeach data if XBeach was run for this timestep
        if (self.xbeach_times[timestep_id] == 1.0):
            # Use a separate index for XBeach time (could be fewer timesteps)
            self.nc_writer.append(i_xb, {
                "time_xbeach": t_rel,
                "wave_height_xbeach": H,
                "zb_xbeach": zb,
                "zs_xbeach": zs,
                "wave_energy": E,
                "radiation_stress_xx": Sxx,
                "radiation_stress_xy": Sxy,
                "radiation_stress_yy": Syy,
                "velocity_magnitude": vmag,
                "orbital_velocity": urms,
            }, time_dim="time_xbeach")

    
    def save_ground_temp_layers_in_memory(self, timestep_id, layers=[], heat_fluxes=[], write=False):
        """This function saves the ground temperature directly into a single dataframe, 
        which is helpful for validation purposes.

        Args:
            timestep_id (int): id of the current timestep
            layers (list, optional): list of the layers to save. Defaults to [].
            heat_fluxes (list, optional): list of heat fluxes to save. Defaults to [].
            write (bool, optional): whether or not to write. Defaults to False.
        """
        # define colnames
        col_names = ['time'] + ['air_temp[K]'] + [f'temp_{layer}m[K]' for layer in layers] + heat_fluxes
        
        values = [self.timestamps[timestep_id], self.current_air_temp]
        
        # loop through layers to find corresponding temperature
        for layer in layers:
            
            index_x = int(self.temp_matrix.shape[0] // 2)
            index_z = int(layer * self.config.thermal.grid_resolution // self.config.thermal.max_depth)
            
            values.append(self.temp_matrix[index_x, index_z])
            
        # find heat fluxes
        values.append(self.heat_flux[index_x])
        values.append(self.lw_flux[index_x])
        values.append(self.sw_flux[index_x])
        values.append(self.latent_flux[index_x])
        values.append(self.convective_flux[index_x])

        # create dataframe at first timestep
        if timestep_id == 0 or not hasattr(self, "temperature_timeseries"):  # (second case: restart from a spin-up state)

            self.temperature_timeseries = pd.DataFrame(dict(zip(col_names, values)), index=[0])
            
        else:
            
            # add temperature and heat fluxs to dataframe
            self.temperature_timeseries = self.temperature_timeseries._append(
                dict(zip(col_names, values)), ignore_index=True
            )
                
        # write output at final timestep
        if write:
            self.temperature_timeseries.to_csv(
                os.path.join(self.result_dir, f"ground_temperature_timeseries.csv")
            )
            
        return None
        

        
    # functions below are used to quickly obtain values for forcing data
    def _get_sw_flux(self, timestep_id):
        return self.forcing_data["mean_surface_net_short_wave_radiation_flux.csv"].values[timestep_id]
    def _get_lw_flux(self, timestep_id):
        return self.forcing_data["mean_surface_net_long_wave_radiation_flux.csv"].values[timestep_id]
    def _get_latent_flux(self, timestep_id):
        return self.forcing_data["mean_surface_latent_heat_flux.csv"].values[timestep_id]
    def _get_snow_depth(self, timestep_id):
        return self.forcing_data["snow_depth.csv"].values[timestep_id]
    def _get_sea_ice(self, timestep_id):
        return self.forcing_data["sea_ice_cover.csv"].values[timestep_id]
    def _get_2m_temp(self, timestep_id):
        return self.forcing_data["2m_temperature.csv"].values[timestep_id]
    def _get_sea_temp(self, timestep_id):
        return self.forcing_data["sea_surface_temperature.csv"].values[timestep_id]
    def _get_u_wind(self, timestep_id):
        return self.forcing_data["10m_u_component_of_wind.csv"].values[timestep_id]
    def _get_v_wind(self, timestep_id):
        return self.forcing_data["10v_u_component_of_wind.csv"].values[timestep_id]
    def _get_soil_temp(self, timestep_id, level=1):
        if not level in [1, 2, 3, 4]:
            raise ValueError("'level' variable should have a value of 1, 2, 3, or 4")
        return self.forcing_data[f"soil_temperature_level_{level}.csv"].values[timestep_id]

    def _get_timeseries(self, tstart, tend, fpath, flux_shift_hours=0):
        """returns timeseries start from tstart and ending at tend. The filepath has to be specified.

        flux_shift_hours (int): ERA5 'mean_surface_*' fluxes are mean rates over the hour *ending* at the time label, while
        the thermal model applies row t over [t, t+1h]. With flux_shift_hours = 1 the three mean_surface_* columns are shifted
        back one (hourly) record, so row t holds the mean over [t, t+1h]. 0 (default) applies no shift. The shift is done on
        the full file, before the time mask, so the last model hour is also filled.

        returns: pd.DataFrame of length T"""

        # read forcing file
        with open(fpath) as f:
            df = pd.read_csv(f, parse_dates=['time'])

            if flux_shift_hours:
                flux_cols = [c for c in df.columns if c.startswith('mean_surface_')]
                df[flux_cols] = df[flux_cols].shift(-int(flux_shift_hours)).ffill()

            # mask out correct time frame
            mask = (df["time"] >= tstart) * (df["time"] < tend)
            
            # repeat time frame if required
            df = pd.concat([df[mask]] * 1, ignore_index=True)
                        
        return df



class NCAppender:
    """
    Create results.nc (once) and append a single time-slice each timestep.
    Dimensions are fixed-length except 'time', which is unlimited.
    """
    
    # ini netcdf
    def __init__(self, path, xgr: np.ndarray, depth_id: np.ndarray, xgr_xb: np.ndarray, tstart: str = "1970-01-01T00:00:00"):
        self.path   = Path(path)
        create      = True
        self.ds     = Dataset(self.path, "w", format="NETCDF4")         # overwrite old netcdf

        # Define all
        if create:

            # Add global attributes
            self.ds.setncattr("Producer", "Arctic XBeach")
            self.ds.setncattr("Title", "Arctic XBeach Thermo-Morphological Model Results")
            self.ds.setncattr("Institution", "Deltares")
            self.ds.setncattr("Source", "Arctic XBeach - Coupled thermo-morphological coastal model")
            self.ds.setncattr("History", f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            self.ds.setncattr("Conventions", "CF-1.8")
            self.ds.setncattr("References", "https://github.com/openearth/xbeach-toolbox")
            self.ds.setncattr("Comment", "Arctic XBeach model output combining thermal and morphological processes")
            
            self.ds.setncattr("Version", __version__)
            self.ds.setncattr("Revision", git_revision())
            
            # Add creation date
            self.ds.setncattr("Date_created", datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))

            # dimensions
            self.ds.createDimension("time", None)                       # unlimited time dimension for thermal parameters
            self.ds.createDimension("time_xbeach", None)                # unlimited time dimension for xbeach parameters
            self.ds.createDimension("xgr", int(len(xgr)))               # dimension of thermal x
            self.ds.createDimension("xgr_xb", int(len(xgr_xb)))         # dimension of XBeach grid (only 1D supprt)
            self.ds.createDimension("depth_id", int(len(depth_id)))     # dimension of thermal z (depth)

            # coordinate variables
            v = self.ds.createVariable("time", "f8", ("time",))     # seconds since tstart (thermal)
            v.units = f"seconds since {tstart}"
            v.calendar = "standard"
            v = self.ds.createVariable("time_xbeach", "f8", ("time_xbeach",))     # seconds since tstart (xbeach)
            v.units = f"seconds since {tstart}"
            v.calendar = "standard"
            v = self.ds.createVariable("xgr", "f4", ("xgr",))
            v[:] = xgr.astype("f4")
            v.units = "m"
            v.long_name = "cross-shore distance"
            v.standard_name = "distance"
            v.axis = "X"
            v = self.ds.createVariable("depth_id", "i4", ("depth_id",))
            v[:] = depth_id.astype("i4")
            v.units = "1"
            v.long_name = "depth level index"
            v.standard_name = "depth_level"
            v.axis = "Z"
            v = self.ds.createVariable("xgr_xb", "f4", ("xgr_xb",))
            v[:] = xgr_xb.astype("f4")
            v.units = "m"
            v.long_name = "cross-shore distance"
            v.standard_name = "distance"
            v.axis = "X"

            # helper to make time-varying vars (thermal time dimension)
            def v1(name, dims_tail, dtype="f4", **kw):
                # Calculate optimal chunk sizes for time-series data
                # For 3D data (time, xgr, depth_id), use chunks of (1, len(xgr), len(depth_id))
                # For 2D data (time, xgr), use chunks of (1, len(xgr))
                # For 1D data (time,), use chunks of (100,)
                if dims_tail:
                    chunk_shape = (1,) + tuple(self.ds.dimensions[d].size if d in self.ds.dimensions else 1 for d in dims_tail)
                else:
                    chunk_shape = (100,)  # For scalar time series
                return self.ds.createVariable(name, dtype, ("time",) + tuple(dims_tail), 
                                             zlib=True, complevel=4, shuffle=True, 
                                             chunksizes=chunk_shape, **kw)
            
            # helper to make time-varying vars (xbeach time dimension)
            def v1_xb(name, dims_tail, dtype="f4", **kw):
                if dims_tail:
                    chunk_shape = (1,) + tuple(self.ds.dimensions[d].size if d in self.ds.dimensions else 1 for d in dims_tail)
                else:
                    chunk_shape = (100,)
                return self.ds.createVariable(name, dtype, ("time_xbeach",) + tuple(dims_tail), 
                                             zlib=True, complevel=4, shuffle=True,
                                             chunksizes=chunk_shape, **kw)

            # time/meta (thermal time)
            self.v = {
                "time":         self.ds.variables["time"],
                "time_xbeach":  self.ds.variables["time_xbeach"],
                "timestep_id":  v1("timestep_id", (), "i4"),
                "cumtime":      v1("cumtime", (), "f4"),
            }

            # geometry (time-varying to allow morphodynamics) - use thermal time
            self.v["zgr"]                       = v1("zgr", ("xgr",))
            self.v["zgr"].coordinates           = "xgr"
            self.v["angles"]                    = v1("angles", ("xgr",))
            self.v["angles"].coordinates        = "xgr"
            self.v["abs_xgr"]                   = v1("abs_xgr", ("xgr","depth_id"))
            self.v["abs_xgr"].coordinates       = "xgr depth_id"
            self.v["abs_zgr"]                   = v1("abs_zgr", ("xgr","depth_id"))
            self.v["abs_zgr"].coordinates       = "xgr depth_id"

            # hydrodynamics on xbeach grid - use xbeach time
            for name, dims in {
                "wave_height_xbeach": ("xgr_xb",),
                "zb_xbeach": ("xgr_xb",),
                "zs_xbeach": ("xgr_xb",),
                "wave_energy": ("xgr_xb",),
                "radiation_stress_xx": ("xgr_xb",),
                "radiation_stress_xy": ("xgr_xb",),
                "radiation_stress_yy": ("xgr_xb",),
                "velocity_magnitude": ("xgr_xb",),
                "orbital_velocity": ("xgr_xb",),
            }.items():
                self.v[name] = v1_xb(name, dims)
                self.v[name].coordinates = "xgr_xb"

            # thermo (xgr, depth_id) - use thermal time
            for name in ("ground_temperature_distribution","ground_enthalpy_distribution","nb","k","rho"):
                self.v[name] = v1(name, ("xgr","depth_id"))
                self.v[name].coordinates = "xgr depth_id"

            # fluxes on xgr - use thermal time
            flux_attrs = {
                "solar_radiation_factor": dict(units="1", long_name="solar radiation factor", standard_name="solar_radiation_factor"),
                "solar_radiation_flux": dict(units="W m-2", long_name="solar radiation flux", standard_name="surface_downwelling_shortwave_flux_in_air"),
                "long_wave_radiation_flux": dict(units="W m-2", long_name="net long wave radiation flux", standard_name="surface_net_longwave_flux"),
                "latent_heat_flux": dict(units="W m-2", long_name="surface latent heat flux", standard_name="surface_upward_latent_heat_flux"),
                "convective_heat_flux": dict(units="W m-2", long_name="surface convective heat flux", standard_name="surface_upward_sensible_heat_flux"),
                "total_heat_flux": dict(units="W m-2", long_name="total surface heat flux", standard_name="surface_upward_heat_flux"),
                "thaw_depth": dict(units="m", long_name="thaw depth below surface", standard_name="thaw_depth"),
            }
            for name, attrs in flux_attrs.items():
                self.v[name] = v1(name, ("xgr",))
                self.v[name].coordinates = "xgr"
                for k, v_attr in attrs.items():
                    setattr(self.v[name], k, v_attr)

            # forcings (scalars) - use thermal time
            for name, dtype in {
                "air_temperature_2m": "f4",
                "sea_surface_temperature": "f4",
                "sea_ice_cover": "f4",
                "wind_velocity": "f4",
                "wind_direction": "f4",
                "water_level_offshore": "f4",
                "wave_height_offshore": "f4"
            }.items():
                self.v[name] = v1(name, (), dtype)

            # water/runup scalars & xgr fields - use thermal time for all since they're written with thermal time
            self.v["beta_f"]      = v1("beta_f", ())
            self.v["run_up2pct"]  = v1("run_up2pct", ())

        else:
            # reopen handles
            self.v = {name: self.ds.variables[name] for name in self.ds.variables}

    # Reopening netcdf
    def _reopen(self):
        # Only reopen if self.ds is None
        if getattr(self, "ds", None) is None:
            self.ds = Dataset(self.path, "a", format="NETCDF4")
            self.v = {name: self.ds.variables[name] for name in self.ds.variables}

    # Appending netcdf
    def append(self, i: int, arrays: dict, time_dim="time"):
        """
        i: index in the time dimension to write
        arrays: mapping var_name -> np.ndarray or scalar
        time_dim: either "time" (thermal) or "time_xbeach" (xbeach)
        Shapes must match variable dims (excluding time dimension).
        """
        self._reopen()
        
        # Update the appropriate time coordinate
        if time_dim == "time":
            self.v["time"][i] = arrays.get("time", i)
        elif time_dim == "time_xbeach":
            self.v["time_xbeach"][i] = arrays.get("time_xbeach", i)
        
        for name, data in arrays.items():
            if name in ["time", "time_xbeach"]:  # Skip time coordinates, handled above
                continue
                
            var = self.v[name]
            
            # Check which time dimension this variable uses
            var_time_dim = var.dimensions[0] if var.ndim > 0 else None
            
            # Only write if this variable uses the current time dimension
            if var_time_dim == time_dim:
                if var.ndim == 1:          # ("time",) or ("time_xbeach",)
                    var[i] = data
                elif var.ndim == 2:        # ("time", X) or ("time_xbeach", X)
                    var[i, :] = data
                elif var.ndim == 3:        # ("time", X, Y) or ("time_xbeach", X, Y)
                    var[i, :, :] = data
                else:
                    var[i, ...] = data

    # Closing netcdf
    def close(self):
        self.ds.sync()
        self.ds.close()
