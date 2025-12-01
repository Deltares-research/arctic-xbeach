import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.signal import convolve


def textbox(text):
    """Function to quickly generate text boxes.

    Args:
        text (str): string of the text that should be in the textbox.
        
    Returns:
        str: String representation of the input text with a textbox around it.
    """
    row_length = len(text) + 6
    
    row0 = row_length * "%" + "\n"
    row1 = "%" + (row_length-2) * " " + "%" + "\n"
    row2 = f"%  {text}  %" + "\n"
    row3 = row1
    row4 = row_length * "%"

    total_text = row0 + row1 + row2 + row3 + row4
    
    return total_text

def interpolate_points(x, y, num_points):
    """
    Interpolate points linearly between given points (x, y).
    
    Parameters:
        x (array-like): X coordinates of the given points.
        y (array-like): Y coordinates of the given points.
        num_points (int): Number of points to interpolate between the given points.
    
    Returns:
        np.ndarray: Array of interpolated points.
    """
    # Generate interpolated points using linspace
    interpolated_x = np.linspace(x[0], x[-1], num_points)
    interpolated_y = np.interp(interpolated_x, x, y)
    
    # Combine provided points and interpolated points
    all_points = np.column_stack((interpolated_x, interpolated_y))
    
    return all_points

def get_A_matrix(n):
    """ This function is used to get the 'A' matrix, which is used to make the numerical scheme faster. It is based on second order central differences for internal points.
        Att the border points, the grid is extended with an identical point (i.e. mirrored), in order to calculate the second derivative.

    Args:
        n (int): number of points in the grid
    """
    # Initialize an empty square matrix
    matrix = np.zeros((n+2, n))

    # Fill the super diagonal with 1
    np.fill_diagonal(matrix[:-2,:], 1)

    # Fill the main diagonal with -2
    np.fill_diagonal(matrix[1:-1,:], -2)

    # Fill the sub-diagonal with 1
    np.fill_diagonal(matrix[2:,:], 1)

    return matrix

def count_nonzero_until_zero(matrix):
    """Returns the number of grid points with a nonzero input, counted for each row from the lowest index until the first zero input.

    Args:
        matrix (np.array): a matrix

    Returns:
        result: number of grid points for each row before a zero input (-1 if no zeros in the entire row)
    """
    
    matrix = np.column_stack((matrix, np.zeros(matrix.shape[0])))
    
    indices = np.argmax(matrix == 0, axis=1)
    
    mask = (indices == matrix.shape[1]-1)
    
    indices[mask] = -1
    
    return indices

def count_nonzero_until_n_zeros(matrix, dN=1):
    """Returns the number of grid points with a nonzero input, counted for each row from the lowest index until the first dN consecutive zero inputs.

    Args:
        matrix (np.array): a matrix
        dN (int): integer indicating the minimum amount of consecutive zeros needed to stop counting (from the top)

    Returns:
        result: number of grid points for each row before dN consecutive zero inputs (-1 if no dN consecutive zeros in the entire row)
    """
    
    # Create the convolution kernel to detect dN consecutive zeros
    kernel = np.ones(dN)
    
    # Convolve the matrix rows with the kernel
    convolved = convolve(matrix == 0, kernel[None, :], mode='same')
    
    # Find the index of the first occurrence of dN consecutive zeros in each row
    indices = np.argmax(convolved >= dN, axis=1)
    
    # If the first occurrence is at the end, we need to check if it's a valid detection
    valid_indices = convolved[np.arange(convolved.shape[0]), indices] >= dN
    indices[~valid_indices] = -1
    indices[valid_indices] -= dN//2
    
    return indices

def generate_perpendicular_grids(xgr, zgr, resolution=30, max_depth=3):
    """This function takes an xgrid and a zgrid, as well as a resolution and maximum depth, and returns a (temperature) grid perpendicular to the existing x and z-grid.
    ----------
    xgr: array
        1D array of x-values
    zgr: array
        1D array of z-values
    resolution: integer (default: 30)
        number of desired grid points in the 1D models
    max_depth: float (default: 3)
        maximum depth to be modelled
    ---------Returns----------
    Returns: x_matrix, z_matrix
    
    x_matrix: array
        2D array of x-values. Each row contains the x-values for a 1D model
    z_matrix: array
        2D array of z-values. Each row contains the z-values for a 1D model
    """
    temp_depth_grid = np.linspace(0, max_depth, resolution)
    initial_temp = np.linspace(-5, 5, resolution)

    thermal_matrix = np.zeros((xgr.shape[0], temp_depth_grid.shape[0]))

    for i in range(len(thermal_matrix)):
        thermal_matrix[i,:] = initial_temp
        
    gradient = np.gradient(zgr, xgr)

    orientation = np.arctan(gradient) - 0.5 * np.pi
        
    x_matrix = np.tile(xgr, (len(temp_depth_grid), 1)).T + np.outer(np.cos(orientation), temp_depth_grid)
    z_matrix = np.tile(zgr, (len(temp_depth_grid), 1)).T + np.outer(np.sin(orientation), temp_depth_grid)
    
    return x_matrix, z_matrix

def linear_interp_with_nearest(xgr, zgr, values, new_xgr, new_zgr):
    """This function takes an old x-grid and z-grid, and associated values (e.g., temperature), and casts them to a new grid of x and z values. 
    Linear interpolation is used for new grid points within the convex hull of the old grid, and a nearest value is assigned for values outside of the convex hull.

    Args:
        xgr (np.Array): x-grid (m x n), with every 1D model being assigned to a row
        zgr (np.Array): z-grid (m x n), with every 1D model being assigned to a row
        values (np.Array): matrix of same shape as xgr and zgr with associated valeus
        new_xgr (np.Array): new x-grid (p x q)
        new_zgr (np.Array): new z-grid (q x q)

    Returns:
        np.Array: new matrix of associated values for the new grid (shape p x q)
    """
    # Flatten the input arrays
    x_flat = xgr.flatten()
    z_flat = zgr.flatten()
    
    new_x_flat = new_xgr.flatten()
    new_z_flat = new_zgr.flatten()
    
    temperature_flat = values.flatten()

    # Create the interpolator
    interp_func = LinearNDInterpolator(list(zip(x_flat, z_flat)), temperature_flat)

    # Interpolate temperature values for new points
    new_temperature_values = interp_func(new_x_flat, new_z_flat)
    
    for i in range(len(new_temperature_values)):
        if np.isnan(new_temperature_values[i]):
            new_temperature_values[i] = temperature_flat[np.argmin(np.sqrt((x_flat - new_x_flat[i])**2 + (z_flat - new_z_flat[i])**2))]
             
    # Reshape the interpolated temperature values to match the shape of new_x_points and new_y_points
    new_temperature_values = new_temperature_values.reshape(new_xgr.shape)

    return new_temperature_values

def linear_interp_z(abs_xgr, abs_zgr, temp_matrix, abs_xgr_new, abs_zgr_new, water_level, fill_value_top_water, fill_value_top_air='nearest', fill_bottom='nearest'):
    """This function takes an old x-grid and z-grid, and associated values (e.g., temperature), and casts them to a new grid of x and z values. 
    For each new 1D perpendicular model, the closest surface coordinate from the old grid is taken, and new temperature values are based on the 
    old perpendicular model related to this surface node. New values are computed through interpolation based on depth only. Grid points above the
    surface get a new set temperature (given as argument), and grid cells below the old grid get set to the nearest value.

    Args:
        xgr (np.Array): x-grid (m x n), with every 1D model being assigned to a row
        zgr (np.Array): z-grid (m x n), with every 1D model being assigned to a row
        values (np.Array): matrix of same shape as xgr and zgr with associated valeus
        new_xgr (np.Array): new x-grid (p x q)
        new_zgr (np.Array): new z-grid (q x q)
        water_level (float): current water level
        fill_value_top_water (float): value given to grid cells above the old grid (below the water level)
        fill_value_top_air (float or string): if 'nearest', set temperature to nearest grid cell. If float, use that value.
        fill_bottom (string): strategy to use for setting bottom values below the old grid. Defaults to 'nearest'.

    Returns:
        np.Array: new matrix of associated values for the new grid (shape p x q)
    """
    # initialize new temperature matrix
    new_temperature_values = np.zeros(abs_xgr_new.shape)
    
    for i in range(len(new_temperature_values)):
        
        grid_id = np.argmin(np.abs(abs_xgr_new[i, 0] - abs_xgr[:, 0]))
        
        z_array = abs_zgr[grid_id,:]
        temp_array = temp_matrix[grid_id,:]
        
        sort_id = np.argsort(z_array)
        
        new_z_array = abs_zgr_new[i,:]
        
        if new_z_array[0] < water_level:
            new_temp_array = np.interp(
                x=new_z_array, 
                xp=z_array[sort_id], 
                fp=temp_array[sort_id], 
                left=(temp_array[sort_id][0] if fill_bottom=='nearest' else 0),
                right=fill_value_top_water if not fill_value_top_water=='nearest' else temp_array[sort_id][-1]
                )
            
        else:
            new_temp_array = np.interp(
                x=new_z_array, 
                xp=z_array[sort_id], 
                fp=temp_array[sort_id], 
                left=(temp_array[sort_id][0] if fill_bottom=='nearest' else 0),
                right=(temp_array[sort_id][-1] if fill_value_top_air=='nearest' else fill_value_top_air)
            )
        
        new_temperature_values[i,:] = new_temp_array
    
    return new_temperature_values

def datetime_from_timestamp(t):
    return datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
    
    
# def get_time_vector(datasets):
#     '''Returns array with correct temporal stamps'''

#     time_stop = datasets.t.shape[-1]
#     temporal_res = datasets[0].temporal_res

#     time_vector = np.linspace(0, (time_stop*temporal_res), time_stop)

#     return time_vector

# def plot_line_with_deviation(time_vector, variable, ax=None, **plt_kwargs):
#     """
#     >>>>>FROM<<<<<

#     @Article{hess-27-4227-2023,
#     AUTHOR = {Bentivoglio, R. and Isufi, E. and Jonkman, S. N. and Taormina, R.},
#     TITLE = {Rapid spatio-temporal flood modelling via hydraulics-based graph neural networks},
#     JOURNAL = {Hydrology and Earth System Sciences},
#     VOLUME = {27},
#     YEAR = {2023},
#     NUMBER = {23},
#     PAGES = {4227--4246},
#     URL = {https://hess.copernicus.org/articles/27/4227/2023/},
#     DOI = {10.5194/hess-27-4227-2023}
#     }
    
#     """
#     ax = ax or plt.gca()

#     df = pd.DataFrame(np.vstack((time_vector, variable))).T
#     df = df.rename(columns={0: "time"})
#     df = df.set_index('time')

#     mean = df.mean(1)
#     std = df.std(1)
#     under_line = (mean - std)
#     over_line = (mean + std)

#     p = ax.plot(mean, linewidth=2, marker='o', **plt_kwargs)
#     color = p[0].get_color()
#     ax.fill_between(std.index, under_line, over_line, color=color, alpha=.3)
#     return p