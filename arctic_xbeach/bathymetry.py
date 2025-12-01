import os

import numpy as np


def calculate_shoreline_position(xgr, zgr, cross_value=0):
    """Returns the x value at which the zgrid passes some horizontal line.

    Args:
        xgr (array): array containing x coordinates
        zgr (array): array containing z coordinates
        cross_value (float, optional): the value that the z grid should cross.

    Returns:
        float: the x-coordinate of the crossing.
    """
    
    zgr_adjusted = zgr - cross_value
    
    idx = np.where(np.diff(np.sign(zgr_adjusted)))[0]

    x1, x2 = xgr[idx], xgr[idx+1]
    z1, z2 = zgr_adjusted[idx], zgr_adjusted[idx+1]

    x_intersection = x1 + (0 - z1)/(z2 - z1) * (x2 - x1)
    
    return x_intersection
    
def calculate_bluff_edge_toe_position(xgr, zgr):
    """Calculates the bluff edge and toe position for a given profile. Method by Palaseanu-Lovejoy et al (2016).

    Args:
        xgr (array): array containing x coordinates
        zgr (array): array containing z coordinates

    Returns:
        tuple: tuple of length 2, with the first being the x value of the bluff edge and 
               the second being the x value of the bluff toe.
    """
    p1 = np.array((xgr[0], zgr[0]))
    p2 = np.array((xgr[-1], zgr[-1]))

    distances = np.zeros(xgr.shape)

    for i in range(len(xgr)):
        
        p3 = np.array((xgr[i], zgr[i]))
            
        distances[i] = np.cross(p2-p1, p3-p1) / np.linalg.norm(p2-p1)
        
    bluff_edge_id = np.argmax(distances)
    bluff_toe_id = np.argmin(distances)
    
    return xgr[bluff_edge_id], xgr[bluff_toe_id], distances

def generate_schematized_bathymetry(
        bluff_flat_length,
        
        bluff_height, 
        bluff_slope,
        
        beach_width, 
        beach_slope,
        
        nearshore_max_depth, 
        nearshore_slope,
        
        offshore_max_depth, 
        offshore_slope,
        
        contintental_flat_width,
        
        with_artificial=False,
        artificial_max_depth=500,
        artificial_slope=1/50,
        artificial_flat=60, # meters
        
        N=100
    ):
    """This function is used to generate a schematized bathymetry, consisting of a flat, 
    bluff, beach, nearshore, offshore, continental flat, and possibly an artificial extension of the grid.

    Returns:
        tuple: two arrays representing x and z (grid) values.
    """
    x_wl = 0
    x_toe = x_wl - beach_width
    x_bluff_edge = x_toe - bluff_height / bluff_slope
    x_start = x_bluff_edge - bluff_flat_length
    x_near_offs = x_wl + nearshore_max_depth / nearshore_slope
    x_offs_cont = x_near_offs + (offshore_max_depth - nearshore_max_depth) / offshore_slope
    x_cont_arti = x_offs_cont + contintental_flat_width
    x_end = x_cont_arti + artificial_max_depth / artificial_slope
    x_flat_extension = x_end + artificial_flat
    
    z_wl = 0
    z_toe = z_wl + beach_width * beach_slope
    z_bluff_edge = z_toe + bluff_height
    z_start = z_bluff_edge
    z_near_offs = z_wl - nearshore_max_depth
    z_offs_cont = z_wl - offshore_max_depth
    z_cont_arti = z_wl - offshore_max_depth
    z_end = z_wl - artificial_max_depth
    z_flat_extension = z_end
    
    xs = [
        x_start,
        x_bluff_edge,
        x_toe,
        x_wl,
        x_near_offs,
        x_offs_cont,
        x_cont_arti,
    ]
    
    zs = [
        z_start,
        z_bluff_edge,
        z_toe,
        z_wl,
        z_near_offs,
        z_offs_cont,
        z_cont_arti,
    ]
    
    if with_artificial:
        xs.append(x_end)
        zs.append(z_end)
        
        xs.append(x_flat_extension)
        zs.append(z_flat_extension)
        
        
    
    total_x = np.array([])
    total_z = np.array([])
    
    for i in range(len(xs)-1):
        x1, x2 = xs[i], xs[i+1]
        z1, z2 = zs[i], zs[i+1]
        
        total_x = np.append(total_x, np.linspace(x1, x2, N))
        total_z = np.append(total_z, np.linspace(z1, z2, N))
        
    # x=0 should be the furthest offshore point, and increase with distance to shore
    total_x = total_x[::-1] * -1 + np.max(total_x)
    total_z = total_z[::-1]

    return total_x, total_z


