# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from datetime import datetime, date, time
import pandas as pd
import os
import matplotlib.pyplot as plt

def is_linux():  #returns true if you're using linux, otherwise false
    try:
        test = os.uname()
        if test[0] == "Linux":
            return True
    except AttributeError:
        return False

def path_finding():
    """
    Function that takes a path and returns the path of the traj.mat
    """
    #TODO Set these to a list of prefixes for bulk analysis, keep folder info as the variable name?
    path_prefix_windows = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/20220721T163616_StandardizedConditions_C5_CAM1_Tracking_Video/"
    path_prefix_linux = "/home/admin/Desktop/Camera_setup_analysis/Tracking_Video/"

    if is_linux():
        path_prefix = path_prefix_linux
    else:
        path_prefix = path_prefix_windows

    return path_prefix + "traj.mat"


def trajmat_to_pandas(path_of_mat):
    """
    Takes the path of a .mat object, and returns a pandas dataframe containing the same data
    """
    # Loads .mat file into a dictionnary with meta info
    trajmat = loadmat(path_of_mat) # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
    # Structure of traj.mat: traj.mat[0] = one line per worm, with their x,y positions at t_0
    # So if you call traj.mat[:,0] you get all the positions of the first worm.
    nb_of_worms = len(trajmat.get('traj')[0])
    trajectories_worm_xy_time = [] #first list index is worm, second is x/t, third is time
    for worm in range(nb_of_worms):
        trajectories_worm_xy_time.append(pd.DataFrame(trajmat.get('traj')[:,worm]))
    return trajectories_worm_xy_time

def draw(trajectories):
    """
    Function that takes in a list of series of positions and draws them.
    :param trajectory: list of series of (x,y) positions ([[x0,x1,x2...] [y0,y1,y2...])
    :return: trajectory plot
    """
    for i_traj in range(len(trajectories)):
        plt.plot(trajectories[i_traj][0],trajectories[i_traj][1])
    plt.show()

def in_patch(position, patch, ):
    """
    returns true if position = [x,y] is inside the patch
    tolerance is that the worm is still considered inside the patch when its center is sticking out but only a bit
    """
    center = patch[0]
    radius = patch[1]
    distance = sqrt((position[0]-center[0])**2 + (position[1]-center[1])**2)
    return distance < radius + radial_tolerance

def patch_visits(trajectory, list_of_patches):
    """
    returns a list [[d0,d1,...], [d0,d1,...],...] with one list per patch
    each sublist contains the durations of visits to each patch
    """
    is_in_patch = np.ones(len(list_of_patches)) # Bool list to store where the worm is currently
    list_of_durations = [list(i) for i in np.zeros((10,1), dtype='int')] # List with the right format, 0 for each patch

    #We go through the whole trajectory
    for t in range(len(trajectory)):
        was_in_patch = is_in_patch # Keeping in memory where the worm was
        for i_patch in list_of_patches:
            is_in_patch[i_patch] = in_patch(trajectory[t], list_of_patches[i])




draw(trajmat_to_pandas(path_finding()))

# Parameters
radial_tolerance = 0.1
fake_patch = [[1400, 1200], 100]  # [[x,y], radius] with x y = position of the center

