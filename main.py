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
        reformatted_trajectory = list(zip(*trajectories[i_traj])) # our trajectories are not in the right format, easier to convert them here
        plt.plot(reformatted_trajectory[0],reformatted_trajectory[1])
    plt.show()

def in_patch(position, patch):
    """
    returns true if position = [x,y] is inside the patch
    tolerance is that the worm is still considered inside the patch when its center is sticking out but only a bit
    """
    center = patch[0]
    radius = patch[1]
    distance = np.sqrt((position[0]-center[0])**2 + (position[1]-center[1])**2)
    return distance < radius + radial_tolerance

def patch_visits_single_traj(trajectory, list_of_patches):
    """
    returns a list [[d0,d1,...], [d0,d1,...],...] with one list per patch
    each sublist contains the durations of visits to each patch
    """
    is_in_patch = np.ones(len(list_of_patches)) # Bool list to store where the worm is currently
    list_of_durations = [list(i) for i in np.zeros((10,1), dtype='int')] # List with the right format [[0],[0],...,[0]]
    #In list_of_durations, zero means "the worm was not in the patch in the previous timestep"
    #As soon as the worm enters the patch, this zero starts being incremented
    #As soon as the worm leaves the patch, a new zero is added to this patch's list

    patch_where_it_is = -1 #initialization
    #We go through the whole trajectory
    for time in range(len(trajectory)):
        was_in_patch = is_in_patch # Keeping in memory where the worm was [0 0 1 0] = in patch 2
        patch_where_it_was = patch_where_it_is #Index of the patch where it is
        patch_where_it_is = -1 # By default the worm is not in a patch
        for i_patch in range(len(list_of_patches)):
            is_in_patch[i_patch] = in_patch(trajectory[time], list_of_patches[i_patch])
            if is_in_patch[i_patch] == True:
                patch_where_it_is = i_patch
        if patch_where_it_is==-1: # Worm currently out
            if patch_where_it_was != patch_where_it_is:  # Worm exited a patch
                list_of_durations[patch_where_it_was].append(0) # Add a zero because previous visit was interrupted
        if patch_where_it_is!=-1: # Worm currently inside, no matter whether it just entered or stayed inside
            list_of_durations[patch_where_it_is]+=1
    return list_of_durations

def patch_visits_multiple_traj(list_of_trajectories, list_of_patches):
    """
    returns a list of outputs from the single_traj function, one output per trajectory
    """
    list_of_durations = []
    for traj in list_of_trajectories:
        list_of_durations.append(patch_visits_single_traj(traj, list_of_patches))
    return list_of_durations

def reformat_trajectories(bad_trajectory):
    """
    Very specific to our file format. Removes NaN lines, and reformats the trajectory file
    From [x0 x1 ... xN] [y0 ... yN]
    To [x0 y0] [x1 y1] ... [xN yN]
    """
    cleaned_trajectories = []
    for i_traj in range(len(bad_trajectory)):
        current_trajectory = bad_trajectory[i_traj]
        reformatted_trajectory = list(zip(current_trajectory[0],current_trajectory[1]))
        cleaned_trajectory = [tuple for tuple in reformatted_trajectory if not np.isnan(tuple[0]) and not np.isnan(tuple[1])]
        cleaned_trajectories.append(cleaned_trajectory)
    return cleaned_trajectories

# Parameters
radial_tolerance = 0.1
fake_patch1 = [[1400, 1200], 100]  # [[x,y], radius] with x y = position of the center
fake_patch2 = [[500, 1000], 30]  # [[x,y], radius] with x y = position of the center
patch_list = [fake_patch1, fake_patch2]

# Function tests
trajectories_with_nans = trajmat_to_pandas(path_finding())
# Initial format of the trajectories:
# List of trajectories, and each trajectory:
# [x0 x1 ... xN] [y0 y1 ... yN]
# New format:
# No NaNs and [x0 y0] [x1 y1] ... [xN yN]
trajectories = reformat_trajectories(trajectories_with_nans)

draw(trajectories)
patch_visits_multiple_traj(trajectories, patch_list)