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
    '''
    Function to set the paths of the tracking outputs.
    :return:
    '''
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

draw(trajmat_to_pandas(path_finding()))


