import os
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import numpy as np
import pandas as pd
import glob


def is_linux():  # returns true if you're using linux, otherwise false
    try:
        test = os.uname()
        if test[0] == "Linux":
            return True
    except AttributeError:
        return False

def path_finding_traj():
    """
    Function that takes a folder prefix and returns the paths of the traj.mat files
    It will look through subfolders if necessary
    """
    path_prefix_windows = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/20220721T163616_StandardizedConditions_C5_CAM1_Tracking_Video/"
    path_prefix_linux = "/home/admin/Desktop/Camera_setup_analysis/Tracking_Video/"

    if is_linux():
        path_prefix = path_prefix_linux
    else:
        path_prefix = path_prefix_windows

    return glob.glob(path_prefix + "*/traj.mat")


def trajmat_to_pandas(paths_of_mat):
    """
    Takes a list of paths for .mat objects, and returns a pandas dataframe containing the data
    It keeps path info in a column
    """
    dataframe = pd.DataFrame() #dataframe where we'll put everything

    for i_file in len(paths_of_mat): #for every file
        current_path = paths_of_mat[i_file]

        # Loadmat function loads .mat file into a dictionnary with meta info
        # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
        trajmat = loadmat(current_path)

        #Finding the Holes.mat file with condition info
        lentoremove = len('Tracking_Video/traj.mat')
        path_for_holes = current_path[:-lentoremove]+"holes.mat"


        #syntax for python 3.9 or higher
        #path_for_holes = current_path.removesuffix('Tracking_Video/traj.mat')+'holes.mat'


        # Structure of traj.mat: traj.mat[0] = one line per worm, with their x,y positions at t_0
        # So if you call traj.mat[:,0] you get all the positions of the first worm.
        nb_of_worms = len(trajmat.get('traj')[0])
        trajectories_worm_xy_time = []  # first list index is worm, second is x/t, third is time
        for worm in range(nb_of_worms):
            dataframe["worm"].append(pd.DataFrame(trajmat.get('traj')[:, worm]))

    return trajectories_worm_xy_time


def reformat_trajectories(bad_trajectory):
    """
    Very specific to our file format. Removes NaN lines, and reformats the trajectory file
    From [x0 x1 ... xN] [y0 ... yN]
    To [x0 y0] [x1 y1] ... [xN yN]
    """
    cleaned_trajectories = []
    for i_traj in range(len(bad_trajectory)):
        current_trajectory = bad_trajectory[i_traj]
        reformatted_trajectory = list(zip(current_trajectory[0], current_trajectory[1]))
        cleaned_trajectory = [tuple for tuple in reformatted_trajectory if
                              not np.isnan(tuple[0]) and not np.isnan(tuple[1])]
        cleaned_trajectories.append(cleaned_trajectory)
    return cleaned_trajectories
