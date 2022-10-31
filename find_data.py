import os
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import numpy as np
import pandas as pd
import glob


# def is_linux():  # returns true if you're using linux, otherwise false
#     try:
#         test = os.uname()
#         if test[0] == "Linux":
#             return True
#     except AttributeError:
#         return False

def path_finding_traj(path_prefix):
    """
    Function that takes a folder prefix and returns the paths of the traj.mat files
    It will look through subfolders if necessary
    """
    #These are now arguments of the function to avoid unnecessary hard coded mess
    #path_prefix_windows = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/20220721T163616_StandardizedConditions_C5_CAM1_Tracking_Video"
    #path_prefix_linux = "/home/admin/Desktop/Camera_setup_analysis/Tracking_Video"

    return glob.glob(path_prefix + "*/traj.csv")


def matfiles_to_pandas_dataframe(paths_of_mat, other_data = True):
    """
    Takes a list of paths for .mat objects, and returns a pandas dataframe containing the data
    It keeps path info in a column, and has one line per timestep (so many rows for each worm)
    other_data has to be set to False if you're just working with the trajectory, it looks for other files
    """
    dataframe = pd.DataFrame() #dataframe where we'll put everything

    for i_file in range(len(paths_of_mat)): #for every file
        current_path = paths_of_mat[i_file]

        #Open the csv with trajectory info for each worm, store it in a pandas dataframe
        trajmat = pd.read_csv(current_path)

        if other_data:
            #Finding other files with condition info
            lentoremove = len('traj.csv') # removes traj from the current path, to get to the parent folder
            path_for_holes = current_path[:-lentoremove]+"holes.mat"
            path_for_patches = current_path[:-lentoremove]+"foodpatches.mat"

            # Loadmat function loads .mat file into a dictionnary with meta info
            # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
            holesmat = loadmat(path_for_holes) #load holes in a dictionary using loadmat
            patchesmat = loadmat(path_for_patches) #load patches info

        #### outdated comments but might be useful?? about the old structure of traj.mat
        # Structure of traj.mat: traj.mat[0] = one line per worm, with their x,y positions at t_0
        # So if you call traj.mat[:,0] you get all the positions of the first worm.


        nb_of_timesteps = len(trajmat.get('traj')[0])
        trajectories_worm_xy_time = []  # first list index is worm, second is x/y, third is time

        for worm in range(nb_of_timesteps):
            trajectories_worm_xy_time.append(pd.DataFrame(trajmat.get('traj')[:, worm]))
            #if find_holes:
            #replace those by watever format
            #holelist.append(pd.DataFrame(holesmat.get('pointList')[:, worm]))
            #pathlist.append(pd.DataFrame(current_path))
        #if find_holes:
            #dataframe["condition"]=holelist
            #dataframe["folder"]=pathlist
        dataframe["trajectory"]=pd.DataFrame(trajectories_worm_xy_time)

    return dataframe


def reformat_trajectories(bad_trajectory):
    """
    Very specific to our file format. Removes NaN lines, and reformats the trajectory file
    From [x0 x1 ... xN] [y0 ... yN]
    To [x0 y0] [x1 y1] ... [xN yN]
    This format is a bit less convenient for plotting but a bit more convenient for calling a position
    """
    cleaned_trajectories = []
    for i_traj in range(len(bad_trajectory)):
        current_trajectory = bad_trajectory[i_traj]
        reformatted_trajectory = list(zip(current_trajectory[0], current_trajectory[1]))
        cleaned_trajectory = [tuple for tuple in reformatted_trajectory if
                              not np.isnan(tuple[0]) and not np.isnan(tuple[1])]
        cleaned_trajectories.append(cleaned_trajectory)
    return cleaned_trajectories
