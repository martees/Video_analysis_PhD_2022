import os
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import numpy as np
import pandas as pd
import glob
from param import *


def is_linux():  # returns true if you're using linux, otherwise false
     try:
         test = os.uname()
         if test[0] == "Linux":
             return True
     except AttributeError:
         return False

def path_finding_traj(path_prefix):
    """
    Function that takes a folder prefix and returns a list of paths of the traj.csv files present in that folder
    WARNING: it will look through subfolders if necessary, so it might take a while if your folder is very big!
    """
    #These are now arguments of the function to avoid unnecessary hard coded mess
    #path_prefix_windows = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/20220721T163616_StandardizedConditions_C5_CAM1_Tracking_Video"
    #path_prefix_linux = "/home/admin/Desktop/Camera_setup_analysis/Tracking_Video"

    listofpaths = glob.glob(path_prefix + "/**/traj_mm.csv", recursive=True)

    if not is_linux(): #on windows the glob output uses \\ as separators so remove that
        listofpaths = [name.replace("\\",'/') for name in listofpaths]

    return listofpaths


def trajmat_to_dataframe(paths_of_mat):
    """
    Takes a list of paths for .csv tables, and returns a pandas dataframe containing all their data concatenated
    The dataframe has the same columns as traj.csv : one line per timestep for each worm. See readme.txt for detailed info
        x,y,time: position at a given time
        id_conservative: id of the worm
        folder: path of where the data was extracted from (to keep computer - camera - date info)
    NOTE: it's with the value in folder that you can call other info such as patch positions, using folder_to_metadata()
    """
    folder_list = []
    for i_file in range(len(paths_of_mat)): #for every file
        current_path = paths_of_mat[i_file]
        current_data = pd.read_csv(current_path) #dataframe with all the info
        # We add the file number to the worm identifyers, for them to become unique accross all folders
        current_data["id_conservative"] = [id + 100*i_file for id in current_data["id_conservative"]]

        if i_file == 0:
            dataframe = current_data
        else:
            dataframe = pd.concat([dataframe,current_data]) #add it to the main dataframe

        #In the folder list, add the folder as many times as necessary:
        nb_of_timesteps = len(current_data.get('time'))  # get the length of that
        folder_list += [current_path for i in range(nb_of_timesteps)]

    dataframe["folder"] = folder_list

        #### outdated comments but might be useful?? about the old structure of traj.mat
        # Structure of traj.mat: traj.mat[0] = one line per worm, with their x,y positions at t_0
        # So if you call traj.mat[:,0] you get all the positions of the first worm.

    return dataframe

def folder_to_metadata(path):
    """
    This function takes the path of a traj.csv file and returns a dataframe with the metadata of this video,
    found in the same folder, which in the case of our current dataset is just the condition number.
    :param path: a string with the path leading to the traj.csv whose metadata you want to retrieve (metadata should be in the same folder)
    """
    metadata = pd.DataFrame() #where we'll put everything

    # Finding the path of the other files
    lentoremove = len('traj_mm.csv')  # removes traj from the current path, to get to the parent folder
    path_for_condition = path[:-lentoremove] + "num_condition.mat"

    # Loadmat function loads .mat file into a dictionnary with meta info
    # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
    conditionsmat = loadmat(path_for_condition)  # load holes in a dictionary using loadmat

    metadata["condition"] = list(conditionsmat.get("num_condition"))[0]

    return metadata


#useless functions
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




