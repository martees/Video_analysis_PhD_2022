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
    Function that takes a folder prefix and returns a list of paths of the traj.csv files present in that folder
    It will look through subfolders if necessary
    """
    #These are now arguments of the function to avoid unnecessary hard coded mess
    #path_prefix_windows = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/20220721T163616_StandardizedConditions_C5_CAM1_Tracking_Video"
    #path_prefix_linux = "/home/admin/Desktop/Camera_setup_analysis/Tracking_Video"

    return glob.glob(path_prefix + "*/traj.csv")


def trajmat_to_dataframe(paths_of_mat):
    """
    Takes a list of paths for .mat objects, and returns a pandas dataframe containing the data
    The dataframe has the same columns as traj.mat : one line per timestep per worm. See readme.txt for detailed info
        x,y,time: position at a given time
        id_conservative: id of the worm
        folder: path of where the data was extracted from (to keep computer - camera - date info)
    NOTE: it's with the value in folder that you can call other info such as patch positions, using folder_to_metadata()
    """
    dataframe = pd.DataFrame() #dataframe where we'll put everything
    dataframe["folder"] = pd.DataFrame() #creating column for folder names

    for i_file in range(len(paths_of_mat)): #for every file
        current_path = paths_of_mat[i_file]

        #Open the csv with trajectory info for each worm, store it in a pandas dataframe
        dataframe.append(pd.read_csv(current_path))
        dataframe["folder"].append([current_path for i in range(nb_of_timesteps)])

        #### outdated comments but might be useful?? about the old structure of traj.mat
        # Structure of traj.mat: traj.mat[0] = one line per worm, with their x,y positions at t_0
        # So if you call traj.mat[:,0] you get all the positions of the first worm.

    return dataframe

def folder_to_metadata(path):
    """
    This function takes the path of a traj.csv file and returns a dataframe with the metadata of this video:
        patch_centers: list of coordinates of the patch centers
        patch_densities: list of densities of each patch
        TODO: code and reference from holes positions (see with al)
    """
    metadata = pd.DataFrame() #where we'll put everything

    # Finding the path of the other files
    lentoremove = len('traj.csv')  # removes traj from the current path, to get to the parent folder
    path_for_holes = current_path[:-lentoremove] + "holes.mat"
    path_for_patches = current_path[:-lentoremove] + "foodpatches.mat"

    # Loadmat function loads .mat file into a dictionnary with meta info
    # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
    holesmat = loadmat(path_for_holes)  # load holes in a dictionary using loadmat
    patchesmat = loadmat(path_for_patches)  # load patches info

    holepositions = pd.DataFrame(holesmat.get('pointList')) # gets the holes positions
    condition_number = readcode(holepositions) #thats a fake function for now lmao
    # TODO reference_scale, reference_xyshift, reference_rotation = read_reference_holes(holepositions)

    # We then just add the source folder and condition number in a folder
    nb_of_timesteps = len(dataframe.get('time'))
    dataframe["condition"] = [condition_number for i in range(nb_of_timesteps)]

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


def readcode(holepositions):
    # This should be written if Alfonso doesnt give it lmao
    return holepositions[2][0]


