import os
import shutil
import pandas as pd
import random
import ReferencePoints
import scipy

import parameters
# My code
from parameters import *
import find_data as fd

# Originally, our controls are saved in some folder. In order to have one control per inter-patch distance, we create
#     subfolders inside of those original folders, containing the name of the corresponding distance. Eg : inside the folder
#     XXX, containing the data of a control condition where no food was put on the plate, we create subfolders:
#     - XXX/XXX_control_close
#     - XXX/XXX_control_med
#     - XXX/XXX_control_far


def generate_controls(path):
    """
    Takes a path prefix, finds the files of all control conditions inside.
    Will create subfolders (named parentfolder_control_close, parentfolder_control_med, etc.) each containing:
        - a copy of the traj.csv file from the parent folder
        - a foodpatches.mat folder containing a new condition number
        - a foodpatches_new
    """

    # Full list of paths for the traj.csv files that can be found in the arborescence starting in path
    folder_list = fd.path_finding_traj(path)
    # Select folders that correspond to a control condition (11)
    folder_list = fd.return_folders_condition_list(folder_list, 11)
    # For all control folders
    for folder in folder_list:
        # Find the corresponding control sub-folders (eg ./parent_folder_control_med/traj.csv)
        current_control_folders = fd.control_folders(folder, ["close", "med", "far"])
        for current_control_folder in current_control_folders:
            # First check if the folder exists, and create it if it doesn't
            if not os.path.isdir(current_control_folder):
                # Create folder (name is parentfolder_control_close for example)
                os.mkdir(current_control_folder)
            # Check if there is a traj.csv file in the current_control_folder, otherwise copy it from parent
            if not os.path.isfile(current_control_folder+"/traj.csv"):
                # Copy the traj.csv from folder into current_control_folder
                # (all folder names have /traj.csv in the end, but not current_control_folder) (output of fd.control_folders)
                shutil.copy(folder, current_control_folder)
            # Make the foodpatches.csv files
            current_distance_condition = current_control_folder.split("_")[-1]  # eg "control_med" => "med" will be the last element of "_" split
            foodpatches = return_control_patches(path, current_control_folder, current_distance_condition)
            foodpatches.to_csv(current_control_folder)


def return_control_patches(path, folder, distance):
    """
    Input:
        :folder: folder path of a control experiment
        :distance: a distance condition, for example "close" or "cluster"
    Output:
        A foodpatches dataframe with all the info about the patches (condition, patch centers, densities, spline breaks, spline coefs)
    What it does:
        - Chooses a random folder with condition :condition:
        - Takes the foodpatches_new.mat from there
        - Converts that to the current reference points, and changes condition
    """

    all_folders = fd.path_finding_traj(path)
    # Find the folders of the experiments with the same distance between the patches
    same_distance_folders = fd.return_folders_condition_list(all_folders, parameters.name_to_nb_list(distance))
    # Pick a random one
    n = random.randint(0, len(same_distance_folders) - 1)
    the_chosen_one = same_distance_folders[n]
    # Find its patch information
    metadata = fd.folder_to_metadata(the_chosen_one)



    # metadata["patch_centers"]
    # metadata["spline_guides"]
    # metadata["spline_breaks"]
    # metadata["spline_coefs"]
    # metadata["patch_densities"]
    # metadata["condition"]


    return pd.DataFrame()
