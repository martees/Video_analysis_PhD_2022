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


def rename_original_control_traj(path):
    """
    Look for condition 11 folders, and replace the name of the traj.csv inside by traj_control.csv.
    This is because we rebuild a "med" control in a subfolder, and don't want the old control to appear in future searches.
    This should only do anything if you have just imported new folders.
    """
    # Full list of paths for the traj.csv files that can be found in the arborescence starting in path
    folder_list = fd.path_finding_traj(path, include_fake_control_folders=False)
    # Select folders that correspond to a control condition (11)
    folder_list = fd.return_folders_condition_list(folder_list, 11)
    # Rename them allllll
    for folder in folder_list:
        os.rename(folder, folder[:-"traj.csv"]+"traj_control.csv")


def generate_controls(path):
    """
    Takes a path prefix, finds the files of all control conditions inside.
    Will create subfolders (named parentfolder_control_close, parentfolder_control_med, etc.) each containing:
        - a copy of the traj.csv file from the parent folder
        - a foodpatches.mat folder containing a new condition number
        - a foodpatches_new
    """
    # Rename any (new) control traj.csv into traj_control.csv
    rename_original_control_traj(path)
    # Full list of paths for the traj_control.csv files that can be found in the arborescence starting in path
    folder_list = fd.path_finding_traj(path, target_name="traj_control.csv", include_fake_control_folders=False)
    # For all control folders
    for i_folder in range(len(folder_list)):
        if i_folder % 1 == 0:
            print("Generating control patches... ", i_folder, " / ", len(folder_list))
            print(folder_list[i_folder])
        folder = folder_list[i_folder]
        # Find the corresponding control sub-folders (eg ./parent_folder/parent_folder_control_far/traj.csv)
        current_control_folders = fd.control_folders(folder, ["close", "med", "far", "cluster"])
        for current_control_folder in current_control_folders:
            # First check if the folder exists, and create it if it doesn't
            if not os.path.isdir(current_control_folder):
                # Create folder (name is parentfolder_control_close for example)
                os.mkdir(current_control_folder)
            # Make the metadata.csv files
            current_distance_condition = current_control_folder.split("_")[-1]  # eg "control_med" => "med" will be the last element of "_" split
            metadata = return_control_metadata(path, folder, current_distance_condition)
            metadata.to_csv(current_control_folder+"/metadata.csv")
            # Check if there is a traj.csv file in the current_control_folder, otherwise copy it from parent
            # Do this AFTER creating metadata otherwise find_data functions can find a traj.csv file with no metadata in the folder (and get pissed)
            if not os.path.isfile(current_control_folder + "/traj.csv"):
                # Copy the traj.csv from folder into current_control_folder
                # (all folder names have /traj.csv in the end, but not current_control_folder) (output of fd.control_folders)
                shutil.copy(folder, current_control_folder)
                # folder_without_traj = folder[:-len("traj.csv")]
                # shutil.copy(folder_without_traj+"composite.tif", current_control_folder)
                # shutil.copy(folder_without_traj+"background.tif", current_control_folder)


def return_control_metadata(path, parent_folder, distance):
    """
    Input:
        :path: path where all the experimental folders live
        :parent_folder: folder path of a control experiment
        :distance: a distance condition, for example "close" or "cluster"
    Output:
        A foodpatches dataframe with all the info about the patches (condition, patch centers, densities, spline breaks, spline coefs)
    What it does:
        - Chooses a random folder with condition :condition:
        - Takes the foodpatches_new.mat from there
        - Converts that to the current reference points, and changes condition
    """

    all_folders = fd.path_finding_traj(path, include_fake_control_folders=False)
    # Find the folders of the experiments with the same distance between the patches
    same_distance_folders = fd.return_folders_condition_list(all_folders, parameters.name_to_nb_list[distance])
    # Pick a random one
    the_chosen_one = random.choice(same_distance_folders)
    # Load patch information for source folder
    source_folder_metadata = fd.folder_to_metadata(the_chosen_one)
    #TODO If reference points are not good, look for another one?
    parent_folder_metadata = fd.folder_to_metadata(parent_folder)

    # Load reference points for re-alignment
    source_xy_holes = source_folder_metadata["holes"][0]
    # We load the holes of the parent_folder (the control_folder traj.csv is a copy from the parent_folder's, so same ref)
    target_xy_holes = parent_folder_metadata["holes"][0]
    # Convert it to a ReferencePoints object and use one of the class methods to convert patch centers to this reference
    source_reference_points = ReferencePoints.ReferencePoints(source_xy_holes)
    target_reference_points = ReferencePoints.ReferencePoints(target_xy_holes)
    patch_centers = source_reference_points.pixel_to_mm(parameters.distance_to_xy[distance])


    # Store it all in a metadata dataframe
    control_metadata = pd.DataFrame()
    control_metadata["patch_centers"] = patch_centers.tolist()
    control_metadata["holes"] = [target_xy_holes for _ in range(len(patch_centers))]
    control_metadata["spline_guides"] = list(source_folder_metadata["spline_guides"])
    control_metadata["spline_breaks"] = list(source_folder_metadata["spline_breaks"])
    control_metadata["spline_coefs"] = list(source_folder_metadata["spline_coefs"])
    control_metadata["patch_densities"] = [0 for i in range(len(control_metadata["patch_centers"]))]
    control_metadata["condition"] = parameters.name_to_nb[distance+" 0"]
    # metadata["patch_centers"] = [patch_centers[i][0] for i in range(len(patch_centers))]
    # metadata["spline_guides"] = [spline_guides[i].tolist() for i in range(len(spline_guides))]
    # metadata["spline_breaks"] = [list(spline_objects[i][0][0][1][0]) for i in range(len(spline_objects))]
    # metadata["spline_coefs"] = [[list(spline_objects[j][0][0][2][i]) for i in range(len(spline_objects[j][0][0][2]))]
    # metadata["patch_densities"] = list(patchesmat.get("densities_patches"))
    # metadata["condition"]

    return control_metadata
