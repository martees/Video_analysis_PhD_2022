import os
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import numpy as np
import pandas as pd
import glob
import json

# My code
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

    listofpaths = glob.glob(path_prefix + "/**/traj.csv", recursive=True)

    if not is_linux(): #on windows the glob output uses \\ as separators so remove that
        listofpaths = [name.replace("\\",'/') for name in listofpaths]

    print("Finished path finding")
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
    found in the same folder:
        patch_centers: list of coordinates of the patch centers
        patch_densities: list of densities of each patch
    :param path: a string with the path leading to the traj.csv whose metadata you want to retrieve (metadata should be in the same folder)
    """
    metadata = pd.DataFrame() #where we'll put everything

    # Finding the path of the other files
    lentoremove = len('traj.csv')  # removes traj from the current path, to get to the parent folder
    path_for_holes = path[:-lentoremove] + "holes.mat"
    path_for_patches = path[:-lentoremove] + "foodpatches.mat"
    path_for_patch_splines = path[:-lentoremove] + "foodpatches_new.mat"

    # Loadmat function loads .mat file into a dictionnary with meta info
    # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
    holesmat = loadmat(path_for_holes)  # load holes in a dictionary using loadmat
    patchesmat = loadmat(path_for_patches)  # load old patch objects (just centers and densities)
    splinesmat = loadmat(path_for_patch_splines)  # load alfonso's patch objects (with spline info)

    # Extract patch objects
    patch_objects = splinesmat.get("fp_struct")[0]
    # In patch_object[0] are 8x2 lists of interpolation points that were used to extract the spline
    # In patch_object[1] is a 1x2 list with the patch center coordinates
    # In patch_object[2] is a matlab spline object stuck in two [[[]]], so we access it like that:
    spline_objects = [patch_objects[i][2] for i in range(len(patch_objects))]
    # In spline_objects[0] is useless strings
    # In spline_objects[1] is the breaks which give the intervals for each spline equation
    # In spline_objects[2] are the coefficients of the spline

    # Extract the data into the dataframe
    # holepositions = pd.DataFrame(holesmat.get('pointList')) # gets the holes positions
    # condition_number = readcode(holepositions) #get the condition from that

    metadata["patch_centers"] = [patch_objects[i][1][0] for i in range(len(patch_objects))]
    metadata["spline_breaks"] = [list(spline_objects[i][0][0][1][0]) for i in range(len(spline_objects))]
    metadata["spline_coefs"] = [[list(spline_objects[j][0][0][2][i]) for i in range(len(spline_objects[j][0][0][2]))] for j in range(len(spline_objects))]

    metadata["patch_densities"] = list(patchesmat.get("densities_patches"))
    metadata["condition"] = list(patchesmat.get("num_condition"))[0][0]

    return metadata


def perfect_square_parameters(x1, y1, x2, y2, x3, y3, x4, y4):
    # Create numpy arrays for the points
    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Translate the points so that the centroid is at the origin
    translated_points = points - centroid

    # Perform singular value decomposition (SVD) on the translated points
    _, _, vh = np.linalg.svd(translated_points)

    # Extract the rotation matrix from the right singular vectors
    rotation_matrix = vh.T @ vh

    # Calculate the scaling factor as the square root of the largest eigenvalue of the rotation matrix
    scaling_factor = np.sqrt(np.max(np.linalg.eigvals(rotation_matrix)))

    # Calculate the rotation angle from the rotation matrix
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return scaling_factor, rotation_angle


def return_folders_condition_list(full_folder_list, condition_list):
    condition_folders = []
    for folder in full_folder_list:
        current_condition = folder_to_metadata(folder).reset_index()["condition"][0]
        if current_condition in condition_list:
            condition_folders.append(folder)
    return condition_folders


def load_list(results, column_name):
    """
    For lists that are stored as strings in the results.csv table
    """
    return list(json.loads(results[column_name][0]))

def load_condition(folder_name):
    return folder_to_metadata(folder_name)["condition"][0]