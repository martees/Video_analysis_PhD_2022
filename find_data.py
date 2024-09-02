import os
import shutil
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import numpy as np
import pandas as pd
import glob
import json
import re

# My code
from Parameters.parameters import *


def is_linux():  # returns true if you're using linux, otherwise false
     try:
         test = os.uname()
         if test[0] == "Linux":
             return True
     except AttributeError:
         return False


def path_finding_traj(path_prefix, target_name="traj.csv", include_fake_control_folders=True):
    """
    Function that takes a folder prefix and returns a list of paths of the "target_name" files present in that folder.
    :include_fake_control_folders: if TRUE, the function will return all the paths. If FALSE, function will exclude
                                   folders that have the word "control" in it.
    """
    listofpaths = glob.glob(path_prefix + "/**/" + target_name, recursive=True)

    if not include_fake_control_folders:
        listofpaths = [path for path in listofpaths if "control_" not in path]

    if not is_linux():  # On windows the glob output uses \\ as separators so remove that
        listofpaths = [name.replace("\\",'/') for name in listofpaths]

    if verbose:
        print("Finished path finding")
    return listofpaths


def trajcsv_to_dataframe(paths_of_mat):
    """
    Takes a list of paths for traj.csv tables, and returns a pandas dataframe containing all their data concatenated
    The output dataframe has the same columns as trajectories.csv : one line per timestep for each worm. See readme.txt for detailed info
        x,y,time: position at a given time
        id_conservative: id of the worm
        folder: path of where the data was extracted from (to keep computer - camera - date info)
    NOTE: it's with the value in folder that you can call other info such as patch positions, using folder_to_metadata()
    """
    folder_list = []
    for i_file in range(len(paths_of_mat)): #for every file
        if i_file % 100 == 0:
            print("Retrieving trajectories for file ", i_file, " / ", len(paths_of_mat), "!")
        current_path = paths_of_mat[i_file]
        current_data = pd.read_csv(current_path) #dataframe with all the info
        if len(np.unique(current_data["id_conservative"])) > 100000:
            print("Problem in trajcsv_to_dataframe! id's will overlap")
        # We add the file number to the worm identifyers, for them to become unique accross all folders
        current_data["id_conservative"] = [id + 100000*i_file for id in current_data["id_conservative"]]

        if i_file == 0:
            dataframe = current_data
        else:
            dataframe = pd.concat([dataframe,current_data]) #add it to the main dataframe

        #In the folder list, add the folder as many times as necessary:
        nb_of_timesteps = len(current_data.get('time'))  # get the length of that
        folder_list += [current_path for i in range(nb_of_timesteps)]

    dataframe["folder"] = folder_list

        # outdated comments but might be useful?? about the old structure of traj.mat
        # Structure of traj.mat: traj.mat[0] = one line per worm, with their x,y positions at t_0
        # So if you call traj.mat[:,0] you get all the positions of the first worm.
    return dataframe


def folder_to_metadata(path):
    """
    This function takes the path of a traj.csv file and returns a dataframe with the metadata of this video,
    found in the same folder:
        "patch_centers": list of coordinates of the patch centers
        "patch_densities": list of densities of each patch
        "condition": condition number
        "spline_guides": coordinates of the points that were used to build the patch splines (for checkup purposes)
        "spline_breaks": breaks in the spline (points at which the polynomial switches, see MATLAB splines)
        "spline_coefs": coefficients of the polynomials for each inter-break interval
    :param path: a string with the path leading to the traj.csv whose metadata you want to retrieve (metadata should be in the same folder)
    """
    # If the path contains the word control, which means it's a folder created during analysis to contain fake patches
    if "control" in path:
        metadata = pd.read_csv(path[:-len("traj.csv")]+"metadata.csv")
        # In metadata each patch_centers & spline guides element is like "[560 890]", so to give it to json.loads we first change it into "[560,890]"
        # First convert all multiple spaces into single spaces
        patch_centers = [re.sub(r"\s+", " ", metadata["patch_centers"][i]) for i in range(len(metadata["patch_centers"]))]
        spline_guides = [re.sub(r"\s+", " ", metadata["spline_guides"][i]) for i in range(len(metadata["spline_guides"]))]
        # Then remove spaces that are near brackets
        patch_centers = [patch_centers[i].replace("[ ", "[").replace(" ]", "]") for i in range(len(patch_centers))]
        spline_guides = [spline_guides[i].replace("[ ", "[").replace(" ]", "]") for i in range(len(spline_guides))]
        # In case the strings already have commas, replace them with single space
        patch_centers = [patch_centers[i].replace(", ", " ").replace(",", " ") for i in range(len(patch_centers))]
        spline_guides = [spline_guides[i].replace(", ", " ").replace(",", " ") for i in range(len(spline_guides))]
        # For the spline breaks and coefs, sometimes it comes with a weird format "[[np.float64(-0.2), ...]", remove that
        spline_breaks = [metadata["spline_breaks"][i].replace("np.float64", "").replace("(", "").replace(")", "") for i in range(len(metadata["spline_breaks"]))]
        spline_coefs = [metadata["spline_coefs"][i].replace("np.float64", "").replace("(", "").replace(")", "") for i in range(len(metadata["spline_coefs"]))]
        # Finally, replace spaces by commas and use json.loads
        metadata["patch_centers"] = [json.loads(patch_centers[i].replace(" ", ",")) for i in range(len(patch_centers))]
        metadata["spline_guides"] = [json.loads(spline_guides[i].replace(" ", ",").replace("\n", "")) for i in range(len(spline_guides))]
        metadata["spline_breaks"] = [json.loads(spline_breaks[i]) for i in range(len(spline_breaks))]
        metadata["spline_coefs"] = [json.loads(spline_coefs[i]) for i in range(len(spline_coefs))]
        metadata["holes"] = [json.loads(metadata["holes"][i]) for i in range(len(metadata["patch_centers"]))]

        return metadata

    # Else if it's a normal folder that went through Alfonso's tracking pipeline (metadata stored in .mat files)
    else:
        metadata = pd.DataFrame() #where we'll put everything

        # Finding the path of the other files
        lentoremove = len('traj.csv')  # removes traj from the current path, to get to the parent folder
        if "traj_parent.csv" in path:
            lentoremove = len("traj_parent.csv")

        path_for_holes = path[:-lentoremove] + "holes.mat"
        path_for_patches = path[:-lentoremove] + "foodpatches.mat"

        if not os.path.isfile(path_for_patches):
            sub_directory = path.split("/")[-2]
            directory = path.split("/")[-3]
            shutil.move(path[:-lentoremove-1], path[:-lentoremove-len(sub_directory)-len(directory)-3]+"/Results_minipatches_no_fp/"+sub_directory)
            return False

        else:
            # Replace path by reviewed food patches if they exist
            if os.path.isfile(path[:-lentoremove] + "foodpatches_reviewed.mat"):
                path_for_patches = path[:-lentoremove] + "foodpatches_reviewed.mat"
            path_for_info = path_for_patches

            # For old dataset: replaced by foodpatches_new if available, because the non-new one does not contain splines
            if os.path.isfile(path[:-lentoremove] + "foodpatches_new.mat"):
                path_for_patches = path[:-lentoremove] + "foodpatches_new.mat"

            # Loadmat function loads .mat file into a dictionnary with meta info
            # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
            holesmat = loadmat(path_for_holes)  # load holes in a dictionary using loadmat
            patchesmat = loadmat(path_for_patches)  # load patch objects
            infomat = loadmat(path_for_info) # load condition & patch densities (same as patchesmat for new dataset)

            # Extract patch objects
            patch_objects = patchesmat.get("fp_struct")[0]
            # In patch_object[0] are 8x2 lists of interpolation points that were used to extract the spline
            # In patch_object[1] is a 1x2 list with the patch center coordinates
            # In patch_object[2] is a matlab spline object stuck in two [[[]]], so we access it like that:
            spline_guides = [patch_objects[i][0] for i in range(len(patch_objects))]
            patch_centers = [patch_objects[i][1] for i in range(len(patch_objects))]
            spline_objects = [patch_objects[i][2] for i in range(len(patch_objects))]
            # In spline_objects[0] is useless strings
            # In spline_objects[1] is the breaks which give the intervals for each spline equation
            # In spline_objects[2] are the coefficients of the spline

            # Extract the data into the dataframe
            # holepositions = pd.DataFrame(holesmat.get('pointList')) # gets the holes positions
            # condition_number = readcode(holepositions) #get the condition from that

            metadata["patch_centers"] = [patch_centers[i][0] for i in range(len(patch_centers))]
            metadata["spline_guides"] = [spline_guides[i].tolist() for i in range(len(spline_guides))]
            metadata["spline_breaks"] = [list(spline_objects[i][0][0][1][0]) for i in range(len(spline_objects))]
            metadata["spline_coefs"] = [[list(spline_objects[j][0][0][2][i]) for i in range(len(spline_objects[j][0][0][2]))] for j in range(len(spline_objects))]

            metadata["patch_densities"] = list(infomat.get("densities_patches"))
            metadata["condition"] = list(infomat.get("num_condition"))[0][0]
            metadata["holes"] = [[hole[0:2] for hole in holesmat.get("pointList").tolist() if hole[2] == 2] for i_patch in range(len(metadata["patch_centers"]))]

    return metadata


def load_silhouette(path):
    """
    Takes a folder path, returns the content of the silhouette matrix found in this path.
    """
    silhouette_path = load_file_path(path, "silhouettes.mat")
    matrix = loadmat(silhouette_path)

    # Get the different arrays from the dictionnary output of loadmat
    frame_size = matrix.get("frame_size")
    pixels = matrix.get("pixels")
    intensities = matrix.get("intensities")

    # Reformat
    pixels = [[pixels[i][0][j][0] for j in range(len(pixels[i][0]))] for i in range(len(pixels))]
    intensities = [[intensities[i][0][j][0] for j in range(len(intensities[i][0]))] for i in range(len(intensities))]
    frame_size = frame_size[0]

    return pixels, intensities, frame_size


def unravel_index_list(index_list, frame_size):
    unraveled = []
    for i in range(len(index_list)):
        unraveled.append(np.unravel_index(index_list[i], frame_size))
    return unraveled


def reindex_silhouette(pixels, frame_size):
    """
    Take a pixel table, with one set of pixels per frame, and the set of pixels is indexed linearly by MATLAB.
    Return reindexed pixels (from MATLAB linear indexing to (x,y) coordinates).
    """
    for frame in range(len(pixels)):
        pixels[frame] = unravel_index_list(pixels[frame], frame_size)
    reformatted_pixels = []
    for frame in range(len(pixels)):
        x_list = []
        y_list = []
        for pixel in range(len(pixels[frame])):
            x_list.append(pixels[frame][pixel][0])
            y_list.append(pixels[frame][pixel][1])
        reformatted_pixels.append([x_list, y_list])
    return reformatted_pixels


def return_folders_condition_list(full_folder_list, condition_list, return_conditions=False):
    """
    Takes a list of folders, and a list of conditions, and returns the list of folders that correspond to the
    conditions in the list of conditions.
    @param full_folder_list: a list of strings leading to "./traj.csv" files
    @param condition_list: a list of numbers (see ./Parameters/parameters.py for corresponding names)
    @param return_conditions: if TRUE, will also return a list of the conditions of the folders.
    @return: a list of folder names.
    """
    if type(condition_list) is int:
        condition_list = [condition_list]
    condition_folders = []
    if return_conditions:
        folder_conditions = []
    for folder in full_folder_list:
        current_condition = folder_to_metadata(folder).reset_index()["condition"][0]
        if current_condition in condition_list:
            condition_folders.append(folder)
            if return_conditions:
                folder_conditions.append(current_condition)
    if return_conditions:
        return condition_folders, folder_conditions
    else:
        return condition_folders


def load_list(results, column_name):
    """
    For lists that are stored as strings in the results.csv table
    """
    if column_name in results.columns:
        try:
            return list(json.loads(results[column_name][0].replace("np.float64(", "").replace(")", "")))
        except KeyError:
            results = results.reset_index()
            return list(json.loads(results[column_name][0]))
    else:
        print("Column ", column_name, " does not exist in results.")


def load_condition(folder):
    return folder_to_metadata(folder)["condition"][0]


def load_index(trajectories, folder, time):
    """
    Will load the part of trajectories that corresponds to the folder, and find at which index of the table it is the
    time stamp time of the tracking (if there are holes in the tracking, time 800 could be at index 750, because of
    50 frames with no tracking)
    """
    current_traj = trajectories[trajectories["folder"] == folder]
    index = find_closest(current_traj["time"], time)
    return index


def load_time(trajectories, folder, index):
    """
    Will load the traj.csv matrix in folder, and find at which frame of the table it is the index-th time stamp of the
    tracking (if there are holes in the tracking, time 800 could be at index 750, because of 50 frames with no tracking)
    """
    current_traj = trajectories[trajectories["folder"] == folder].reset_index()
    return current_traj["time"][index]


def find_closest(iterable, value):
    """
    Find index of closest value to "value" in "iterable"
    """
    return min(enumerate(iterable), key=lambda x: abs(x[1] - value))[0]


def control_folders(path, condition_names):
    """
    See generate_controls.py for documentation of why this function is useful.
    This script will return the paths for the control sub-folders that should be in path. Does not check if they exist.
    Example:
        input: Results_minipatches_20221108_clean_fp/20221014T101512_SmallPatches_C1-CAM1/traj.csv, ["close", "med"]
        output: - Results_minipatches_20221108_clean_fp/20221014T101512_SmallPatches_C1-CAM1/20221014T101512_SmallPatches_C1-CAM1_control_close
                - Results_minipatches_20221108_clean_fp/20221014T101512_SmallPatches_C1-CAM1/20221014T101512_SmallPatches_C1-CAM1_control_med
    """
    # Goes from "XXX/YYY/traj.csv" to ["XXX", "YYY", "traj.csv"]
    parse_path = path.split("/")
    # Remove the final "traj.csv" from the path
    path = path[:-len(parse_path[-1])]
    # Add parent_folder name to path
    path = path + parse_path[-2]
    # Make a list with the controls added to path
    path_list = []
    for condition_name in condition_names:
        path_list.append(path + "_control_" + condition_name)
    return path_list


def load_file_path(folder, file_name):
    """
    Take a folder path ending in /traj.csv, and return the path of the corresponding composite_patches.tif / background.tif / silhouettes.mat.
    In non-control conditions they are just in the same folder so it's easy, but for control sub-folders, you have to
    go look into the parent folder.
    image_name = "composite_patches.tif" or "background.tif"
    """
    folder = folder[:-len(folder.split("/")[-1])]  # in any case, remove traj.csv (or traj_parent.csv, only used for s202405_random_walk_experimental_plates.py)
    # Add extension or _patches if it has been forgotten
    if "composite" in file_name:
        file_name = "composite_patches.tif"
    if "background" in file_name:
        file_name = "background.tif"

    # For model folders, look for the "original_folder.npy" string and load the image from there
    if "model" in folder:
        if "control" not in folder:
            return load_file_path(np.load(folder + "original_folder.npy")[0], file_name)
        else:  # if it's a control subfolder from a model folder, go look for the original_folder.npy in the parent
            # len(folder.split("/")[-2]) is the length of the last subfolder (because split[-1] is just '' when the
            # path ends with a "/". And then we remove one more character than that because of the final "/"
            return load_file_path(np.load(folder[:-len(folder.split("/")[-2]) - 1] + "original_folder.npy")[0], file_name)
    # For non-control experiments it's just in the same folder
    elif "control" not in folder:
        return folder + file_name
    # For control experiments, it's in the parent folder
    else:
        folder = folder[:-1]  # remove end "/" (leftover after we only removed "traj.csv")
        parse_path = folder.split("/")  # get a list of the folders along the path
        return folder[:-len(parse_path[-1])] + file_name  # remove last folder to get to parent, then add composite

