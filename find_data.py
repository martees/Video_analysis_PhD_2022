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
        current_data["id_conservative"] = pd.DataFrame([id + 100*i_file for id in current_data["id_conservative"]])

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
        TODO: code and reference from holes positions (see with al)
    :param path: a string with the path leading to the traj.csv whose metadata you want to retrieve (metadata should be in the same folder)
    """
    metadata = pd.DataFrame() #where we'll put everything

    # Finding the path of the other files
    lentoremove = len('traj.csv')  # removes traj from the current path, to get to the parent folder
    path_for_holes = path[:-lentoremove] + "holes.mat"
    path_for_patches = path[:-lentoremove] + "foodpatches.mat"

    # Loadmat function loads .mat file into a dictionnary with meta info
    # the data is stored as a value for the key with the original table name ('traj' for traj.mat)
    holesmat = loadmat(path_for_holes)  # load holes in a dictionary using loadmat
    patchesmat = loadmat(path_for_patches)  # load patches info

    # Extract the data into the dataframe
    # holepositions = pd.DataFrame(holesmat.get('pointList')) # gets the holes positions
    # condition_number = readcode(holepositions) #get the condition from that
    # TODO reference_scale, reference_xyshift, reference_rotation = read_reference_holes(holepositions)

    metadata["patch_centers"] = list(patchesmat.get("centers_patches"))
    metadata["patch_densities"] = list(patchesmat.get("densities_patches"))
    metadata["condition"] = list(patchesmat.get("num_condition"))

    return metadata

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
    '''
    Thats a shitty function that takes hole positions, only looks at code ones (1 in third column), and spits out
    an integer conversion of [line1, line2, line3] as a binary. So not the real codes but it's a bijection so who cares.
    '''
    codepositions = []
    for i in range(len(holepositions)):
        if holepositions[2][i] == 1:
            codepositions.append([holepositions[0][i],holepositions[1][i]])

    # Vertical boundaries
    left_border = min(holepositions[0][:])
    right_border = max(holepositions[0][:])
    vboundary1 = left_border + ((right_border - left_border)/3)
    vboundary2 = left_border + 2*((right_border - left_border)/3)

    # Horizontal boundaries
    bottom_border = min(holepositions[1][:])
    top_border = max(holepositions[1][:])
    hboundary1 = bottom_border + 2 * ((top_border - bottom_border) / 3) #multiply by 2 here because its more convenient to have boundary1 higher
    hboundary2 = bottom_border + ((top_border - bottom_border) / 3)

    code_matrix = np.zeros((3, 3))

    for i in range(len(codepositions)):
        curr_x = codepositions[i][0]
        curr_y = codepositions[i][1]
        if curr_x <= vboundary1:
            if curr_y >= hboundary1:
                code_matrix[0][0] = 1
            elif curr_y >= hboundary2:
                code_matrix[1][0] = 1
            elif curr_y < hboundary2:
                code_matrix[2][0] = 1
        elif curr_x <= vboundary2:
            if curr_y >= hboundary1:
                code_matrix[0][1] = 1
            elif curr_y >= hboundary2:
                code_matrix[1][1] = 1
            elif curr_y < hboundary2:
                code_matrix[2][1] = 1
        elif curr_x > vboundary2:
            if curr_y >= hboundary1:
                code_matrix[0][2] = 1
            elif curr_y >= hboundary2:
                code_matrix[1][2] = 1
            elif curr_y < hboundary2:
                code_matrix[2][2] = 1

    line_code_matrix = code_matrix[0][:] + code_matrix[1][:] + code_matrix[2][:]
    to_code = int("".join(str(int(x)) for x in line_code_matrix), 2)

    return to_code


