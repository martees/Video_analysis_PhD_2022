# In this code, I will generate random trajectories (in the sense that at each time step, the agent reorients itself
# in a random direction), with a step-length that depends on whether the agent is inside or outside the patch.
# I will copy metadata (about condition, and patch location) from existing data folders, but replace the traj.csv file
# containing x-y-time data points by my modelled trajectory.

import os
import shutil
import pandas as pd
import numpy as np
import random
from Generating_data_tables import generate_trajectories as gt
from Generating_data_tables import main as gen
import find_data as fd


def generate_rw_trajectory(speed_in, speed_out, model_folder, duration):
    """
    Function that generates a random walk with the parameters of some duration, with the parameters speed_in and
    speed_out, and the information available in folder (patch map, initial position of the worm).
    @param speed_in: step length of the agent inside food patches
    @param speed_out: step length of the agent outside food patches
    @param model_folder: folder in which to take the in_patch_map which says which pixels belong to which patch
    @param duration: length of the simulated trajectory
    @return: returns a traj.csv table with x-y-time info
    """

    # Load matrix with patch information for every pixel of the arena
    if os.path.isfile(model_folder[:-len("traj.csv")] + "in_patch_matrix.csv"):
        patch_map = pd.read_csv(model_folder[:-len("traj.csv")] + "in_patch_matrix.csv")
    else:
        patch_map = gt.in_patch_all_pixels(model_folder)

    # Load initial position of the worm
    original_folder_path = np.load(model_folder[:-len("traj.csv")] + "original_folder.npy")[0]
    initial_position = [pd.read_csv(original_folder_path)["x"][0], pd.read_csv(original_folder_path)["y"][0]]

    # Initialize outputs
    x_list = [-2 for _ in range(duration)]
    y_list = [-2 for _ in range(duration)]
    time_list = range(duration)

    current_x = initial_position[0]
    current_y = initial_position[1]
    for time in range(duration):
        x_list[time] = current_x
        y_list[time] = current_y
        current_patch = patch_map[current_y][current_x]
        turning_angle = 360 * random.random()
        # If worm is outside
        if current_patch == -1:
            current_x += speed_out * np.cos(turning_angle)
            current_y += speed_out * np.sin(turning_angle)
        # If worm is inside
        else:
            current_x += speed_in * np.cos(turning_angle)
            current_y += speed_in * np.sin(turning_angle)

    trajectory = pd.DataFrame()
    trajectory["frame"] = time_list
    trajectory["time"] = time_list
    trajectory["x"] = x_list
    trajectory["y"] = y_list

    return trajectory


def generate_model_folders(data_folder_list, modeled_data_path):
    """
    Function that takes a list of folders containing experimental worm data, and copies their content to a new
    model_folder, creating an "original.npy" file with the source folder path, and replaces the original traj.csv
    by a traj.csv containing a modelled trajectory.

    @param data_folder_list: a list of paths to experimental data. They should lead to a traj.csv file.
    @param modeled_data_path: a path to a folder in which to copy experimental folders and generate modeling data.
    @return: None.
    """
    for i_folder, folder in enumerate(data_folder_list):
        source_folder_name = folder.split("/")[-2]  # take the last subfolder (split[-1] is "traj.csv")
        experimental_data_path = folder[:-len("traj.csv")]
        model_folder_path = modeled_data_path + source_folder_name + "_model"
        shutil.copytree(experimental_data_path, model_folder_path)
        
        model_trajectory = generate_rw_trajectory(1.4, 3.5, model_folder_path, 30000)
        model_trajectory.to_csv(model_folder_path+"/traj.csv")
    return 0


data_path = gen.generate("", test_pipeline=True)
model_path = gen.generate("", modeled_data=True)
list_of_experimental_folders = fd.path_finding_traj(data_path)
# Create modelled data
generate_model_folders(list_of_experimental_folders, model_path)
# Generate same datasets as for experimental data
gen.generate("controls", modeled_data=True)




