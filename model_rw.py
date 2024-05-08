## Random walk model
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
import plots


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
    # Load initial position of the worm
    original_folder_path = np.load(model_folder[:-len(model_folder.split("/")[-1])] + "original_folder.npy")[0]
    initial_position = [pd.read_csv(original_folder_path)["x"][0], pd.read_csv(original_folder_path)["y"][0]]

    # Load matrix with patch information for every pixel of  the arena
    if not os.path.isfile(original_folder_path[:-len(original_folder_path.split("/")[-1])] + "in_patch_matrix.csv"):
        gt.in_patch_all_pixels(original_folder_path)
    patch_map = pd.read_csv(original_folder_path[:-len(original_folder_path.split("/")[-1])] + "in_patch_matrix.csv")

    # Initialize outputs
    x_list = [-2 for _ in range(duration)]
    y_list = [-2 for _ in range(duration)]

    current_x = int(initial_position[0])
    current_y = int(initial_position[1])
    for time in range(duration):
        x_list[time] = current_x
        y_list[time] = current_y

        if type(current_x) != int:
            print("jjoojo")

        current_patch = patch_map[str(current_y)][current_x]
        turning_angle = 360 * random.random()
        # If worm is outside
        if current_patch == -1:
            current_x += speed_out * np.cos(turning_angle)
            current_y += speed_out * np.sin(turning_angle)
        # If worm is inside
        else:
            current_x += speed_in * np.cos(turning_angle)
            current_y += speed_in * np.sin(turning_angle)
        # Force the worm to stay inside the camera field
        current_x = int(np.clip(current_x, 0, 1943))
        current_y = int(np.clip(current_y, 0, 1943))

    trajectory = pd.DataFrame()
    trajectory["id_conservative"] = [0 for _ in range(duration)]
    trajectory["frame"] = list(range(duration))
    trajectory["time"] = list(range(duration))
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
        # First, copy the folders from the experimental part of the world
        source_folder_name = folder.split("/")[-2]  # take the last subfolder (split[-1] is "traj.csv")
        model_folder_path = modeled_data_path + source_folder_name + "_model"
        # If the model folder does not exist yet, create folder
        if not os.path.isdir(model_folder_path):
            os.mkdir(model_folder_path)
        # Copy pasta some metadata from the original folder
        list_of_metadata_stuff = ["/holes.mat", "/foodpatches.mat", "/foodpatches_new.mat"]
        for file in list_of_metadata_stuff:
            if not os.path.isfile(model_folder_path + file):
                shutil.copy(folder[:-len(folder.split("/")[-1])] + file, model_folder_path + file)
        # Add to the folders a .npy containing their origin (that's just for the generate_rw_trajectory function,
        # and XXXXX function, which will use it to build the "folder" column of the trajectories.csv dataframe,
        # so that we don't have to copy all the heavy metadata from the original folder)
        original_folder = [folder]
        np.save(model_folder_path + "/original_folder.npy", original_folder)
        # If the trajectory was named traj_parent, rename it into

    # Then take all the empty folders that you created
    model_folder_list = os.listdir(modeled_data_path)
    model_folder_list = [modeled_data_path + model_folder_list[i] + "/traj.csv" for i in range(len(model_folder_list))]
    # And replace them by new, modeled traj.csv
    for i_folder, folder in enumerate(model_folder_list):
        # Find length of silhouettes to find out how many tracked time points the original video had
        current_silhouettes, _, _ = fd.load_silhouette(folder)
        model_trajectory = generate_rw_trajectory(1.4, 3.5, folder, len(current_silhouettes))
        model_trajectory.to_csv(folder)

    return 0


regenerate = True

data_path = gen.generate("", test_pipeline=True)
model_path = gen.generate("", modeled_data=True)

if regenerate:
    # Find the path of non-control folders, and then add the control folders by looking for "traj_parent.csv" folders
    list_of_experimental_folders = fd.path_finding_traj(data_path, include_fake_control_folders=False)
    list_of_experimental_folders += fd.path_finding_traj(data_path, target_name="traj_parent.csv")
    # Create modelled data
    generate_model_folders(list_of_experimental_folders, model_path)
    # Generate same datasets as for experimental data
    gen.generate("beginning", modeled_data=True)

# Look at the trajectoriessss
trajectories = pd.read_csv(model_path + "trajectories.csv")
plots.trajectories_1condition(trajectories, [0, 1, 2, 3])


