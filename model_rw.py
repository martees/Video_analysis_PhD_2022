## Random walk model
# In this code, I will generate random trajectories (in the sense that at each time step, the agent reorients itself
# in a random direction), with a step-length that depends on whether the agent is inside or outside the patch.
# I will copy metadata (about condition, and patch location) from existing data folders, but replace the traj.csv file
# containing x-y-time data points by my modelled trajectory.

import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from Generating_data_tables import generate_trajectories as gt
from Generating_data_tables import main as gen
import find_data as fd
import plots
import video
import ReferencePoints
import main


def add_mask_to_patch_map(folder, patch_map):
    """
    This function takes a folder and the corresponding patch map, and will add a distinction to the matrix, with pixels
    outside the plate labeled as -2 (pixels that are inside the plate, but outside any patch, are already labeled as
    -1 in patch_map). I only need this distinction for modeled trajectories (to prevent agent from escaping),
    which is why I do not integrate it to the main pipeline. It still uses the same map_from_boundaries function.
    (See map_from_boundaries() and in_patch_all_pixels() for detailed algorithm documentation)
    @param folder: a path to a ./traj.csv file
    @param patch_map: matrix with in each cell the patch to which the corresponding pixel belongs. -1 stands for outside
    any food patch
    @return: a matrix with the same format as patch_map, but also with -2 for pixels that are outside the plate
    """
    # In order to do that, load the position of the four reference points at each corner of the plate
    source_folder_metadata = fd.folder_to_metadata(folder)
    source_xy_holes = source_folder_metadata["holes"][0]
    source_reference_points = ReferencePoints.ReferencePoints(source_xy_holes)

    if len(source_reference_points.xy_holes) == 4:
        [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = source_reference_points.xy_holes

        # Compute an approximate plate center (average of all the coordinates)
        plate_center = [np.mean([x0, x1, x2, x3]), np.mean([y0, y1, y2, y3])]
        # Compute an approximate plate radius (average of the two diagonals of the reference square)
        plate_diameter = np.mean([np.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2), np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)])
        plate_radius = plate_diameter / 2

        # In order to build the mask, we use the method described in in_patch_all pixels: we make a list of points that
        # are at the plate boundary, and then map_from_boundaries() colors the matrix using them.
        # In order for the method to work, we need the boundary to be continuous after the boundary points are set to
        # integers, so at least one point per boundary pixel. The videos are ~2000 px wide, so let's say the plate radius
        # is around ~1000. It means the circumference is around 62 830 px, so we will take 100 000 points just in case.

        # List of discrete positions for the patch boundaries
        boundary_position_list = []
        # Range of 100 000 angular positions
        angular_pos = np.linspace(-np.pi, np.pi, 100000)
        # Add to position list discrete (int) cartesian positions
        # (due to discretization, positions will be added multiple times, but we don't care)
        for point in range(len(angular_pos)):
            x = int(plate_center[0] + (plate_radius * np.cos(angular_pos[point])))
            y = int(plate_center[1] + (plate_radius * np.sin(angular_pos[point])))
            boundary_position_list.append((x, y, 0))  # 3rd tuple element is patch number, I put 0 arbitrarily

        # This function will generate a matrix from the boundary points, containing 0's inside the plate, and -1's outside
        out_of_arena_mask, _ = gt.map_from_boundaries(folder, boundary_position_list, [plate_center])

    else:
        # In the case where there aren't enough reference points, just put an empty out_of_arena_mask
        # Will get excluded later in the clean_results anyway
        _, _, frame_size = fd.load_silhouette(folder)
        out_of_arena_mask = -1 * np.ones((frame_size[0], frame_size[1]))  # for now only -1

    # Sum the two patch maps => inside the plate, patch_map values are unchanged, while outside -1's become -2's
    return patch_map + out_of_arena_mask


def generate_rw_trajectory(speed_in, speed_out, model_folder, silhouettes):
    """
    Function that generates a random walk with the parameters of some duration, with the parameters speed_in and
    speed_out, and the information available in folder (patch map, initial position of the worm).
    @param speed_in: step length of the agent inside food patches
    @param speed_out: step length of the agent outside food patches
    @param model_folder: folder in which to take the in_patch_map which says which pixels belong to which patch
    @param silhouettes: gives the length of the simulated trajectory, and worm silhouettes from the experiments
    @return: returns a traj.csv table with x-y-time info
    """
    duration = len(silhouettes)

    # Load initial position of the worm
    original_folder_path = np.load(model_folder[:-len(model_folder.split("/")[-1])] + "original_folder.npy")[0]
    if len(pd.read_csv(original_folder_path)["x"]) + len(pd.read_csv(original_folder_path)["y"]) > 1:
        initial_position = [pd.read_csv(original_folder_path)["x"][0], pd.read_csv(original_folder_path)["y"][0]]

        # Load matrix with patch information for every pixel of  the arena
        if not os.path.isfile(original_folder_path[:-len(original_folder_path.split("/")[-1])] + "in_patch_matrix.csv"):
            gt.in_patch_all_pixels(original_folder_path)
        patch_map = pd.read_csv(original_folder_path[:-len(original_folder_path.split("/")[-1])] + "in_patch_matrix.csv")

        # Update this matrix by adding -2 in points that are outside the plate
        patch_map = add_mask_to_patch_map(model_folder, patch_map)

        # Initialize outputs
        x_list = [-2 for _ in range(duration)]
        y_list = [-2 for _ in range(duration)]

        # Generate the actual trajectory
        current_x = int(initial_position[0])
        current_y = int(initial_position[1])
        x_list[0] = current_x
        y_list[0] = current_y
        for time in range(1, duration):
            current_patch = patch_map[str(int(np.clip(current_x, 0, len(patch_map["0"]) - 1)))][
                int(np.clip(current_y, 0, len(patch_map) - 1))]

            heading_angle = np.random.rand() * 2 * np.pi
            # If worm is inside
            if current_patch >= 0:
                current_x += speed_in * np.cos(heading_angle)
                current_y += speed_in * np.sin(heading_angle)
            # If worm is outside
            elif current_patch == -1:
                current_x += speed_out * np.cos(heading_angle)
                current_y += speed_out * np.sin(heading_angle)
            # If worm is escaping the plate (current_patch == -2)
            else:
                # While it's escaped, draw a new direction by progressively rotating the heading_angle
                i_while = 0
                while current_patch == -2 and i_while < 70:
                    heading_angle += 0.1
                    x = current_x + speed_out * np.cos(heading_angle)
                    y = current_y + speed_out * np.sin(heading_angle)
                    current_patch = patch_map[str(int(np.clip(x, 0, len(patch_map["0"]) - 1)))][
                        int(np.clip(y, 0, len(patch_map) - 1))]
                    i_while += 1
                # In case the previous, coarse-grained loop didn't work
                i_while = 0
                while current_patch == -2 and i_while < 700:
                    heading_angle += 0.01
                    x = current_x + speed_out * np.cos(heading_angle)
                    y = current_y + speed_out * np.sin(heading_angle)
                    current_patch = patch_map[str(int(np.clip(x, 0, len(patch_map["0"]) - 1)))][
                        int(np.clip(y, 0, len(patch_map) - 1))]
                    i_while += 1
                current_x += speed_out * np.cos(heading_angle)
                current_y += speed_out * np.sin(heading_angle)
            x_list[time] = current_x
            y_list[time] = current_y

        # Shift the silhouettes to match with the new centroids

        trajectory = pd.DataFrame()
        trajectory["id_conservative"] = [0 for _ in range(duration)]
        trajectory["frame"] = list(range(duration))
        trajectory["time"] = list(range(duration))
        trajectory["x"] = x_list
        trajectory["y"] = y_list

        # plt.plot(x_list, y_list)
        # plt.show()

    # In the case where there isn't any data point, just put an empty dataframe
    # Will get excluded later in the clean_results anyway
    else:
        trajectory = pd.DataFrame(columns=["id_conservative", "frame", "time", "x", "y"])

    # return trajectory, shifted_silhouettes
    return trajectory


def generate_model_folders(data_folder_list, modeled_data_path, speed_inside, speed_outside):
    """
    Function that takes a list of folders containing experimental worm data, and copies their content to a new
    model_folder, creating an "original.npy" file with the source folder path, and replaces the original traj.csv
    by a traj.csv containing a modelled trajectory.


    @param data_folder_list: a list of paths to experimental data. They should lead to a traj.csv file.
    @param modeled_data_path: a path to a folder in which to copy experimental folders and generate modeling data.
    @param speed_inside:
    @param speed_outside:
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
        original_folder = [folder, speed_inside, speed_outside]
        np.save(model_folder_path + "/original_folder.npy", original_folder)
        # If the trajectory was named traj_parent, rename it into

    # Then take all the directories contained in the modeled data path (os.walk returns a list with dirpath, dirnames, filenames)
    model_folder_list = next(os.walk(modeled_data_path))[1]
    model_folder_list = [modeled_data_path + model_folder_list[i] + "/traj.csv" for i in range(len(model_folder_list))]
    # And replace them by new, modeled traj.csv, and adapt the silhouettes accordingly
    for i_folder, folder in enumerate(model_folder_list):
        if i_folder % 10 == 0:
            print("Modeling trajectory + silhouettes for folder ", i_folder, " / ", len(data_folder_list))

        if folder == "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp_model_rw/20221011T111318_SmallPatches_C2-CAM2_model/traj.csv":
            print("hey")

        # Find length of silhouettes to find out how many tracked time points the original video had
        current_silhouettes, _, _ = fd.load_silhouette(folder)
        #model_trajectory, shifted_silhouettes = generate_rw_trajectory(speed_inside, speed_outside, folder, current_silhouettes)
        model_trajectory = generate_rw_trajectory(speed_inside, speed_outside, folder, current_silhouettes)
        model_trajectory.to_csv(folder)
        # shifted_silhouettes.to_csv(folder)

    return 0


regenerate_model = False
regenerate_results = False

data_path = gen.generate("", test_pipeline=False)
model_path = gen.generate("", modeled_data=True)

if regenerate_model:
    # Find the path of non-control folders, and then add the control folders by looking for "traj_parent.csv" folders
    list_of_experimental_folders = fd.path_finding_traj(data_path, include_fake_control_folders=False)
    list_of_experimental_folders += fd.path_finding_traj(data_path, target_name="traj_parent.csv")
    # Create modelled data
    # Average speed inside and outside in our experimental plates: 1.4 and 3.5
    generate_model_folders(list_of_experimental_folders, model_path, 1, 40)

    # Regenerate smoothed trajectories.csv table
    gen.generate("beginning", modeled_data=True)

if regenerate_results:
    # Generate same datasets as for experimental data
    gen.generate("results_per_id", modeled_data=True)

# Look at the trajectoriessss
results = pd.read_csv(model_path + "clean_results.csv")
trajectories = pd.read_csv(model_path + "trajectories.csv")

# plots.trajectories_1condition(trajectories, [9], plot_lines=True, save_fig=True, is_plot=False)
# plots.trajectories_1condition(trajectories, [10], plot_lines=True, save_fig=True, is_plot=False)

main.plot_graphs(results, "visit_duration", [["close 0", "med 0", "far 0"]])
main.plot_graphs(results, "visit_duration", [["close 0.2", "med 0.2", "far 0.2"]])
main.plot_graphs(results, "visit_duration", [["close 0.5", "med 0.5", "far 0.5"]])


#TODO handle when the model folder does not exist yet

