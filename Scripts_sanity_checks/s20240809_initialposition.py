# This is a script to plot the theoretical + recorded initial position for the worms,
# in the idealized landscapes.

import matplotlib.pyplot as plt
import numpy as np
import datatable as dt
import pandas as pd
import os

from Generating_data_tables import main as gen
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
import find_data as fd
from Parameters import parameters as param

path = gen.generate("")
results = dt.fread(path + "clean_results.csv")
trajectories = dt.fread(path + "clean_trajectories.csv")
full_list_of_folders = results[:, "folder"].to_list()[0]
print("Finished loading tables!")

distances = ["close", "med", "far", "superfar", "cluster", "control"]
# Lists to fill in following loop
list_of_first_recorded_x = [[] for _ in range(len(distances))]
list_of_first_recorded_y = [[] for _ in range(len(distances))]
background_each_distance = []
for i_distance, distance in enumerate(distances):
    print("Computing the first recorded position for distance " + distance + "!")
    if distance != "control":
        current_conditions = param.name_to_nb_list[distance]
        # Remove controls because they don't have patches
        for cond in current_conditions:
            if param.nb_to_density[cond] == "0":
                current_conditions.remove(cond)
    else:  # for the controls just take one condition because they all correspond to the same experimental data
        current_conditions = param.name_to_nb_list["0"][0]
    current_folder_list = fd.return_folders_condition_list(full_list_of_folders, current_conditions)
    curr_list_x = [0 for _ in range(len(current_folder_list))]
    curr_list_y = [0 for _ in range(len(current_folder_list))]
    for i_folder, folder in enumerate(current_folder_list):
        if i_folder % (len(current_folder_list) // 10) == 0:
            print("> Folder "+str(i_folder)+" / "+str(len(current_folder_list)))
        if i_folder == 0:
            bg_path = fd.load_file_path(folder, "background.tif")
            background_each_distance.append(plt.imread(bg_path))
        # Load matrix with, in each cell, the corresponding idealized coordinates
        xp_to_perfect_path = folder[:-len("traj.csv")] + "xp_to_perfect.npy"
        if not os.path.isfile(xp_to_perfect_path):
            print("Idealized landscapes have not been generated for plate ", folder)
            print("To generate it, go to the global_presence_heatmap.py script.")
        xp_to_perfect_indices = np.load(xp_to_perfect_path)
        # Load first position for this worm
        first_x = trajectories[dt.f.folder == folder, "x"][0, 0]
        first_y = trajectories[dt.f.folder == folder, "y"][0, 0]
        # Fill the list
        curr_list_x[i_folder] = first_x
        curr_list_y[i_folder] = first_y
    # Fill the table
    list_of_first_recorded_x[i_distance] = curr_list_x
    list_of_first_recorded_y[i_distance] = curr_list_y
print("Finished retrieving initial positions!")

# If it's not already done, compute the average patch radius
if not os.path.isfile(path + "perfect_heatmaps/average_patch_radius_each_condition.csv"):
    heatmap_script.generate_average_patch_radius_each_condition(path, full_list_of_folders)
average_patch_radius_each_cond = pd.read_csv(
    path + "perfect_heatmaps/average_patch_radius_each_condition.csv")
average_radius = np.mean(average_patch_radius_each_cond["avg_patch_radius"])

# Compute the idealized patch positions by converting the robot xy data to mm in a "perfect" reference frame
ideal_patch_centers_each_cond = heatmap_script.idealized_patch_centers_mm(path, full_list_of_folders, 1944)

print("Starting the plot!")
# Then, plotto dayou
fig, axs = plt.subplots(1, len(distances))
for i_distance, distance in enumerate(distances):
    if distance == "control":
        current_patch_centers = []
    else:
        one_cond_with_that_distance = 0
        while param.nb_to_distance[one_cond_with_that_distance] != distance:
            one_cond_with_that_distance += 1
            if one_cond_with_that_distance == 11:  # because condition 11 does not exist
                one_cond_with_that_distance += 1
        current_patch_centers = ideal_patch_centers_each_cond[one_cond_with_that_distance]
    for i_patch, curr_xy in enumerate(current_patch_centers):
        patch = plt.Circle((curr_xy[0], curr_xy[1]), radius=average_radius, color="black", fill=False)
        axs[i_distance].add_patch(patch)
    axs[i_distance].imshow(background_each_distance[i_distance], cmap="gray")
    axs[i_distance].scatter(list_of_first_recorded_x[i_distance], list_of_first_recorded_y[i_distance],
                            color=param.name_to_color[distance], alpha=0.3)
    axs[i_distance].set_title(distance)

plt.show()
