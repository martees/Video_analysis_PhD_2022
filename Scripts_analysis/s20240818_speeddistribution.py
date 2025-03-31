# This is a script to look at the distribution of speeds inside vs. outside in all conditions
import numpy as np
import find_data as fd
import matplotlib.pyplot as plt
import time
from Parameters import parameters as param
from Generating_data_tables import main as gen
import pandas as pd
import datatable as dt


def load_speeds_one_plate(trajectory_this_plate, only_first_visit):
    nb_of_time_steps = trajectory_this_plate.shape[0]
    list_of_speeds_inside = -1 * np.ones(nb_of_time_steps)
    list_of_speeds_outside = -1 * np.ones(nb_of_time_steps)
    already_visited_patches = []
    currently_visited_patch = -1
    for i_time in range(nb_of_time_steps):
        current_patch = trajectory_this_plate[i_time, "patch_silhouette"]
        if current_patch == -1:
            list_of_speeds_outside[i_time] = trajectory_this_plate[i_time, "speeds"]
        else:
            if only_first_visit:
                # If this is the patch being visited or if it was not visited yet
                if (current_patch == currently_visited_patch) or (current_patch not in already_visited_patches):
                    list_of_speeds_inside[i_time] = trajectory_this_plate[i_time, "speeds"]
                # If this patch is different from the "patch currently visited"
                if current_patch != currently_visited_patch:
                    already_visited_patches.append(current_patch)  # then add the previous to already visited
                    currently_visited_patch = current_patch  # and update the currently visited patch
            else:
                list_of_speeds_inside[i_time] = trajectory_this_plate[i_time, "speeds"]
    # Remove the -1's, as they were just there to avoid using appends
    list_of_speeds_inside = [l for l in list_of_speeds_inside if l != -1]
    list_of_speeds_outside = [l for l in list_of_speeds_outside if l != -1]
    return list_of_speeds_inside, list_of_speeds_outside


def plot_speed_distribution(full_folder_list, trajectories, condition_list, first_visit=False):
    tic = time.time()
    # Create figure
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Speed inside" + " (1st visit each patch)"*first_visit)
    axs[1].set_title("Speed outside")

    # Compute and draw the curves
    folder_list, condition_each_folder = fd.return_folders_condition_list(full_folder_list, condition_list, return_conditions=True)
    for i_condition, condition in enumerate(condition_list):
        print("Computing speed distribution for condition ", param.nb_to_name[condition])
        current_folder_list = np.array(folder_list)[np.array(condition_each_folder) == condition]
        speeds_inside_this_condition = []
        speeds_outside_this_condition = []
        for folder in current_folder_list:
            current_trajectory = trajectories[dt.f.folder == folder, :]
            speeds_inside_this_plate, speeds_outside_this_plate = load_speeds_one_plate(current_trajectory, only_first_visit=first_visit)
            speeds_inside_this_plate = [s for s in speeds_inside_this_plate if not np.isnan(s)]
            speeds_outside_this_condition = [s for s in speeds_outside_this_condition if not np.isnan(s)]
            speeds_inside_this_condition += speeds_inside_this_plate
            speeds_outside_this_condition += speeds_outside_this_plate

        condition_name = param.nb_to_name[condition]
        condition_color = param.name_to_color[condition_name]
        axs[0].hist(speeds_inside_this_condition, histtype="step", density=True, cumulative=-1, label=condition_name,
                 color=condition_color, bins=list(np.linspace(0, 4, 41)), linewidth=2)
        axs[1].hist(speeds_outside_this_condition, histtype="step", density=True, cumulative=-1, label=condition_name,
                 color=condition_color, bins=list(np.linspace(0, 4, 41)), linewidth=2)

    plt.legend()
    plt.show()


path = gen.generate("")
results = pd.read_csv(path + "clean_results.csv")
traj = dt.fread(path + "clean_trajectories.csv")
full_list_of_folders = list(results["folder"])

plot_speed_distribution(full_list_of_folders, traj, [0, 1, 2, 14], first_visit=False)
plot_speed_distribution(full_list_of_folders, traj, [17, 18, 19, 20], first_visit=False)
plot_speed_distribution(full_list_of_folders, traj, [4, 5, 6, 15], first_visit=False)
plot_speed_distribution(full_list_of_folders, traj, [12, 8, 13, 16], first_visit=False)
