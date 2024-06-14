# In this script, I will look at speed and visit time to pixels as a function of the distance to the edge of a patch

from scipy import ndimage
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt
from Parameters import parameters as param
import find_data as fd
import analysis as ana


def generate_patch_distance_map(folder):
    in_patch_matrix_path = folder[:-len("traj.csv")] + "in_patch_matrix.csv"
    in_patch_matrix = pd.read_csv(in_patch_matrix_path)
    in_patch_matrix = in_patch_matrix.to_numpy()

    zeros_inside = np.zeros(in_patch_matrix.shape)
    zeros_outside = np.zeros(in_patch_matrix.shape)
    for i in range(len(in_patch_matrix)):
        for j in range(len(in_patch_matrix[i])):
            if in_patch_matrix[i, j] == -1:
                zeros_inside[i, j] = 1
                zeros_outside[i, j] = 0
            else:
                zeros_inside[i, j] = 0
                zeros_outside[i, j] = 1

    # Create a distance matrix with 0 inside food patches and distance to boundary outside
    distance_transform_outside = ndimage.distance_transform_edt(zeros_inside)
    # Create a distance matrix with 0 outside food patches and distance to boundary inside
    distance_transform_inside = ndimage.distance_transform_edt(zeros_outside)
    # Subtract them from one another so that distance to patch boundary is positive outside, negative inside
    distance_transform = distance_transform_outside - distance_transform_inside

    np.save(folder[:-len(folder.split("/")[-1])] + "distance_to_patch_map.npy", distance_transform)


def pixel_visits_vs_distance_to_boundary(folder_list, traj, total_or_average_or_number="Total", return_nb_of_visited_pixels=False):
    visit_values = []
    distance_values = []
    for i_folder, folder in enumerate(folder_list):
        print(">>> Folder ", i_folder, " / ", len(folder_list))
        # Load the distance map
        distance_map_path = folder[:-len(folder.split("/")[-1])] + "distance_to_patch_map.csv"
        if not os.path.isdir(distance_map_path):
            generate_patch_distance_map(folder)
        distance_map = np.load(folder[:-len(folder.split("/")[-1])] + "distance_to_patch_map.npy")

        print(">>>>>> Loaded distance map")

        # If it's not already done, compute the pixel visit durations
        pixelwise_visits_path = folder[:-len("traj.csv")] + "pixelwise_visits.npy"
        if not os.path.isfile(pixelwise_visits_path):
            gr.generate_pixelwise_visits(traj, folder)
        # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
        pixel_wise_visits = np.load(pixelwise_visits_path, allow_pickle=True)

        print(">>>>>> Loaded pixel visits ")

        # Go through both and add values for visit duration and distance
        for i_line in range(len(distance_map)):
            if i_line % 400 == 0:
                print(">>>>>>>>> Image line ", i_line, " / ", len(distance_map))
            for i_col in range(len(distance_map[0])):
                current_visits = pixel_wise_visits[i_line, i_col]
                if current_visits:
                    visit_durations = ana.convert_to_durations(current_visits)
                    if total_or_average_or_number == "Total":
                        visit_values.append(np.sum(visit_durations))
                    elif total_or_average_or_number == "Average":
                        visit_values.append(np.mean(visit_durations))
                    elif total_or_average_or_number == "Number":
                        visit_values.append(len(visit_durations))
                    distance_values.append(distance_map[i_line, i_col])

    return visit_values, distance_values


def plot_visit_duration_vs_distance(full_folder_list, trajectories, curve_list, curve_names, average_or_total_or_number="Total"):
    """
    Function that will make a plot with the duration of visits as a function of distance to the closest patch boundary
    (negative distance => worm is inside the patch). Average is made over all pixels pooled together for each curve.
    @param full_folder_list:
    @param trajectories:
    @param curve_list:
    @param curve_names:
    @param average_or_total_or_number:
    @return:
    """
    tic = time.time()
    for i_curve, curve in enumerate(curve_list):
        print(int(time.time() - tic), "s: Curve ", i_curve + 1, " / ", len(curve_list))
        folder_list = fd.return_folders_condition_list(full_folder_list, curve)
        visit_values, distance_values = pixel_visits_vs_distance_to_boundary(folder_list, trajectories, total_or_average_or_number=average_or_total_or_number)

        distance_bins, avg_visit_values, [_, _], binned_visits = ana.xy_to_bins(distance_values, visit_values, do_not_edit_xy=False, bin_size=10, custom_bins=[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 80, 120], compute_bootstrap=False)

        condition_name = curve_names[i_curve]
        condition_color = param.name_to_color[condition_name]

        # Plot error bars
        plt.plot(distance_bins, avg_visit_values, color=condition_color, linewidth=4,
                 label=condition_name)
        #plt.errorbar(distance_bins, avg_visit_values, [errors_inf, errors_sup], fmt='.k', capsize=5)

    print("Total time: ", int((time.time() - tic) // 60), "min")

    plt.title(average_or_total_or_number + " pixel visit duration as a function of distance to the edge of the patch in " + condition_name)
    plt.ylabel("Average visit to pixel")
    plt.xlabel("Distance to closest patch boundary ( <0 = inside)")
    plt.legend()
    plt.show()


# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=True)
results = pd.read_csv(path + "clean_results.csv")
trajectories = pd.read_csv(path + "clean_trajectories.csv")
full_list_of_folders = results["folder"]

import cProfile
import pstats

#profiler = cProfile.Profile()
#profiler.enable()
#plot_visit_duration_vs_distance(full_list_of_folders, [[0], [1], [2]], ['close 0.2', 'med 0.2', 'far 0.2'], average_or_total="Average")
#plot_visit_duration_vs_distance(full_list_of_folders, [[4], [5], [6]], ['close 0.5', 'med 0.5', 'far 0.5'], average_or_total="Average")
#plot_nb_of_visits_vs_distance(full_list_of_folders, [[0], [1], [2]], ['close 0.2', 'med 0.2', 'far 0.2'])
#plot_nb_of_visits_vs_distance(full_list_of_folders, [[4], [5], [6]], ['close 0.5', 'med 0.5', 'far 0.5'])
#profiler.disable()
#stats = pstats.Stats(profiler).sort_stats('cumtime')
#stats.print_stats()

plot_visit_duration_vs_distance(full_list_of_folders, trajectories, [[0], [1], [2]], ['close 0.2', 'med 0.2', 'far 0.2'], average_or_total_or_number="Number")
plot_visit_duration_vs_distance(full_list_of_folders, trajectories, [[4], [5], [6]], ['close 0.5', 'med 0.5', 'far 0.5'], average_or_total_or_number="Number")
plot_visit_duration_vs_distance(full_list_of_folders, trajectories, [[12], [13], [14]], ['close 0', 'med 0', 'far 0'], average_or_total_or_number="Number")
plot_visit_duration_vs_distance(full_list_of_folders, trajectories, [[0, 1, 2, 3], [4, 5, 6, 7], [8], [12, 13, 14, 15]], ["0.2", "0.5", "1.25", "control"], average_or_total_or_number="Number")







