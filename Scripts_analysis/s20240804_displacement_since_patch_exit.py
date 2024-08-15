from typing import Callable, Any

import os
import numpy as np
import matplotlib.pyplot as plt
import datatable as dt
from scipy import ndimage

from main import *
import find_data as fd
from Generating_data_tables import main as gen
from Generating_data_tables import generate_trajectories as gt
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script

# Analysis of the worm's displacement evolution after leaving a food patch

path = gen.generate("")
results = pd.read_csv(path + "clean_results.csv")
trajectories = dt.fread(path + "clean_trajectories.csv")

# Parameters
curve_list = ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]
#curve_list = ["close 0", "med 0", "far 0", "superfar 0"]
#curve_list = ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]
#curve_list = ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]
# curve_list = ["close 1.25"]
bin_list = [1, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
min_length = 1  # minimal transit length to get considered
min_nb_data_points = 4  # minimal number of transits for a point to get plotted

time_window = max(bin_list)
nb_of_bins = len(bin_list)

for i_curve in range(len(curve_list)):
    curve_name = curve_list[i_curve]
    current_condition_list = param.name_to_nb_list[curve_name]
    print(current_condition_list, curve_name)
    folder_list = fd.return_folders_condition_list(results["folder"], current_condition_list)

    # Init: lists to fill with average displacement for each time step pre/post entry/exit for each plate
    avg_displacement_each_bin_each_plate = np.empty((nb_of_bins, len(folder_list)))
    avg_displacement_each_bin_each_plate[:] = np.nan
    nb_of_points_each_bin_each_plate = np.empty((nb_of_bins, len(folder_list)))
    nb_of_points_each_bin_each_plate[:] = np.nan
    for i_folder, current_folder in enumerate(folder_list):
        if i_folder % 20 == 0:
            print("Folder ", i_folder, " / ", len(folder_list))
        # Load visits
        current_results = results[results["folder"] == current_folder].reset_index(drop=True)
        current_traj = trajectories[dt.f.folder == current_folder, :]
        list_of_transits = fd.load_list(current_results, "aggregated_raw_transits")
        # Load the matrix with patch to which each pixel belongs
        in_patch_matrix_path = current_folder[:-len("traj.csv")] + "in_patch_matrix.csv"
        if not os.path.isfile(in_patch_matrix_path):
            gt.in_patch_all_pixels(in_patch_matrix_path)
        in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()
        # Load the polar map matrix, which contains [i, r_b, theta], where we're interested in i,
        # the index of the closest food patch
        # Then, if it's not already done, or has to be redone, compute the polar coordinates for the plate
        polar_map_path = current_folder[:-len("traj.csv")] + "polar_map.npy"
        if not os.path.isfile(polar_map_path):
            heatmap_script.generate_polar_map(current_folder)
        current_polar_map = np.load(polar_map_path)
        closest_patch_index = current_polar_map[:, :, 0]
        # For each patch, create a map with, for each pixel, the distance to the boundary of that patch
        nb_of_patches = len(np.unique(in_patch_matrix)) - 1
        distance_map_each_patch = [[] for _ in range(nb_of_patches)]
        for i_patch in range(nb_of_patches):
            zeros_inside = np.where((closest_patch_index == i_patch) & (in_patch_matrix != -1), 0, 1)
            distance_map_each_patch[i_patch] = ndimage.distance_transform_edt(zeros_inside)

        # Lists of frames where worm exits patches (visit and transit starts) and it lasts more than time window
        # (any visit or transit shorter than time window would lead to an "impure" behavior => excluded from analysis)
        long_enough_transits = [list_of_transits[i] for i in range(len(list_of_transits)) if
                                list_of_transits[i][1] - list_of_transits[i][0] >= min_length]
        exit_frames = [long_enough_transits[i][0] for i in range(len(long_enough_transits))]
        end_frames = [long_enough_transits[i][1] for i in range(len(long_enough_transits))]

        # Init
        current_folder_values = [[] for _ in range(nb_of_bins)]
        # Fill displacement list
        for i_exit, current_exit_frame in enumerate(exit_frames):
            current_end_frame = end_frames[i_exit]
            exit_index = fd.find_closest(current_traj[:, dt.f.frame].to_list()[0], current_exit_frame)
            end_index = fd.find_closest(current_traj[:, dt.f.frame].to_list()[0], current_end_frame)
            exit_from = current_traj[exit_index, dt.f.patch_silhouette][0, 0]
            distance_this_patch = distance_map_each_patch[exit_from]
            # Check if frames are continuous around exit: otherwise, exclude it completely (for now because I'm tired)
            if end_index - exit_index >= current_end_frame - current_exit_frame:
                x_list = current_traj[exit_index:end_index, dt.f.x].to_list()[0]
                y_list = current_traj[exit_index:end_index, dt.f.y].to_list()[0]
                xy_list = np.stack((x_list, y_list), axis=1)
                distance_function: Callable[[Any], int] = lambda xy: distance_this_patch[int(xy[1])][int(xy[0])]
                displacement_list = list(map(distance_function, xy_list))
                current_bin = 0
                i_time = 0
                while i_time < len(displacement_list) and current_bin < nb_of_bins:
                    if i_time > bin_list[current_bin]:
                        current_bin += 1
                    else:
                        #print("len folder val: ", len(current_folder_values), "curr b: ", current_bin)
                        #print("len disp list: ", len(displacement_list), "i time: ", i_time)
                        current_folder_values[current_bin].append(displacement_list[i_time])
                        if displacement_list[i_time] > 2000:
                            print("wtf")
                        i_time += 1

        # At this point, speed_before_entry, speed_after_... etc. are filled with one sublist per bin
        # and each sublist contains the worms' speeds during those time steps. Now we average for each bin
        for i_bin in range(nb_of_bins):
            if i_bin == 0:
                current_bin_size = 1
            else:
                current_bin_size = bin_list[i_bin] - bin_list[i_bin - 1]
            if len(current_folder_values[i_bin]) > 0:
                avg_displacement_each_bin_each_plate[i_bin][i_folder] = np.nanmean(current_folder_values[i_bin])
                nb_of_points_each_bin_each_plate[i_bin][i_folder] = len(current_folder_values[i_bin]) / current_bin_size

    # Now that we have the full list of averages for each time step before/after entry/exit, average and bootstrap all that
    avg_displacement_each_bin = np.empty(nb_of_bins)
    avg_displacement_each_bin[:] = np.nan
    # Errors
    errors_inf = np.empty(nb_of_bins)
    errors_sup = np.empty(nb_of_bins)
    errors_inf[:] = np.nan
    errors_sup[:] = np.nan

    for i_bin in range(nb_of_bins):
        # Rename
        values_this_time = avg_displacement_each_bin_each_plate[i_bin]
        # and remove nan values for bootstrapping
        values_this_time = [values_this_time[i] for i in range(len(values_this_time)) if
                            not np.isnan(values_this_time[i])]
        if values_this_time:
            current_avg = np.nanmean(values_this_time)
            avg_displacement_each_bin[i_bin] = current_avg
            bootstrap_ci = ana.bottestrop_ci(values_this_time, 1000)
            errors_inf[i_bin] = current_avg - bootstrap_ci[0]
            errors_sup[i_bin] = bootstrap_ci[1] - current_avg

    y_list = avg_displacement_each_bin[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
    x_list = np.array(bin_list) + 10*i_curve
    x_list = x_list[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
    errors_inf = errors_inf[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
    errors_sup = errors_sup[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]

    plt.plot(x_list, y_list, color=param.name_to_color[curve_name], label=curve_name, linewidth=3)
    plt.errorbar(x_list, y_list, [errors_inf, errors_sup], fmt='.k', capsize=5)

plt.title("Average displacement from patch boundary as a function of time since exit")
plt.xticks(bin_list, [str(b) for b in bin_list])
plt.ylabel("Average displacement")
plt.xlabel("Time post exit")
plt.legend()
plt.show()
