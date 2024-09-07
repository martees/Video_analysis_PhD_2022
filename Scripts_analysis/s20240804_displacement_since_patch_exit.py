from typing import Callable, Any

import os
import numpy as np
import matplotlib.pyplot as plt
import datatable as dt
from scipy import ndimage
import time

from main import *
import find_data as fd
from Generating_data_tables import main as gen
from Generating_data_tables import generate_trajectories as gt
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
from Scripts_sanity_checks import s20240810_interpatch_distance as interpatch_script

# Analysis of the worm's displacement evolution after leaving a food patch
# Used to plot displacement as a function of time since leaving, now plots time to reach a certain displacement
# (to avoid sampling biases)

path = gen.generate("", test_pipeline=False)
results = pd.read_csv(path + "clean_results.csv")
trajectories = dt.fread(path + "clean_trajectories.csv")

# Parameters
#curve_list = ["close ", "med", "far", "superfar"]
#curve_list = ["close 0", "med 0", "far 0", "superfar 0"]
#curve_list = ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]
#curve_list = ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]
#curve_list = ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]
#curve_list = ["close 0", "close 0.2", "close 0.5", "close 1.25"]
#curve_list = ["med 0", "med 0.2", "med 0.5", "med 1.25"]
#curve_list = ["far 0", "far 0.2", "far 0.5", "far 1.25"]
curve_list = ["superfar 0", "superfar 0.2", "superfar 0.5", "superfar 1.25"]
displacement_bin_list = [1, 10, 20, 30, 40, 50, 70, 100, 200, 400, 800]
min_length = 1  # minimal transit length to get considered
min_nb_data_points = 10  # minimal number of transits for a point to get plotted
recompute = False

# Extract the average distance between patches in the conditions to analyze.
# That is because, to avoid sampling biases, when comparing a "close" condition to a "med" or "far" one,
# we exclude transits once they have exceeded the point where there could've been a food patch in the "close".
# (This also affects close trajectories, since we ignore worms after they cross that radius, but we found that it was
# the cleanest way of not having the fastest worms in the "close" only in short transits, because they find a patch.)
# if not os.path.isfile(gen.generate("") + "interpatch_distance.csv"):
#     interpatch_script.generate_patch_distances()
# interpatch_dataframe = pd.read_csv(gen.generate("") + "interpatch_distance.csv")
# # Remove the radius!
# if not os.path.isfile(path + "perfect_heatmaps/average_patch_radius_each_condition.csv"):
#     heatmap_script.generate_average_patch_radius_each_condition(path, results["folder"])
# average_patch_radius_each_cond = pd.read_csv(
#     path + "perfect_heatmaps/average_patch_radius_each_condition.csv")
# average_radius = np.mean(average_patch_radius_each_cond["avg_patch_radius"])
# # Compute the average interpatch distance from boundary to boundary (so remove twice the radius)
# smallest_distance = np.min(interpatch_dataframe["interpatch_distance"]) - 2 * average_radius

#Init
tic = time.time()
nb_of_bins = len(displacement_bin_list)
for i_curve in range(len(curve_list)):
    curve_name = curve_list[i_curve]
    current_condition_list = param.name_to_nb_list[curve_name]
    print(current_condition_list, curve_name)
    folder_list = fd.return_folders_condition_list(results["folder"], current_condition_list)

    analysis_subfolder = path + "distance_since_exit_analysis/"
    if not os.path.isfile(analysis_subfolder + "conditions_"+curve_name+"_avg_time_each_bin.npy") or recompute:
        # Init: lists to fill with average for each plate
        avg_time_each_bin_each_plate = np.empty((nb_of_bins, len(folder_list)))
        avg_time_each_bin_each_plate[:] = np.nan
        nb_of_points_each_bin_each_plate = np.empty((nb_of_bins, len(folder_list)))
        nb_of_points_each_bin_each_plate[:] = np.nan
        for i_folder, current_folder in enumerate(folder_list):
            if i_folder % 1 == 0:
                print(int((time.time() - tic) // 60), "min: Folder ", i_folder, " / ", len(folder_list))
            # Load visits
            current_results = results[results["folder"] == current_folder].reset_index(drop=True)
            current_traj = trajectories[dt.f.folder == current_folder, :]
            list_of_transits = fd.load_list(current_results, "aggregated_raw_transits")
            # Load the matrix with patch to which each pixel belongs
            in_patch_matrix_path = current_folder[:-len("traj.csv")] + "in_patch_matrix.csv"
            if not os.path.isfile(in_patch_matrix_path):
                gt.in_patch_all_pixels(in_patch_matrix_path)
            in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()
            # For each patch, create a map with, for each pixel, the distance to the boundary of that patch
            print(int((time.time() - tic) // 60), "min: Computing distance map each patch...")
            nb_of_patches = len(np.unique(in_patch_matrix)) - 1
            distance_map_each_patch = [[] for _ in range(nb_of_patches)]
            for i_patch in range(nb_of_patches):
                zeros_inside = np.where(in_patch_matrix == i_patch, 0, 1)
                # Create a distance matrix with 0 inside food patches and distance to boundary outside
                distance_map_each_patch[i_patch] = ndimage.distance_transform_edt(zeros_inside)
            # Lists of frames where worm exits patches (visit and transit starts) and it lasts more than time window
            # (used to exclude transits that are so short that we might want to consider them as artifacts)
            # (but for now it's set at 1 so does not do anything lol)
            long_enough_transits = [list_of_transits[i] for i in range(len(list_of_transits)) if
                                    list_of_transits[i][1] - list_of_transits[i][0] >= min_length]
            exit_frames = [long_enough_transits[i][0] for i in range(len(long_enough_transits))]
            end_frames = [long_enough_transits[i][1] for i in range(len(long_enough_transits))]
            print(int((time.time() - tic) // 60), "min: Starting to fill lists...")
            # Init
            current_folder_times = [[] for _ in range(nb_of_bins + 1)]
            # Fill time list
            for i_exit, current_exit_frame in enumerate(exit_frames):
                current_end_frame = end_frames[i_exit]
                exit_index = fd.find_closest(current_traj[:, dt.f.frame].to_list()[0], current_exit_frame)
                exit_time = fd.load_time(current_traj.to_pandas(), current_folder, exit_index)
                end_index = fd.find_closest(current_traj[:, dt.f.frame].to_list()[0], current_end_frame)
                exit_from = current_traj[exit_index, dt.f.patch_silhouette][0, 0]
                if exit_from != -1:  # if the video starts with a transit, then the worm is not exiting any patch
                    distance_map_this_patch = distance_map_each_patch[exit_from]
                    # Check if frames are continuous around exit: otherwise, exclude it completely (for now because I'm tired)
                    if end_index - exit_index >= current_end_frame - current_exit_frame:
                        x_list = current_traj[exit_index:end_index, dt.f.x].to_list()[0]
                        y_list = current_traj[exit_index:end_index, dt.f.y].to_list()[0]
                        xy_list = np.stack((x_list, y_list), axis=1)
                        distance_function: Callable[[Any], int] = lambda xy: distance_map_this_patch[int(xy[1])][int(xy[0])]
                        displacement_list = np.array(list(map(distance_function, xy_list)))
                        # Add elements smaller than first bin
                        index_each_bin = [list(np.where(displacement_list < displacement_bin_list[0])[0])]
                        # Find the index of the displacements that fit in each bin
                        index_each_bin += [list(np.where((displacement_bin_list[i_bin] <= displacement_list)
                                                         & (displacement_list < displacement_bin_list[i_bin + 1]))[0])
                                           for i_bin in range(len(displacement_bin_list) - 1)]
                        # Then convert those indices to times post exit
                        time_function = lambda i: fd.load_time(current_traj.to_pandas(), current_folder,
                                                               exit_index + i) - exit_time
                        current_folder_times = [list(map(time_function, indices)) for indices in index_each_bin]
                        #for i_distance, distance in enumerate(displacement_list):
                        #    i_bin = np.searchsorted(displacement_bin_list, distance)
                        #    curr_time = fd.load_time(current_traj.to_pandas(), current_folder, exit_index + i_distance)
                        #    post_exit_time = curr_time - exit_time
                        #    current_folder_times[i_bin].append(post_exit_time)
                        # Compute index at which to stop (because we exclude trajectories once they reach some radius, see
                        # explanation near definition of smallest_distance
                        #if np.max(displacement_list) > smallest_distance:
                        #    index_to_stop = fd.find_closest(displacement_list, smallest_distance)
                        #else:
                        #    index_to_stop = len(displacement_list)
                        #current_bin = 0
                        #i_time = 0
                        #while i_time < index_to_stop and current_bin < nb_of_bins:
                        #    if i_time > displacement_bin_list[current_bin]:
                        #        current_bin += 1
                        #    else:
                        #        current_folder_times[current_bin].append(displacement_list[i_time])
                        #        i_time += 1

            # At this point, avg_displacement_each_bin_each_plate is filled with one sublist per bin
            # and each sublist contains the worms' displacement during those time steps. Now we average for each bin
            for i_bin in range(nb_of_bins):
                if i_bin == 0:
                    current_bin_size = 1
                else:
                    current_bin_size = displacement_bin_list[i_bin] - displacement_bin_list[i_bin - 1]
                if len(current_folder_times[i_bin]) > 0:
                    avg_time_each_bin_each_plate[i_bin][i_folder] = np.nanmean(current_folder_times[i_bin])
                    nb_of_points_each_bin_each_plate[i_bin][i_folder] = len(current_folder_times[i_bin]) / current_bin_size

        # Save it to a csv table
        if not os.path.isdir(analysis_subfolder):
            os.mkdir(analysis_subfolder)
        np.save(analysis_subfolder + "conditions_"+curve_name+"_avg_time_each_bin.npy", avg_time_each_bin_each_plate)
        np.save(analysis_subfolder + "conditions_"+curve_name+"_nb_of_points_each_bin.npy", nb_of_points_each_bin_each_plate)

    avg_time_each_bin_each_plate = np.load(analysis_subfolder + "conditions_" + curve_name + "_avg_time_each_bin.npy")
    nb_of_points_each_bin_each_plate = np.load(analysis_subfolder + "conditions_" + curve_name + "_nb_of_points_each_bin.npy")

    # Now that we have the full list of averages for each distance reached after exit, average and bootstrap all that
    avg_time_each_bin = np.empty(nb_of_bins)
    avg_time_each_bin[:] = np.nan
    # Errors
    errors_inf = np.empty(nb_of_bins)
    errors_sup = np.empty(nb_of_bins)
    errors_inf[:] = np.nan
    errors_sup[:] = np.nan

    for i_bin in range(nb_of_bins):
        # Rename
        values_this_distance = avg_time_each_bin_each_plate[i_bin]
        # and remove nan values for bootstrapping
        values_this_distance = [values_this_distance[i] for i in range(len(values_this_distance)) if
                                not np.isnan(values_this_distance[i])]
        if values_this_distance:
            current_avg = np.nanmean(values_this_distance)
            avg_time_each_bin[i_bin] = current_avg
            bootstrap_ci = ana.bottestrop_ci(values_this_distance, 1000)
            errors_inf[i_bin] = current_avg - bootstrap_ci[0]
            errors_sup[i_bin] = bootstrap_ci[1] - current_avg

    y_list = avg_time_each_bin[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
    x_list = np.array(displacement_bin_list) + i_curve * 0.2
    x_list = x_list[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
    errors_inf = errors_inf[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
    errors_sup = errors_sup[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]

    plt.plot(x_list, y_list, color=param.name_to_color[curve_name], label=curve_name, linewidth=3)
    plt.errorbar(x_list, y_list, [errors_inf, errors_sup], fmt='.k', capsize=5)

plt.title("Average time to reach each radius around the patches")
plt.xticks(displacement_bin_list, [str(b) for b in displacement_bin_list])
plt.ylabel("Average time post exit (s)")
plt.xlabel("Radius around patch")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()
