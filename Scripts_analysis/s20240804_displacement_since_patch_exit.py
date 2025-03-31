from typing import Callable, Any

import os
import numpy as np
import matplotlib.pyplot as plt
import datatable as dt
from scipy import ndimage
import time

import Parameters.parameters
from main import *
import find_data as fd
from Generating_data_tables import main as gen
from Generating_data_tables import generate_trajectories as gt
# from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
# from Scripts_sanity_checks import s20240810_interpatch_distance as interpatch_script

# Analysis of the worm's displacement evolution after leaving a food patch
# Used to plot displacement as a function of time since leaving, now plots time to reach a certain displacement
# (to avoid sampling biases)


def msd_analysis(results, trajectories, curve_list, displacement_bin_list, min_length, min_nb_data_points, recompute,
                 time_or_probability="time", is_plot=True):
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
            nb_of_transits_each_plate = np.empty(len(folder_list))
            nb_of_transits_each_plate[:] = np.nan
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
                # For each visited patch, create a map with, for each pixel, the distance to the boundary of that patch
                print(int((time.time() - tic) // 60), "min: Computing distance map each patch...")
                visits = fd.load_list(current_results, "no_hole_visits")
                visited_patches = np.unique([int(visit[2]) for visit in visits])
                nb_of_patches = len(np.unique(in_patch_matrix)) - 1
                distance_map_each_patch = [[] for _ in range(nb_of_patches)]
                for i_patch in visited_patches:
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
                for i_exit, current_exit_time in enumerate(exit_frames):
                    current_end_time = end_frames[i_exit]
                    exit_index = fd.load_index(current_traj.to_pandas(), current_folder, current_exit_time)
                    end_index = fd.load_index(current_traj.to_pandas(), current_folder, current_end_time)
                    exit_from = current_traj[exit_index, dt.f.patch_silhouette][0, 0]
                    if exit_from != -1:  # only for the case where the video starts with a transit
                        distance_map_this_patch = distance_map_each_patch[exit_from]
                        # Check if frames are continuous around exit: otherwise, exclude it completely (for now because I'm tired)
                        if end_index - exit_index >= current_end_time - current_exit_time:
                            x_list = current_traj[exit_index:end_index, dt.f.x].to_list()[0]
                            y_list = current_traj[exit_index:end_index, dt.f.y].to_list()[0]
                            xy_list = np.stack((x_list, y_list), axis=1)
                            distance_function: Callable[[Any], int] = lambda xy: distance_map_this_patch[int(xy[1])][int(xy[0])]
                            displacement_list = np.array(list(map(distance_function, xy_list)))
                            # Find the index of where the displacement matches each bin
                            index_each_bin = np.zeros(len(displacement_bin_list))
                            index_each_bin[:] = np.nan
                            i_bin = 0
                            i_point = 0
                            while i_point < len(displacement_list) and i_bin < len(displacement_bin_list):
                                current_bin = displacement_bin_list[i_bin]
                                while i_point < len(displacement_list) and displacement_list[i_point] < current_bin:
                                    i_point += 1
                                # At this point, either the condition was met before i_point reached the end, or
                                # the condition was met exactly at the last point
                                if i_point < len(displacement_list) or displacement_list[-1] >= current_bin :
                                    index_each_bin[i_bin] = i_point
                                i_bin += 1
                            # Handle NaN values: if there is any gap in index_each_bin (NaN values between numeric
                            # values), it means that the worm has jumped two bins at once. Then put fill this bin
                            # with the same time as the subsequent bin
                            # To test for that we go through index_each_bin backwards, so that we can change the
                            # algorithm once the first value has been encountered.
                            first_value_encountered = False
                            for k_bin in range(0, len(index_each_bin), -1):
                                if not first_value_encountered:
                                    if not np.isnan(index_each_bin[i_bin]):
                                        first_value_encountered = True
                                else:
                                    if np.isnan(index_each_bin[i_bin]):
                                        index_each_bin[i_bin] = index_each_bin[i_bin + 1]
                            # Then convert those indices to times post exit and invite them to the party
                            time_function = lambda i: np.round(fd.load_time(current_traj.to_pandas(), current_folder,
                                                                   exit_index + i) - current_exit_time, 2)
                            for j_bin, index in enumerate(index_each_bin):
                                if not np.isnan(index):
                                    current_folder_times[j_bin].append(time_function(index))

                # At this point, displacement_bin_list is filled with one sublist per bin
                # and each sublist contains the worms' time post-exit at that displacement. Now we average for each bin
                for i_bin in range(nb_of_bins):
                    if len(current_folder_times[i_bin]) > 0:
                        avg_time_each_bin_each_plate[i_bin][i_folder] = np.nanmean(current_folder_times[i_bin])
                        nb_of_points_each_bin_each_plate[i_bin][i_folder] = len(current_folder_times[i_bin])
                nb_of_transits_each_plate[i_folder] = len(long_enough_transits)

            # Save it to a csv table
            if not os.path.isdir(analysis_subfolder):
                os.mkdir(analysis_subfolder)
            np.save(analysis_subfolder + "conditions_"+curve_name+"_avg_time_each_bin.npy", avg_time_each_bin_each_plate)
            np.save(analysis_subfolder + "conditions_"+curve_name+"_nb_of_points_each_bin.npy", nb_of_points_each_bin_each_plate)
            np.save(analysis_subfolder + "conditions_"+curve_name+"_nb_of_transits_each_plate.npy", nb_of_transits_each_plate)

        avg_time_each_bin_each_plate = np.load(analysis_subfolder + "conditions_" + curve_name + "_avg_time_each_bin.npy")
        nb_of_points_each_bin_each_plate = np.load(analysis_subfolder + "conditions_" + curve_name + "_nb_of_points_each_bin.npy")
        nb_of_transits_each_plate = np.load(analysis_subfolder + "conditions_" + curve_name + "_nb_of_transits_each_plate.npy")

        # Now that we have the full list of averages for each distance reached after exit, average and bootstrap all that
        avg_each_bin = np.empty(nb_of_bins)
        avg_each_bin[:] = np.nan
        # Errors
        errors_inf = np.empty(nb_of_bins)
        errors_sup = np.empty(nb_of_bins)
        errors_inf[:] = np.nan
        errors_sup[:] = np.nan

        for i_bin in range(nb_of_bins):
            # Rename
            if time_or_probability == "time":
                values_this_distance = avg_time_each_bin_each_plate[i_bin]
            else:
                values_this_distance = np.divide(nb_of_points_each_bin_each_plate[i_bin], nb_of_transits_each_plate)
            # Remove nan values for bootstrapping
            values_this_distance = [values_this_distance[i] for i in range(len(values_this_distance)) if
                                    not np.isnan(values_this_distance[i])]
            if values_this_distance:
                current_avg = np.nanmean(values_this_distance)
                avg_each_bin[i_bin] = current_avg
                bootstrap_ci = ana.bottestrop_ci(values_this_distance, 1000)
                errors_inf[i_bin] = current_avg - bootstrap_ci[0]
                errors_sup[i_bin] = bootstrap_ci[1] - current_avg

        # Plotting lists
        y_list = avg_each_bin[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
        errors_inf = errors_inf[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
        errors_sup = errors_sup[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
        x_list = np.add(np.array(displacement_bin_list), i_curve * 0.02 * np.array(displacement_bin_list))
        x_list = x_list[np.nansum(nb_of_points_each_bin_each_plate, axis=1) > min_nb_data_points]
        x_list *= param.one_pixel_in_mm  # Conversion to mm

        # Plot
        plt.errorbar(x_list, y_list, [errors_inf, errors_sup], fmt='-o', capsize=5,
                     color=param.name_to_color[curve_name], label=curve_name, linewidth=2.3, elinewidth=1.2)

    if is_plot:
        plt.title("Average " + time_or_probability + " to reach each radius around the patches")
        plt.xticks(displacement_bin_list, [str(b) for b in displacement_bin_list])
        if time_or_probability == "time":
            plt.ylabel("Average time post exit (seconds)")
        else:
            plt.ylabel("Probability to reach")

        plt.xlabel("Radius around patch (mm)")
        plt.xscale("log")
        plt.yscale("log")

        # Plot inter_patch distances as vertical lines
        inter_patch_distances_table = pd.read_csv(gen.generate() + "interpatch_distance.csv")
        average_patch_radius_each_cond = pd.read_csv(
            path + "perfect_heatmaps/average_patch_radius_each_condition.csv")
        average_radius = np.mean(average_patch_radius_each_cond["avg_patch_radius"])
        inter_patch_distances_names = inter_patch_distances_table["distance"]
        inter_patch_distances = inter_patch_distances_table["interpatch_distance"] - 2*average_radius*Parameters.parameters.one_pixel_in_mm
        _, _, img_ymin, img_ymax = plt.axis()
        plt.vlines(inter_patch_distances, ymin=img_ymin, ymax=img_ymax,
                   colors=[param.name_to_color[d] for d in inter_patch_distances_names],
                   alpha=0.3,
                   linestyle="--")
        plt.ylim(img_ymin, img_ymax)

        plt.legend()
        plt.show()



path = gen.generate("", test_pipeline=False)
clean_results = pd.read_csv(path + "clean_results.csv")
# clean_trajectories = dt.fread(path + "clean traj/clean_trajectories.csv")
clean_trajectories = dt.fread(path + "clean_trajectories.csv")
print("Finished retrieving tables")

# Parameters
bin_list = [10, 20, 35, 55, 75, 100, 200, 400, 800]

# msd_analysis(clean_results, clean_trajectories, ["close 0"], bin_list, 1, 10, True, "probability")

msd_analysis(clean_results, clean_trajectories, ["0", "0.2", "0.5", "1.25"],
            bin_list, 1, 10, False, time_or_probability="probability", is_plot=True)
msd_analysis(clean_results, clean_trajectories, ["0", "0.2", "0.5", "1.25"],
            bin_list, 1, 10, False, time_or_probability="time", is_plot=True)

msd_analysis(clean_results, clean_trajectories, ["close", "med", "far", "superfar"],
            bin_list, 1, 10, False, time_or_probability="probability", is_plot=True)
msd_analysis(clean_results, clean_trajectories, ["close", "med", "far", "superfar"],
            bin_list, 1, 10, False, time_or_probability="time", is_plot=True)

# Probability to reach radius
msd_analysis(clean_results, clean_trajectories, ["close 0", "med 0", "far 0", "superfar 0"], bin_list, 1, 4, False, "probability", True)
msd_analysis(clean_results, clean_trajectories, ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], bin_list, 1, 4, False, "probability", True)
msd_analysis(clean_results, clean_trajectories, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], bin_list, 1, 4, False, "probability", True)
msd_analysis(clean_results, clean_trajectories, ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], bin_list, 1, 4, False, "probability", True)

msd_analysis(clean_results, clean_trajectories, ["close 0", "close 0.2", "close 0.5", "close 1.25"], bin_list, 1, 4, False, "probability", True)
msd_analysis(clean_results, clean_trajectories, ["med 0", "med 0.2", "med 0.5", "med 1.25"], bin_list, 1, 4, False, "probability", True)
msd_analysis(clean_results, clean_trajectories, ["far 0", "far 0.2", "far 0.5", "far 1.25"], bin_list, 1, 4, False, "probability", True)
msd_analysis(clean_results, clean_trajectories, ["superfar 0", "superfar 0.2", "superfar 0.5", "superfar 1.25"], bin_list, 1, 4, False, "probability", True)

# Time to reach radius
msd_analysis(clean_results, clean_trajectories, ["close 0", "med 0", "far 0", "superfar 0"], bin_list, 1, 4, False, "time", True)
msd_analysis(clean_results, clean_trajectories, ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], bin_list, 1, 4, False, "time", True)
msd_analysis(clean_results, clean_trajectories, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], bin_list, 1, 4, False, "time", True)
msd_analysis(clean_results, clean_trajectories, ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], bin_list, 1, 4, False, "time", True)

msd_analysis(clean_results, clean_trajectories, ["close 0", "close 0.2", "close 0.5", "close 1.25"], bin_list, 1, 4, False, "time", True)
msd_analysis(clean_results, clean_trajectories, ["med 0", "med 0.2", "med 0.5", "med 1.25"], bin_list, 1, 4, False, "time", True)
msd_analysis(clean_results, clean_trajectories, ["far 0", "far 0.2", "far 0.5", "far 1.25"], bin_list, 1, 4, False, "time", True)
msd_analysis(clean_results, clean_trajectories, ["superfar 0", "superfar 0.2", "superfar 0.5", "superfar 1.25"], bin_list, 1, 4, False, "time", True)
