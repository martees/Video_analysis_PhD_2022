
import os
import numpy as np
import matplotlib.pyplot as plt
import datatable as dt
from scipy import ndimage
import time

from main import *
import find_data as fd
from Generating_data_tables import main as gen

# Analysis of the number of visited foodpatches throughout the videos

def evolution_food_patches(curve_list, time_bins, min_length, min_nb_data_points, plot_diff):
    # Init
    tic = time.time()
    nb_of_bins = len(time_bins)
    for i_curve in range(len(curve_list)):
        print("Computing for curve ", curve_list[i_curve], "!")
        curve_name = curve_list[i_curve]
        current_condition_list = param.name_to_nb_list[curve_name]
        print(current_condition_list, curve_name)
        folder_list = fd.return_folders_condition_list(results["folder"], current_condition_list)

        # Init: lists to fill with nb of patches discovered for each plate in each bin
        # (one line per bin, one column per folder)
        nb_each_bin_each_plate = np.zeros((nb_of_bins, len(folder_list)))

        for i_folder, folder in enumerate(folder_list):
            if i_folder % 10 == 0:
                print("Folder ", i_folder, "/", len(folder_list), "!!!")
            current_results = results[results["folder"] == folder]
            visit_list = np.array(fd.load_list(current_results, "no_hole_visits"))
            if len(visit_list) > 0:
                order_patch_visits = visit_list[:, 2]
                visited_patches = pd.unique(order_patch_visits)
                for patch in visited_patches:
                    # Find in the order of patch visits where the patch was first visited
                    # [2, 2, 4, 1] => first visit to 1 is visit of index 3
                    index_first_visit = fd.find_closest(order_patch_visits, patch)
                    # Find when this first visit was made (when it started)
                    first_visit_start = visit_list[index_first_visit, 0]
                    # Find where it is in the bin list
                    discovery_bin_index = np.searchsorted(time_bins, first_visit_start)
                    # Then add one to the number of discovered patches in this bin + all subsequent ones
                    i_bin = discovery_bin_index
                    while i_bin < len(time_bins):
                        nb_each_bin_each_plate[i_bin][i_folder] += 1
                        i_bin += 1

        # At this point, nb_each_bin_each_plate has one value per bin/folder in each cell => average and bootstrap all that
        avg_each_bin = np.empty(nb_of_bins)
        avg_each_bin[:] = np.nan  # just a convenient way of having a table full of NaNs
        # Errors
        errors_inf = np.empty(nb_of_bins)
        errors_sup = np.empty(nb_of_bins)
        errors_inf[:] = np.nan
        errors_sup[:] = np.nan
        for i_bin in range(nb_of_bins):
            # Rename
            values_this_time_bin = nb_each_bin_each_plate[i_bin]
            # and remove nan values for bootstrapping
            values_this_time_bin = [values_this_time_bin[i] for i in range(len(values_this_time_bin)) if
                                    not np.isnan(values_this_time_bin[i])]
            if values_this_time_bin:
                current_avg = np.nanmean(values_this_time_bin)
                avg_each_bin[i_bin] = current_avg
                bootstrap_ci = ana.bottestrop_ci(values_this_time_bin, 1000)
                errors_inf[i_bin] = current_avg - bootstrap_ci[0]
                errors_sup[i_bin] = bootstrap_ci[1] - current_avg

        x_list = np.array(time_bins) + i_curve * 100

        if plot_diff:
            plt.plot(x_list[1:], avg_each_bin[1:] - avg_each_bin[:-1], color=param.name_to_color[curve_name], label=curve_name, linewidth=3)
        else:
            plt.plot(x_list, avg_each_bin, color=param.name_to_color[curve_name], label=curve_name, linewidth=3)
            plt.errorbar(x_list, avg_each_bin, [errors_inf, errors_sup], fmt=param.name_to_color[curve_name], capsize=5)

    if plot_diff:
        plt.title("New discovered food patch at different times in the videos", fontsize=12)
        plt.ylabel("Number of new discovered food patches", fontsize=12)
    else:
        plt.title("Average nb of discovered food patch at different times in the videos", fontsize=12)
        plt.ylabel("Total number of discovered food patches", fontsize=12)

    plt.gcf().set_size_inches(6.2, 6.6)
    plt.xticks(time_bins, [str(np.round(b/3600, 1)) for b in time_bins], rotation=50)  # hour conversion is here
    plt.xlabel("Time in video (hours)", fontsize=12)
    plt.legend()
    plt.show()



path = gen.generate("", test_pipeline=False)
results = pd.read_csv(path + "clean_results.csv")
trajectories = dt.fread(path + "clean_trajectories.csv")


time_bins = [0, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400]
min_length = 1  # minimal transit length to get considered
min_nb_data_points = 10  # minimal number of transits for a point to get plotted

plot_diff = False  # set this to true to plot the y difference between points instead of the points
evolution_food_patches(["close", "med", "far", "superfar"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["0", "0.2", "0.5", "1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0", "med 0", "far 0", "superfar 0"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0", "close 0.2", "close 0.5", "close 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["med 0", "med 0.2", "med 0.5", "med 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["far 0", "far 0.2", "far 0.5", "far 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["superfar 0.2", "superfar 0.5", "superfar 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)

plot_diff = True  # set this to true to plot the y difference between points instead of the points
evolution_food_patches(["close", "med", "far", "superfar"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["0", "0.2", "0.5", "1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0", "med 0", "far 0", "superfar 0"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["close 0", "close 0.2", "close 0.5", "close 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["med 0", "med 0.2", "med 0.5", "med 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["far 0", "far 0.2", "far 0.5", "far 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)
evolution_food_patches(["superfar 0.2", "superfar 0.5", "superfar 1.25"], time_bins, min_length, min_nb_data_points, plot_diff)




