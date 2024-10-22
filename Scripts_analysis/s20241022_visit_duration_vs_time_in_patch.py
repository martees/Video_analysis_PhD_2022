# In this script, we will plot the average duration of visits as a function of the time already spent in the food
# patch when the visit starts.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import find_data as fd
import analysis as ana
from Parameters import parameters as param


def add_visit_vs_time_in_patch_1_folder(visit_list, bin_list, values_each_bin):
    """
    Will return a list of visit durations, binned as a function of the total time already spent in the patch when the
    visit starts, with one bin per value in bin_list (values are the right edge of the bins).
    :param visit_list: a list of visits in the format [t0, t1, idx] where t0 is the start of the visit, t1 the end, and
                       idx the index of the visited food patch.
    :param bin_list: a list of numbers.
    :param values_each_bin: a list of lists, with one list per value in bin_list.
    :return: a list of lists of numbers.
    """
    total_time_each_patch = [0 for _ in range(52)]  # there's never more than 52 patches so just create a big list
    i_bin_each_patch = [0 for _ in range(52)]
    current_bin_each_patch = [bin_list[i] for i in i_bin_each_patch]
    for i_visit, visit in enumerate(visit_list):
        current_patch = visit[2]
        current_visit_duration = visit[1] - visit[0] + 1
        while total_time_each_patch[current_patch] > current_bin_each_patch[current_patch]:
            i_bin_each_patch[current_patch] += 1
            current_bin_each_patch[current_patch] = bin_list[i_bin_each_patch[current_patch]]
        values_each_bin[i_bin_each_patch[current_patch]].append(current_visit_duration)
        total_time_each_patch[current_patch] += current_visit_duration
    return values_each_bin


def plot_visit_duration_vs_time_in_patch(results, curve_list, bin_list):
    full_list_of_folders = results["folder"]
    for i_curve, curve in enumerate(curve_list):
        # Extract the values for all folders
        all_visit_lengths_each_bin = [[] for _ in range(len(bin_list))]
        current_folder_list = fd.return_folders_condition_list(full_list_of_folders, param.name_to_nb_list[curve])
        for i_folder, folder in enumerate(current_folder_list):
            current_results = results[results["folder"] == folder]
            current_visits = fd.load_list(current_results, "no_hole_visits")
            all_visit_lengths_each_bin = add_visit_vs_time_in_patch_1_folder(current_visits, bin_list, all_visit_lengths_each_bin)
        # Compute stats for each bin
        average_visit_length_each_bin = [np.nan for _ in range(len(bin_list))]
        error_inf_visit_length_each_bin = [np.nan for _ in range(len(bin_list))]
        error_sup_visit_length_each_bin = [np.nan for _ in range(len(bin_list))]
        for i_bin in range(len(bin_list)):
            current_values = all_visit_lengths_each_bin[i_bin]
            if len(current_values) > 0:  # if there's any data to work with
                average_visit_length_each_bin[i_bin] = np.mean(current_values)
                [error_inf, error_sup] = ana.bottestrop_ci(current_values, 1000)
                error_inf_visit_length_each_bin[i_bin] = average_visit_length_each_bin[i_bin] - error_inf
                error_sup_visit_length_each_bin[i_bin] = error_sup - average_visit_length_each_bin[i_bin]

        # Then, keep only the bins that do not have NaN averages
        bin_with_values = [bin_list[i] for i in range(len(bin_list)) if
                           not np.isnan(average_visit_length_each_bin[i])]
        error_inf_visit_length_each_bin = [error_inf_visit_length_each_bin[i] for i in
                                           range(len(error_inf_visit_length_each_bin)) if
                                           not np.isnan(average_visit_length_each_bin[i])]
        error_sup_visit_length_each_bin = [error_sup_visit_length_each_bin[i] for i in
                                           range(len(error_sup_visit_length_each_bin)) if
                                           not np.isnan(average_visit_length_each_bin[i])]
        average_visit_length_each_bin = [avg for avg in average_visit_length_each_bin if not np.isnan(avg)]

        # Then, plot it
        plt.plot(bin_with_values, average_visit_length_each_bin,
                 color=param.name_to_color[curve],
                 label=curve, linewidth=4)
        plt.errorbar(bin_with_values, average_visit_length_each_bin,
                     [error_inf_visit_length_each_bin, error_sup_visit_length_each_bin],
                     color=param.name_to_color[curve], capsize=5)
    plt.show()