# In this script, we will plot the average duration of visits as a function of the time already spent in the food
# patch when the visit starts.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Generating_data_tables import main as gen
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
    current_bin_each_patch = [bin_list[0] for _ in range(52)]
    for i_visit, visit in enumerate(visit_list):
        current_patch = int(visit[2])
        if visit[1] < visit[0]:
            print("NEGATIVE VISIT AAAAAAA")
        current_visit_duration = visit[1] - visit[0]
        while total_time_each_patch[current_patch] > current_bin_each_patch[current_patch] and i_bin_each_patch[
            current_patch] < len(bin_list) - 1:
            i_bin_each_patch[current_patch] += 1
            current_bin_each_patch[current_patch] = bin_list[i_bin_each_patch[current_patch]]
        values_each_bin[i_bin_each_patch[current_patch]].append(current_visit_duration)
        total_time_each_patch[current_patch] += current_visit_duration
    return values_each_bin


def plot_visit_duration_vs_time_in_patch(results, curve_list, bin_list, min_nb_each_bin, only_show_density=False):
    full_list_of_folders = results["folder"]
    for i_curve, curve in enumerate(curve_list):
        # Extract the values for all folders
        all_visit_lengths_each_bin = [[] for _ in range(len(bin_list))]
        current_folder_list = fd.return_folders_condition_list(full_list_of_folders, param.name_to_nb_list[curve])
        for i_folder, folder in enumerate(current_folder_list):
            current_results = results[results["folder"] == folder]
            current_visits = fd.load_list(current_results, "no_hole_visits")
            all_visit_lengths_each_bin = add_visit_vs_time_in_patch_1_folder(current_visits, bin_list,
                                                                             all_visit_lengths_each_bin)
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

        # Then, keep only the bins that have enough worms
        bin_with_values = [bin_with_values[i] for i in range(len(bin_with_values)) if
                           len(all_visit_lengths_each_bin[i]) > min_nb_each_bin]
        error_inf_visit_length_each_bin = [error_inf_visit_length_each_bin[i] for i in
                                           range(len(error_inf_visit_length_each_bin)) if
                                           len(all_visit_lengths_each_bin[i]) > min_nb_each_bin]
        error_sup_visit_length_each_bin = [error_sup_visit_length_each_bin[i] for i in
                                           range(len(error_sup_visit_length_each_bin)) if
                                           len(all_visit_lengths_each_bin[i]) > min_nb_each_bin]
        average_visit_length_each_bin = [average_visit_length_each_bin[i] for i in
                                         range(len(average_visit_length_each_bin)) if
                                         len(all_visit_lengths_each_bin[i]) > min_nb_each_bin]

        # Then, plot it
        density = param.nb_to_density[param.name_to_nb[curve]]
        color_of_density = param.name_to_color[density]
        color_of_condition = param.name_to_color[curve]
        if only_show_density:
            color = color_of_density
            label = "OD = "+density
        else:
            color = color_of_condition
            label = curve

        # Set the bins to be in the middle instead of right edge of bin
        bin_with_values = [0] + bin_with_values
        bin_with_values = [(bin_with_values[i-1]+bin_with_values[i])/2 for i in range(1, len(bin_with_values))]

        plt.plot(np.array(bin_with_values)/3600, np.array(average_visit_length_each_bin)/60,
                 color=color, label=label, linewidth=4)
        plt.errorbar(np.array(bin_with_values)/3600, np.array(average_visit_length_each_bin)/60,
                     [np.array(error_inf_visit_length_each_bin)/60, np.array(error_sup_visit_length_each_bin)/60],
                     color=color, capsize=5)

    plt.yscale("log")
    # plt.xscale("log")
    plt.title(str(curve_list), fontsize=20)
    #plt.xscale("log")
    plt.ylabel("Average visit duration (minutes)", fontsize=16)
    plt.xlabel("Time spent in patch before this visit (hours)", fontsize=16)
    plt.legend(fontsize=14)
    plt.show()


results_path = gen.generate("", shorten_traj=False)
clean_results = pd.read_csv(results_path + "clean_results.csv")
bins = [100, 400, 1000, 1900, 3100, 4600, 6400, 8500, 10900, 13600]

# plot_visit_duration_vs_time_in_patch(clean_results, ["med 0", "med 0.2", "med 0.5", "med 1.25"], bins, 20, only_show_density=True)

plot_visit_duration_vs_time_in_patch(clean_results, ["close 0", "close 0.2", "close 0.5", "close 1.25"], bins, 20)
plot_visit_duration_vs_time_in_patch(clean_results, ["med 0", "med 0.2", "med 0.5", "med 1.25"], bins, 20)
plot_visit_duration_vs_time_in_patch(clean_results, ["far 0", "far 0.2", "far 0.5", "far 1.25"], bins, 20)
plot_visit_duration_vs_time_in_patch(clean_results, ["superfar 0", "superfar 0.2", "superfar 0.5", "superfar 1.25"], bins, 20)

plot_visit_duration_vs_time_in_patch(clean_results, ["close 0", "med 0", "far 0", "superfar 0"], bins, 20)
plot_visit_duration_vs_time_in_patch(clean_results, ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], bins, 20)
plot_visit_duration_vs_time_in_patch(clean_results, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], bins, 20)
plot_visit_duration_vs_time_in_patch(clean_results, ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], bins, 20)
