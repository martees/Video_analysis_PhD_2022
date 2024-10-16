import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Generating_data_tables import main as gen
import find_data as fd
from Parameters import parameters as param
import analysis as ana

# Script to return new stuff that Alfonso wants for his model
# He now wants the average total time spent in patch as a function of the number of visits already made for the food patch
# In bin x of the plot, have the average total time spent in the patches that have at least x visits

results_path = gen.generate("", shorten_traj=True)
results = pd.read_csv(results_path + "clean_results.csv")
full_folder_list = results["folder"]

# Columns in the table to return in the end: one line per point in the graphs. For each line:
# condition, right edge of the visit nb bin, and average + error bars
final_table_conditions = []
final_table_bins = []
final_table_avg = []
final_table_error_sup = []
final_table_error_inf = []
final_table_nb_of_points = []

# curve_list = param.name_to_nb_list.keys()
curve_list = ["close 0", "med 0", "far 0", "superfar 0",
              "close 0.2", "med 0.2", "far 0.2", "superfar 0.2",
              "close 0.5", "med 0.5", "far 0.5", "superfar 0.5",
              "close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]
nb_of_visit_bins = list(range(1, 300, 1))  # note: the code bug if there's a value superior to the last bin
nb_of_bins = len(nb_of_visit_bins)
for i_curve, current_curve_name in enumerate(curve_list):
    print("Condition ", current_curve_name, ", ", i_curve, " / ", len(curve_list))
    total_time_each_bin_each_plate_this_cond = [[] for _ in range(nb_of_bins)]
    nb_of_points_each_bin_each_plate_this_cond = [[] for _ in range(nb_of_bins)]
    current_curve_nb = param.name_to_nb_list[current_curve_name]
    folder_list = fd.return_folders_condition_list(full_folder_list, current_curve_nb)
    for i_folder, folder in enumerate(folder_list):
        total_time_each_bin_this_plate = [[] for _ in range(nb_of_bins)]
        # Load stuff
        current_data = results[results["folder"] == folder].reset_index()
        visit_list = fd.load_list(current_data, "no_hole_visits")
        if len(visit_list) > 0:
            # Just take the highest patch index to create big enough lists (with one item per patch)
            max_visited_patch = np.max(np.array(visit_list)[:, 2]) + 1
            # Initialize lists at zero, and as the for loop runs, the nb of visits / time in each patch will be incremented
            nb_of_visits_each_patch = np.zeros(max_visited_patch)
            total_time_each_patch = np.zeros(max_visited_patch)
            for i_visit, visit in enumerate(visit_list):
                current_patch = visit[2]
                nb_of_visits_each_patch[current_patch] += 1
                total_time_each_patch[current_patch] += visit[1] - visit[0] + 1
                # Then, add this total time in the right bin. Only add it when there's a change in bins, or if this is
                # the first bin of the bin list
                current_bin_index = np.searchsorted(nb_of_visit_bins, nb_of_visits_each_patch[current_patch])
                previous_bin_index = np.searchsorted(nb_of_visit_bins, nb_of_visits_each_patch[current_patch] - 1)
                if current_bin_index != previous_bin_index or current_bin_index == np.argmin(nb_of_visit_bins):
                    total_time_each_bin_this_plate[current_bin_index].append(total_time_each_patch[current_patch])
        # Add this folder's average values to the condition data table
        for i_bin in range(nb_of_bins):
            if len(total_time_each_bin_this_plate[i_bin]) > 0:
                total_time_each_bin_each_plate_this_cond[i_bin].append(np.mean(total_time_each_bin_this_plate[i_bin]))
                nb_of_points_each_bin_each_plate_this_cond[i_bin].append(len(total_time_each_bin_this_plate[i_bin]))
    # Then, average and bootstrap for this condition + count the number of patches for each bin
    average_each_bin_this_cond = [np.nan for _ in range(nb_of_bins)]
    error_inf_each_bin_this_cond = [np.nan for _ in range(nb_of_bins)]
    error_sup_each_bin_this_cond = [np.nan for _ in range(nb_of_bins)]
    nb_of_points_each_bin_this_cond = [np.nan for _ in range(nb_of_bins)]
    for i_bin in range(nb_of_bins):
        if len(total_time_each_bin_each_plate_this_cond[i_bin]) > 0:  # if there's any data to work with
            current_bin_values = total_time_each_bin_each_plate_this_cond[i_bin]
            average_each_bin_this_cond[i_bin] = np.mean(current_bin_values)
            [error_inf, error_sup] = ana.bottestrop_ci(current_bin_values, 1000)
            error_inf_each_bin_this_cond[i_bin] = average_each_bin_this_cond[i_bin] - error_inf
            error_sup_each_bin_this_cond[i_bin] = error_sup - average_each_bin_this_cond[i_bin]
            nb_of_points_each_bin_this_cond[i_bin] = np.sum(nb_of_points_each_bin_each_plate_this_cond[i_bin])

    # Then, keep only the bins that do not have NaN averages
    bin_with_values = [nb_of_visit_bins[i] for i in range(len(nb_of_visit_bins)) if
                       not np.isnan(average_each_bin_this_cond[i])]
    error_inf_each_bin_this_cond = [error_inf_each_bin_this_cond[i] for i in range(len(error_inf_each_bin_this_cond)) if
                                    not np.isnan(average_each_bin_this_cond[i])]
    error_sup_each_bin_this_cond = [error_sup_each_bin_this_cond[i] for i in range(len(error_sup_each_bin_this_cond)) if
                                    not np.isnan(average_each_bin_this_cond[i])]
    nb_of_points_each_bin_this_cond = [nb_of_points_each_bin_this_cond[i] for i in range(len(nb_of_points_each_bin_this_cond)) if
                                       not np.isnan(average_each_bin_this_cond[i])]
    average_each_bin_this_cond = [avg for avg in average_each_bin_this_cond if not np.isnan(avg)]
    # Then, plot it
    plt.plot(bin_with_values, average_each_bin_this_cond,
             color=param.name_to_color[current_curve_name],
             label=current_curve_name, linewidth=4)
    plt.errorbar(bin_with_values, average_each_bin_this_cond,
                 [error_inf_each_bin_this_cond, error_sup_each_bin_this_cond],
                 fmt='.k', capsize=5)
    for i_bin in range(len(bin_with_values)):
        plt.text(bin_with_values[i_bin], average_each_bin_this_cond[i_bin] + 60, str(nb_of_points_each_bin_this_cond[i_bin]), fontsize=24)
    # But also add it to the bloody lists for the effing table
    final_table_conditions += [current_curve_name for _ in range(len(average_each_bin_this_cond))]
    final_table_bins += bin_with_values
    final_table_avg += average_each_bin_this_cond
    final_table_error_inf += error_inf_each_bin_this_cond
    final_table_error_sup += error_sup_each_bin_this_cond
    final_table_nb_of_points += nb_of_points_each_bin_this_cond

plt.xlabel("Nb of visits already made to the patch")
plt.xscale("log")
plt.ylabel("Average total time spent at that point (s)")
plt.legend()
plt.show()

datatable = pd.DataFrame({"condition": final_table_conditions,
                          "nb_of_visits_bin": final_table_bins,
                         "nb_of_data_points": final_table_nb_of_points,
                          "average_total_time": final_table_avg,
                          "error_inf": final_table_error_inf,
                          "error_sup": final_table_error_sup})

datatable.to_csv(results_path + "total_time_vs_nb_of_visits_from_alid.csv")


