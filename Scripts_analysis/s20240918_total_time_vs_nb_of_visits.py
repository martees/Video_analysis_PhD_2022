import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.patheffects as pe

from Generating_data_tables import main as gen
import find_data as fd
from Parameters import parameters as param
import analysis as ana
from Parameters import custom_legends
import plots

# Script to return new stuff that Alfonso wants for his model
# He now wants the average total time spent in patch as a function of the number of visits already made
# for the food patch
# In bin x of the plot, have the average total time spent in the patches that have at least x visits



def plot_Tp_vs_Nv(curve_list, is_plot=True, is_save=False):
    results_path = gen.generate("", shorten_traj=False)
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

    # nb_of_visit_bins = list(range(1, 300, 1))  # note: the code bugs if there's a value superior to the last bin
    nb_of_visit_bins = list(np.logspace(0, 2.5, 20))  # note: the code bugs if there's a value superior to the last bin
    nb_of_bins = len(nb_of_visit_bins)
    for i_curve, current_curve_name in enumerate(curve_list):
        print("Condition ", current_curve_name, ", ", i_curve, " / ", len(curve_list))

        # Load the average total time in patch and average number of visits to each patch
        avg_total_time, _, avg_total_time_errors = plots.plot_selected_data(results, "",
                                                                            [param.name_to_nb[current_curve_name]],
                                                                            "total_visit_time",
                                                                            divided_by="nb_of_visited_patches",
                                                                            is_plot=False,
                                                                            show_stats=False,
                                                                            remove_censored_patches=True,
                                                                            hard_cut=False)
        avg_visit_per_patch, _, avg_visit_per_patch_errors = plots.plot_selected_data(results, "",
                                                                            [param.name_to_nb[current_curve_name]],
                                                                            "nb_of_visits",
                                                                            divided_by="nb_of_visited_patches",
                                                                            is_plot=False,
                                                                            show_stats=False,
                                                                            remove_censored_patches=True,
                                                                            hard_cut=False)


        total_time_each_bin_each_plate_this_cond = [[] for _ in range(nb_of_bins)]
        nb_of_points_each_bin_each_plate_this_cond = [[] for _ in range(nb_of_bins)]
        current_curve_nb = param.name_to_nb_list[current_curve_name]
        folder_list = fd.return_folders_condition_list(full_folder_list, current_curve_nb)
        for i_folder, folder in enumerate(folder_list):
            total_time_each_bin_this_plate = [[] for _ in range(nb_of_bins)]
            # Load stuff
            current_data = results[results["folder"] == folder].reset_index()
            visit_list = fd.load_list(current_data, "visits_to_uncensored_patches")
            if len(visit_list) > 0:
                # Just take the highest patch index to create big enough lists (with one item per patch)
                max_visited_patch = int(np.max(np.array(visit_list)[:, 2]) + 1)
                # Initialize lists at zero, and as the for loop runs, the nb of visits / time in each patch will be incremented
                nb_of_visits_each_patch = np.zeros(max_visited_patch)
                total_time_each_patch = np.zeros(max_visited_patch)
                for i_visit, visit in enumerate(visit_list):
                    current_patch = int(visit[2])
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

        # Then, keep only the bins with more than 10 data points
        bin_with_values = [bin_with_values[i] for i in range(len(bin_with_values)) if
                           nb_of_points_each_bin_this_cond[i] > 10]
        error_inf_each_bin_this_cond = [error_inf_each_bin_this_cond[i] for i in range(len(error_inf_each_bin_this_cond)) if
                                        nb_of_points_each_bin_this_cond[i] > 10]
        error_sup_each_bin_this_cond = [error_sup_each_bin_this_cond[i] for i in range(len(error_sup_each_bin_this_cond)) if
                                        nb_of_points_each_bin_this_cond[i] > 10]
        average_each_bin_this_cond = [average_each_bin_this_cond[i] for i in range(len(average_each_bin_this_cond)) if
                                      nb_of_points_each_bin_this_cond[i] > 10]
        nb_of_points_each_bin_this_cond = [nb_of_points_each_bin_this_cond[i] for i in range(len(nb_of_points_each_bin_this_cond)) if
                                           nb_of_points_each_bin_this_cond[i] > 10]

        # Convert to hours
        average_each_bin_this_cond = np.array(average_each_bin_this_cond) / 3600
        error_inf_each_bin_this_cond = np.array(error_inf_each_bin_this_cond) / 3600
        error_sup_each_bin_this_cond = np.array(error_sup_each_bin_this_cond) / 3600

        # Then, plot it
        if is_plot:
            # Curve
            plt.errorbar(bin_with_values, average_each_bin_this_cond, [error_inf_each_bin_this_cond, error_sup_each_bin_this_cond],
                         color=param.name_to_color[current_curve_name], capsize=5, capthick=2,
                         marker=param.distance_to_marker[param.nb_to_distance[param.name_to_nb[current_curve_name]]],
                         markersize=10, linewidth=3)
            # Average total time and nb of visits per patch
            plt.scatter(avg_visit_per_patch, avg_total_time,
                         color=param.name_to_color[current_curve_name], s=92, zorder=4,
                         marker=param.distance_to_marker[param.nb_to_distance[param.name_to_nb[current_curve_name]]],
                         path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])
            plt.errorbar(avg_visit_per_patch, avg_total_time,
                         yerr=avg_total_time_errors,
                         xerr=avg_visit_per_patch_errors,
                         elinewidth=2, capthick=2,
                         color="black", capsize=4, marker="*", markersize=8, zorder=3)

            #for i_bin in range(len(bin_with_values)):
            #    plt.text(bin_with_values[i_bin], average_each_bin_this_cond[i_bin] + 60, str(nb_of_points_each_bin_this_cond[i_bin]), fontsize=24)

            # Then, plot as a dashed line Alfonso's model
            # bin_with_values = np.array(bin_with_values)
            # t_first = t_first_model[current_curve_name]
            # constant = 2.4909
            # equation_values = (t_first / (np.log(1 + constant))) * np.log(1 + constant * bin_with_values)
            # plt.plot(bin_with_values, equation_values, color=param.name_to_color[current_curve_name], linestyle="dashed", linewidth=2)

        # But also add it to the bloody lists for the effing table
        final_table_conditions += [current_curve_name for _ in range(len(average_each_bin_this_cond))]
        final_table_bins += list(bin_with_values)
        final_table_avg += list(average_each_bin_this_cond)
        final_table_error_inf += list(error_inf_each_bin_this_cond)
        final_table_error_sup += list(error_sup_each_bin_this_cond)
        final_table_nb_of_points += nb_of_points_each_bin_this_cond


    if is_save:
        datatable = pd.DataFrame({"condition": final_table_conditions,
                                  "nb_of_visits_bin": final_table_bins,
                                 "nb_of_data_points": final_table_nb_of_points,
                                  "average_total_time": final_table_avg,
                                  "error_inf": final_table_error_inf,
                                  "error_sup": final_table_error_sup})

        datatable.to_csv(results_path + "total_time_vs_nb_of_visits_from_alid.csv")

    else:
        # Custom legend with doodles as labels
        # Empty lines for the legend
        ax = plt.gca()
        lines = []
        for i_cond, cond in enumerate(curve_list):
            line, = ax.plot([], [], color=param.name_to_color[cond], label=cond, linewidth=6,
                             marker=param.distance_to_marker[param.nb_to_distance[param.name_to_nb[cond]]],
                             markersize=14)
            # line, = plt.plot([], [], color=param.name_to_color[param.nb_list_to_name[str(curve)]], linewidth=6,
            #                  marker=param.distance_to_marker[param.nb_to_distance[curve[0]]], markersize=4,
            #                  path_effects=[pe.Stroke(offset=(-0.2, 0.2), linewidth=8,
            #                                          foreground=param.name_to_color[param.nb_to_distance[curve[0]]]),
            #                                pe.Normal()])
            lines.append(line)
        lines = [lines[len(lines) - i] for i in range(1, len(lines) + 1)]  # invert it for nicer order in legend
        curve_list = [curve_list[len(curve_list) - i] for i in
                      range(1, len(curve_list) + 1)]  # invert it for nicer order in legend
        plt.legend(lines, ["" for _ in range(len(lines))],
                   handler_map={lines[i]: custom_legends.HandlerLineImage(
                       "icon_" + param.nb_to_distance[param.name_to_nb[curve_list[i]]] + ".png") for i in
                       range(len(lines))},
                   handlelength=1.6, labelspacing=0.0, fontsize=46, borderpad=0.10, loc=2,
                   handletextpad=0.05, borderaxespad=0.15)

        plt.title("OD = " + param.nb_to_density[param.name_to_nb[curve_list[0]]], fontsize=24)
        plt.xlabel("Number of visits", fontsize=20)
        plt.xscale("log")
        plt.ylabel("Total time in patch (hours)", fontsize=24)
        plt.ylim(0, 2.72)
        plt.show()


def plot_t_first_each_density():
    t_first_each_density = {"0.2": [], "0.5": [], "1.25": []}
    for k, v in t_first_model.items():
        for K in t_first_each_density.keys():
            if K in k:
                t_first_each_density[K].append(v)
    for k, v in t_first_each_density.items():
        plt.plot(range(len(v)), v, color=param.name_to_color[k], linewidth=4, marker="o", markersize=10, label="OD = "+k)

    plt.ylabel("Fitted duration of first visit (hours)", fontsize=16)

    distance_list = ["close", "med", "far", "superfar"]
    # Set the x labels to the distance icons!
    # Stolen from https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings
    for i in range(len(distance_list)):
        ax = plt.gcf().gca()
        ax.set_xticks([])

        # Image to use
        arr_img = plt.imread(os.getcwd().replace("\\", "/")[:-len("Scripts_analysis/")] + "/Parameters/icon_" + distance_list[i] + '.png')

        # Image box to draw it!
        imagebox = OffsetImage(arr_img, zoom=0.6)
        imagebox.image.axes = ax

        x_annotation_box = AnnotationBbox(imagebox, (i, 0),
                                          xybox=(0, -8),
                                          # that's the shift that the image will have compared to (i, 0)
                                          xycoords=("data", "axes fraction"),
                                          boxcoords="offset points",
                                          box_alignment=(.5, 1),
                                          bboxprops={"edgecolor": "none"})

        ax.add_artist(x_annotation_box)

    plt.legend(fontsize=12)
    plt.show()


t_first_model = {"close 0.2": 0.1099, "med 0.2": 0.2021, "far 0.2": 0.2869, "superfar 0.2": 0.2477,
                 "close 0.5": 0.2465, "med 0.5": 0.3513, "far 0.5": 0.4433, "superfar 0.5": 0.4627,
                 "close 1.25": 0.3123, "med 1.25": 0.5296, "far 1.25": 0.5504, "superfar 1.25": 0.5635}

#list_of_curves = param.name_to_nb_list.keys()
list_of_curves = ["close 0", "med 0", "far 0", "superfar 0",
                  "close 0.2", "med 0.2", "far 0.2", "superfar 0.2",
                  "close 0.5", "med 0.5", "far 0.5", "superfar 0.5",
                  "close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]
#plot_Tp_vs_Nv(list_of_curves, False, True)

plot_Tp_vs_Nv(["close 0", "med 0", "far 0", "superfar 0"], True)
plot_Tp_vs_Nv(["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], True)
# plot_Tp_vs_Nv(["close 0.5"], True)
plot_Tp_vs_Nv(["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], True)
plot_Tp_vs_Nv(["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], True)
# plot_t_first_each_density()







