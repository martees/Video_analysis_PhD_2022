# Analysis of visit duration dynamics (look at visit duration but by looking only at visits that start/end
# at specific times of the experiments)

from main import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
import find_data as fd


def visit_duration_dynamic(result_table, condition_list, min_visit_start, max_visit_start, nb_of_bins):
    # Find conditions and folders
    all_condition_names = [param.nb_to_name[nb] for nb in condition_list]
    full_folder_list = np.unique(result_table["folder"])

    # Time bin list (to plot dynamic, one point per bin will be plotted)
    time_bins = [int(min_visit_start + i * (max_visit_start - min_visit_start)/nb_of_bins) for i in range(nb_of_bins + 1)]

    # These dictionaries will have one key per density in each condition, and as a value the time series of avg/errors
    avg_visit_each_time = {}
    errors_inf_each_time = {}
    errors_sup_each_time = {}

    for i_time in range(len(time_bins) - 1):
        # Note: it would probably be much better to have the time loop inside a folder loop, but... this was historically easier
        print("computing for time ", i_time, "/", nb_of_bins)

        # Dictionary with condition name as key, and list of results as values.
        avg_visit_duration_all_plates = {all_condition_names[i]: [] for i in range(len(all_condition_names))}

        # Go through all conditions and fill dictionaries with average visit duration for each plate
        for i_cond in range(len(condition_list)):
            # Load folders
            current_condition = condition_list[i_cond]
            current_condition_name = param.nb_to_name[current_condition]
            current_condition_folders = fd.return_folders_condition_list(full_folder_list,
                                                                         current_condition)  # folder list for that condition

            # Create lists to be filled, that will have one sublist for each patch density
            # Should produce two sublists for mixed conditions and one sublist for pure ones, without need for an if
            current_densities = np.sort(np.unique(param.nb_to_density[current_condition].split("+")))  # lower density first
            for i_folder in range(len(current_condition_folders)):
                # List of values for this folder
                current_folder_duration_list = []
                # Load tables and patch info
                current_folder = current_condition_folders[i_folder]
                current_results = result_table[result_table["folder"] == current_folder]
                # Load lists from the tables
                list_of_visits = fd.load_list(current_results, "no_hole_visits")
                list_of_visit_durations = ana.convert_to_durations(list_of_visits)
                for i_visit in range(len(list_of_visits)):
                    if time_bins[i_time] <= list_of_visits[i_visit][0] < time_bins[i_time + 1]:
                        current_visit_duration = list_of_visit_durations[i_visit]
                        current_folder_duration_list.append(current_visit_duration)
                # At this point current_folder_duration_list should have the visit durations of all visits for current plate
                # Fill list of plate averages
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_visit_duration_all_plates[current_condition_name].append(np.nanmean(current_folder_duration_list))

        # Go through the result dictionaries from this time step, and add them to dictionaries with one list per condition, one item per time step
        for condition in all_condition_names:
            # For first loop, initialize the dictionaries with lists with one element per number of time bins
            if condition not in avg_visit_each_time.keys():
                avg_visit_each_time[condition] = np.zeros(nb_of_bins)
                errors_inf_each_time[condition] = np.zeros(nb_of_bins)
                errors_sup_each_time[condition] = np.zeros(nb_of_bins)
            # Fill 'em up
            list_of_values = avg_visit_duration_all_plates[condition]
            list_of_values = [list_of_values[i] for i in range(len(list_of_values)) if not np.isnan(list_of_values[i])]
            # I expect to see RuntimeWarnings in this block
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                current_avg = np.nanmean(list_of_values)
            avg_visit_each_time[condition][i_time] = current_avg
            bootstrap_ci = ana.bottestrop_ci(list_of_values, 1000)
            errors_inf_each_time[condition][i_time] = current_avg - bootstrap_ci[0]
            errors_sup_each_time[condition][i_time] = bootstrap_ci[1] - current_avg

    fig = plt.gcf()
    plt.title("Evolution of visit duration in all densities, visits starting between "+str(min_visit_start)+" & "+str(max_visit_start))
    fig.set_size_inches(len(condition_list)+0.5, 5)

    if nb_of_bins > 1:
        # Plot condition averages, one curve per possible density
        for condition in all_condition_names:
            plt.plot(time_bins[:-1], avg_visit_each_time[condition], color=param.name_to_color[condition.split(" ")[1]], label=condition)
            # Plot error bars
            plt.errorbar(time_bins[:-1], avg_visit_each_time[condition], [errors_inf_each_time[condition], errors_sup_each_time[condition]], fmt='.k', capsize=5)
        plt.legend()

    else:
        # If only one bin, make a bar plot
        # For color, the name.split part is to have density override distance for the color
        plt.bar(range(len(all_condition_names)), [avg_visit_each_time[cond][0] for cond in all_condition_names], color=[param.name_to_color[name.split(" ")[1]] for name in all_condition_names])
        # Plot error bars
        plt.errorbar(range(len(all_condition_names)), [avg_visit_each_time[cond][0] for cond in all_condition_names],
                     [[errors_inf_each_time[cond][0] for cond in all_condition_names],
                      [errors_sup_each_time[cond][0] for cond in all_condition_names]], fmt='.k', capsize=5)

        for i in range(len(condition_list)):
            plt.scatter([range(len(condition_list))[i] for _ in range(len(avg_visit_duration_all_plates[all_condition_names[i]]))],
                       avg_visit_duration_all_plates[all_condition_names[i]], color="red", zorder=2)

        ax = fig.gca()
        ax.set_xticks(range(len(all_condition_names)))
        ax.set_xticklabels(all_condition_names, rotation=60)

    plt.show()


results = pd.read_csv(path + "/clean_results.csv")

visit_duration_dynamic(results, [12, 13, 14], 0, 10000, 1)
visit_duration_dynamic(results, [0, 1, 2], 0, 10000, 1)
visit_duration_dynamic(results, [4, 5, 6], 0, 10000, 1)
visit_duration_dynamic(results, [12, 13, 14], 10000, 40000, 1)
visit_duration_dynamic(results, [0, 1, 2], 10000, 40000, 1)
visit_duration_dynamic(results, [4, 5, 6], 10000, 40000, 1)

