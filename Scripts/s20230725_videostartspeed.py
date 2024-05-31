# Analysis of speed in the first N time steps of our videos
from main import *
import numpy as np
import matplotlib.pyplot as plt
import warnings


def speed_dynamic(result_table, trajectory_table, condition_list, min_visit_start, max_visit_start, nb_of_bins, speed_inside=True, speed_outside=True):
    # Find conditions and folders
    all_condition_names = [param.nb_to_name[nb] for nb in condition_list]
    full_folder_list = np.unique(result_table["folder"])

    # Time bin list (to plot dynamic, one point per bin will be plotted)
    time_bins = [int(min_visit_start + i * (max_visit_start - min_visit_start)/nb_of_bins) for i in range(nb_of_bins + 1)]

    # These dictionaries will have one key per density in each condition, and as a value the time series of avg/errors
    avg_speed_each_bin = {}
    errors_inf_each_time = {}
    errors_sup_each_time = {}

    for i_time in range(len(time_bins) - 1):
        # Note: it would probably be much better to have the time loop inside a folder loop, but... this was historically easier
        print("computing for time ", i_time, "/", nb_of_bins)

        # Dictionary with condition name as key, and list of results as values.
        avg_speed_all_plates = {all_condition_names[i]: [] for i in range(len(all_condition_names))}

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
                # Load tables and patch info
                current_folder = current_condition_folders[i_folder]
                current_traj = trajectory_table[trajectory_table["folder"] == current_folder]
                # Load list of speeds from trajectory_table
                current_speeds = current_traj["speeds"]
                bin_start = fd.find_closest(current_traj["frame"], time_bins[i_time])
                bin_end = fd.find_closest(current_traj["frame"], time_bins[i_time + 1])
                if speed_inside and not speed_outside:
                    current_speeds = current_speeds[current_traj["patch_silhouette"] != -1]
                if speed_outside and not speed_inside:
                    current_speeds = current_speeds[current_traj["patch_silhouette"] == -1]
                current_folder_avg_speed = np.mean(current_speeds.iloc[bin_start:bin_end])
                # At this point current_folder_duration_list should have the visit durations of all visits for current plate
                # Fill list of plate averages
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_speed_all_plates[current_condition_name].append(np.nanmean(current_folder_avg_speed))

        # Go through the result dictionaries from this time step, and add them to dictionaries with one list per condition, one item per time step
        for condition in all_condition_names:
            # For first loop, initialize the dictionaries with lists with one element per number of time bins
            if condition not in avg_speed_each_bin.keys():
                avg_speed_each_bin[condition] = np.zeros(nb_of_bins)
                errors_inf_each_time[condition] = np.zeros(nb_of_bins)
                errors_sup_each_time[condition] = np.zeros(nb_of_bins)
            # Fill 'em up
            list_of_values = avg_speed_all_plates[condition]
            list_of_values = [list_of_values[i] for i in range(len(list_of_values)) if not np.isnan(list_of_values[i])]
            # I expect to see RuntimeWarnings in this block
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                current_avg = np.nanmean(list_of_values)
            avg_speed_each_bin[condition][i_time] = current_avg
            bootstrap_ci = ana.bottestrop_ci(list_of_values, 1000)
            errors_inf_each_time[condition][i_time] = current_avg - bootstrap_ci[0]
            errors_sup_each_time[condition][i_time] = bootstrap_ci[1] - current_avg

    fig = plt.gcf()
    plt.title("Evolution of speed from time "+str(min_visit_start)+" to time "+str(max_visit_start)+", inside="+str(speed_inside)+", outside="+str(speed_outside))
    fig.set_size_inches(9, 6)

    if nb_of_bins > 1:
        # Plot condition averages, one curve per possible density
        for condition in all_condition_names:
            plt.plot(time_bins[:-1], avg_speed_each_bin[condition], color=param.name_to_color[condition.split(" ")[0]], label=condition, linewidth=3)
            # Plot error bars
            plt.errorbar(time_bins[:-1], avg_speed_each_bin[condition], [errors_inf_each_time[condition], errors_sup_each_time[condition]], fmt='.k', capsize=4)
        plt.legend()

    else:
        # If only one bin, make a bar plot
        plt.bar(range(len(all_condition_names)), [avg_speed_each_bin[cond][0] for cond in all_condition_names], color=[param.name_to_color[name] for name in all_condition_names])
        # Plot error bars
        plt.errorbar(range(len(all_condition_names)), [avg_speed_each_bin[cond][0] for cond in all_condition_names],
                     [[errors_inf_each_time[cond][0] for cond in all_condition_names],
                      [errors_sup_each_time[cond][0] for cond in all_condition_names]], fmt='.k', capsize=5)
        ax = fig.gca()
        ax.set_xticks(range(len(all_condition_names)))
        ax.set_xticklabels(all_condition_names, rotation=60)

    plt.show()


clean_results = pd.read_csv(path + "/clean_results.csv")
clean_trajectories = pd.read_csv(path + "/clean_trajectories.csv")

speed_dynamic(clean_results, clean_trajectories, param.name_to_nb_list["all"], 0, 10000, 1, speed_inside=True, speed_outside=False)
speed_dynamic(clean_results, clean_trajectories, param.name_to_nb_list["all"], 0, 10000, 1, speed_inside=False, speed_outside=True)
speed_dynamic(clean_results, clean_trajectories, param.name_to_nb_list["all"], 10000, 40000, 1, speed_inside=True, speed_outside=False)
speed_dynamic(clean_results, clean_trajectories, param.name_to_nb_list["all"], 10000, 40000, 1, speed_inside=False, speed_outside=True)

