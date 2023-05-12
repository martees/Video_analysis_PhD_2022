# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from scipy.stats import bootstrap
import random

# My code
import generate_results as gr
import find_data as fd
from param import *
import json


def results_per_condition(result_table, column_name, divided_by=""):
    """
    Function that takes our result table and a column name (as a string)
    Returns the list of values of that column pooled by condition, a list of the average value for each condition, and a
    bootstrap confidence interval for each value.
    Can take in a third argument, column name by which you want to divide the main column, plate by plate
    eg: divide duration sum by nb of visits for each plate to get average visit duration for each plate
    """

    # Initializing a list
    list_of_conditions = np.unique(result_table["condition"])
    list_of_plates = np.unique(result_table["folder"])

    # Full list
    full_list_of_values = [list(i) for i in np.zeros((len(list_of_conditions), 1), dtype='int')]

    # List of average
    list_of_avg_values = np.zeros(len(list_of_conditions))

    # Initializing errors
    errors_inf = np.zeros(len(list_of_conditions))
    errors_sup = np.zeros(len(list_of_conditions))

    for i_condition in range(len(list_of_conditions)):
        # Extracting and slicing
        current_condition = list_of_conditions[i_condition]
        current_data = result_table[result_table["condition"] == current_condition]
        list_of_plates = np.unique(current_data["folder"])

        # Compute average for each plate of the current condition, save it in a list
        list_of_values = np.zeros(len(list_of_plates))

        for i_plate in range(len(list_of_plates)):
            # Take only one plate
            current_plate = current_data[current_data["folder"] == list_of_plates[i_plate]]
            if divided_by != "":  # In this case, we want to divide column name by another one
                if np.sum(current_plate[divided_by]) != 0:  # Non zero check for division
                    list_of_values[i_plate] = np.sum(current_plate[column_name]) / np.sum(current_plate[divided_by])
                else:
                    print("Trying to divide by 0... what a shame")
                # if divided_by == "nb_of_visits" and column_name == "total_visit_time" and current_condition == 2: #detecting extreme far 0.2 cases
                #    if list_of_values[i_plate]>800:
                #        print(list_of_plates[i_plate])
                #        print(list_of_values[i_plate])
            else:  # No division has to be made
                if column_name == "average_speed_inside" or column_name == "average_speed_outside":
                    # Exclude the 0's which are the cases were the worm didnt go to a patch / out of a patch for a full track
                    list_speed_current_plate = [nonzero for nonzero in current_plate[column_name] if int(nonzero) != 0]
                    if list_speed_current_plate:  # If any non-zero speed was recorded for that plate
                        list_of_values[i_plate] = np.average(list_speed_current_plate)
                elif column_name == "proportion_of_visited_patches" or column_name == "nb_of_visited_patches":  # Special case: divide by total nb of patches in plate
                    current_plate = current_plate.reset_index()
                    list_of_visited_patches = [json.loads(current_plate["list_of_visited_patches"][i]) for i in
                                               range(len(current_plate["list_of_visited_patches"]))]
                    list_of_visited_patches = [i for liste in list_of_visited_patches for i in liste]
                    if column_name == "nb_of_visited_patches":
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches))
                    else:
                        list_total_patch = [52, 24, 7, 25, 52, 24, 7, 25, 24, 24, 24, 24]
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches)) \
                                                  / list_total_patch[i_condition]
                elif column_name == "furthest_patch_distance":  # in this case we want the maximal value and not the average
                    list_of_values[i_plate] = np.max(current_plate[column_name])
                else:  # in any other case
                    list_of_values[i_plate] = np.sum(current_plate[column_name])

        # In the case of speed, 0 values are for plates where there was no speed inside/outside recorded so we remove their values
        # (idk if this case happens but at least it's taken care of)
        if column_name == "average_speed_inside" or column_name == "average_speed_outside":
            list_of_values = [nonzero for nonzero in list_of_values if int(nonzero) != 0]

        # Keep in memory the full list of averages
        full_list_of_values[i_condition] = list_of_values

        # Average for the current condition
        list_of_avg_values[i_condition] = np.mean(list_of_values)

        # Bootstrapping on the plate avg duration
        bootstrap_ci = bottestrop_ci(list_of_values, 1000)
        errors_inf[i_condition] = list_of_avg_values[i_condition] - bootstrap_ci[0]
        errors_sup[i_condition] = bootstrap_ci[1] - list_of_avg_values[i_condition]

    return list_of_conditions, full_list_of_values, list_of_avg_values, [list(errors_inf), list(errors_sup)]


def bottestrop_ci(data, nb_resample):
    """
    Function that takes a dataset and returns a confidence interval using nb_resample samples for bootstrapping
    """
    bootstrapped_means = []
    # data = [x for x in data if str(x) != 'nan']
    for i in range(nb_resample):
        y = []
        for k in range(len(data)):
            y.append(random.choice(data))
        avg = np.mean(y)
        bootstrapped_means.append(avg)
    bootstrapped_means.sort()
    return [np.percentile(bootstrapped_means, 5), np.percentile(bootstrapped_means, 95)]


def plot_traj(traj, i_condition, n_max=4, is_plot_patches=False, show_composite=True, plot_in_patch=False,
              plot_continuity=False, plot_speed=False, plot_time=False, plate_list=[]):
    """
    Function that takes in our dataframe format, using columns: "x", "y", "id_conservative", "folder"
    and extracting "condition" info in metadata
    Extracts list of series of positions from indicated condition and draws them, with one color per id
    :param traj: dataframe containing the series of (x,y) positions ([[x0,x1,x2...] [y0,y1,y2...])
    :return: trajectory plot
    """
    if plate_list:
        worm_list = []
        for i_plate in range(len(plate_list)):
            worm_list.append(traj[traj["folder"] == plate_list[i_plate]]["id_conservative"])
        worm_list = np.unique(worm_list)
    else:
        worm_list = np.unique(traj["id_conservative"])
    nb_of_worms = len(worm_list)
    colors = plt.cm.jet(np.linspace(0, 1, nb_of_worms))
    previous_folder = 0
    n_plate = 1
    for i_worm in range(nb_of_worms):
        current_worm = worm_list[i_worm]
        current_traj = traj[traj["id_conservative"] == current_worm]
        current_list_x = current_traj.reset_index()["x"]
        current_list_y = current_traj.reset_index()["y"]
        current_folder = list(current_traj["folder"])[0]
        metadata = fd.folder_to_metadata(current_folder)
        current_condition = metadata["condition"][0]
        plt.suptitle("Trajectories for condition " + str(i_condition))
        if plate_list or current_condition == i_condition:
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                if n_plate > n_max:
                    plt.show()
                    n_plate = 1
                if len(plate_list) != 1:
                    plt.subplot(n_max // 2, n_max // 2, n_plate)
                    n_plate += 1
                # Show background and patches
                fig = plt.gcf()
                ax = fig.gca()
                fig.set_tight_layout(True)  # make the margins tighter
                if show_composite:  # show composite with real patches
                    composite = plt.imread(current_folder[:-len("traj.csv")] + "composite_patches.tif")
                    ax.imshow(composite)
                else:  # show cleaner background without the patches
                    background = plt.imread(current_folder[:-len("traj.csv")] + "background.tif")
                    ax.imshow(background, cmap='gray')
                ax.set_title(str(current_folder[-48:-9]))
                # Plot them patches
                if is_plot_patches:
                    patch_densities = metadata["patch_densities"]
                    patch_centers = metadata["patch_centers"]
                    x_list, y_list = plot_patches([current_folder], show_composite=False, is_plot=False)
                    for i_patch in range(len(patch_centers)):
                        ax.plot(x_list[i_patch], y_list[i_patch], color='yellow', alpha=patch_densities[i_patch])
                        ax.annotate(str(i_patch), xy=(patch_centers[i_patch][0] + 80, patch_centers[i_patch][1] + 80),
                                    color='white')

            # Plot worm trajectory
            # Plot the trajectory with a colormap based on the speed of the worm
            if plot_speed:
                distance_list = current_traj.reset_index()["distances"]
                normalize = mplcolors.Normalize(vmin=0, vmax=3.5)
                plt.scatter(current_list_x, current_list_y, c=distance_list, cmap="hot", norm=normalize, s=1)
                if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                    plt.colorbar()

            # Plot the trajectory with a colormap based on time
            if plot_time:
                nb_of_timepoints = len(current_list_x)
                bin_size = 100
                # colors = plt.cm.jet(np.linspace(0, 1, nb_of_timepoints//bin_size))
                for bin in range(nb_of_timepoints // bin_size):
                    lower_bound = bin * bin_size
                    upper_bound = min((bin + 1) * bin_size, len(current_list_x))
                    plt.scatter(current_list_x[lower_bound:upper_bound], current_list_y[lower_bound:upper_bound],
                                c=range(lower_bound, upper_bound), cmap="hot", s=0.5)
                if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                    plt.colorbar()

            # Plot black dots when the worm is inside
            if plot_in_patch:
                plt.scatter(current_list_x, current_list_y, color=colors[i_worm], s=.5)
                indexes_in_patch = np.where(current_traj["patch"] != -1)
                plt.scatter(current_list_x.iloc[indexes_in_patch], current_list_y.iloc[indexes_in_patch], color='black',
                            s=.5)

            # Plot markers where the tracks start, interrupt and restart
            if plot_continuity:
                # Tracking stops
                plt.scatter(current_list_x.iloc[-1], current_list_y.iloc[-1], marker='X', color="red")
                # Tracking restarts
                if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                    plt.scatter(current_list_x[0], current_list_y[0], marker='*', color="black", s=100)
                    previous_folder = current_folder
                # First tracked point
                else:
                    plt.scatter(current_list_x[0], current_list_y[0], marker='*', color="green")

            # Plot the trajectory, one color per worm
            else:
                plt.scatter(current_list_x, current_list_y, color=colors[i_worm], s=.5)

    plt.show()


def plot_speed_time_window_list(traj, list_of_time_windows, nb_resamples, in_patch=False, out_patch=False):
    # TODO take care of holes in traj.csv
    """
    Will take the trajectory dataframe and exit the following plot:
    x-axis: time-window size
    y-axis: average proportion of time spent on food over that time-window
    color: current speed
    in_patch / out_patch: only take time points such as the worm is currently inside / outside a patch
                          if both are True or both are False, will take any time point
    This function will show nb_resamples per time window per plate.
    """
    plate_list = np.unique(traj["folder"])
    nb_of_plates = len(plate_list)
    normalize = mplcolors.Normalize(vmin=0, vmax=3.5)
    for i_window in range(len(list_of_time_windows)):
        window_size = list_of_time_windows[i_window]
        print(window_size)
        average_food_list = np.zeros(nb_of_plates * nb_resamples)
        current_speed_list = np.zeros(nb_of_plates * nb_resamples)
        for i_plate in range(nb_of_plates):
            plate = plate_list[i_plate]
            current_traj = traj[traj["folder"] == plate].reset_index()
            # Pick a random time to look at, cannot be before window_size otherwise not enough past for avg food
            for i_resample in range(nb_resamples):
                if len(current_traj) > 10000:
                    random_time = random.randint(window_size, len(current_traj) - 1)

                    # Only take current times such as the worm is inside a patch
                    if in_patch and not out_patch:
                        n_trials = 0
                        while current_traj["patch"][random_time] == -1 and n_trials < 100:
                            random_time = random.randint(window_size, len(current_traj) - 1)
                            n_trials += 1
                        if n_trials == 100:
                            print("No in_patch position was  for window, plate:", str(window_size), ", ", plate)

                    # Only take current times such as the worm is outside a patch
                    if out_patch and not in_patch:
                        n_trials = 0
                        while current_traj["patch"][random_time] != -1 and n_trials < 100:
                            random_time = random.randint(window_size, len(current_traj) - 1)
                            n_trials += 1
                        if n_trials == 100:
                            print("No out_patch position was  for window, plate:", str(window_size), ", ", plate)

                    # Look for first index of the trajectory where the frame is at least window_size behind
                    # This is because if we just take random time - window-size there might be a tracking hole in the traj
                    first_index = current_traj[current_traj["frame"] <= current_traj["frame"][random_time] - window_size].index.values
                    if len(first_index) > 0:  # if there is such a set of indices
                        first_index = first_index[-1]  # take the latest one
                        traj_window = traj[first_index: random_time]
                        # Compute average feeding rate over that window and current speed
                        average_food_list[i_plate * nb_resamples + i_resample] = len(traj_window[traj_window["patch"] != -1]) / window_size
                        current_speed_list[i_plate * nb_resamples + i_resample] = current_traj["distances"][random_time]
                    else:  # otherwise it means the video is not long enough
                        average_food_list[i_plate + i_resample] = -1
                        current_speed_list[i_plate + i_resample] = -1

                else:
                    average_food_list[i_plate + i_resample] = -1
                    current_speed_list[i_plate + i_resample] = -1
        # Sort all lists according to speed (in one line sorry oopsie)
        current_speed_list, average_food_list = zip(*sorted(zip(current_speed_list, average_food_list)))
        # Plot for this window size
        plt.scatter([window_size + current_speed_list[i] for i in range(len(current_speed_list))], average_food_list,
                    c=current_speed_list, cmap="viridis", norm=normalize)
    plt.colorbar()
    plt.show()
    return 0


def plot_speed_time_window_continuous(traj, time_window_min, time_window_max, step_size, nb_resamples, current_speed,
                                      speed_history, past_speed):
    #TODO take care of holes in traj.csv
    """
    === Will take the trajectory dataframe and:
    start and end for the time windows
    step size by which to increase time window
    nb of times to do a random resample in the video
    3 bool values to describe which speed is plotted as color
    === Exits the following plot:
    x-axis: time-window size
    y-axis: average proportion of time spent on food over that time-window
    color: current speed
    Note: it will show one point per time window per plate because otherwise wtf
    """
    plate_list = np.unique(traj["folder"])
    random_plate = plate_list[random.randint(0, len(plate_list))]
    random_traj = traj[traj["folder"] == random_plate].reset_index()
    condition = fd.folder_to_metadata(random_plate)["condition"].reset_index()
    for n in range(nb_resamples):
        present_time = random.randint(0, len(random_traj))
        normalize = mplcolors.Normalize(vmin=0, vmax=3.5)
        window_size = time_window_min - step_size
        while window_size < min(len(random_traj), time_window_max):
            window_size += step_size
            traj_window = traj[present_time - window_size:present_time]
            average_food = len(traj_window[traj_window["patch"] != -1]) / window_size
            speed = 0
            if current_speed:
                speed = random_traj["distances"][present_time]
            if speed_history:
                speed = np.mean(traj_window["distances"])
            if past_speed:
                speed = np.mean(traj_window["distances"][0:window_size])
            # Plot for this window size
            plt.scatter(window_size, average_food, c=speed, cmap="viridis", norm=normalize)
    plt.colorbar()
    plt.ylabel("Average feeding rate")
    plt.xlabel("Time window to compute past average feeding rate")
    plt.title(str(condition["condition"][0]) + ", " + str(random_plate)[-48:-9])
    plt.show()
    return 0


def binned_speed_as_a_function_of_time_window(traj, condition_list, list_of_time_windows, list_of_food_bins, nb_resamples, in_patch=False, out_patch=False):
    """
    Function that takes a table of trajectories, a list of time windows and food bins,
    and will plot the CURRENT SPEED for each time window and for each average food during that time window
    FOR NOW, WILL TAKE nb_resamples RANDOM TIMES IN EACH PLATE
    """
    # Prepare plate list
    full_plate_list = np.unique(traj["folder"])
    plate_list = []
    for condition in condition_list:
        plate_list += fd.return_folder_list_one_condition(full_plate_list, condition)
    nb_of_plates = len(plate_list)

    # This is for x ticks for the final plot
    list_of_x_positions = []

    # Fill lists with info for each plate
    for i_window in range(len(list_of_time_windows)):
        window_size = list_of_time_windows[i_window]
        average_food_list = np.zeros(nb_of_plates * nb_resamples)
        current_speed_list = np.zeros(nb_of_plates * nb_resamples)
        for i_plate in range(nb_of_plates):
            if i_plate % 20 == 0:
                print("Computing for plate ", i_plate, "/", nb_of_plates)
            plate = plate_list[i_plate]
            current_traj = traj[traj["folder"] == plate].reset_index()

            # Pick a random time to look at, cannot be before window_size otherwise not enough past for avg food
            for i_resample in range(nb_resamples):
                #TODO correct this interval, it should be between last frame and first frame that is at least window size
                random_time = random.randint(window_size, len(current_traj)-1)

                # Only take current times such as the worm is inside a patch
                if in_patch and not out_patch:
                    n_trials = 0
                    while current_traj["patch"][random_time] == -1 and n_trials < 100:
                        random_time = random.randint(window_size, len(current_traj)-1)
                        n_trials += 1
                    if n_trials == 100:
                        print("No in_patch position was  for window, plate:", str(window_size), ", ", plate)

                # Only take current times such as the worm is outside a patch
                if out_patch and not in_patch:
                    n_trials = 0
                    while current_traj["patch"][random_time] != -1 and n_trials < 100:
                        random_time = random.randint(window_size, len(current_traj)-1)
                        n_trials += 1
                    if n_trials == 100:
                        print("No out_patch position was  for window, plate:", str(window_size), ", ", plate)

                # Look for first index of the trajectory where the frame is at least window_size behind
                # This is because if we just take random time - window-size there might be a tracking hole in the traj
                first_index = current_traj[current_traj["frame"] <= current_traj["frame"][random_time] - window_size].index.values
                if len(first_index) > 0:  # if there is such a set of indices
                    first_index = first_index[-1]  # take the latest one
                    traj_window = traj[first_index: random_time]
                    # Compute average feeding rate over that window and current speed
                    average_food_list[i_plate * nb_resamples + i_resample] = len(traj_window[traj_window["patch"] != -1]) / window_size
                    current_speed_list[i_plate * nb_resamples + i_resample] = current_traj["distances"][random_time]
                else:  # otherwise it means the video is not long enough
                    average_food_list[i_plate + i_resample] = -1
                    current_speed_list[i_plate + i_resample] = -1

        # Sort all lists according to average_food (in one line sorry oopsie)
        average_food_list, current_speed_list = zip(*sorted(zip(average_food_list, current_speed_list)))
        print("Finished computing for window = ", window_size)

        # Fill the binsss
        binned_avg_speeds = np.zeros(len(list_of_food_bins))
        errorbars_sup = []
        errorbars_inf = []
        i_food = 0
        for i_bin in range(len(list_of_food_bins)):
            list_curr_speed_this_bin = []
            # While avg food is not above bin, continue filling it
            while i_food < len(average_food_list) and average_food_list[i_food] <= list_of_food_bins[i_bin]:
                list_curr_speed_this_bin.append(current_speed_list[i_food])
                i_food += 1
            # Once the bin is over, fill stat info for global plot
            binned_avg_speeds[i_bin] = np.mean(list_curr_speed_this_bin)
            errors = bottestrop_ci(list_curr_speed_this_bin, 1000)
            errorbars_inf.append(errors[0])
            errorbars_sup.append(errors[1])
            # and plot individual points
            plt.scatter([2 * i_window + list_of_food_bins[i_bin] for _ in range(len(list_curr_speed_this_bin))],
                        list_curr_speed_this_bin, zorder=2, color="gray")
            try:
                plt.violinplot(list_curr_speed_this_bin, positions=[2 * i_window + list_of_food_bins[i_bin]])
            except ValueError:
                pass
            # Indicate on graph the nb of points in each bin
            if list_curr_speed_this_bin:
                ax = plt.gca()
                ax.annotate(str(len(list_curr_speed_this_bin)), xy=(2 * i_window + list_of_food_bins[i_bin], max(list_curr_speed_this_bin)+0.5))
            print("Finished binning for bin ", i_bin, "/", len(list_of_food_bins))

        # Plot for this window size
        x_positions = [2 * i_window + list_of_food_bins[i] for i in range(len(list_of_food_bins))]  # for the bars
        list_of_x_positions += x_positions  # for the final plot
        plt.bar(x_positions, binned_avg_speeds, width=min(0.1, 1/len(list_of_food_bins)), label=str(window_size))
        plt.errorbar(x_positions, binned_avg_speeds, [errorbars_inf, errorbars_sup], fmt='.k', capsize=5)

    ax = plt.gca()
    ax.set_xticks(list_of_x_positions)
    ax.set_xticklabels([str(np.round(list_of_food_bins[i], 2)) for i in range(len(list_of_food_bins))]*len(list_of_time_windows))
    plt.xlabel("Average amount of food during time window")
    plt.ylabel("Average speed")
    plt.legend(title="Time window size")

    plt.show()
    return 0

# for i_traj in range(len(trajectories)):
#     reformatted_trajectory = list(zip(*trajectories[i_traj])) # converting from [x y][x y][x y] format to [x x x] [y y y]
#     plt.plot(reformatted_trajectory[0],reformatted_trajectory[1])


def visit_time_as_a_function_of(result_table, variable):
    if variable == "last_travel_time":
        list_of_visit_lengths = []
        list_of_previous_transit_lengths = []
        starts_with_visit = False
        ends_with_transit = False
        for i_plate in range(len(result_table)):
            list_of_visits = list(json.loads(result_table["aggregated_raw_visits"][i_plate]))
            list_of_transits = list(json.loads(result_table["aggregated_raw_transits"][i_plate]))
            if list_of_visits and list_of_transits:  # if there's at least one visit and one transit
                # Check whether the plate starts and ends with a visit or a transit
                if list_of_visits[0][0] < list_of_transits[0][0]:
                    starts_with_visit = True
                # If it starts with a visit we only start at visit 1 (visit 0 has no previous transit)
                i_visit = starts_with_visit
                # If there are consecutive visits/transits, we count them to still look at temporally consecutive visits and transits
                double_transits = 0
                double_visits = 0
                while i_visit + double_visits < len(list_of_visits):
                    if verbose:
                        print("Nb of visits = ", len(list_of_visits), ", nb of transits = ", len(list_of_transits), ", i_visit = ", i_visit, "starts_with = ", starts_with_visit)
                        print("double_transits = ", double_transits, ", double_visits = ", double_visits)
                    current_visit = list_of_visits[i_visit + double_visits]  # True = 1 in Python
                    # When the video starts with a visit, visit 1 has to be compared to transit 0
                    # Otherwise, visit 0 has to be compared to transit 0
                    current_transit = list_of_transits[i_visit + double_transits - double_visits + starts_with_visit]
                    # Check that this is the right transit:
                    if current_visit[0] == current_transit[1]:
                        list_of_visit_lengths.append(current_visit[1]-current_visit[0]+1)
                        list_of_previous_transit_lengths.append(current_transit[1]-current_transit[0]+1)
                    else:
                        # Take care of any extra visit/transit that's in the way
                        while current_visit[0] > current_transit[1]:  # there were two consecutive transits
                            double_transits += 1
                            current_transit = list_of_transits[i_visit - starts_with_visit - double_visits + double_transits]
                            # We add this extra transit to the previous transit length
                            list_of_previous_transit_lengths[-1] += current_transit[1]-current_transit[0]+1
                        while current_visit[0] < current_transit[1]:  # there were two consecutive visits
                            double_visits += 1
                            current_visit = list_of_visits[i_visit + double_visits]
                            # We add this extra transit to the previous transit length
                            list_of_visit_lengths[-1] += current_visit[1]-current_visit[0]+1
                    i_visit += 1
        plt.scatter(list_of_previous_transit_lengths, list_of_visit_lengths)
        plt.show()


def plot_patches(folder_list, show_composite=True, is_plot=True):
    """
    Function that takes a folder list, and for each folder, will either:
    - plot the patch positions on the composite patch image, to check if our metadata matches our actual data (is_plot = True)
    - return a list of border positions for each patch (is_plot = False)
    """
    for folder in folder_list:
        metadata = fd.folder_to_metadata(folder)
        patch_centers = metadata["patch_centers"]

        lentoremove = len('traj.csv')  # removes traj from the current path, to get to the parent folder
        folder = folder[:-lentoremove]

        if is_plot:
            fig, ax = plt.subplots()
            if show_composite:
                composite = plt.imread(folder + "composite_patches.tif")
                composite = ax.imshow(composite)
            else:
                background = plt.imread(folder + "background.tif")
                background = ax.imshow(background, cmap='gray')

        patch_centers = metadata["patch_centers"]
        patch_densities = metadata["patch_densities"]
        patch_spline_breaks = metadata["spline_breaks"]
        patch_spline_coefs = metadata["spline_coefs"]

        colors = plt.cm.jet(np.linspace(0, 1, len(patch_centers)))
        x_list = []
        y_list = []
        # For each patch
        for i_patch in range(len(patch_centers)):
            # For a range of 100 angular positions
            angular_pos = np.linspace(0, 2 * np.pi, 100)
            radiuses = np.zeros(len(angular_pos))
            # Compute the local spline value for each of those radiuses
            for i_angle in range(len(angular_pos)):
                radiuses[i_angle] = gr.spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                                    patch_spline_coefs[i_patch])

            fig = plt.gcf()
            ax = fig.gca()

            # Create lists of cartesian positions out of this
            x_pos = []
            y_pos = []
            for point in range(len(angular_pos)):
                x_pos.append(patch_centers[i_patch][0] + (radiuses[point] * np.sin(angular_pos[point])))
                y_pos.append(patch_centers[i_patch][1] + (radiuses[point] * np.cos(angular_pos[point])))

            # Either plot them
            if is_plot:
                plt.plot(x_pos, y_pos, color=colors[i_patch])
            # Or add them to a list for later
            else:
                x_list.append(x_pos)
                y_list.append(y_pos)

        if is_plot:
            plt.title(folder)
            plt.show()
        else:
            return x_list, y_list


def plot_selected_data(plot_title, condition_list, column_name, condition_names, divided_by="", mycolor="blue"):
    """
    This function will plot a selected part of the data. Selection is described as follows:
    - condition_low, condition_high: bounds on the conditions (0,3 => function will plot conditions 0, 1, 2, 3)
    - column_name:
    """
    # Getting results
    list_of_conditions, list_of_avg_each_plate, average_per_condition, errorbars = results_per_condition(results,
                                                                                                         column_name,
                                                                                                         divided_by)

    # Slicing to get condition we're interested in (only take indexes from condition_list)
    list_of_conditions = [list_of_conditions[i] for i in condition_list]
    list_of_avg_each_plate = [list_of_avg_each_plate[i] for i in condition_list]
    average_per_condition = [average_per_condition[i] for i in condition_list]
    errorbars[0] = [errorbars[0][i] for i in condition_list]
    errorbars[1] = [errorbars[1][i] for i in condition_list]

    # Plotttt
    plt.title(plot_title)
    fig = plt.gcf()
    ax = fig.gca()
    fig.set_size_inches(5, 6)

    # Plot condition averages as a bar plot
    ax.bar(range(len(list_of_conditions)), average_per_condition, color=mycolor)
    ax.set_xticks(range(len(list_of_conditions)))
    ax.set_xticklabels(condition_names, rotation=45)
    ax.set(xlabel="Condition number")

    # Plot plate averages as scatter on top
    for i in range(len(list_of_conditions)):
        ax.scatter([range(len(list_of_conditions))[i] for _ in range(len(list_of_avg_each_plate[i]))],
                   list_of_avg_each_plate[i], color="red", zorder=2)
    ax.errorbar(range(len(list_of_conditions)), average_per_condition, errorbars, fmt='.k', capsize=5)
    plt.show()


def plot_data_coverage(traj):
    """
    Takes a dataframe with the trajectories implemented as in our trajectories.csv folder.
    Returns a plot with plates in y, time in x, and a color depending on whether:
    - there is or not a data point for this frame
    - the worm in this frame is in a patch or not
    """
    list_of_plates = np.unique(traj["folder"])
    nb_of_plates = len(list_of_plates)
    list_of_frames = [list(i) for i in np.zeros((nb_of_plates, 1),
                                                dtype='int')]  # list of list of frames for each plate [[0],[0],...,[0]]
    list_of_coverages = np.zeros(len(list_of_plates))  # proportion of coverage for each plate
    # to plot data coverage
    list_x = []
    list_y = []
    for i_plate in range(nb_of_plates):
        current_plate = list_of_plates[i_plate]
        current_plate_data = traj[traj["folder"] == current_plate]  # select one plate
        current_list_of_frames = list(current_plate_data["frame"])  # extract its frames
        current_coverage = len(current_list_of_frames) / current_list_of_frames[-1]  # coverage
        list_of_coverages[i_plate] = current_coverage
        if current_coverage > 0.85:
            for frame in current_list_of_frames:
                list_x.append(frame)
                list_y.append(current_plate)
    plt.scatter(list_x, list_y, s=.8)
    plt.show()


def plot_graphs(plot_quality=False, plot_speed=False, plot_visit_duration=False, plot_visit_rate=False,
                plot_proportion=False, plot_full=False):
    # Data quality
    if plot_quality:
        plot_selected_data("Average proportion of double frames in all densities", 0, 11,
                           "avg_proportion_double_frames",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="",
                           mycolor="green")
        plot_selected_data("Average number of bad events in all densities", 0, 11, "nb_of_bad_events",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="",
                           mycolor="green")

    # Speed plots
    if plot_speed:
        plot_selected_data("Average speed in all densities (inside)", range(12), "average_speed_inside",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="",
                           mycolor="green")
        plot_selected_data("Average speed in all densities (outside)", range(12), "average_speed_outside",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="",
                           mycolor="green")

    # Visits plots
    if plot_visit_duration:
        plot_selected_data("Average duration of visits in low densities", [0, 1, 2, 11], "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by="nb_of_visits", mycolor="brown")
        plot_selected_data("Average duration of visits in medium densities", [4, 5, 6, 11], "total_visit_time",
                           ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by="nb_of_visits", mycolor="orange")
        plot_selected_data("Average duration of MVT visits in low densities", [0, 1, 2, 11], "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by="mvt_nb_of_visits",
                           mycolor="brown")
        plot_selected_data("Average duration of MVT visits in medium densities", [4, 5, 6, 11], "total_visit_time",
                           ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by="mvt_nb_of_visits",
                           mycolor="orange")

    if plot_visit_rate:
        plot_selected_data("Average visit rate in low densities", [0, 1, 2, 11], "nb_of_visits",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], divided_by="total_video_time",
                           mycolor="brown")
        plot_selected_data("Average visit rate in medium densities", [4, 5, 6, 11], "nb_of_visits",
                           ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], divided_by="total_video_time",
                           mycolor="orange")
        plot_selected_data("Average visit rate MVT in low densities", [0, 1, 2, 11], "mvt_nb_of_visits",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], divided_by="total_video_time",
                           mycolor="brown")
        plot_selected_data("Average visit rate MVT in medium densities", [4, 5, 6, 11], "mvt_nb_of_visits",
                           ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], divided_by="total_video_time",
                           mycolor="orange")

    if plot_proportion:
        plot_selected_data("Average proportion of time spent in patches in low densities", [0, 1, 2, 11],
                           "total_visit_time", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"],
                           divided_by="total_video_time", mycolor="brown")
        plot_selected_data("Average proportion of time spent in patches in mediun densities", [4, 5, 6, 11],
                           "total_visit_time", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"],
                           divided_by="total_video_time", mycolor="orange")

        # plot_selected_data("Average number of visits in low densities", 0, 3, "nb_of_visits", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
        # plot_selected_data("Average furthest visited patch distance in low densities", 0, 3, "furthest_patch_distance", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
        # plot_selected_data("Average proportion of visited patches in low densities", 0, 3, "proportion_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
        # plot_selected_data("Average number of visited patches in low densities", 0, 3, "nb_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")

        # plot_selected_data("Average number of visits in medium densities", 4, 7, "nb_of_visits", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
        # plot_selected_data("Average furthest visited patch distance in medium densities", 4, 7, "furthest_patch_distance", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
        # plot_selected_data("Average proportion of visited patches in medium densities", 4, 7, "proportion_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
        # plot_selected_data("Average number of visited patches in medium densities", 4, 7, "nb_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")

    # Full plots
    if plot_full:
        plot_selected_data("Average duration of visits in all densities", 0, 11, "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                           divided_by="nb_of_visits", mycolor="brown")
        plot_selected_data("Average duration of MVT visits in all densities", 0, 11, "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                           divided_by="mvt_nb_of_visits", mycolor="brown")


# Data path
if fd.is_linux():  # Linux path
    path = "/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/"
else:  # Windows path
    path = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/Results_minipatches_20221108_clean_fp_less/"

# Extracting data, the function looks for all "traj.csv" files in the indicated path (will look into subfolders)
# It will then generate a "results" table, with one line per worm, and these info:
# NOTE: lists are stored as strings in the csv so we retrieve the values with json loads function

# Only generate the results in the beginning of your analysis!
### Saves the results in path:
####### "trajectories.csv": raw trajectories, one line per tracked point
####### "results_per_id.csv":
####### "results_per_plate.csv":
####### "clean_results.csv":
# Will regenerate the dataset from the first True boolean
regenerate_trajectories = False
regenerate_results_per_id = False
regenerate_results_per_plate = False
regenerate_clean_results = False

if regenerate_trajectories:
    gr.generate_trajectories(path)
    gr.generate_results_per_id(path)
    gr.generate_results_per_plate(path)
    gr.generate_clean_results(path)

elif regenerate_results_per_id:
    gr.generate_results_per_id(path)
    gr.generate_results_per_plate(path)
    gr.generate_clean_results(path)

elif regenerate_results_per_plate:
    gr.generate_results_per_plate(path)
    gr.generate_clean_results(path)

elif regenerate_clean_results:
    gr.generate_clean_results(path)

# Retrieve results from what generate_and_save has saved
trajectories = pd.read_csv(path + "clean_trajectories.csv")
results = pd.read_csv(path + "clean_results.csv")

print("finished retrieving stuff")

# plot_patches(fd.path_finding_traj(path))
# plot_avg_furthest_patch()
# plot_data_coverage(trajectories)
# plot_traj(trajectories, 2, n_max = 4, is_plot_patches = True, show_composite = True, plot_in_patch = True, plot_continuity = True, plot_speed = False, plot_time = False)
# plot_graphs(plot_visit_duration=True)
# plot_speed_time_window_list(trajectories, [100, 1000, 10000], 1, out_patch=True)
# plot_speed_time_window_continuous(trajectories, 1, 120, 1, 100, current_speed=False, speed_history=True, past_speed=False)
# binned_speed_as_a_function_of_time_window(trajectories, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 100, 1000], [0, 0.6, 0.8, 0.9, 1], 1, out_patch=True)
visit_time_as_a_function_of(results, "last_travel_time")

# TODO function find frame that returns index of a frame in a traj with two options: either approach from below, or approach from top
# TODO function that shows speed as a function of time since patch has been entered (ideally, concatenate all visits)
# TODO function that shows length of (first) visit to a patch as a function of last travel time / average feeding rate in window

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
