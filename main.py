# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

def results_per_condition(result_table, column_name, divided_by = ""):
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
            if divided_by != "": # In this case, we want to divide column name by another one
                if np.sum(current_plate[divided_by]) != 0: # Non zero check for division
                    list_of_values[i_plate] = np.sum(current_plate[column_name]) / np.sum(current_plate[divided_by])
                else:
                    print("Trying to divide by 0... what a shame")
                #if divided_by == "nb_of_visits" and column_name == "total_visit_time" and current_condition == 2: #detecting extreme far 0.2 cases
                #    if list_of_values[i_plate]>800:
                #        print(list_of_plates[i_plate])
                #        print(list_of_values[i_plate])
            else: # No division has to be made
                if column_name == "average_speed_inside" or column_name == "average_speed_outside":
                    # Exclude the 0's which are the cases were the worm didnt go to a patch / out of a patch for a full track
                    list_speed_current_plate = [nonzero for nonzero in current_plate[column_name] if int(nonzero) != 0]
                    if list_speed_current_plate:  # If any non-zero speed was recorded for that plate
                        list_of_values[i_plate] = np.average(list_speed_current_plate)
                elif column_name == "proportion_of_visited_patches" or column_name == "nb_of_visited_patches": # Special case: divide by total nb of patches in plate
                    current_plate = current_plate.reset_index()
                    list_of_visited_patches = [json.loads(current_plate["list_of_visited_patches"][i]) for i in range(len(current_plate["list_of_visited_patches"]))]
                    list_of_visited_patches = [i for liste in list_of_visited_patches for i in liste]
                    if column_name == "nb_of_visited_patches":
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches))
                    else:
                        list_total_patch = [52, 24, 7, 25, 52, 24, 7, 25, 24, 24, 24, 24]
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches))\
                                              /list_total_patch[i_condition]
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

    return list_of_conditions, full_list_of_values, list_of_avg_values, [errors_inf, errors_sup]

def bottestrop_ci(data, nb_resample):
    '''
    Function that takes a dataset and returns a confidence interval using nb_resample samples for bootstrapping
    '''
    bootstrapped_means = []
    #data = [x for x in data if str(x) != 'nan']
    for i in range(nb_resample):
        y = []
        for k in range(len(data)):
            y.append(random.choice(data))
        avg = np.mean(y)
        bootstrapped_means.append(avg)
    bootstrapped_means.sort()
    return [np.percentile(bootstrapped_means, 5), np.percentile(bootstrapped_means, 95)]

def plot_traj(traj, i_condition, n_max = 4, plot_patches = False, show_composite = True, plot_in_patch = False, plot_continuity = False, plot_speed = False, plate_list = []):
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
        if current_condition == i_condition:
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                if n_plate > n_max:
                    plt.show()
                    n_plate = 1
                plt.subplot(n_max//2, n_max//2, n_plate)
                n_plate += 1
                # if previous_folder != 0: #if its not the first (if its the first theres nothing to show)
                # plt.show()
                # Show background and patches
                patches = metadata["patch_centers"]
                patch_densities = metadata["patch_densities"]
                fig = plt.gcf()
                ax = fig.gca()
                fig.set_tight_layout(True)
                if show_composite :
                    composite = plt.imread(current_folder[:-len("traj.csv")] + "composite_patches.tif")
                    ax.imshow(composite)
                else :
                    background = plt.imread(current_folder[:-len("traj.csv")] + "background.tif")
                    ax.imshow(background, cmap='gray')
                ax.set_title(str(current_folder[-48:-9]))
                for i_patch in range(len(patches)):
                    circle = plt.Circle((patches[i_patch][0], patches[i_patch][1]), patch_radius, color="grey",
                                        alpha=min(1, patch_densities[i_patch][0]))
                    fig = plt.gcf()
                    ax = fig.gca()
                    if plot_patches:
                        ax.add_patch(circle)

            #     # Plot first and last position of the worm
            # first_pos = json.loads(current_traj["first_recorded_position"])
            # last_pos = json.loads(current_traj["last_tracked_position"])
            # plt.scatter(first_pos[0], first_pos[1], marker = '*')
            # plt.scatter(last_pos[0], last_pos[1], marker = 'X')

            # Plot worm trajectory
            indexes_in_patch = np.where(current_traj["patch"]!=-1)
            if not plot_speed:
                plt.scatter(current_list_x, current_list_y, color=colors[i_worm], s = .5)
            else:
                distance_list = current_traj.reset_index()["distances"]
                normalize = mplcolors.Normalize(vmin=0, vmax=3.5)
                plt.scatter(current_list_x, current_list_y, c = distance_list, cmap = "hot", norm = normalize, s = 1)
                if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                    plt.colorbar()

            if plot_in_patch:
                plt.scatter(current_list_x.iloc[indexes_in_patch], current_list_y.iloc[indexes_in_patch], color='black', s = .5)

            if plot_continuity:
                plt.scatter(current_list_x.iloc[-1], current_list_y.iloc[-1], marker='X', color="red")
                if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                    plt.scatter(current_list_x[0], current_list_y[0], marker='*', color = "black", s = 100)
                    previous_folder = current_folder
                else:
                    plt.scatter(current_list_x[0], current_list_y[0], marker='*', color = "green")
    plt.show()

def plot_speed_through_time(traj, time_window):
    plate_list = np.unique(traj["folder"])




# for i_traj in range(len(trajectories)):
    #     reformatted_trajectory = list(zip(*trajectories[i_traj])) # converting from [x y][x y][x y] format to [x x x] [y y y]
    #     plt.plot(reformatted_trajectory[0],reformatted_trajectory[1])

def check_patches(folder_list):
    """
    Function that takes a folder list, and for each folder, will:
    plot the patch positions on the composite patch image, to check if our metadata matches our actual data
    """
    for folder in folder_list:
        metadata = fd.folder_to_metadata(folder)
        patches = metadata["patch_centers"]

        lentoremove = len('traj.csv')  # removes traj from the current path, to get to the parent folder
        folder = folder[:-lentoremove]

        background = plt.imread(folder + "background.tif")
        composite = plt.imread(folder + "composite_patches.tif")

        fig, ax = plt.subplots()
        background = ax.imshow(background, cmap = 'gray')
        # composite = ax.imshow(composite)

        patches = metadata["patch_centers"]
        patch_densities = metadata["patch_densities"]
        for i_patch in range(len(patches)):
            circle = plt.Circle((patches[i_patch][0], patches[i_patch][1]), 50, color="white", alpha=0.5)
            fig = plt.gcf()
            ax = fig.gca()
            ax.add_patch(circle)

        plt.title(folder)

        plt.show()

def plot_selected_data(plot_title, condition_list, column_name, condition_names, divided_by = "", mycolor = "blue"):
    """
    This function will plot a selected part of the data. Selection is described as follows:
    - condition_low, condition_high: bounds on the conditions (0,3 => function will plot conditions 0, 1, 2, 3)
    - column_name:
    """
    # Getting results
    list_of_conditions, list_of_avg_each_plate, average_per_condition, errorbars = results_per_condition(results, column_name, divided_by)

    # Slicing to get condition we're interested in
    list_of_conditions = [list_of_conditions[i] for i in condition_list]
    list_of_avg_each_plate = [list_of_avg_each_plate[i] for i in condition_list]
    average_per_condition = [average_per_condition[i] for i in condition_list]
    errorbars[0] = [errorbars[0][i] for i in condition_list]
    errorbars[1] =  [errorbars[1][i] for i in condition_list]

    # Plotttt
    plt.title(plot_title)
    fig = plt.gcf()
    ax = fig.gca()
    fig.set_size_inches(5, 6)
    # Plot condition averages as a bar plot
    ax.bar(range(len(list_of_conditions)), average_per_condition, color = mycolor)
    ax.set_xticks(range(len(list_of_conditions)))
    ax.set_xticklabels(condition_names, rotation = 45)
    ax.set(xlabel="Condition number")
    # Plot plate averages as scatter on top
    for i in range(len(list_of_conditions)):
        ax.scatter([range(len(list_of_conditions))[i] for j in range(len(list_of_avg_each_plate[i]))], list_of_avg_each_plate[i], color="red", zorder = 2)
    ax.errorbar(range(len(list_of_conditions)), average_per_condition, errorbars, fmt='.k', capsize=5)
    plt.show()

def plot_data_coverage(trajectories):
    """
    Takes a dataframe with the trajectories implemented as in our trajectories.csv folder.
    Returns a plot with plates in y, time in x, and a color depending on whether:
    - there is or not a data point for this frame
    - the worm in this frame is in a patch or not
    """
    list_of_plates = np.unique(trajectories["folder"])
    nb_of_plates = len(list_of_plates)
    list_of_frames = [list(i) for i in np.zeros((nb_of_plates, 1), dtype='int')]  #list of list of frames for each plate [[0],[0],...,[0]]
    list_of_coverages = np.zeros(len(list_of_plates)) #proportion of coverage for each plate
    # to plot data coverage
    list_x = []
    list_y = []
    for i_plate in range(nb_of_plates):
        print(i_plate," / ",nb_of_plates)
        current_plate = list_of_plates[i_plate]
        current_plate_data = trajectories[trajectories["folder"] == current_plate] #select one plate
        current_list_of_frames = list(current_plate_data["frame"]) #extract its frames
        current_coverage = len(current_list_of_frames)/current_list_of_frames[-1] #coverage
        list_of_coverages[i_plate] = current_coverage
        if current_coverage > 0.85:
            for frame in current_list_of_frames:
                list_x.append(frame)
                list_y.append(current_plate)
    plt.scatter(list_x, list_y, s = .8)
    plt.show()

def plot_graphs(plot_quality = False, plot_speed = False, plot_low_density = True, plot_medium_density = True, plot_full = False):
    # Data quality
    if plot_quality:
        plot_selected_data("Average proportion of double frames in all densities", 0, 11, "avg_proportion_double_frames", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by= "", mycolor = "green")
        plot_selected_data("Average number of bad events in all densities", 0, 11, "nb_of_bad_events", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by= "", mycolor = "green")

    # Speed plots
    if plot_speed:
        plot_selected_data("Average speed in all densities (inside)", 0, 11, "average_speed_inside", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by= "", mycolor = "green")
        plot_selected_data("Average speed in all densities (outside)", 0, 11, "average_speed_outside", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by= "", mycolor = "green")

    # Low density plots
    if plot_low_density:
        plot_selected_data("Average duration of visits in low densities", [0,1,2,11], "total_visit_time", ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by= "nb_of_visits", mycolor = "brown")
        plot_selected_data("Average proportion of time spent in patches in low densities", [0,1,2,11], "total_visit_time", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], divided_by= "total_video_time", mycolor = "brown")
        plot_selected_data("Average visit rate in low densities", [0,1,2,11], "nb_of_visits", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], divided_by= "total_video_time", mycolor = "brown")
        #plot_selected_data("Average number of visits in low densities", 0, 3, "nb_of_visits", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
        #plot_selected_data("Average furthest visited patch distance in low densities", 0, 3, "furthest_patch_distance", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
        plot_selected_data("Average duration of MVT visits in low densities", [0,1,2,11], "total_visit_time", ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by= "mvt_nb_of_visits", mycolor = "brown")
        plot_selected_data("Average visit rate MVT in low densities", [0,1,2,11], "mvt_nb_of_visits", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], divided_by= "total_video_time", mycolor = "brown")
        #plot_selected_data("Average proportion of visited patches in low densities", 0, 3, "proportion_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
        #plot_selected_data("Average number of visited patches in low densities", 0, 3, "nb_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")

    # Medium density plots
    if plot_medium_density:
        plot_selected_data("Average duration of visits in medium densities", [4,5,6,11], "total_visit_time", ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by= "nb_of_visits", mycolor = "orange")
        plot_selected_data("Average proportion of time spent in patches in mediun densities", [4,5,6,11], "total_visit_time", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], divided_by= "total_video_time", mycolor = "orange")
        plot_selected_data("Average visit rate in medium densities", [4,5,6,11], "nb_of_visits", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], divided_by= "total_video_time", mycolor = "orange")
        #plot_selected_data("Average number of visits in medium densities", 4, 7, "nb_of_visits", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
        #plot_selected_data("Average furthest visited patch distance in medium densities", 4, 7, "furthest_patch_distance", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
        plot_selected_data("Average duration of MVT visits in medium densities", [4,5,6,11], "total_visit_time", ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by= "mvt_nb_of_visits", mycolor = "orange")
        plot_selected_data("Average visit rate MVT in medium densities", [4,5,6,11], "mvt_nb_of_visits", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], divided_by= "total_video_time", mycolor = "orange")
        #plot_selected_data("Average proportion of visited patches in medium densities", 4, 7, "proportion_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
        #plot_selected_data("Average number of visited patches in medium densities", 4, 7, "nb_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")


    # Full plots
    if plot_full:
        plot_selected_data("Average duration of visits in all densities", 0, 11, "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="nb_of_visits", mycolor="brown")
        plot_selected_data("Average duration of MVT visits in all densities", 0, 11, "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="mvt_nb_of_visits", mycolor="brown")


# Data path
if fd.is_linux():  # Linux path
    path = "/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/"
else:  # Windows path
    path = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/Results_minipatches_20221108_clean_fp/"

# Extracting data, the function looks for all "traj.csv" files in the indicated path (will look into subfolders)
# It will then generate a "results" table, with one line per worm, and these info:
# NOTE: lists are stored as strings in the csv so we retrieve the values with json loads function
#         results_table["folder"] = folder from which the worm comes (so plate identifier)
#         results_table["condition"] = condition written on the plate of the worm
#         results_table["track_id"] = number of the track (100 times the file number + id attributed by tracking algorithm)
#         results_table["total_time"] = total number of frames for this worm
#         results_table["raw_visits"] = list outputed by patch_visits_single_traj (see its description)
#         results_table["order_of_visits"] = list of order of visits [2 3 0 1] = first patch 2, then patch 3, etc
#         results_table["total_visit_time"] = total duration of visits for each worm
#         results_table["nb_of_visits"] = nb of visits to patches this worm did
#         results_table["nb_of_visited_patches"] = nb of different patches it visited
#         results_table["furthest_patch_distance"] = furthest patch visited
#         results_table["total_transit_time"] = total time spent outside of patches (same as total_time - total_visit_time)
#         results_table["adjusted_raw_visits"] = adjusted: consecutive visits to the same patch are counted as one
#         results_table["adjusted_total_visit_time"] = should be the same as duration sum (did this to check)
#         results_table["adjusted_nb_of_visits"] = nb of adjusted visits

# Only generate the results in the beginning of your analysis!
### Saves the results in path:
####### "trajectories.csv": raw trajectories, one line per tracked point
####### "results_per_id.csv":
####### "results_per_plate.csv":
####### "clean_results.csv":
# Will regenerate the dataset from the first True boolean
regenerate_trajectories = True
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
#trajectories = pd.read_csv(path + "trajectories.csv")
results = pd.read_csv(path + "clean_results.csv")

print("finished retrieving stuff")

# check_patches(fd.path_finding_traj(path))
# plot_avg_furthest_patch()
# plot_traj(trajectories, 2, n_max = 4, plot_patches = True, show_composite = True, plot_in_patch = False, plot_continuity = True, plot_speed = True, plate_list=["/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean/20221013T115434_SmallPatches_C5-CAM8/traj.csv"])
plot_graphs()

# plot_data_coverage(trajectories)

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
