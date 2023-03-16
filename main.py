# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                #if divided_by == "nb_of_visits" and column_name == "duration_sum" and current_condition == 2: #detecting extreme far 0.2 cases
                #    if list_of_values[i_plate]>800:
                #        print(list_of_plates[i_plate])
                #        print(list_of_values[i_plate])
            else: # No division has to be made
                if column_name == "proportion_of_visited_patches" or column_name == "nb_of_visited_patches": # Special case: divide by total nb of patches in plate
                    current_plate = current_plate.reset_index()
                    list_of_visited_patches = [json.loads(current_plate["list_of_visited_patches"][i]) for i in range(len(current_plate["list_of_visited_patches"]))]
                    list_of_visited_patches = [i for liste in list_of_visited_patches for i in liste]
                    if column_name == "nb_of_visited_patches":
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches))
                    else:
                        list_total_patch = [52, 24, 7, 25, 52, 24, 7, 25, 24, 24, 24, 24]
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches))\
                                              /list_total_patch[i_condition]
                if column_name == "furthest_patch_distance":
                    list_of_values[i_plate] = np.max(current_plate[column_name])
                else:
                    list_of_values[i_plate] = np.sum(current_plate[column_name])

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

def plot_traj(data, i_condition, n_max = 4, plot_patches = False):
    """
    Function that takes in our dataframe format, using columns: "x", "y", "id_conservative", "folder"
    and extracting "condition" info in metadata
    Extracts list of series of positions from indicated condition and draws them, with one color per id
    :param data: dataframe containing the series of (x,y) positions ([[x0,x1,x2...] [y0,y1,y2...])
    :return: trajectory plot
    """
    worm_list = np.unique(data["id_conservative"])
    nb_of_worms = len(worm_list)
    colors = plt.cm.jet(np.linspace(0, 1, nb_of_worms))
    previous_folder = 0
    n_plate = 1
    for i_worm in range(nb_of_worms):
        current_worm = worm_list[i_worm]
        current_list_x = data[data["id_conservative"] == current_worm]["x"]
        current_list_y = data[data["id_conservative"] == current_worm]["y"]
        current_folder = list(data["folder"][data["id_conservative"] == worm_list[i_worm]])[0]
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
                previous_folder = current_folder
                patches = metadata["patch_centers"]
                patch_densities = metadata["patch_densities"]
                # composite = plt.imread(current_folder[:-len("traj.csv")] + "composite_patches.tif")
                background = plt.imread(current_folder[:-len("traj.csv")] + "background.tif")
                fig = plt.gcf()
                ax = fig.gca()
                fig.set_tight_layout(True)
                # ax.imshow(composite)
                ax.imshow(background, cmap='gray')
                ax.set_title(str(current_folder[-48:-9]))
                for i_patch in range(len(patches)):
                    circle = plt.Circle((patches[i_patch][0], patches[i_patch][1]), patch_radius, color="grey",
                                        alpha=min(1, patch_densities[i_patch][0]))
                    fig = plt.gcf()
                    ax = fig.gca()
                    if plot_patches:
                        ax.add_patch(circle)
            # Plot worm trajectory
            plt.plot(current_list_x, current_list_y, color=colors[i_worm])
    plt.show()
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

def plot_selected_data(plot_title, condition_low, condition_high, column_name, condition_names, divided_by = "", mycolor = "blue"):
    """
    This function will plot a selected part of the data. Selection is described as follows:
    - condition_low, condition_high: bounds on the conditions (0,3 => function will plot conditions 0, 1, 2, 3)
    - column_name:
    """
    # Getting results
    list_of_conditions, list_of_avg_each_plate, average_per_condition, errorbars = results_per_condition(results, column_name, divided_by)

    # Slicing to get condition we're interested in
    list_of_conditions = list_of_conditions[condition_low:condition_high+1]
    list_of_avg_each_plate = list_of_avg_each_plate[condition_low:condition_high+1]
    average_per_condition = average_per_condition[condition_low:condition_high+1]
    errorbars[0] = errorbars[0][condition_low:condition_high+1]
    errorbars[1] = errorbars[1][condition_low:condition_high+1]

    # Plotttt
    plt.title(plot_title)
    fig = plt.gcf()
    ax = fig.gca()
    # Plot condition averages as a bar plot
    ax.bar(list_of_conditions, average_per_condition, color = mycolor)
    ax.set_xticks(range(len(list_of_conditions)))
    ax.set_xticklabels(condition_names)
    ax.errorbar(list_of_conditions, average_per_condition, errorbars, fmt='.k', capsize=5)
    ax.set(xlabel="Condition number")
    # Plot plate averages as scatter on top
    for i in range(len(list_of_conditions)):
        ax.scatter([list_of_conditions[i] for j in range(len(list_of_avg_each_plate[i]))], list_of_avg_each_plate[i], color="red")
    plt.show()

#TODO once we have an exclusion protocol, add a "data cleaning" step to the generate_results.py script
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

def plot_graphs():
    # Low density plots
    plot_selected_data("Average duration of visits in low densities", 0, 3, "duration_sum", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], divided_by= "nb_of_visits", mycolor = "brown")
    plot_selected_data("Average proportion of time spent in patches in low densities", 0, 3, "duration_sum", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], divided_by= "total_time", mycolor = "brown")
    plot_selected_data("Average visit rate in low densities", 0, 3, "nb_of_visits", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], divided_by= "total_time", mycolor = "brown")
    #plot_selected_data("Average number of visits in low densities", 0, 3, "nb_of_visits", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
    #plot_selected_data("Average furthest visited patch distance in low densities", 0, 3, "furthest_patch_distance", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
    plot_selected_data("Average duration of MVT visits in low densities", 0, 3, "duration_sum", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], divided_by= "adjusted_nb_of_visits", mycolor = "brown")
    plot_selected_data("Average visit rate MVT in low densities", 0, 3, "adjusted_nb_of_visits", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], divided_by= "total_time", mycolor = "brown")
    #plot_selected_data("Average proportion of visited patches in low densities", 0, 3, "proportion_of_visited_patches", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
    #plot_selected_data("Average number of visited patches in low densities", 0, 3, "nb_of_visited_patches", ["close 0.2", "medium 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")


    # Medium density plots
    plot_selected_data("Average duration of visits in medium densities", 4, 7, "duration_sum", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], divided_by= "nb_of_visits", mycolor = "orange")
    plot_selected_data("Average proportion of time spent in patches in mediun densities", 4, 7, "duration_sum", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], divided_by= "total_time", mycolor = "orange")
    plot_selected_data("Average visit rate in medium densities", 4, 7, "nb_of_visits", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], divided_by= "total_time", mycolor = "orange")
    #plot_selected_data("Average number of visits in medium densities", 4, 7, "nb_of_visits", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
    #plot_selected_data("Average furthest visited patch distance in medium densities", 4, 7, "furthest_patch_distance", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
    plot_selected_data("Average duration of MVT visits in medium densities", 4, 7, "duration_sum", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], divided_by= "adjusted_nb_of_visits", mycolor = "orange")
    plot_selected_data("Average visit rate MVT in medium densities", 4, 7, "adjusted_nb_of_visits", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], divided_by= "total_time", mycolor = "orange")
    #plot_selected_data("Average proportion of visited patches in medium densities", 4, 7, "proportion_of_visited_patches", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
    #plot_selected_data("Average number of visited patches in medium densities", 4, 7, "nb_of_visited_patches", ["close 0.5", "medium 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")


    # Full plots
    #plot_selected_data(0, 11, "adjusted_duration_sum", [], "Average duration of visits", divided_by= "nb_of_visits", mycolor = "green")
    #plot_selected_data(0, 11, "nb_of_visited_patches", [], "Average proportion of visited patches", divided_by= "", mycolor = "green")
    #plot_selected_data(0, 11, "total_time", [], "Total measured time", divided_by= "", mycolor = "green")
    #plot_selected_data(0, 11, "duration_sum", [], "Average duration of visits", divided_by= "nb_of_visits", mycolor = "green")


# I have two lines, one for Windows and the other for Linux:
if fd.is_linux():
    path = "/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean/"
else:
    path = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/Results_minipatches_Nov2022_clean/"

# Extracting data, the function looks for all "traj.csv" files in the indicated path (will look into subfolders)
# It will then generate a "results" table, with one line per worm, and these info:
# NOTE: lists are stored as strings in the csv so we retrieve the values with json loads function
#         results_table["folder"] = folder from which the worm comes (so plate identifier)
#         results_table["condition"] = condition written on the plate of the worm
#         results_table["worm_id"] = number of the worm (100 times the file number + id attributed by tracking algorithm)
#         results_table["total_time"] = total number of frames for this worm
#         results_table["raw_visits"] = list outputed by patch_visits_single_traj (see its description)
#         results_table["order_of_visits"] = list of order of visits [2 3 0 1] = first patch 2, then patch 3, etc
#         results_table["duration_sum"] = total duration of visits for each worm
#         results_table["nb_of_visits"] = nb of visits to patches this worm did
#         results_table["nb_of_visited_patches"] = nb of different patches it visited
#         results_table["furthest_patch_distance"] = furthest patch visited
#         results_table["total_transit_time"] = total time spent outside of patches (same as total_time - duration_sum)
#         results_table["adjusted_raw_visits"] = adjusted: consecutive visits to the same patch are counted as one
#         results_table["adjusted_duration_sum"] = should be the same as duration sum (did this to check)
#         results_table["adjusted_nb_of_visits"] = nb of adjusted visits

# Only run this once in the beginning of your analysis!
### Saves these results in a "results.csv" file in path, so no need to run this line every time!
regenerate_data = True # Set to True to regenerate the dataset, otherwise use the saved one
if regenerate_data:
    gr.generate_and_save(path)  # run this once, will save results under path+"results.csv"

# Retrieve results from what generate_and_save has saved
trajectories = pd.read_csv(path + "trajectories.csv")
results = pd.read_csv(path + "results.csv")

print("finished retrieving stuff")

# check_patches(fd.path_finding_traj(path))
# plot_avg_furthest_patch()
#traj_draw(trajectories, 2, plot_patches = True)
#plot_graphs()

plot_data_coverage(trajectories)

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
# TODO for now furthest patch is useless because should be computed depending on ORIGIN and not first recorded position
