import find_data as fd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import pandas as pd
from param import *


def in_patch(position, patch):
    """
    returns True if position = [x,y] is inside the patch
    uses general parameter radial_tolerance: the worm is still considered inside the patch when its center is sticking out by that distance or less
    """
    center = [patch[0], patch[1]]
    radius = patch_radius
    distance_from_center = np.sqrt((position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2)
    return distance_from_center < radius + radial_tolerance


def patch_visits_single_traj(list_x, list_y, patch_centers):
    """
    Takes a trajectory under the format: [x0 x1 ... xN] [y0 y1 ... yN] and a list of patch centers
    For now, uses the unique patch_radius defined as a global parameter.
    Returns a list [[d0,d1,...], [d0,d1,...],...] with one list per patch
    each sublist contains the durations of visits to each patch
    so len(output[0]) is the number of visits to the first patch
    """

    # Variables for the loop and the output
    is_in_patch = np.zeros(
        len(patch_centers))  # Bool list to store where the worm is currently (1 where it currently is)
    list_of_durations = [list(i) for i in
                         np.zeros((len(patch_centers), 1), dtype='int')]  # List with the right format [[0],[0],...,[0]]
    # In list_of_durations, zero means "the worm was not in the patch in the previous timestep"
    # As soon as the worm enters the patch, this zero starts being incremented
    # As soon as the worm leaves the patch, a new zero is added to this patch's list
    # These 0 are added for computational ease and will be removed in the end
    patch_where_it_is = -1  # initializing variable with index of patch where the worm currently is

    # We go through the whole trajectory
    for time in range(len(list_x)):
        was_in_patch = is_in_patch  # keeping in memory where the worm was, [0, 0, 1, 0] = in patch 2
        patch_where_it_was = patch_where_it_is  # index of the patch where it is
        patch_where_it_is = -1  # resetting the variable
        for i_patch in range(len(patch_centers)):  # for every patch
            is_in_patch[i_patch] = in_patch([list_x[time], list_y[time]],
                                            patch_centers[i_patch])  # check if the worm is in
            if is_in_patch[i_patch]:  # remember where it is
                patch_where_it_is = i_patch
        if patch_where_it_is == -1:  # Worm currently out
            if patch_where_it_was != patch_where_it_is:  # Worm exited a patch
                list_of_durations[patch_where_it_was].append(0)  # Add a zero because previous visit was interrupted
        if patch_where_it_is != -1:  # Worm currently inside, no matter whether it just entered or stayed inside
            list_of_durations[patch_where_it_is][-1] += 1  # add one to the last element of the patch list

    duration_sum = 0  # this is to compute the avg duration of visits
    nb_of_visits = 0
    first_recorded_worm_position = [list_x[0], list_y[0]]
    furthest_patch_distance = 0

    # Remove the zeros because they're just here for the duration algorithm
    # in the same loop we compute the average visit duration, and the furthest visited patch
    for i_patch in range(len(list_of_durations)):
        list_of_durations[i_patch] = [nonzero for nonzero in list_of_durations[i_patch] if nonzero != 0]

        if len(list_of_durations[i_patch]) > 0:  # if the patch was visited at least once
            patch_distance_to_center = distance.euclidean(first_recorded_worm_position, patch_centers[i_patch])
            furthest_patch_distance = max(patch_distance_to_center, furthest_patch_distance)

        duration_sum += sum(list_of_durations[i_patch])
        nb_of_visits += len(list_of_durations[i_patch])

    return list_of_durations, duration_sum, nb_of_visits, furthest_patch_distance


def patch_visits_multiple_traj(data):
    """
    (tldr: returns a list of outputs from the single_traj function, one list item per trajectory)
    Takes our data table and returns a series of analysis regarding patch visits durations
    """
    worm_list = np.unique(data["id_conservative"])
    nb_of_worms = len(worm_list)

    results_table = pd.DataFrame()
    results_table["folder"] = [-1 for i in range(nb_of_worms)]
    results_table["condition"] = [-1 for i in range(nb_of_worms)]
    results_table["worm_id"] = [-1 for i in range(nb_of_worms)]
    results_table["raw_visits"] = [-1 for i in range(nb_of_worms)]
    results_table["duration_sum"] = [-1 for i in range(nb_of_worms)]
    results_table["nb_of_visits"] = [-1 for i in range(nb_of_worms)]
    results_table["furthest_patch_distance"] = [-1 for i in range(nb_of_worms)]

    for i_worm in range(nb_of_worms):
        print(i_worm)

        # Data from the dataframe
        current_worm = worm_list[i_worm]
        current_data = data[data["id_conservative"] == current_worm]
        current_list_x = current_data["x"]
        current_list_y = current_data["y"]
        current_folder = list(current_data["folder"])[0]

        # Getting to the metadata through the folder name in the data
        current_metadata = fd.folder_to_metadata(current_folder)
        list_of_densities = current_metadata["patch_densities"]

        # Computing the visit durations
        raw_durations, duration_sum, nb_of_visits, furthest_patch_distance = patch_visits_single_traj(
                                                                                                list(current_list_x),
                                                                                                list(current_list_y),
                                                                                                current_metadata[
                                                                                                    "patch_centers"])

        # Fill up results table
        results_table.loc[i_worm, "folder"] = current_folder
        results_table.loc[i_worm, "condition"] = current_metadata["condition"][0]
        results_table.loc[i_worm, "worm_id"] = current_worm
        results_table.loc[i_worm, "raw_visits"] = pd.DataFrame(raw_durations)  # all visits of all patches
        results_table.loc[i_worm, "duration_sum"] = duration_sum  # all visits of all patches
        results_table.loc[i_worm, "nb_of_visits"] = nb_of_visits  # all visits of all patches
        results_table.loc[i_worm, "furthest_patch_distance"] = furthest_patch_distance  # all visits of all patches

    return results_table


def generate_and_save(dataframe, path):
    results = patch_visits_multiple_traj(dataframe)
    results.to_csv(path + "results.csv")
    return 0
