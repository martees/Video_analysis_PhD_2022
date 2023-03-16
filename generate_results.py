import find_data as fd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import pandas as pd
from param import *
from numba import njit
from itertools import groupby


def in_patch(position, patch):
    """
    returns True if position = [x,y] is inside the patch
    uses general parameter radial_tolerance: the worm is still considered inside the patch when its center is sticking out by that distance or less
    """
    center = [patch[0], patch[1]]
    radius = patch_radius
    distance_from_center = np.sqrt((position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2)
    return distance_from_center < radius + radial_tolerance


def speed_analysis(traj):
    """
    Function that takes in our trajectories dataframe, and returns a column with the distance covered by the worm since
    last timestep, in order to compute speed
    """
    array_x_r = np.array(traj["x"][1:])
    array_y_r = np.array(traj["y"][1:])
    array_x_l = np.array(traj["x"][:-1])
    array_y_l = np.array(traj["y"][:-1])

    list_of_distances = np.sqrt((array_x_l - array_x_r) ** 2 + (array_y_l - array_y_r) ** 2)

    return list_of_distances


def in_patch_list(traj):
    """
    Function that takes in our trajectories dataframe, and returns a column with the patch where the worm is.
    """
    list_of_plates = np.unique(traj["folder"])
    nb_of_plates = len(list_of_plates)

    # List where we'll put the patch where the worm is at each timestep
    list_of_patches = [-1 for i in range(len(traj["x"]))]
    i = 0  # global counter

    for i_plate in range(nb_of_plates):  # for every plate
        # Handmade progress bar
        print(i_plate, "/", nb_of_plates)

        # Extract patch information
        current_plate = list_of_plates[i_plate]
        current_metadata = fd.folder_to_metadata(current_plate)
        patch_centers = current_metadata["patch_centers"]

        # Extract positions
        current_data = traj[traj["folder"] == current_plate]
        list_x = current_data["x"]
        list_y = current_data["y"]

        # Analyze
        patch_where_it_is = -1  # initializing variable with index of patch where the worm currently is
        # We go through the whole trajectory
        for time in range(len(list_x)):
            # First we figure out where the worm is
            patch_where_it_was = patch_where_it_is  # index of the patch where it is
            patch_where_it_is = -1  # resetting the variable to "worm is out"

            # In case the worm is in the same patch, don't try all the patches:
            if in_patch([list_x[time], list_y[time]], patch_centers[patch_where_it_was]):
                patch_where_it_is = patch_where_it_was

            # If the worm changed patch, then look for it
            else:
                for i_patch in range(len(patch_centers)):  # for every patch
                    if in_patch([list_x[time], list_y[time]], patch_centers[i_patch]):  # check if the worm is in it:
                        patch_where_it_is = i_patch  # if it's in it, keep that in mind
            list_of_patches[i] = patch_where_it_is  # still -1 if patch wasn't found
            i = i+1

    return list_of_patches


# @njit(parallel=True)
def single_traj_analysis(which_patch_list, list_of_frames, patch_centers, first_xy):
    """
    Takes a list containing the patch where the worm is at each timestep, a list of frames to which each data point
    corresponds, a list of patch centers ([[x0,y0],...,[xN,yN]]), and the first recorded position for that worm [x,y]
    For now, uses the unique patch_radius defined as a global parameter.
    Returns a list [[d0,d1,...], [d0,d1,...],...] with one list per patch
    each sublist contains the durations of visits to each patch
    so len(output[0]) is the number of visits to the first patch
    """
    # Variables for the loop and the output

    # In list_of_timestamps, we will have one list per patch, containing the beginning and end of successive visits to that patch
    # [0,0] means "the worm was not in that patch in the previous timestep"
    # As soon as the worm enters the patch, first zero is updated to be current frame
    # As soon as the worm leaves the patch, second zero becomes current frame, and a new [0,0] is added to this patch's list
    # These [0,0] are added for computational ease and will be removed in the end
    list_of_timestamps = [[list(i)] for i in np.zeros((len(patch_centers), 2), dtype='int')]
    # List with the right format [[[0,0]],[[0,0]],...,[[0,0]]]
    # => each [0,0] is an empty visit that begins and ends in 0
    # in the end we want

    # This is the list of timestamps we will use to test the Marginal Value Theorem
    # In order to do so, if the worm visits the same patch multiple times in a row, we count that as one visit
    # The encoding is the same as list_of_timestamps, 0 = worm was not in that patch in its previous visit
    adjusted_list_of_timestamps = [list(i) for i in np.zeros((len(patch_centers), 1), dtype='int')]  # List with the right format [[0,0],[0,0],...,[0,0]]

    # Order in which the patches were visited (should have as many elements as list_of_timestamps)
    # (function that removes consecutive duplicates, [1,2,2,3] => [1,2,3]
    order_of_visits = [i[0] for i in groupby(which_patch_list)]

    # In order to compute when visits to patches start, end, and how long they last, we avoid looking at every line by
    # only looking at the indexes where the patch value changes:
    which_patch_array = np.array(which_patch_list)
    event_indexes = np.where(which_patch_array[:-1] != which_patch_array[1:])[0]
    # (this formula works by looking at differences between the list shifted by one to the left or to the right)
    # (for [1,1,2,2,3,3] it will return [1,3])

    # We go through every event
    patch_where_it_is = which_patch_list[0]  # initializing variable with index of patch where the worm currently is
    for time in event_indexes:
        patch_where_it_was = patch_where_it_is  # index of the patch where it is
        patch_where_it_is = which_patch_list[time]

        # Worm just exited a patch
        if patch_where_it_is == -1:  # worm currently out
            if patch_where_it_was != patch_where_it_is:  # was inside before
                list_of_timestamps[patch_where_it_was][-1][1] = list_of_frames[time]  # end the previous visit
                list_of_timestamps[patch_where_it_was].append([0, 0])  # add new visit to the previous patch

        # Worm just entered a patch
        if patch_where_it_is != -1:  # worm currently in
            if order_of_visits[-1] == -1:  # was outside before
                order_of_visits[-1] = patch_where_it_is  # add this patch to the visit order
                list_of_timestamps[patch_where_it_is][-1][0] = list_of_frames[time]  # begin visit in current patch sublist
                if len(order_of_visits) >= 2 and order_of_visits[-1] != order_of_visits[-2]:  # if it's not the same patch as the previous visit
                    adjusted_list_of_timestamps[patch_where_it_is][-1][1] = list_of_frames[time]  # end the old visit in the previous patch
                    adjusted_list_of_timestamps[order_of_visits[-2]].append([0, 0])  # start a new visit in the previous patch

    duration_sum = 0  # this is to compute the avg duration of visits
    nb_of_visits = 0
    adjusted_duration_sum = 0
    adjusted_nb_of_visits = 0
    list_of_visited_patches = []
    furthest_patch_distance = 0
    furthest_patch_position = [0, 0]
    list_of_transit_durations = []

    # Run through each patch to compute global variables
    for i_patch in range(len(list_of_timestamps)):
        # Remove the zeros because they're just here for the duration algorithm
        list_of_timestamps[i_patch] = [nonzero for nonzero in list_of_timestamps[i_patch] if nonzero != [0,0]]
        adjusted_list_of_timestamps[i_patch] = [nonzero for nonzero in adjusted_list_of_timestamps[i_patch] if nonzero != [0, 0]]

        # Update list of visited patches and the furthest patch visited
        if len(list_of_timestamps[i_patch]) > 0:  # if the patch was visited at least once in this trajectory
            patch_distance_to_center = distance.euclidean(first_xy, patch_centers[i_patch])
            if patch_distance_to_center > furthest_patch_distance:
                furthest_patch_position = patch_centers[i_patch]
                furthest_patch_distance = distance.euclidean(first_xy, furthest_patch_position)
            list_of_visited_patches.append(i_patch)

        # Visits info for average visit duration
        duration_sum += sum(list_of_timestamps[i_patch])
        nb_of_visits += len(list_of_timestamps[i_patch])

        # Same but adjusted for multiple consecutive visits to same patch
        adjusted_duration_sum += sum(adjusted_list_of_timestamps[i_patch])
        adjusted_nb_of_visits += len(adjusted_list_of_timestamps[i_patch])

    total_transit_time = np.sum(list_of_transit_durations)

    return list_of_timestamps, order_of_visits, duration_sum, nb_of_visits, list_of_visited_patches, furthest_patch_position, total_transit_time, adjusted_list_of_timestamps, adjusted_duration_sum, adjusted_nb_of_visits


def make_results_table(data):
    """
    Takes our data table and returns a series of analysis regarding patch visits, one line per worm
    """
    global first_pos
    worm_list = np.unique(data["id_conservative"])
    nb_of_worms = len(worm_list)

    results_table = pd.DataFrame()
    results_table["folder"] = [-1 for i in range(nb_of_worms)]
    results_table["condition"] = [-1 for i in range(nb_of_worms)]
    results_table["worm_id"] = [-1 for i in range(nb_of_worms)]
    results_table["total_time"] = [-1 for i in range(nb_of_worms)]
    results_table["raw_visits"] = [-1 for i in range(nb_of_worms)]
    results_table["order_of_visits"] = [-1 for i in range(nb_of_worms)]
    results_table["duration_sum"] = [-1 for i in range(nb_of_worms)]
    results_table["nb_of_visits"] = [-1 for i in range(nb_of_worms)]
    results_table["list_of_visited_patches"] = [-1 for i in range(nb_of_worms)]
    results_table["first_recorded_position"] = [-1 for i in range(nb_of_worms)]
    results_table["furthest_patch_position"] = [-1 for i in range(nb_of_worms)]
    results_table["total_transit_time"] = [-1 for i in range(nb_of_worms)]
    results_table["adjusted_raw_visits"] = [-1 for i in range(nb_of_worms)]  # consecutive visits to same patch= 1 visit
    results_table["adjusted_duration_sum"] = [-1 for i in range(nb_of_worms)]  # THIS SHOULD BE THE SAME AS DURATION SUM
    results_table["adjusted_nb_of_visits"] = [-1 for i in range(nb_of_worms)]

    old_folder = "caca"
    for i_worm in range(nb_of_worms):
        # Handmade progress bar
        print(i_worm, "/", nb_of_worms)

        # Data from the dataframe
        current_worm = worm_list[i_worm]
        current_data = data[data["id_conservative"] == current_worm]
        current_list_x = current_data["x"]
        current_list_y = current_data["y"]
        current_folder = list(current_data["folder"])[0]

        # First recorded position of each plate is first position of the first worm of the plate
        if current_folder != old_folder:
            first_pos = [current_data["x"][0], current_data["y"][0]]
        old_folder = current_folder

        # Getting to the metadata through the folder name in the data
        current_metadata = fd.folder_to_metadata(current_folder)
        list_of_densities = current_metadata["patch_densities"]

        # Computing the visit durations
        raw_durations, order_of_visits, duration_sum, nb_of_visits, list_of_visited_patches, furthest_patch_position, \
            total_transit_time, adjusted_raw_visits, adjusted_duration_sum, adjusted_nb_of_visits = single_traj_analysis(
            current_data["patches"], current_data["frame"], current_metadata["patch_centers"], first_pos)

        # Fill up results table
        results_table.loc[i_worm, "folder"] = current_folder
        results_table.loc[i_worm, "condition"] = current_metadata["condition"][0]
        results_table.loc[i_worm, "worm_id"] = current_worm
        results_table.loc[i_worm, "total_time"] = len(current_list_x)
        results_table.loc[i_worm, "raw_visits"] = str(raw_durations)  # all visits of all patches
        results_table.loc[i_worm, "order_of_visits"] = str(order_of_visits)  # patch order of visits
        results_table.loc[i_worm, "duration_sum"] = duration_sum  # total duration of visits
        results_table.loc[i_worm, "nb_of_visits"] = nb_of_visits  # total nb of visits
        results_table.loc[i_worm, "list_of_visited_patches"] = str(list_of_visited_patches)  # index of patches visited
        results_table.loc[i_worm, "first_recorded_position"] = str(first_pos)
        results_table.loc[i_worm, "furthest_patch_position"] = str(furthest_patch_position)  # distance
        results_table.loc[i_worm, "total_transit_time"] = total_transit_time
        results_table.loc[i_worm, "adjusted_raw_visits"] = str(adjusted_raw_visits)
        results_table.loc[i_worm, "adjusted_duration_sum"] = adjusted_duration_sum
        results_table.loc[i_worm, "adjusted_nb_of_visits"] = adjusted_nb_of_visits

    return results_table


def generate_and_save(path):
    trajectories = fd.trajmat_to_dataframe(fd.path_finding_traj(path))  # run this to retrieve trajectories
    print("Finished retrieving trajectories")
    print("Computing distances...")
    trajectories["distances"] = speed_analysis(trajectories)
    print("Finished computing distance covered by the worm at each time step")
    print("Computing where the worm is...")
    trajectories["patch"] = in_patch_list(trajectories)
    print("Finished computing in which patch the worm is at each time step")
    trajectories.to_csv(path + "trajectories.csv")
    print("Starting to build results from trajectories...")
    results = make_results_table(trajectories)
    print("Finished!")
    results.to_csv(path + "results.csv")
    return 0
