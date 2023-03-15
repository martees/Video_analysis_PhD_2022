import find_data as fd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import pandas as pd
from param import *
from numba import njit


def in_patch(position, patch):
    """
    returns True if position = [x,y] is inside the patch
    uses general parameter radial_tolerance: the worm is still considered inside the patch when its center is sticking out by that distance or less
    """
    center = [patch[0], patch[1]]
    radius = patch_radius
    distance_from_center = np.sqrt((position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2)
    return distance_from_center < radius + radial_tolerance


# @njit(parallel=True)
def patch_visits_single_traj(list_x, list_y, first_pos, patch_centers):
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

    # In list_of_durations, we will have one list per patch, containing the duration of successive visits to that patch
    # Zero means "the worm was not in that patch in the previous timestep"
    # As soon as the worm enters the patch, this zero starts being incremented
    # As soon as the worm leaves the patch, a new zero is added to this patch's list
    # These 0 are added for computational ease and will be removed in the end
    list_of_durations = [list(i) for i in
                         np.zeros((len(patch_centers), 1), dtype='int')]  # List with the right format [[0],[0],...,[0]]

    # Order in which the patches were visited (should have as many elements as list_of_durations)
    # In order_of_visits: if last element is -1, worm was out in the previous timestep
    # otherwise, it means it was already in that patch in the previous timestep, so it's not a new visit
    order_of_visits = [-1]

    # List of transit durations, to compute average transit time between patches
    list_of_transit_durations = [0]

    # This is the list of durations we will use to test the Marginal Value Theorem
    # In order to do so, if the worm visits the same patch multiple times in a row, we count that as one visit
    # The encoding is the same as list_of_durations, 0 = worm was not in that patch in its previous visit
    adjusted_list_of_durations = [list(i) for i in
                                  np.zeros((len(patch_centers), 1),
                                           dtype='int')]  # List with the right format [[0],[0],...,[0]]

    patch_where_it_is = -1  # initializing variable with index of patch where the worm currently is
    # We go through the whole trajectory
    for time in range(len(list_x)):
        patch_where_it_was = patch_where_it_is  # index of the patch where it is
        patch_where_it_is = -1  # resetting the variable of current patch position
        for i_patch in range(len(patch_centers)):  # for every patch
            is_in_patch[i_patch] = in_patch([list_x[time], list_y[time]],
                                            patch_centers[i_patch])  # check if the worm is in it
            if is_in_patch[i_patch]:  # if it's in it, keep that in mind
                patch_where_it_is = i_patch

        # Worm currently out
        if patch_where_it_is == -1:
            list_of_transit_durations[-1] += 1  # add 1 to current transit duration
            if patch_where_it_was != patch_where_it_is:  # Worm just exited a patch
                list_of_durations[patch_where_it_was].append(0)  # Add a zero because previous visit was interrupted
                order_of_visits.append(-1)

        # Worm currently inside, no matter whether it just entered or stayed inside
        if patch_where_it_is != -1:
            if order_of_visits[-1] == -1:  # if the worm just entered the patch
                order_of_visits[-1] = patch_where_it_is  # add this patch to the visit order
                list_of_transit_durations.append(0)  # start a new transit
                if len(order_of_visits) > 2 and order_of_visits[-1] != order_of_visits[
                    -2]:  # if it's not the same patch as the previous visit
                    adjusted_list_of_durations[order_of_visits[-2]].append(0)  # start a new visit in the previous patch
            list_of_durations[patch_where_it_is][-1] += 1  # add one to the last element of the current patch sublist
            adjusted_list_of_durations[patch_where_it_is][-1] += 1  # same for adjusted

    duration_sum = 0  # this is to compute the avg duration of visits
    nb_of_visits = 0
    adjusted_duration_sum = 0
    adjusted_nb_of_visits = 0
    list_of_visited_patches = []
    furthest_patch_distance = 0
    furthest_patch_position = [0, 0]

    # Run through each patch to compute global variables
    for i_patch in range(len(list_of_durations)):
        # Remove the zeros because they're just here for the duration algorithm
        list_of_durations[i_patch] = [nonzero for nonzero in list_of_durations[i_patch] if nonzero != 0]
        adjusted_list_of_durations[i_patch] = [nonzero for nonzero in adjusted_list_of_durations[i_patch] if
                                               nonzero != 0]

        # Update list of visited patches and the furthest patch visited
        if len(list_of_durations[i_patch]) > 0:  # if the patch was visited at least once in this trajectory
            patch_distance_to_center = distance.euclidean(first_pos, patch_centers[i_patch])
            if patch_distance_to_center > furthest_patch_distance:
                furthest_patch_position = patch_centers[i_patch]
                furthest_patch_distance = distance.euclidean(first_pos, furthest_patch_position)
            list_of_visited_patches.append(i_patch)

        # Visits info for average visit duration
        duration_sum += sum(list_of_durations[i_patch])
        nb_of_visits += len(list_of_durations[i_patch])

        # Same but adjusted for multiple consecutive visits to same patch
        adjusted_duration_sum += sum(adjusted_list_of_durations[i_patch])
        adjusted_nb_of_visits += len(adjusted_list_of_durations[i_patch])

    total_transit_time = np.sum(list_of_transit_durations)

    return list_of_durations, order_of_visits, duration_sum, nb_of_visits, list_of_visited_patches, furthest_patch_position, total_transit_time, adjusted_list_of_durations, adjusted_duration_sum, adjusted_nb_of_visits


def patch_visits_multiple_traj(data):
    """
    Takes our data table and returns a series of analysis regarding patch visits, one line per worm
    """
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
            total_transit_time, adjusted_raw_visits, adjusted_duration_sum, adjusted_nb_of_visits = patch_visits_single_traj(
            list(current_list_x), list(current_list_y), first_pos, current_metadata["patch_centers"])

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


def speed_analysis(traj):
    """
    Function that takes in our trajectories dataframe, and adds a column with the distance covered by the worm since
    last timestep, in order to compute speed
    """
    list_of_positions = pd.DataFrame(zip(traj["x"], traj["y"]))
    list_of_distances = list(list_of_positions.apply(lambda x, y: distance.euclidean(x, y)))
    traj["distances"] = list_of_distances
    return traj


def generate_and_save(path):
    trajectories = fd.trajmat_to_dataframe(fd.path_finding_traj(path))  # run this to retrieve trajectories
    trajectories = speed_analysis(trajectories)
    trajectories.to_csv(path + "trajectories.csv")
    results = patch_visits_multiple_traj(trajectories)
    results.to_csv(path + "results.csv")
    return 0
