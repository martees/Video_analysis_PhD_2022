# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from datetime import datetime, date, time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

#My code
import find_data as fd

def in_patch(position, patch):
    """
    returns True if position = [x,y] is inside the patch
    uses general parameter radial_tolerance: the worm is still considered inside the patch when its center is sticking out by that distance or less
    """
    center = [patch[0],patch[1]]
    radius = patch_radius
    distance = np.sqrt((position[0]-center[0])**2 + (position[1]-center[1])**2)
    return distance < radius + radial_tolerance

def patch_visits_single_traj(list_x, list_y, patch_centers):
    """
    Takes a trajectory under the format: [x0 x1 ... xN] [y0 y1 ... yN] and a list of patch centers
    For now, uses the unique patch_radius defined as a global parameter.
    Returns a list [[d0,d1,...], [d0,d1,...],...] with one list per patch
    each sublist contains the durations of visits to each patch
    so len(output[0]) is the number of visits to the first patch
    """

    #Variables for the loop and the output
    is_in_patch = np.zeros(len(patch_centers)) # Bool list to store where the worm is currently (1 where it currently is)
    list_of_durations = [list(i) for i in np.zeros((len(patch_centers),1), dtype='int')] # List with the right format [[0],[0],...,[0]]
    # In list_of_durations, zero means "the worm was not in the patch in the previous timestep"
    # As soon as the worm enters the patch, this zero starts being incremented
    # As soon as the worm leaves the patch, a new zero is added to this patch's list
    # These 0 are added for computational ease and will be removed in the end
    patch_where_it_is = -1 #initializing variable with index of patch where the worm currently is

    # We go through the whole trajectory
    for time in range(len(list_x)):
        was_in_patch = is_in_patch # keeping in memory where the worm was, [0, 0, 1, 0] = in patch 2
        patch_where_it_was = patch_where_it_is # index of the patch where it is
        patch_where_it_is = -1 # resetting the variable
        for i_patch in range(len(patch_centers)): #for every patch
            is_in_patch[i_patch] = in_patch([list_x[time],list_y[time]], patch_centers[i_patch]) #check if the worm is in
            if is_in_patch[i_patch] == True: #remember where it is
                patch_where_it_is = i_patch
        if patch_where_it_is==-1: # Worm currently out
            if patch_where_it_was != patch_where_it_is:  # Worm exited a patch
                list_of_durations[patch_where_it_was].append(0) # Add a zero because previous visit was interrupted
        if patch_where_it_is!=-1: # Worm currently inside, no matter whether it just entered or stayed inside
            list_of_durations[patch_where_it_is][-1]+=1 #add one to the last element of the patch list

    duration_sum = 0  # this is to compute the avg duration of visits
    nb_of_visits = 0
    first_recorded_worm_position = [list_x[0],list_y[0]]
    furthest_patch_distance = 0
    # Remove the zeros because they're just here for the duration algorithm
    # in the same loop we compute the average visit duration, and the furthest visited patch
    for i_patch in range(len(list_of_durations)):
        list_of_durations[i_patch] = [nonzero for nonzero in list_of_durations[i_patch] if nonzero != 0]

        if len(list_of_durations[i_patch])>0: #if the patch was visited at least once
            patch_distance_to_center = distance.euclidean(first_recorded_worm_position,patch_centers[i_patch])
            furthest_patch_distance = max(patch_distance_to_center, furthest_patch_distance)

        duration_sum += sum(list_of_durations[i_patch])
        nb_of_visits += len(list_of_durations[i_patch])
    if nb_of_visits != 0:
        avg_duration =  duration_sum/nb_of_visits
    else:
        avg_duration = 0

    return list_of_durations, avg_duration, furthest_patch_distance

def patch_visits_multiple_traj(data):
    """
    (tldr: returns a list of outputs from the single_traj function, one list item per trajectory)
    Takes our data table and returns a series of analysis regarding patch visits durations
    """
    worm_list = np.unique(data["id_conservative"])
    nb_of_worms = len(worm_list)

    results_table = pd.DataFrame()
    results_table["condition"] = [-1 for i in range(nb_of_worms)]
    results_table["worm_id"] = [-1 for i in range(nb_of_worms)]
    results_table["raw_visits"] = [-1 for i in range(nb_of_worms)]
    results_table["avg_visit_duration"] = [-1 for i in range(nb_of_worms)]
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
        raw_durations, avg_duration, furthest_patch_distance = patch_visits_single_traj(list(current_list_x), list(current_list_y), current_metadata["patch_centers"])

        # Fill up results table
        results_table.loc[i_worm, "condition"] = current_metadata["condition"][0]
        results_table.loc[i_worm, "worm_id"] = current_worm
        results_table.loc[i_worm, "raw_visits"] = pd.DataFrame(raw_durations)  # all visits of all patches
        results_table.loc[i_worm, "avg_visit_duration"] = avg_duration  # all visits of all patches
        results_table.loc[i_worm, "furthest_patch_distance"] = furthest_patch_distance  # all visits of all patches

    return results_table


def traj_draw(data):
    """
    Function that takes in our dataframe, using columns: "x", "y", "id_conservative"
    Extracts list of series of positions and draws them, with one color per id
    :param trajectory: list of series of (x,y) positions ([[x0,x1,x2...] [y0,y1,y2...])
    :return: trajectory plot
    """
    worm_list = np.unique(data["id_conservative"])
    nb_of_worms = len(worm_list)
    colors = plt.cm.jet(np.linspace(0, 1, nb_of_worms))
    for i_worm in range(nb_of_worms):
        current_worm = worm_list[i_worm]
        current_list_x = data[data["id_conservative"]==current_worm]["x"]
        current_list_y = data[data["id_conservative"]==current_worm]["y"]
        plt.plot(current_list_x,current_list_y,color=colors[i_worm])
    # for i_traj in range(len(trajectories)):
    #     reformatted_trajectory = list(zip(*trajectories[i_traj])) # converting from [x y][x y][x y] format to [x x x] [y y y]
    #     plt.plot(reformatted_trajectory[0],reformatted_trajectory[1])
    plt.show()

def landscape_draw(patch_centers):
    """
    Function that draws the patches
    """

    return 0

def avg_duration_per_condition(result_table):
    list_of_conditions = np.unique(result_table["condition"])
    list_of_avg = np.zeros(len(list_of_conditions))
    sum = 0
    nb = 0
    for i_condition in range(len(list_of_conditions)):
        condition = list_of_conditions[i_condition]
        current = result_table[result_table["condition"] == condition]["avg_visit_duration"]
        sum = np.sum(current)
        nb = len(current)
        list_of_avg[i_condition] = sum/nb
    return list_of_avg

# Parameters
radial_tolerance = 0.1
patch_radius = 20

# Stuff for tests
fake_patch1 = [[1400, 1200], 100]  # [[x,y], radius] with x y = position of the center
fake_patch2 = [[500, 1000], 30]  # [[x,y], radius] with x y = position of the center
patch_list = [fake_patch1, fake_patch2]

# Function tests

#Extracting data, the function looks for all "traj.mat" files in the indicated path (will look into subfolders)
#I have two lines, one for Windows and the other for Linux:
# dataframe = fd.trajmat_to_dataframe(fd.path_finding_traj("C:/Users/Asmar/Desktop/Thèse/2022_summer_videos"))
dataframe = fd.trajmat_to_dataframe(fd.path_finding_traj("/home/admin/Desktop/Camera_setup_analysis/"))


# trajectories = fd.reformat_trajectories(dataframe["trajectories"])

# traj_draw(dataframe)
results = patch_visits_multiple_traj(dataframe)
print(avg_duration_per_condition(results))

plt.bar(results["worm_id"], results["avg_visit_duration"])
plt.show()