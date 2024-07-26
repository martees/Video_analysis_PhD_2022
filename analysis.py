import numpy as np
import random
from scipy import stats
import time
import copy
import os
import pandas as pd
from itertools import groupby

# My code
from Parameters import parameters as param
import find_data as fd
from Scripts_models import s2023_mvt_null_revisits as model
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt


def r2(x, y):
    """
    R squared for linear regression between x and y vectors
    """
    return stats.pearsonr(x, y)[0] ** 2


def distance(xy0, xy1):
    """
    Returns the Euclidean distance between the two points xy0 and xy1.
    xy0 and xy1 are two lists containing x and y for the two points.
    """
    return np.sqrt((xy0[0] - xy1[0]) ** 2 + (xy0[1] - xy1[1]) ** 2)


def random_sample(array_1D):
    # This returns a list of random samples from array_1D (same size as the original array)
    # It does so by applying random_choice to a table with the values in array_1D repeated to make a square array
    return np.apply_along_axis(random.choices, 0, array_1D, k=len(array_1D))


def bottestrop_ci(data, nb_resample, operation="mean"):
    """
    Function that takes a dataset and returns a confidence interval using nb_resample samples for bootstrapping
    """
    # This returns a list of random samples from data (nb_resample lines, with each as many elements as data)
    random_samples = np.apply_along_axis(random_sample, 1, np.array([data] * nb_resample))
    # Depending on the statistic wanted, do a different operation
    if operation == "mean":
        bootstrapped_stat = np.apply_along_axis(np.nanmean, 1, random_samples)
    if operation == "leaving_probability":
        bootstrapped_stat = np.apply_along_axis(compute_leaving_probability, 0, random_samples)
    bootstrapped_stat.sort()
    return [np.percentile(bootstrapped_stat, 5), np.percentile(bootstrapped_stat, 95)]


def results_per_condition(result_table, list_of_conditions, column_name, divided_by="",
                          normalize_by_video_length=False):
    """
    Function that takes our result table, a list of conditions, and a column name (as a string)
    Returns the list of values of that column pooled by condition, a list of the average value for each condition, and a
    bootstrap confidence interval for each value.
    Can take in a third argument, column name by which you want to divide the main column, plate by plate
    eg: divide duration sum by nb of visits for each plate to get average visit duration for each plate
    """

    # Full list (to be filled)
    full_list_of_values = [list(i) for i in np.zeros((len(list_of_conditions), 1), dtype='int')]

    # List of average (to be filled)
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

            # When we want to divide column name by another one
            if divided_by != "":
                if divided_by == "nb_of_patches":
                    list_total_patch = list(param.nb_to_nb_of_patches.values())  # total nb of patches for each cond
                    list_of_values[i_plate] = np.sum(current_plate[column_name]) / list_total_patch[i_condition]
                elif divided_by == "nb_of_visited_patches":
                    current_plate = current_plate.reset_index()
                    visited_patches_list = list_of_visited_patches(fd.load_list(current_plate, "no_hole_visits"))
                    list_of_values[i_plate] = np.sum(current_plate[column_name]) / max(1, len(visited_patches_list))
                elif np.sum(current_plate[divided_by]) != 0:  # Non zero check for division
                    if np.sum(current_plate[column_name]) < 0:
                        print("Negative ", column_name, " for plate: ", current_plate["folder"].iloc[0])
                    list_of_values[i_plate] = np.sum(current_plate[column_name]) / np.sum(current_plate[divided_by])
                else:
                    print("Trying to divide by 0 in plot_selected_data for plate" + str(
                        list_of_plates[i_plate]) + "... what a shame")

            # When no division has to be made
            else:
                if column_name == "average_speed_inside" or column_name == "average_speed_outside":
                    # Exclude the 0's which are the cases were the worm didn't go to a patch / out of a patch for a full track
                    list_speed_current_plate = [nonzero for nonzero in current_plate[column_name] if int(nonzero) != 0]
                    if list_speed_current_plate:  # If any non-zero speed was recorded for that plate
                        list_of_values[i_plate] = np.average(list_speed_current_plate)
                elif column_name == "proportion_of_visited_patches" or column_name == "nb_of_visited_patches":  # Special case: divide by total nb of patches in plate
                    current_plate = current_plate.reset_index()
                    visited_patches_list = list_of_visited_patches(fd.load_list(current_plate, "no_hole_visits"))
                    if column_name == "nb_of_visited_patches":
                        list_of_values[i_plate] = len(visited_patches_list)
                    else:
                        list_total_patch = list(param.nb_to_nb_of_patches.values())  # total nb of patches for each cond
                        list_of_values[i_plate] = len(np.unique(visited_patches_list)) / list_total_patch[
                            i_condition]
                elif column_name == "furthest_patch_distance":  # in this case we want the maximal value and not the average
                    list_of_values[i_plate] = np.max(current_plate[column_name])
                else:  # in any other case
                    list_of_values[i_plate] = np.sum(current_plate[column_name])

            if normalize_by_video_length:
                list_of_values[i_plate] /= current_plate["total_tracked_time"]

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

    return full_list_of_values, list_of_avg_values, [list(errors_inf), list(errors_sup)]


def model_per_condition(result_table, list_of_conditions, column_name, divided_by=""):
    """
    Function that should return model predictions for column name values (divided by divided_by values), pooled for
    each condition in condition_list.
    """
    revisit_probability, _, _, _, _, average_same_patch, average_cross_patch = transit_properties(result_table,
                                                                                                  list_of_conditions,
                                                                                                  split_conditions=True)

    if column_name == "total_visit_time" and (divided_by == "nb_of_visits" or divided_by == "mvt_nb_of_visits"):
        model_avg_visit_length_list = np.zeros(len(list_of_conditions))
        for i_condition in range(len(list_of_conditions)):
            t1 = int(average_cross_patch[i_condition])
            t2 = int(average_same_patch[i_condition])
            t_min = 0
            p_rev = revisit_probability[i_condition]
            time_constant = model.tauu

            parameter_list = [t1, t2, t_min, p_rev, time_constant]

            resolution = model.nb_of_leaving_rates_to_test
            nb_of_time_steps = model.nb_of_time_steps_per_leaving_rate

            model_avg_visit_length_list[i_condition] = model.opt_mvt("", resolution, nb_of_time_steps, 1,
                                                                     parameter_list, is_plot=False, is_print=False)

        return model_avg_visit_length_list
    else:
        print("We have no model for that yet darling.")
        return np.zeros(len(list_of_conditions))


def visit_time_as_a_function_of(results, traj, condition_list, variable, patch_or_pixel="patch", only_first_visit=True):
    """
    Takes a results and trajectories table, a condition list, and a variable. Returns a list of visit durations and the
    corresponding values of variable.
    NOTE: for variable="last_travel_time", the only parameter values supported for now are patch_or_pixel="patch" and
          only_first_visit = True!!!
    @param results: loaded from clean_results.csv (see readme)
    @param traj: loaded from clean_trajectories.csv (see readme)
    @param condition_list: a list of numbers corresponding to condition in the results.csv (see /Parameters/parameters.py)
    @param variable: the variable that you want to compare visit durations with.
           "last_travel_time" = duration of the closest previous transit between patches
           "visit_start" = point in the video where the visit happens (to see if later visits are shorter for example)
           "speed_when_entering" = worm speed when the visit starts
    @param patch_or_pixel: if set to "patch", the function will return a list of patch-level visit durations
                           if set to "pixel", the function will return a list of pixel-level visit durations
    @param only_first_visit: if set to True, only look at first visit to a patch / to a pixel
    """

    # Fill the folder list up (list of folders corresponding to the conditions in condition_list)
    folder_list = fd.return_folders_condition_list(np.unique(results["folder"]), condition_list)

    # Initialize the lists that will be returned
    full_visit_list = [[] for _ in range(len(condition_list))]
    full_variable_list = [[] for _ in range(len(condition_list))]

    # Fill up the lists depending on the variable specified as an argument
    if variable == "last_travel_time":
        # TODO if we ever use this again, switch it to a much simpler version where we fuse visit and transit lists, sort by time, and then look at each element
        starts_with_visit = False
        for i_folder in range(len(folder_list)):
            # Initialization
            # Slice to one plate
            current_plate = results[results["folder"] == folder_list[i_folder]].reset_index()
            # Visit and transit lists
            list_of_visits = fd.load_list(current_plate, "no_hole_visits")
            list_of_transits = fd.load_list(current_plate, "aggregated_raw_transits")
            # Lists that we'll fill up for this plate
            list_of_visit_lengths = []
            list_of_previous_transit_lengths = []

            if list_of_visits and list_of_transits:  # if there's at least one visit and one transit
                last_tracked_frame = max(list_of_visits[-1][1], list_of_transits[-1][1])  # for later computations
                # Check whether the plate starts and ends with a visit or a transit
                if list_of_visits[0][0] < list_of_transits[0][0]:
                    starts_with_visit = True
                # If it starts with a visit we only start at visit 1 (visit 0 has no previous transit)
                i_visit = int(starts_with_visit)
                # If there are consecutive visits/transits, we count them to still look at temporally consecutive visits and transits
                double_transits = 0
                double_visits = 0
                while i_visit < len(list_of_visits):
                    current_visit = list_of_visits[i_visit]
                    # Index to find the previous transit
                    # When the video starts with a visit, visit 1 has to be compared to transit 0 (True = 1 in Python)
                    # Otherwise, visit 0 has to be compared to transit 0
                    # We remove double_visits because successive visits increase i_visit without going through transits
                    # We add double_transits to account for multiple successive transits that don't go through the visits
                    i_transit = i_visit - double_visits + double_transits - starts_with_visit
                    current_transit = list_of_transits[i_transit]

                    # Debugging shit
                    if param.verbose:
                        print(current_plate["folder"][0])
                        print("Nb of visits = ", len(list_of_visits), ", nb of transits = ", len(list_of_transits),
                              ", i_visit = ", i_visit, "starts_with = ", starts_with_visit)
                        print("double_transits = ", double_transits, ", double_visits = ", double_visits)
                        print("current transit : [", current_transit[0], ", ", current_transit[1], "]")
                        print("current visit : [", current_visit[0], ", ", current_visit[1], "]")

                    # First start a new entry in the duration lists
                    list_of_previous_transit_lengths.append(current_transit[1] - current_transit[0] + 1)
                    list_of_visit_lengths.append(current_visit[1] - current_visit[0] + 1)

                    # Then check that we're comparing the right visit(s) with the right transit(s)
                    # (because there can be multiple consecutive transits w/o visits and vice versa)

                    # If the current visit doesn't start when the current transit ends it means that there are extra transits
                    if current_visit[0] > current_transit[1]:
                        # Take care of any extra transit that's in the way
                        # This transit exists if the current visit starts after the current transit ends
                        while current_visit[0] > current_transit[1] and i_transit + 1 < len(list_of_transits):
                            # Compute next transit
                            current_transit = list_of_transits[i_transit + 1]
                            if param.verbose:
                                print("additional transit : [", current_transit[0], ", ", current_transit[1], "]")
                            # IF this next transit ends before the current visit starts
                            # THEN it means there's really a double transit (otherwise it means there's a hole in the tracking)
                            if current_visit[0] > current_transit[1]:
                                i_transit += 1  # we add one to this counter to remember there was a double transit for this loop only
                                double_transits += 1  # we add one to this counter to remember there was a double transit next time we update i_transit
                                # We add it to the previous transit length
                                list_of_previous_transit_lengths[-1] += current_transit[1] - current_transit[0] + 1

                    # Now let's check if we have all the consecutive visits before the next transit
                    next_transit_start = 0
                    if i_visit + double_transits - starts_with_visit + 1 < len(list_of_transits):
                        next_transit_start = list_of_transits[i_transit + 1][0]
                    # While it is not the last visit, and it doesn't either end with the video or ends at the same time as the next transit
                    while i_visit + 1 < len(list_of_visits) and not (
                            current_visit[1] == last_tracked_frame or current_visit[1] >= next_transit_start):
                        # There are no transits before this next visit, so we account for it here
                        i_visit += 1
                        double_visits += 1
                        current_visit = list_of_visits[i_visit]
                        if param.verbose:
                            print("additional visit : [", current_visit[0], ", ", current_visit[1], "]")
                        # We add this extra transit to the previous transit length
                        list_of_visit_lengths[-1] += current_visit[1] - current_visit[0] + 1

                    # Go to next visit!
                    i_visit += 1

            condition = fd.load_condition(folder_list[i_folder])
            i_condition = condition_list.index(condition)  # for the condition-label correspondence we need the index
            # plt.scatter(list_of_previous_transit_lengths, list_of_visit_lengths, color=colors[i_condition], label=str(condition_names[i_condition]), zorder=i_condition)

            # For plotting
            full_visit_list[i_condition] += list_of_visit_lengths
            full_variable_list[i_condition] += list_of_previous_transit_lengths

    else:
        for i_folder, folder in enumerate(folder_list):
            time_start = time.time()
            # Initialization
            # Information about condition
            condition = fd.load_condition(folder_list[i_folder])
            i_condition = condition_list.index(condition)  # for the condition-label correspondence we need the index
            current_traj = traj[traj["folder"] == folder].reset_index()

            if patch_or_pixel == "patch":
                # Slice to one plate
                current_plate = results[results["folder"] == folder].reset_index()
                # Visit list
                list_of_visits = fd.load_list(current_plate, "no_hole_visits")
                # If only first visits, keep track of already visited patches (simpler than triaging the visit list)
                if only_first_visit:
                    already_visited = []

            if patch_or_pixel == "pixel":
                # If it's not already done, compute the pixel visit durations
                pixelwise_durations_path = folder[:-len("traj.csv")] + "pixelwise_visits.npy"
                if not os.path.isfile(pixelwise_durations_path):
                    gr.generate_pixelwise_visits(traj, folder)
                # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
                matrix_of_visits = np.load(pixelwise_durations_path, allow_pickle=True)
                # Load patch info for this folder
                in_patch_matrix_path = folder[:-len("traj.csv")] + "in_patch_matrix.csv"
                if not os.path.isfile(in_patch_matrix_path):
                    gt.in_patch_all_pixels(folder)
                in_patch_matrix = pd.read_csv(in_patch_matrix_path)
                # Separate inside / outside food patch visit durations (this transforms the matrix into a 1 line array)
                matrix_of_visits = matrix_of_visits[in_patch_matrix != -1]
                # Remove the matrix structure, and add a unique identifier to every pixel
                # So we go from a matrix containing [[t0, tf], ...] for every pixel (t0 = visit start, tf = visit end)
                # to a plain list with [[t0, tf, idx], [t0, tf, idx], ...] where idx is an index corresponding to the pixel
                if only_first_visit:
                    list_of_visits = [matrix_of_visits[i][v] + [i]
                                      for i in range(len(matrix_of_visits))
                                      for v in range(min(1, len(matrix_of_visits[i])))]
                else:
                    list_of_visits = [matrix_of_visits[i][v] + [i]
                                      for i in range(len(matrix_of_visits))
                                      for v in range(len(matrix_of_visits[i]))]

            # Same code for both pixel and patch level visits
            for i_visit in range(len(list_of_visits)):
                current_visit = list_of_visits[i_visit]
                if current_visit[1] < current_visit[0]:
                    print("oulah")
                # Only do the following if we're not in the case where we should take only 1st visit to patches,
                # and the current patch has already been visited
                if not (patch_or_pixel == "patch" and only_first_visit and current_visit[2] in already_visited):
                    full_visit_list[i_condition].append(current_visit[1] - current_visit[0] + 1)
                    if patch_or_pixel == "patch" and only_first_visit:
                        already_visited.append(current_visit[2])
                    if variable == "visit_start":
                        full_variable_list[i_condition].append(current_visit[0])
                    if variable == "speed_when_entering":
                        visit_start = current_traj[current_traj["frame"] == current_visit[0]].reset_index()
                        if len(visit_start) == 0:
                            print("gey")
                        speed_when_entering = visit_start["speeds"][0]
                        full_variable_list[i_condition].append(speed_when_entering)

            print("It took ", int(time.time() - time_start), " sec to analyse plate ", i_folder, " / ",
                  len(folder_list))

    return full_visit_list, full_variable_list


def convert_to_durations(list_of_time_stamps):
    """
    Function that takes a list of timestamps in the format [[t0,t1,...],[t2,t3,...],...] (uses t0 and t1 only)
    And will return the corresponding list of durations [d0,d1,...] (where d0 = t1 - t0 + 1)
    """
    # Equivalent imperative code:
    #nb_of_events = len(list_of_time_stamps)
    #list_of_durations = np.zeros(nb_of_events)
    #for i_event in range(nb_of_events):
    #    list_of_durations[i_event] = list_of_time_stamps[i_event][1] - list_of_time_stamps[i_event][0]
    # Code using lambda function instead:
    if list_of_time_stamps:
        return list(np.apply_along_axis(lambda x: x[1] - x[0] + 1, 1, list_of_time_stamps))
    else:  # If there are no time stamps
        return []


def select_transits(list_of_transits, list_of_visits, to_same_patch=False, to_different_patch=False):
    """
    Function that will take a transit and visit list, and return a new transit list following some property.
        to_same_patch: will return transits that go from a patch to itself
        to_different_patch: will return transits that go from a patch to another
    """
    # Event structure = [x,y,z] with x start of event, y end of event and z patch where the worm is (-1 for outside)
    # Sometimes there are multiple successive visits / transits, so I use while loop to go through them.

    # Fuse the two lists, and sort them depending on the time when they begin
    list_of_events = list_of_transits + list_of_visits
    list_of_events.sort(key=lambda x: x[0])
    new_list_of_transits = []
    current_transit_index = 0

    # If the list starts with a transit, skip it
    if list_of_events[0][2] == -1:
        while list_of_events[current_transit_index][2] == -1 and current_transit_index < len(list_of_events) - 1:
            current_transit_index += 1
        # At this point current_transit_index should be pointing to a visit but it's okay

    # Then go through the whole list
    while current_transit_index < len(list_of_events) - 1:
        # Look for a transit
        while current_transit_index < len(list_of_events) - 1 and list_of_events[current_transit_index][2] != -1:
            current_transit_index += 1

        # If a transit was found
        if list_of_events[current_transit_index][2] == -1:
            # Current_transit_index is the first transit found, so it its preceded by a visit
            previous_visit_location = list_of_events[current_transit_index - 1][2]
            # Look for the next visit
            next_visit_index = current_transit_index + 1
            while next_visit_index < len(list_of_events) and list_of_events[next_visit_index][2] == -1:
                next_visit_index += 1
            # If a visit was found
            if next_visit_index != len(list_of_events):
                # Check where this next visit is happening
                next_visit_location = list_of_events[next_visit_index][2]
                # Fill the new transit list accordingly
                if to_same_patch and previous_visit_location == next_visit_location:
                    new_list_of_transits += list_of_events[current_transit_index:next_visit_index]
                if to_different_patch and previous_visit_location != next_visit_location:
                    new_list_of_transits += list_of_events[current_transit_index:next_visit_index]
            # Next transit is at least after the next visit
            current_transit_index = next_visit_index + 1

        # Else if no transit was found it means it's over so do nothin'

    return new_list_of_transits


def array_division_ignoring_zeros(a, b):
    return np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b != 0)


def return_value_list(results, column_name, condition_list=None, convert_to_duration=True, only_first=False, end_time=False):
    """
    Will return a list of values of column_name in results, pooled for all conditions in condition_list.
    For transits, can return only_same_patch_transits or only_cross_patch_transits if they're set to True.
    to_same_patch: if False, will only plot transits that go from one patch to another
    to_different_patch: if False, will only plot transits that leave and come back to the same patch
    convert_to_duration: by default, it will convert visits and transits to durations, but if set False it will return
                         the list of visit / transit time stamps (same format as in results)
    only_first: if set to True, will only return the first value of column_name for each folder in results
    end_time: if False, just take all values. if = 3000, will only return values that start before 3000.
    """
    # If no condition_list was given, just take all folders from results
    if condition_list is None:
        folder_list = np.unique(results["folder"])
    else:
        if type(condition_list) == int:  # if there's just one condition, make it a list for the rest to work
            condition_list = [condition_list]
        folder_list = fd.return_folders_condition_list(np.unique(results["folder"]), condition_list)

    list_of_values = []
    if column_name == "same transits" or column_name == "cross transits":
        for i_plate in range(len(folder_list)):
            plate_name = folder_list[i_plate]
            plate_results = results[results["folder"] == plate_name].reset_index()
            current_transit_list = fd.load_list(plate_results, "aggregated_raw_transits")
            current_visit_list = fd.load_list(plate_results, "no_hole_visits")
            if column_name == "same transits":
                current_transit_list = select_transits(current_transit_list, current_visit_list, to_same_patch=True)
            if column_name == "cross transits":
                current_transit_list = select_transits(current_transit_list, current_visit_list,
                                                       to_different_patch=True)
            if only_first and current_transit_list:
                current_transit_list = [current_transit_list[0]]
            if end_time and current_transit_list:
                print("Implement end_time in return_value_list for transits!!!")
            if convert_to_duration:
                list_of_values += convert_to_durations(current_transit_list)
            else:
                list_of_values.append(current_transit_list)

    elif column_name == "patch_sequence":
        for i_plate in range(len(folder_list)):
            current_plate = folder_list[i_plate]
            current_results = results[results["folder"] == current_plate].reset_index()
            current_visits = fd.load_list(current_results, "no_hole_visits")
            if current_visits:
                patch_sequence = np.array(current_visits)[:, 2]
                #patch_sequence = [i[0] for i in groupby(np.array(current_visits)[:, 2])]
                if only_first and patch_sequence:
                    patch_sequence = patch_sequence[0]
                if end_time and patch_sequence:
                    print("Implement end_time in return_value_list for patch_sequence!!!")
                list_of_values.append(patch_sequence)

    else:
        if column_name == "visits":
            column_name = "no_hole_visits"
        if column_name == "transits":
            column_name = "aggregated_raw_transits"

        for i_plate in range(len(folder_list)):
            current_plate = folder_list[i_plate]
            current_results = results[results["folder"] == current_plate].reset_index()
            current_values = fd.load_list(current_results, column_name)
            if only_first and current_values:
                list_of_found_patches = []
                first_value_each_patch = []
                for value in current_values:
                    if value[2] not in list_of_found_patches:
                        list_of_found_patches.append(value[2])
                        first_value_each_patch.append(value)
                current_values = first_value_each_patch
            if end_time and current_values:
                current_values = [current_values[i] for i in range(len(current_values)) if current_values[i][0] <= end_time]
            if convert_to_duration:
                list_of_values += convert_to_durations(current_values)
            else:
                list_of_values += current_values

    return list_of_values


def transit_properties(results, condition_list, split_conditions, is_print=False):
    """
    Take a condition or a list of conditions, and look at the transits to compute, for each condition:
        - the probability of coming back to a patch when the worm exits it (same patch transits / total transits)
        - the average duration of same patch transits
        - the average duration of cross patch transits
    """
    if split_conditions:
        revisit_probability = np.zeros(len(condition_list))
        cross_transit_probability = np.zeros(len(condition_list))
        exponential_leaving_probability = np.zeros(len(condition_list))
        min_visit = np.zeros(len(condition_list))
        average_visit = np.zeros(len(condition_list))
        average_same_patch = np.zeros(len(condition_list))
        average_cross_patch = np.zeros(len(condition_list))
        for i_cond in range(len(condition_list)):
            condition = condition_list[i_cond]
            all_visits = return_value_list(results, "visits", [condition])
            all_transits = return_value_list(results, "transits", [condition])
            same_transits = return_value_list(results, "same transits", [condition])
            cross_transits = return_value_list(results, "cross transits", [condition])
            revisit_probability[i_cond] = len(same_transits) / len(all_transits)
            cross_transit_probability[i_cond] = len(cross_transits) / len(all_transits)
            exponential_leaving_probability[i_cond] = 1 / np.mean(all_visits)
            min_visit[i_cond] = np.percentile(all_visits, 15)
            average_visit[i_cond] = np.mean(all_visits)
            average_same_patch[i_cond] = np.mean(same_transits)
            average_cross_patch[i_cond] = np.mean(cross_transits)

    else:
        all_visits = return_value_list(results, "visits", condition_list)
        all_transits = return_value_list(results, "transits", condition_list)
        same_transits = return_value_list(results, "same transits", condition_list)
        cross_transits = return_value_list(results, "cross transits", condition_list)
        revisit_probability = len(same_transits) / len(all_transits)
        cross_transit_probability = len(cross_transits) / len(all_transits)
        exponential_leaving_probability = 1 / np.mean(all_visits)
        min_visit = np.percentile(all_visits, 15)
        average_visit = np.mean(all_visits)
        average_same_patch = np.mean(same_transits)
        average_cross_patch = np.mean(cross_transits)

    if is_print:
        for i_cond in range(len(condition_list)):
            print("Transit properties for condition ", param.nb_to_name[i_cond])
            print("Revisit probability: ", revisit_probability[i_cond])
            print("Cross-patch probability: ", cross_transit_probability[i_cond])
            print("Exponential leaving probability: ", exponential_leaving_probability[i_cond])
            print("Minimal duration of visits: ", min_visit[i_cond])
            print("Average duration of visits: ", average_visit[i_cond])
            print("Average duration of same patch transits: ", average_same_patch[i_cond])
            print("Average duration of cross patch transits: ", average_cross_patch[i_cond])
            print("-----")

    else:
        return revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch


def pool_conditions_by(condition_list, pool_by_variable):
    """
    NOTE: made before the proper condition dictionaries. It would probably be way more efficient to use them.
    Takes a list of condition numbers ([c0,c1,c2]) and will pool them into sublist based on
    variable in argument ([[c0],[c1,c2]]) and the corresponding names (["close", "med"]). Eg by distance or density.
    If pool_conditions = True, then like what's on top.

    """
    # TODO automate those pools using the dictionaries in parameters.py
    if pool_by_variable == "distance":
        pooled_conditions = [[0, 4, 12], [1, 5, 8, 13], [2, 6, 14], [3, 7, 15]]
        pool_names = ["close", "med", "far", "cluster", "control"]
    elif pool_by_variable == "density":
        pooled_conditions = [[0, 1, 2, 3], [4, 5, 6, 7], [8], [12, 13, 14, 15]]
        pool_names = ["0.2", "0.5", "1.25", "0"]
    elif pool_by_variable == "food":
        pooled_conditions = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [12, 13, 14, 15]]
        pool_names = ["food", "control"]
    else:
        pooled_conditions = [[condition_list[i]] for i in range(len(condition_list))]
        pool_names = [param.nb_to_name[condition_list[i]] for i in range(len(condition_list))]

    # We build a new list
    new_condition_list = []
    new_pool_names = []
    # For every pool
    for i_pool in range(len(pooled_conditions)):
        pool = pooled_conditions[i_pool]
        # We copy the pool otherwise iterations do random stuff when we remove elements from the iterable
        new_pool = copy.copy(pool)
        # For every condition
        for cond in pool:
            # Remove the elements that are not in condition_list
            if cond not in condition_list:
                new_pool.remove(cond)
        # Add this new pool to the lists if there's something
        if new_pool:
            new_condition_list.append(new_pool)
            new_pool_names.append(pool_names[i_pool])
        # Else, the pool is empty, so we don't even want its name, booyah

    return new_condition_list, new_pool_names


def aggregate_visits(list_of_visits, condition, aggregation_threshold, return_duration):
    """
    This will aggregate visits to the same patch that are spaced by a transit shorter than aggregation threshold.
    Needs condition to know how many patches there are in total.
    Note: with a very high aggregation threshold, you can get all visits to a patch to be pooled.
    Example:
        INPUT: visit list = [[10, 20, 1], [22, 40, 1], [122, 200, 2], [210, 220, 1]]
        (first element is visit start, second is visit end, third is visit patch)
        OUTPUT:
        - Will first convert to by_patch visit format: [ [[10,20], [22,40], [210,220]], [[122,200]] ]
          because we want to work on visits patch by patch.
        - if return_duration = FALSE, will return
            - aggregation threshold = 5: [ [ [10, 40, [[20, 22]]], [210, 220, [[]] ], [ [122, 200, []] ] ]
            - aggregation threshold = 10000: [ [ [10, 220, [[20,22], [40, 210]] ], [ [210, 200, []] ] ]
            Where each sublist corresponds to a patch, containing one sublist for each visit sublist, itself containing:
                - the start of the visit
                - the end of the visit
                - a sublist containing the times where the worm was out during this visit
        - if return_duration = TRUE, will return
            - aggregation threshold = 5: [[30, 21], [79]]
            - aggregation threshold = 10000: [[51], [79]]
            Where each sublist corresponds to a patch, and contains the durations of the successive visits, excluding
            intermediate transits.
    """
    visits_per_patch = []
    # We refactor visits from chronological to by_patch format (one sub-list per patch)
    sorted_visits = gr.sort_visits_by_patch(list_of_visits, param.nb_to_nb_of_patches[condition])
    # We don't append because we want to pool all patches from all conditions in the same list (for now)
    visits_per_patch += sorted_visits
    # At the end of this, all visits to a same patch are in the same sublist of visits_per_patch
    # [ [[t0,t1],[t0,t1]], [[t0,t1]], [], ... ]

    aggregated_visits = [[] for _ in range(len(visits_per_patch))]
    for i_patch in range(len(visits_per_patch)):
        this_patch_visits = visits_per_patch[i_patch]
        current_visit_start = 0
        current_visit_end = 0
        current_visit_duration = 0
        ignored_transits = []

        # Initialization
        if this_patch_visits:
            current_visit_start = this_patch_visits[0][0]
            current_visit_end = this_patch_visits[0][1]
            current_visit_duration = current_visit_end - current_visit_start + 1

        # Loopy loop
        for i_visit in range(
                len(this_patch_visits) - 1):  # -1 because we look at i_visit and the next, last visit has no next
            next_visit_start = this_patch_visits[i_visit + 1][0]
            next_visit_end = this_patch_visits[i_visit + 1][1]
            next_visit_duration = next_visit_end - next_visit_start + 1

            if return_duration:
                current_visit_end = this_patch_visits[i_visit][1]
                # If this isn't the last next_visit
                if i_visit < len(this_patch_visits) - 2:
                    # If we have to aggregate
                    if abs(next_visit_start - current_visit_end) < aggregation_threshold:
                        current_visit_duration += next_visit_duration
                    # If we don't have to aggregate
                    else:
                        aggregated_visits[i_patch].append(current_visit_duration)
                        current_visit_duration = next_visit_duration  # create a new visit with the next
                # If this is the last next_visit (so current visit is penultimate)
                else:
                    # If we have to aggregate with the last
                    if abs(next_visit_start - current_visit_end) < aggregation_threshold:
                        aggregated_visits[i_patch].append(current_visit_duration + next_visit_duration)
                    # If we don't have to aggregate current and next visit
                    else:
                        aggregated_visits[i_patch].append(current_visit_duration)
                        aggregated_visits[i_patch].append(next_visit_duration)

            else:
                # If this isn't the last next_visit
                if i_visit < len(visits_per_patch[i_patch]) - 2:
                    # If we have to aggregate
                    if abs(next_visit_start - current_visit_end) < aggregation_threshold:
                        ignored_transits.append([current_visit_end, next_visit_start])
                        current_visit_end = next_visit_end
                    # If we don't have to aggregate
                    else:
                        aggregated_visits[i_patch].append([current_visit_start, current_visit_end, ignored_transits])
                        current_visit_start = next_visit_start
                        current_visit_end = next_visit_end
                        ignored_transits = []
                # If this is the last next_visit (so current visit is penultimate)
                else:
                    # If we have to aggregate with the last
                    if abs(next_visit_start - current_visit_end) < aggregation_threshold:
                        ignored_transits.append([current_visit_end, next_visit_start])
                        aggregated_visits[i_patch].append([current_visit_start, next_visit_end, ignored_transits])
                    # If we don't have to aggregate current and next visit
                    else:
                        aggregated_visits[i_patch].append([current_visit_start, current_visit_end, ignored_transits])
                        aggregated_visits[i_patch].append([next_visit_start, next_visit_end, []])

    return aggregated_visits


def leaving_events_time_stamps(list_of_visits_with_transit_info, in_patch_timeline=True):
    """
    Takes a list of "aggregated visits" as outputted by the aggregate_visits function, with transit info included.
    List structure should be as follows: [ [ [10, 40, [[20, 22], [27, 32]] ], [210, 220, [[]] ], [ [122, 200, []] ] ]
    Where each sublist corresponds to a patch, containing one sublist for each visit sublist, itself containing:
        - the start of the visit
        - the end of the visit
        - a sublist containing the times where the worm was out during this visit
    Should return:
    - For each patch, the time stamps of the leaving events in terms of total time in patch. So for the example sublist
    [10, 40, [[20, 22], [27, 32]], it should transform it to simply [11, 17] (transits happen after 10 time steps inside
    the food patch, and then after 16 time steps (18 time steps from 10 to 27, minus the time spent outside from 20 to
    22).
    """
    events_time_stamps = []
    # For every patch sublist
    for i_patch in range(len(list_of_visits_with_transit_info)):
        events_this_patch = []
        current_patch_list = list_of_visits_with_transit_info[i_patch]
        # For every "aggregated visits" to this patch
        for i_visit in range(len(current_patch_list)):
            current_visit = current_patch_list[i_visit]
            visit_start = current_visit[0]
            list_of_transits = current_visit[2]
            time_outside_of_patch = 0
            for transit in list_of_transits:
                # Add when the leaving happens relatively to visit start, removing total time spent outside of patch
                events_this_patch.append(transit[0] - visit_start + 1 - time_outside_of_patch)
                # Update total time spent outside w/ current transit info
                time_outside_of_patch += transit[1] - transit[0] + 1
            # Add last exit event (visit end)
            events_this_patch.append(current_visit[1] - visit_start + 1)
        events_time_stamps.append(events_this_patch)

    return events_time_stamps


def delays_before_leaving(result_table, condition_list):
    """
    Returns list of average times before next leaving event, for each time point of the "total time in patch" timeline,
    pooled for all patches of condition list
    """
    if condition_list is int:
        condition_list = [condition_list]

    full_folder_list = fd.return_folders_condition_list(np.unique(result_table["folder"]), condition_list)
    # Generate fully aggregated visits (where all visits to a patch are merged together)
    result_table = gr.add_aggregate_visit_info_to_results(result_table, [100000])

    delay_before_leaving_list = []
    corresponding_time_in_patch_list = []

    for plate in full_folder_list:
        current_data = result_table[result_table["folder"] == plate]
        list_of_visits = fd.load_list(current_data, "aggregated_visits_thresh_100000")
        list_of_transits = fd.load_list(current_data, "aggregated_raw_transits")
        for i_visit in range(len(list_of_visits)):
            current_patch_info = list_of_visits[i_visit]
            visit_start = current_patch_info[0]
            visit_end = current_patch_info[1]
            current_patch_transits = current_patch_info[-1]
            time_out_of_patch_counter = 0
            # Add delays that run from beginning of visit to first exit
            current_delays = list(range(current_patch_transits[0][0] - visit_start, 0, -1))
            delay_before_leaving_list += current_delays
            corresponding_time_in_patch_list += list(range(current_patch_transits[0][0] - visit_start))

            # Add delays from end of each transit to beginning of next transit
            for i_transit in range(len(current_patch_transits) - 1):
                this_transit_start = current_patch_transits[i_transit][0]
                this_transit_end = current_patch_transits[i_transit][1]
                next_transit_start = current_patch_transits[i_transit + 1][0]
                time_before_next_transit = next_transit_start - this_transit_end
                time_out_of_patch_counter += this_transit_end - this_transit_start  # update in-patch time counter
                current_delays = list(range(time_before_next_transit, 0, -1))
                delay_before_leaving_list += current_delays
                corresponding_time_in_patch_list += list(
                    range(this_transit_end - visit_start - time_out_of_patch_counter,
                          next_transit_start - visit_start - time_out_of_patch_counter))

            # Add delays that run from end of last transit to end of visit, BUT ONLY if it's not the end of the video!
            # To do so, check if there is a transit that start after this visit end
            last_transit_start = np.max([list_of_transits[i][0] for i in range(len(list_of_transits))])

            if visit_end <= last_transit_start:
                current_delays = list(range(visit_end - current_patch_transits[-1][1], 0, -1))
                delay_before_leaving_list += current_delays
                time_out_of_patch_counter += current_patch_transits[-1][1] - current_patch_transits[-1][
                    0]  # update in-patch time counter
                corresponding_time_in_patch_list += list(
                    range(current_patch_transits[-1][1] - visit_start - time_out_of_patch_counter,
                          visit_end - visit_start - time_out_of_patch_counter))

    return delay_before_leaving_list, corresponding_time_in_patch_list


def compute_leaving_probability(delay_list):
    """
    Outputs a list with the leaving probability, defined as:
                number of delays equal to time_threshold or less (see param for setting time threshold)
    divided by: total number of delays
    """
    return len([delay_list[i] for i in range(len(delay_list)) if delay_list[i] <= param.time_threshold]) / len(
        delay_list)


def leaving_probability(results, condition_list, bin_size, worm_limit, errorbars=True):
    """
    Takes result table and a condition.
    Will compute a list of delays as outputted by delays_before_leaving, or by running xy_to_bins on the output of
    delay_before_leaving (so either a list of delays, or a list of lists of delays).
    Outputs, for each bin, the leaving probability (as defined in compute_leaving_probability), and bootstraps around it.
    Cuts the curve after the point where the data is for less than worm_limit.
    """
    folder_list = np.unique(results["folder"])
    folder_list = fd.return_folders_condition_list(folder_list, condition_list)

    # Compute the results for each worm
    # Initialize those in case there's no folder in folder_list
    binned_times_in_patch = 0
    full_list_of_delays = 0
    # List where we'll store binned delays_before_leaving, one sublist per worm
    binned_delays_each_worm = [[] for _ in range(len(folder_list))]
    times_in_patch_bins_each_worm = [[] for _ in range(len(folder_list))]  # not all worms will have same bins
    # Loop to extract and bin delays
    for i_folder in range(len(folder_list)):
        folder = folder_list[i_folder]
        current_plate = results[results["folder"] == folder].reset_index()
        leaving_delays, corresponding_time_in_patch = delays_before_leaving(current_plate, condition_list)
        if corresponding_time_in_patch:  # if there are no visits, don't do that
            binned_times_in_patch, avg_leaving_delays, y_err_list, full_list_of_delays = xy_to_bins(
                corresponding_time_in_patch, leaving_delays, bin_size, print_progress=False)
            binned_delays_each_worm[i_folder] = full_list_of_delays
            times_in_patch_bins_each_worm[i_folder] = binned_times_in_patch

    # Compute the global stats for all the worms pooled together
    global_leaving_delays, global_times_in_patch = delays_before_leaving(results, condition_list)
    global_time_in_patch_bins, global_avg_leaving_delays, _, full_list_of_delays = xy_to_bins(global_times_in_patch,
                                                                                              global_leaving_delays,
                                                                                              bin_size,
                                                                                              print_progress=True)

    # Reformat binned_delays_each_worm to have one sublist per bin, and in each bin sublist one sublist per worm
    wormed_delays_each_bin = [[[] for _ in range(len(folder_list))] for _ in range(len(global_time_in_patch_bins))]
    for i_folder in range(len(folder_list)):
        current_delays = binned_delays_each_worm[i_folder]
        current_times_in_patch = times_in_patch_bins_each_worm[i_folder]
        for i_bin in range(len(current_times_in_patch)):
            # Current worm may have fewer bins than global pool, but if it's missing some it's the last ones
            # So we can just run through the beginning of the global bins and it should always coincide
            if global_time_in_patch_bins[
                i_bin] in current_times_in_patch:  # still, double check that this worm has this bin
                wormed_delays_each_bin[i_bin][i_folder] = current_delays[i_bin]
    # Compute the list of each worm's average leaving probabilities, in each bin
    wormed_avg_leaving_prob_each_bin = [[] for _ in range(len(global_time_in_patch_bins))]
    for i_bin in range(len(global_time_in_patch_bins)):
        current_delays = wormed_delays_each_bin[i_bin]
        wormed_avg_leaving_prob_each_bin[i_bin] = [compute_leaving_probability(current_delays[i_worm]) for i_worm in
                                                   range(len(current_delays)) if len(current_delays[i_worm]) != 0]

    # If list of delays has no bins
    if full_list_of_delays[0] is int:
        full_list_of_delays = [full_list_of_delays]  # Create a fake unique bin

    # Initialize lists
    probability_list = []
    nb_of_worms_each_bin = []
    errors_inf = []
    errors_sup = []

    # Fill those lists
    for i_bin in range(len(global_time_in_patch_bins)):
        nb_of_worms = len(
            [wormed_delays_each_bin[i_bin][i_worm] for i_worm in range(len(wormed_delays_each_bin[i_bin])) if
             wormed_delays_each_bin[i_bin][i_worm] != []])
        if i_bin % 2 == 0:
            print("Computing stats on leaving probabilities... ", int(100 * i_bin / len(full_list_of_delays)), "% done")
        if nb_of_worms > worm_limit:
            nb_of_worms_each_bin.append(nb_of_worms)
            current_bin_avg_probabilities = wormed_avg_leaving_prob_each_bin[i_bin]
            probability_list.append(np.mean(current_bin_avg_probabilities))
            if errorbars:
                bootstrap_ci = bottestrop_ci(wormed_avg_leaving_prob_each_bin[i_bin], 50, operation="mean")
                errors_inf.append(probability_list[i_bin] - bootstrap_ci[0])
                errors_sup.append(bootstrap_ci[1] - probability_list[i_bin])
        else:
            global_time_in_patch_bins = global_time_in_patch_bins[:i_bin]  # cut it to remove the excluded bins
            break

    return global_time_in_patch_bins, probability_list, [errors_inf, errors_sup], nb_of_worms_each_bin


def list_of_visited_patches(list_of_visits):
    """
    Takes a list of visits and returns a list with the ID number of visited patches.
    @param list_of_visits: list in the [t0, t1, p] format (t0 = start time, t1 = end time, p = patch number)
    @return: list_of_patches: list [p0, p1, p2, ...] of the patches that were visited at least once.
    """
    list_of_patches = []
    for i_visit in range(len(list_of_visits)):
        list_of_patches.append(list_of_visits[i_visit][2])
    return np.unique(list_of_patches)


def xy_to_bins(x, y, bin_size, print_progress=True, custom_bins=None, do_not_edit_xy=True, compute_bootstrap=True):
    """
    Will take an x and a y iterable.
    Will return bins spaced by bin_size for the x values, and the corresponding average y value in each of those bins.
    With errorbars if bootstrap is set to True.
    """
    if do_not_edit_xy:
        x_copy = copy.deepcopy(x)
        y_copy = copy.deepcopy(y)
    else:
        x_copy = x
        y_copy = y

    # Create or load bin list, with left limit of each bin
    if custom_bins is None:
        bin_list = []
        current_bin = np.min(x_copy)
        while current_bin < np.max(x_copy):
            bin_list.append(current_bin)
            current_bin += bin_size
        bin_list.append(current_bin)  # add the last value
        bin_list.sort()
        nb_of_bins = len(bin_list)
    else:
        bin_list = sorted(custom_bins)
        # If the value of the last bin is inferior to the max, there are bugs, so don't do that!
        if bin_list[-1] < np.max(x):
            bin_list.append(int(np.max(x) + 1))
        nb_of_bins = len(bin_list)

    # Create a list with one sublist of y values for each bin
    binned_y_values = [[] for _ in range(nb_of_bins)]
    original_y_len = len(y_copy)
    for i in range(len(y_copy)):
        if print_progress and i % (original_y_len // 10) == 0:
            print("Binning in xy_to_bins... ", int(100 * i / (i + len(y_copy))), "% done")
        current_y = y_copy.pop()
        current_x = x_copy.pop()
        # Looks for first index at which x can be inserted in the bin list
        # So np.searchsorted([0, 1, 2], 1) = 1
        i_bin = np.searchsorted(bin_list, current_x)
        binned_y_values[i_bin].append(current_y)

    # If the bins are custom and some lowest / highest bins are empty, remove them:
    if custom_bins is not None:
        i_low = 0
        while binned_y_values[i_low] == []:
            i_low += 1
        i_high = nb_of_bins - 1
        while binned_y_values[i_high] == []:
            i_high -= 1
        binned_y_values = binned_y_values[i_low:i_high + 1]
        bin_list = bin_list[i_low:i_high + 1]
        nb_of_bins = len(bin_list)

    # Compute averages and bootstrap errorbars
    avg_list = np.zeros(nb_of_bins)
    errors_inf = np.zeros(nb_of_bins)
    errors_sup = np.zeros(nb_of_bins)
    for i_bin in range(nb_of_bins):
        if i_bin % 10 == 0 and print_progress:
            print("Averaging xy_to_bins... ", int(100 * i_bin / nb_of_bins), "% done")
        current_values = binned_y_values[i_bin]
        if current_values:  # if there are values there
            avg_list[i_bin] = np.nanmean(current_values)
            if compute_bootstrap:
                # Bootstrapping on the plate avg duration
                bootstrap_ci = bottestrop_ci(current_values, 100)
                errors_inf[i_bin] = avg_list[i_bin] - bootstrap_ci[0]
                errors_sup[i_bin] = bootstrap_ci[1] - avg_list[i_bin]

    print("Finished xy_to_bins()")

    return bin_list, avg_list, [list(errors_inf), list(errors_sup)], binned_y_values

#bins, avg, _, binned = xy_to_bins([0, 0, 1, 1, 2, 2, 2, 3, 4, 10, 100], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bin_size=0, custom_bins=[0, 1])
