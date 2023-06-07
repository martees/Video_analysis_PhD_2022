import numpy as np
import random
import json
from scipy import stats
import time

import param
# My code
from param import *
import model

import find_data as fd


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


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


def results_per_condition(result_table, list_of_conditions, column_name, divided_by=""):
    """
    Function that takes our result table, a list of conditions, and a column name (as a string)
    Returns the list of values of that column pooled by condition, a list of the average value for each condition, and a
    bootstrap confidence interval for each value.
    Can take in a third argument, column name by which you want to divide the main column, plate by plate
    eg: divide duration sum by nb of visits for each plate to get average visit duration for each plate
    """

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

            # When we want to divide column name by another one
            if divided_by != "":
                if np.sum(current_plate[divided_by]) != 0:  # Non zero check for division
                    list_of_values[i_plate] = np.sum(current_plate[column_name]) / np.sum(current_plate[divided_by])
                else:
                    print("Trying to divide by 0... what a shame")
                # if divided_by == "nb_of_visits" and column_name == "total_visit_time" and current_condition == 2: #detecting extreme far 0.2 cases
                #    if list_of_values[i_plate]>800:
                #        print(list_of_plates[i_plate])
                #        print(list_of_values[i_plate])

            # When no division has to be made
            else:
                if column_name == "average_speed_inside" or column_name == "average_speed_outside":
                    # Exclude the 0's which are the cases were the worm didn't go to a patch / out of a patch for a full track
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
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches)) / list_total_patch[
                            i_condition]
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

    return full_list_of_values, list_of_avg_values, [list(errors_inf), list(errors_sup)]


def model_per_condition(result_table, list_of_conditions, column_name, divided_by=""):
    """
    Function that should return model predictions for column name values (divided by divided_by values), pooled for
    each condition in condition_list.
    """
    revisit_probability, _, _, _, _, average_same_patch, average_cross_patch = transit_properties(result_table, list_of_conditions, split_conditions=True)

    if column_name == "total_visit_time" and divided_by == "nb_of_visits":
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

            model_avg_visit_length_list[i_condition] = model.opt_mvt("", resolution, nb_of_time_steps, 1, parameter_list, is_plot=False, is_print=False)

        return model_avg_visit_length_list
    else:
        print("We have no model for that yet darling.")
        return np.zeros(len(list_of_conditions))

def visit_time_as_a_function_of(results, traj, condition_list, variable):
    """
    Takes a condition list and a variable and will plot visit time against this variable for the selected conditions
    """

    # Fill the folder list up (list of folders corresponding to the conditions in condition_list)
    folder_list = fd.return_folders_condition_list(np.unique(results["folder"]), condition_list)

    # Initialize the lists that will be returned
    full_visit_list = [[] for _ in range(len(condition_list))]
    full_variable_list = [[] for _ in range(len(condition_list))]

    # Fill up the lists depending on the variable specified as an argument
    if variable == "Last travel time":
        # TODO if we ever use this again, switch it to a much simpler version where we fuse visit and transit lists, sort by time, and then look at each element
        starts_with_visit = False
        for i_plate in range(len(folder_list)):
            # Initialization
            # Slice to one plate
            current_plate = results[results["folder"] == folder_list[i_plate]].reset_index()
            # Visit and transit lists
            list_of_visits = list(json.loads(current_plate["aggregated_raw_visits"][0]))
            list_of_transits = list(json.loads(current_plate["aggregated_raw_transits"][0]))
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
                    if verbose:
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
                            if verbose:
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
                        if verbose:
                            print("additional visit : [", current_visit[0], ", ", current_visit[1], "]")
                        # We add this extra transit to the previous transit length
                        list_of_visit_lengths[-1] += current_visit[1] - current_visit[0] + 1

                    # Go to next visit!
                    i_visit += 1

            condition = fd.folder_to_metadata(current_plate["folder"][0])["condition"][0]
            i_condition = condition_list.index(condition)  # for the condition-label correspondence we need the index
            # plt.scatter(list_of_previous_transit_lengths, list_of_visit_lengths, color=colors[i_condition], label=str(condition_names[i_condition]), zorder=i_condition)

            # For plotting
            full_visit_list[i_condition] += list_of_visit_lengths
            full_variable_list[i_condition] += list_of_previous_transit_lengths

    else:
        for i_plate in range(len(folder_list)):
            time_start = time.time()
            # Initialization
            # Slice to one plate
            current_plate = results[results["folder"] == folder_list[i_plate]].reset_index()
            # Visit and transit lists
            list_of_visits = list(json.loads(current_plate["aggregated_raw_visits"][0]))
            # Information about condition
            condition = fd.folder_to_metadata(current_plate["folder"][0])["condition"][0]
            i_condition = condition_list.index(condition)  # for the condition-label correspondence we need the index
            for i_visit in range(len(list_of_visits)):
                current_visit = list_of_visits[i_visit]
                full_visit_list[i_condition].append(current_visit[1] - current_visit[0] + 1)
                if variable == "Visit start":
                    full_variable_list[i_condition].append(current_visit[0])
                if variable == "Speed when entering":
                    current_traj = traj[traj["folder"] == folder_list[i_plate]]
                    visit_start = current_traj[current_traj["frame"] == current_visit[0]].reset_index()
                    speed_when_entering = visit_start["speeds"][0]
                    full_variable_list[i_condition].append(speed_when_entering)
            print("It took ", time.time() - time_start, " sec to analyse plate ", i_plate, " / ", len(folder_list))

    return full_visit_list, full_variable_list


def convert_to_durations(list_of_time_stamps):
    """
    Function that takes a list of timestamps in the format [[t0,t1,...],[t0,t1,...],...] (uses t0 and t1 only)
    And will return the corresponding list of durations [d0,d1,...]
    """
    nb_of_events = len(list_of_time_stamps)
    list_of_durations = np.zeros(nb_of_events)
    for i_event in range(nb_of_events):
        list_of_durations[i_event] = list_of_time_stamps[i_event][1] - list_of_time_stamps[i_event][0]
    return list(list_of_durations)


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


def return_value_list(results, condition_list, column_name):
    """
    Will return a list of values of column_name in results, pooled for all conditions in condition_list.
    For transits, can return only_same_patch_transits or only_cross_patch_transits if they're set to True.
    to_same_patch: if False, will only plot transits that go from one patch to another
    to_different_patch: if False, will only plot transits that leave and come back to the same patch
    """
    list_of_values = []
    folder_list = fd.return_folders_condition_list(np.unique(results["folder"]), condition_list)

    if column_name == "transits":
        for i_plate in range(len(folder_list)):
            current_plate = folder_list[i_plate]
            current_results = results[results["folder"] == current_plate].reset_index()
            current_transit_list = json.loads(current_results["aggregated_raw_transits"][0])
            list_of_values += convert_to_durations(current_transit_list)

    if column_name == "same transits":
        for i_plate in range(len(folder_list)):
            current_plate = folder_list[i_plate]
            current_results = results[results["folder"] == current_plate].reset_index()
            current_transit_list = json.loads(current_results["aggregated_raw_transits"][0])
            current_visit_list = json.loads(current_results["aggregated_raw_visits"][0])
            current_transit_list = select_transits(current_transit_list, current_visit_list, to_same_patch=True)
            list_of_values += convert_to_durations(current_transit_list)

    if column_name == "cross transits":
        for i_plate in range(len(folder_list)):
            current_plate = folder_list[i_plate]
            current_results = results[results["folder"] == current_plate].reset_index()
            current_transit_list = json.loads(current_results["aggregated_raw_transits"][0])
            current_visit_list = json.loads(current_results["aggregated_raw_visits"][0])
            current_transit_list = select_transits(current_transit_list, current_visit_list, to_different_patch=True)
            list_of_values += convert_to_durations(current_transit_list)

    if column_name == "visits":
        for i_plate in range(len(folder_list)):
            current_plate = folder_list[i_plate]
            current_results = results[results["folder"] == current_plate].reset_index()
            current_visit_list = current_results["aggregated_raw_visits"][0]
            list_of_visits = json.loads(current_visit_list)
            list_of_values += convert_to_durations(list_of_visits)

    return list_of_values


def transit_properties(results, condition_list, split_conditions):
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
            all_visits = return_value_list(results, [condition], "visits")
            all_transits = return_value_list(results, [condition], "transits")
            same_transits = return_value_list(results, [condition], "same transits")
            cross_transits = return_value_list(results, [condition], "cross transits")
            revisit_probability[i_cond] = len(same_transits)/len(all_transits)
            cross_transit_probability[i_cond] = len(cross_transits)/len(all_transits)
            exponential_leaving_probability[i_cond] = 1/np.mean(all_visits)
            min_visit[i_cond] = np.percentile(all_visits, 15)
            average_visit[i_cond] = np.mean(all_visits)
            average_same_patch[i_cond] = np.mean(same_transits)
            average_cross_patch[i_cond] = np.mean(cross_transits)

    else:
        all_visits = return_value_list(results, condition_list, "visits")
        all_transits = return_value_list(results, condition_list, "transits")
        same_transits = return_value_list(results, condition_list, "same transits")
        cross_transits = return_value_list(results, condition_list, "cross transits")
        revisit_probability = len(same_transits) / len(all_transits)
        cross_transit_probability = len(cross_transits) / len(all_transits)
        exponential_leaving_probability = 1 / np.mean(all_visits)
        min_visit = np.percentile(all_visits, 15)
        average_visit = np.mean(all_visits)
        average_same_patch = np.mean(same_transits)
        average_cross_patch = np.mean(cross_transits)

    return revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch


def first(iterable, condition=lambda x: True):
    """
    Returns the first item in the `iterable` that satisfies the `condition`.
    If the condition is not given, returns the first item of the iterable.
    Raises `StopIteration` if no item satysfing the condition is found.

    >> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >> first(range(3, 100))
    3
    >> first( () )
    Traceback (most recent call last):
    ...
    StopIteration
    """

    return next(x for x in iterable if condition(x))
