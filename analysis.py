import numpy as np
import random
import json
from scipy import stats
import time
import copy

# My code
import param
import find_data as fd
import model
import generate_results as gr


def r2(x, y):
    """
    R squared for linear regression between x and y vectors
    """
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
                    list_of_visited_patches = [i for sublist in list_of_visited_patches for i in sublist]
                    if column_name == "nb_of_visited_patches":
                        list_of_values[i_plate] = len(np.unique(list_of_visited_patches))
                    else:
                        list_total_patch = list(param.nb_to_nb_of_patches.values())  # total nb of patches for each cond
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

            condition = fd.load_condition(folder_list[i_plate])
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
            list_of_visits = fd.load_list(current_plate, "no_hole_visits")
            # Information about condition
            condition = fd.load_condition(folder_list[i_plate])
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


def return_value_list(results, column_name, condition_list=None, convert_to_duration=True):
    """
    Will return a list of values of column_name in results, pooled for all conditions in condition_list.
    For transits, can return only_same_patch_transits or only_cross_patch_transits if they're set to True.
    to_same_patch: if False, will only plot transits that go from one patch to another
    to_different_patch: if False, will only plot transits that leave and come back to the same patch
    convert_to_duration: by default, it will convert visits and transits to durations, but if set False it will return
                         the list of visit / transit time stamps (same format as in results)
    """
    # If no condition_list was given, just take all folders from results
    if condition_list is None:
        folder_list = np.unique(results["folders"])
    else:
        if type(condition_list) == int:  # if there's just one condition, make it a list for the rest to work
            condition_list = [condition_list]
        list_of_values = []
        folder_list = fd.return_folders_condition_list(np.unique(results["folder"]), condition_list)

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
            if convert_to_duration:
                list_of_values += convert_to_durations(current_transit_list)
            else:
                list_of_values.append(current_transit_list)

    else:
        if column_name == "visits":
            column_name = "no_hole_visits"
            convert_to_duration = convert_to_duration
        if column_name == "transits":
            column_name = "aggregated_raw_transits"
            convert_to_duration = convert_to_duration

        for i_plate in range(len(folder_list)):
            current_plate = folder_list[i_plate]
            current_results = results[results["folder"] == current_plate].reset_index()
            current_values = fd.load_list(current_results, column_name)
            if convert_to_duration:
                list_of_values += convert_to_durations(current_values)
            else:
                list_of_values += current_values

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

    return revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch


def pool_conditions_by(condition_list, pool_by_variable, pool_conditions=True):
    """
    Takes a list of condition numbers ([c0,c1,c2]) and will pool them into sublist based on
    variable in argument ([[c0],[c1,c2]]) and the corresponding names (["close", "med"]). Eg by distance or density.
    If pool_conditions = True, then like what's on top.

    """
    # TODO automate those pools using the dictionaries in param.py
    if pool_by_variable == "distance":
        pooled_conditions = [[0, 4], [1, 5, 8], [2, 6], [3, 7], [11]]
        pool_names = ["close", "med", "far", "cluster", "control"]
    elif pool_by_variable == "density":
        pooled_conditions = [[0, 1, 2, 3], [4, 5, 6, 7], [8], [11]]
        pool_names = ["0.2", "0.5", "1.25", "0"]
    elif pool_by_variable == "food":
        pooled_conditions = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]]
        pool_names = ["food", "control"]
    else:
        pooled_conditions = [[condition_list[i]] for i in range(len(condition_list))]
        pool_names = [[param.nb_to_name[condition_list[i]]] for i in range(len(condition_list))]

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
        for i_visit in range(len(this_patch_visits) - 1):  # -1 because we look at i_visit and the next, last visit has no next
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
    Should return list of average times before next leaving event, for each time point of the "total time in patch" timeline,
    pooled for all patches of condition list
    """

    full_folder_list = fd.return_folders_condition_list(np.unique(result_table["folder"]), condition_list)
    # Generate fully aggregated visits (where all visits to a patch are merged together)
    result_table = gr.add_aggregate_visit_info_to_results(result_table, [100000])

    delay_before_leaving_list = []
    corresponding_time_in_patch_list = []

    for plate in full_folder_list:
        current_data = result_table[result_table["folder"] == plate].reset_index()
        list_of_visits = fd.load_list(current_data, "aggregated_visits_thresh_"+str(100000))
        for i_patch in range(len(list_of_visits)):
            current_patch_info = list_of_visits[i_patch]
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
                next_transit_start = current_patch_transits[i_transit+1][0]
                time_before_next_transit = next_transit_start - this_transit_end
                time_out_of_patch_counter += this_transit_end - this_transit_start  # update in-patch time counter
                current_delays = list(range(time_before_next_transit, 0, -1))
                delay_before_leaving_list += current_delays
                corresponding_time_in_patch_list += list(range(this_transit_end - visit_start - time_out_of_patch_counter, next_transit_start - visit_start - time_out_of_patch_counter))
            # Add delays that run from end of last transit to end of visit
            current_delays = list(range(visit_end - current_patch_transits[-1][1], 0, -1))
            delay_before_leaving_list += current_delays
            corresponding_time_in_patch_list += list(range(current_patch_transits[-1][1] - visit_start - time_out_of_patch_counter, visit_end - visit_start - time_out_of_patch_counter))

    return delay_before_leaving_list, corresponding_time_in_patch_list


def xy_to_bins(x, y, bin_size, bootstrap=True):
    """
    Will take an x and a y iterable.
    Will return bins spaced by bin_size for the x values, and the corresponding average y value in each of those bins.
    With errorbars too.
    """

    # Create bin list, with left limit of each bin
    bin_list = []
    x_values = np.unique(x)
    current_bin = np.min(x_values)
    while current_bin < np.max(x_values):
        bin_list.append(current_bin)
        current_bin += bin_size
    bin_list.append(current_bin)  # add the last value
    nb_of_bins = len(bin_list)

    # Create a list with one sublist of y values for each bin
    binned_y_values = [[] for _ in range(nb_of_bins)]
    for i in range(len(y)):
        if i % 200000 == 0:
            print("Binning leaving delays... ", int(100*i/(i+len(y))), "% done")
        current_y = y.pop()
        current_x = x.pop()
        i_bin = np.searchsorted(bin_list, current_x)  # looks for first index at which x can be inserted in the bin list
        binned_y_values[i_bin].append(current_y)

    # Compute averages and bootstrap errorbars
    avg_list = np.zeros(nb_of_bins)
    errors_inf = np.zeros(nb_of_bins)
    errors_sup = np.zeros(nb_of_bins)
    if bootstrap:
        for i_bin in range(nb_of_bins):
            if i_bin % 10 == 0:
                print("Averaging leaving delays... ", int(100*i_bin/nb_of_bins), "% done")
            current_values = binned_y_values[i_bin]
            if current_values:  # if there are values there
                avg_list[i_bin] = np.mean(current_values)
                # Bootstrapping on the plate avg duration
                bootstrap_ci = bottestrop_ci(current_values, 100)
                errors_inf[i_bin] = avg_list[i_bin] - bootstrap_ci[0]
                errors_sup[i_bin] = bootstrap_ci[1] - avg_list[i_bin]

    return bin_list, avg_list, [list(errors_inf), list(errors_sup)], binned_y_values


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
