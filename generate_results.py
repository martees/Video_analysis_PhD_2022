from scipy.spatial import distance
import pandas as pd
from itertools import groupby
import copy
import json

# My code
import parameters
import analysis as ana
from generate_controls import *


def spline_value(angular_position, spline_breaks, spline_coefs):
    """
    Matlab splines are curves divided in subsections, each subsection defined by a polynomial.

    ::spline_breaks: the limits between the different polynomials
    :spline_coefs: the coefficients for each subsection
    """
    i = 0
    #angular_position = - angular_position
    while i < len(spline_breaks) - 1 and angular_position >= spline_breaks[i]:
        i += 1
    # invert coefficient order (matlab to numpy format)
    coefficients = [spline_coefs[i - 1][j] for j in range(len(spline_coefs[i - 1]) - 1, -1, -1)]
    local_polynomial = np.polynomial.polynomial.Polynomial(coefficients)
    return local_polynomial(angular_position - spline_breaks[i - 1])


def in_patch(position, patch_center, spline_breaks, spline_coefs):
    """
    returns True if position = [x,y] is inside the patch
    uses general parameter radial_tolerance: the worm is still considered inside the patch when its center is sticking out by that distance or less
    """
    # Compute radial coordinates
    # Compute the angle in radians measured counterclockwise from the positive x-axis, returns the angle t in the range (-π, π].
    angular_position = np.arctan2((position[1] - patch_center[1]), (position[0] - patch_center[0]))
    distance_from_center = np.sqrt((position[0] - patch_center[0]) ** 2 + (position[1] - patch_center[1]) ** 2)

    # Compute the local radius of the patch spline
    local_radius = spline_value(angular_position, spline_breaks, spline_coefs)

    return distance_from_center < local_radius


def in_patch_list(traj):
    """
    Function that takes in our trajectories dataframe, and returns a column with the patch where the worm is at
    each time step.
    """
    list_of_plates = pd.unique(traj["folder"])
    nb_of_plates = len(list_of_plates)

    # List where we'll put the patch where the worm is at each timestep
    list_of_patches = [-1 for i in range(len(traj["x"]))]
    i = 0  # global counter

    for i_plate in range(nb_of_plates):  # for every plate
        # Handmade progress bar
        if i_plate % 20 == 0 or i_plate == nb_of_plates:
            print("patch_position for plate ", i_plate, "/", nb_of_plates)

        # Extract patch information
        current_plate = list_of_plates[i_plate]
        current_metadata = fd.folder_to_metadata(current_plate)
        patch_centers = current_metadata["patch_centers"]
        patch_spline_breaks = current_metadata["spline_breaks"]
        patch_spline_coefs = current_metadata["spline_coefs"]

        # Extract positions
        current_data = traj[traj["folder"] == current_plate]
        list_x = current_data["x"]
        list_y = current_data["y"]

        # Analyze
        patch_where_it_is = -1  # initializing variable with index of patch where the worm currently is
        # We go through the whole trajectory
        for time in range(len(list_x)):
            if parameters.verbose and time % 100 == 0:
                print(time, "/", len(list_x))

            # First we figure out where the worm is
            patch_where_it_was = patch_where_it_is  # index of the patch where it is
            patch_where_it_is = -1  # resetting the variable to "worm is out"

            # In case the worm is in the same patch, don't try all the patches (doesn't work if worm is out):
            if patch_where_it_was != -1:
                if in_patch([list_x[time], list_y[time]], patch_centers[patch_where_it_was],
                            patch_spline_breaks[patch_where_it_was], patch_spline_coefs[patch_where_it_was]):
                    patch_where_it_is = patch_where_it_was

            # If the worm is out or changed patch, then look for it
            else:
                for i_patch in range(len(patch_centers)):  # for every patch
                    if in_patch([list_x[time], list_y[time]], patch_centers[i_patch], patch_spline_breaks[i_patch],
                                patch_spline_coefs[i_patch]):  # check if the worm is in it:
                        patch_where_it_is = i_patch  # if it's in it, keep that in mind

            # Update list accordingly
            list_of_patches[i] = patch_where_it_is  # still -1 if patch wasn't found
            i = i + 1
    return list_of_patches


def trajectory_distances(traj):
    """
    Function that takes in our trajectories dataframe, and returns a column with the distance covered by the worm since
    last timestep, for each time step. It should put 0 for the first point of every folder, but record distance even
    when there's a tracking hole.
    """
    # Slice the traj file depending on the folder, because we only want to compare one worm to itself
    folder_list = np.unique(traj["folder"])
    list_of_distances = []

    # For each folder, array operation to compute distance
    for folder in folder_list:
        current_traj = traj[traj["folder"] == folder].reset_index()

        # Generate shifted versions of our position columns, either shifted leftwards or rightwards
        array_x_r = np.array(current_traj["x"].iloc[1:])
        array_y_r = np.array(current_traj["y"].iloc[1:])
        array_x_l = np.array(current_traj["x"].iloc[:-1])
        array_y_l = np.array(current_traj["y"].iloc[:-1])
        # Do the computation
        current_list_of_distances = np.sqrt((array_x_l - array_x_r) ** 2 + (array_y_l - array_y_r) ** 2)
        # Add 0 in the beginning because the first point has no speed
        current_list_of_distances = np.insert(current_list_of_distances, 0, 0)
        # Fill the lists that are to be returned
        list_of_distances += list(current_list_of_distances)

    return list_of_distances


def trajectory_speeds(traj):
    """
    Requires the distances to have been computed!
    Will compute speeds from the distances and frame numbers in the trajectories columns
    """
    # Slice the traj file depending on the folder, because we only want to compare one worm to itself
    folder_list = np.unique(traj["folder"])
    list_of_speeds = []
    # For each folder, array operation to compute speed
    # NOTE: distance and speed should be equal almost all the time, exceptions = points where the tracking is
    #       interrupted, because there can be multiple frames between to consecutive tracked positions
    for folder in folder_list:
        current_traj = traj[traj["folder"] == folder].reset_index()
        current_list_of_distances = current_traj["distances"]

        # Generate shifted versions of our position columns, either shifted leftwards or rightwards
        array_frame_r = np.array(current_traj["frame"].iloc[1:])
        array_frame_l = np.array(current_traj["frame"].iloc[:-1])

        # Compute the number of frames elapsed between each two lines
        current_list_of_time_steps = array_frame_r - array_frame_l
        # Add 1 in the beginning because the first point isn't relevant (not zero to avoid division issues)
        current_list_of_time_steps = np.insert(current_list_of_time_steps, 0, 1)

        if parameters.verbose:
            nb_double_frames = np.count_nonzero(current_list_of_time_steps - np.maximum(current_list_of_time_steps,
                                                                                        0.1 * np.ones(
                                                                                            len(current_list_of_time_steps))))
            if nb_double_frames > 0:
                print("number of double frames:", str(nb_double_frames))

        # Remove the zeros and replace them by 0.1
        current_list_of_time_steps = np.maximum(current_list_of_time_steps,
                                                0.1 * np.ones(len(current_list_of_time_steps)))

        # Compute speeds
        list_of_speeds += list(current_list_of_distances / current_list_of_time_steps)

    return list_of_speeds


def avg_speed_analysis(which_patch_list, list_of_frames, distance_list):
    """
    Parameters:
        - the patch where a worm is at each timestep
        - the list of frame numbers corresponding to each timestep
        - the distance it has crawled for at each timestep
    Returns:
        - the average speed when inside a patch
        - the average speed when outside a patch
    """
    # Concept: sum time inside and outside, distance inside and outside, and then DIVIDE (it's an ancient technique)
    which_patch_list = which_patch_list.reset_index()["patch"]
    list_of_frames = list_of_frames.reset_index()["frame"]
    distance_list = distance_list.reset_index()["distances"]
    distance_inside_sum = 0
    time_inside_sum = 0
    distance_outside_sum = 0
    time_outside_sum = 0
    for i in range(1, len(which_patch_list) - 1):
        if which_patch_list[i] == -1:
            time_outside_sum += list_of_frames[i] - list_of_frames[i - 1]
            distance_outside_sum += distance_list[i]
        else:
            time_inside_sum += list_of_frames[i] - list_of_frames[i - 1]
            distance_inside_sum += distance_list[i]
    if (distance_inside_sum == 0 and time_inside_sum != 0) or (distance_inside_sum != 0 and time_inside_sum == 0) or (
            distance_outside_sum == 0 and time_outside_sum != 0) or (
            distance_outside_sum != 0 and time_outside_sum == 0):
        print("There's an issue with the avg_speed_analysis function!")
    return distance_inside_sum / max(1, time_inside_sum), distance_outside_sum / max(1,
                                                                                     time_outside_sum)  # the max is to prevent division by zero


def single_traj_analysis(which_patch_list, list_of_frames, patch_centers):
    """
    Takes a list containing the patch where the worm is at each timestep, a list of frames to which each data point
    corresponds, a list of patch centers ([[x0,y0],...,[xN,yN]]) (just a handy way of accessing the number of patches in
    that condition)
    Returns a list [[[d0,d1],...], [[d0,d1],...],...] with one list per patch
    each sublist contains the time stamps of the beginning and end of visits to each patch
    so len(output[0]) is the number of visits to the first patch
    Also returns
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

    # Order in which the patches were visited (should have as many elements as list_of_timestamps)
    # (function that removes consecutive duplicates, [1,2,2,3] => [1,2,3]
    # !!! contains -1 whenever the worm went outside !!!
    order_of_visits = [i[0] for i in groupby(which_patch_list)]

    # In order to compute when visits to patches start, end, and how long they last, we avoid looking at every line by
    # only looking at the indexes where the patch value changes:
    which_patch_array = np.array(which_patch_list)
    event_indexes = np.where(which_patch_array[:-1] != which_patch_array[1:])[0]
    # (this formula works by looking at differences between the list shifted by one to the left or to the right)
    # (for [1,1,2,2,6,6,6,6] it will return [1,3])
    event_indexes = np.insert(event_indexes, 0, 0)  # add zero to the events to start the first visit

    # Reset index of frame table, otherwise frames are not found once it's not the first worm
    list_of_frames = list_of_frames.reset_index()

    # We go through every event
    patch_where_it_is = -1  # initializing variable to -1 so that if the worm starts inside it's counted as a new visit
    for time in event_indexes:
        patch_where_it_was = patch_where_it_is  # memorize the patch where it was
        patch_where_it_is = which_patch_array[
            min(len(which_patch_array) - 1, time + 1)]  # update patch, max to not exceed size
        current_frame = int(list_of_frames["frame"][time])
        # Worm just exited a patch
        if patch_where_it_is == -1:  # worm currently out
            if patch_where_it_was != patch_where_it_is:  # was inside before
                list_of_timestamps[patch_where_it_was][-1][1] = current_frame  # end the previous visit
                list_of_timestamps[patch_where_it_was].append([0, 0])  # add new visit to the previous patch
        # Worm just entered a patch
        if patch_where_it_is != -1:  # worm currently inside
            if patch_where_it_was == -1:  # it's a new visit (first patch or was outside before)
                list_of_timestamps[patch_where_it_is][-1][0] = current_frame  # begin visit in current patch sublist
    # Close the last visit
    if which_patch_array[-1] != -1:  # if the worm is inside, last visit hasn't been closed (no exit event)
        list_of_timestamps[patch_where_it_is][-1][1] = int(list_of_frames["frame"].iloc[-1])  # = last frame
    # Clean the table (remove the zeros because they're just here for the duration algorithm)
    for i_patch in range(len(list_of_timestamps)):
        list_of_timestamps[i_patch] = [nonzero for nonzero in list_of_timestamps[i_patch] if nonzero != [0, 0]]

    return list_of_timestamps, order_of_visits


def mvt_patch_visits(list_of_timestamps, order_of_visits, patch_centers):
    """
    Takes a list of time stamps as spitted by single_traj_analysis and returns an adjusted version of those visits
    by aggregating successive visits to a same patch.
    This is the list of durations we will use to test the Marginal Value Theorem.
    """

    adjusted_list_of_durations = [list(i) for i in np.zeros((len(patch_centers), 1), dtype='int')]
    # List with the right format [[0],[0],...,[0]], one list of durations per patch
    # The strategy to compute it:
    # We copy the list_of_timestamps
    # We go through the order of visit table, and for every patch on that:
    # - We sum one duration from the list_of_time_stamp, and add it to current visit duration
    # - If the patch changes we close the previous one and start a new one
    timestamps_copy = copy.deepcopy(list_of_timestamps)
    for i_visit in range(len(order_of_visits)):
        current_patch = order_of_visits[i_visit]
        if timestamps_copy[current_patch]:  # if it's not empty
            current_visit = timestamps_copy[current_patch].pop(-1)  # removes the visit from the list and returns it
            current_duration = current_visit[1] - current_visit[0]
            if current_patch != -1:
                if i_visit > 1:  # if it's not the first visit
                    previous_patch = order_of_visits[i_visit - 2]
                    if current_patch != previous_patch:  # if the worm changed career patch
                        adjusted_list_of_durations[previous_patch].append(0)  # close previous visit
                adjusted_list_of_durations[current_patch][
                    -1] += current_duration  # in any case add duration to relevant patch
    # Clean the adjusted durations
    for i_patch in range(len(adjusted_list_of_durations)):
        adjusted_list_of_durations[i_patch] = [nonzero for nonzero in adjusted_list_of_durations[i_patch] if
                                               nonzero != 0]

    return adjusted_list_of_durations


# Same as before but probably new visit structure?
def new_mvt_patch_visits(list_of_timestamps, patch_centers):
    """
    Takes a list of time stamps as spitted by single_traj_analysis and returns an adjusted version of those visits
    by aggregating successive visits to a same patch.
    This is the list of durations we will use to test the Marginal Value Theorem.
    """

    adjusted_list_of_durations = [list(i) for i in np.zeros((len(patch_centers), 1), dtype='int')]
    # List with the right format [[0],[0],...,[0]], one list of durations per patch
    # The strategy to compute it:
    # We copy the list_of_timestamps
    # We go through the order of visit table, and for every patch on that:
    # - If patch didn't change we compute one duration from the list_of_timestamps, and add it to current visit duration
    # - If the patch changes we close the previous one and start a new one
    for i_visit in range(len(list_of_timestamps)):
        current_patch = list_of_timestamps[i_visit][2]
        current_duration = list_of_timestamps[i_visit][1] - list_of_timestamps[i_visit][0]
        if i_visit > 1:  # if it's not the first visit
            previous_patch = list_of_timestamps[i_visit - 2][2]
            if current_patch != previous_patch:  # if the worm changed career patch
                adjusted_list_of_durations[previous_patch].append(0)  # close previous visit
        adjusted_list_of_durations[current_patch][-1] += current_duration  # in any case add duration to relevant patch
    # Clean the adjusted durations
    for i_patch in range(len(adjusted_list_of_durations)):
        adjusted_list_of_durations[i_patch] = [nonzero for nonzero in adjusted_list_of_durations[i_patch] if
                                               nonzero != 0]

    return adjusted_list_of_durations


def analyse_patch_visits(list_of_timestamps, adjusted_list_of_durations, patch_centers, first_xy):
    """
    Function that takes a list of timestamps as spitted out by single_traj_analysis, and the MVT adjusted version,
    and returns global variables on these visits.
    """

    duration_sum = 0  # this is to compute the avg duration of visits
    nb_of_visits = 0
    adjusted_duration_sum = 0
    adjusted_nb_of_visits = 0
    list_of_visited_patches = []
    furthest_patch_distance = 0
    furthest_patch_position = [0, 0]

    # Run through each patch
    for i_patch in range(len(list_of_timestamps)):
        # Visits info for average visit duration
        current_list_of_timestamps = pd.DataFrame(list_of_timestamps[i_patch])
        current_nb_of_visits = len(current_list_of_timestamps)
        if not current_list_of_timestamps.empty:
            duration_sum += np.sum(current_list_of_timestamps.apply(lambda t: t[1] - t[0], axis=1))
            nb_of_visits += current_nb_of_visits

        # Same but adjusted for multiple consecutive visits to same patch
        current_adjusted_list_of_durations = pd.DataFrame(adjusted_list_of_durations[i_patch])
        if not current_adjusted_list_of_durations.empty:
            adjusted_duration_sum += int(np.sum(current_adjusted_list_of_durations))
            adjusted_nb_of_visits += len(current_adjusted_list_of_durations)

        # Update list of visited patches and the furthest patch visited
        if current_nb_of_visits > 0:  # if the patch was visited at least once in this trajectory
            patch_distance_to_center = distance.euclidean(first_xy, patch_centers[i_patch])
            if patch_distance_to_center > furthest_patch_distance:
                furthest_patch_position = patch_centers[i_patch]
                furthest_patch_distance = distance.euclidean(first_xy, furthest_patch_position)
            list_of_visited_patches.append(i_patch)

    return duration_sum, nb_of_visits, list_of_visited_patches, furthest_patch_position, adjusted_duration_sum, adjusted_nb_of_visits


def make_results_per_id_table(data):
    """
    Takes our data table and returns a series of analysis regarding patch visits, one line per "worm", which in fact
    corresponds to one worm track, each worm/plate usually containing multiple tracks (the tracking is discontinuous
    and when the worm gets lost it starts a new ID).
    """
    track_list = np.unique(data["id_conservative"])
    nb_of_tracks = len(track_list)

    results_table = pd.DataFrame()
    old_folder = "caca"
    first_pos = [0, 0]
    for i_track in range(nb_of_tracks):
        # Handmade progress bar
        if i_track % 100 == 0 or i_track == nb_of_tracks - 1:
            print(i_track, "/", nb_of_tracks - 1)

        # Data from the dataframe
        current_track = track_list[i_track]
        current_data = data[data["id_conservative"] == current_track]
        current_data = current_data.reset_index()
        current_list_x = current_data["x"]
        current_list_y = current_data["y"]
        current_folder = list(current_data["folder"])[0]

        # First recorded position of each plate is first position of the first worm of the plate
        if current_folder != old_folder:
            first_pos = [current_list_x[0], current_list_y[0]]
        old_folder = current_folder

        # Getting to the metadata through the folder name in the data
        current_metadata = fd.folder_to_metadata(current_folder)
        list_of_densities = current_metadata["patch_densities"]

        # Computing the list of visits
        patch_list = current_metadata["patch_centers"]
        raw_visit_timestamps, order_of_visits = single_traj_analysis(current_data["patch"], current_data["frame"],
                                                                     patch_list)
        # Adjusting it for MVT analyses
        adjusted_raw_visits = mvt_patch_visits(raw_visit_timestamps, order_of_visits, patch_list)
        # Computing global variables
        duration_sum, nb_of_visits, list_of_visited_patches, furthest_patch_position, adjusted_duration_sum, adjusted_nb_of_visits = analyse_patch_visits(
            raw_visit_timestamps, adjusted_raw_visits, patch_list, first_pos)

        # Computing average speed
        average_speed_in, average_speed_out = avg_speed_analysis(current_data["patch"], current_data["frame"],
                                                                 current_data["distances"])

        # Fill up results table
        results_table.loc[i_track, "folder"] = current_folder
        results_table.loc[i_track, "condition"] = current_metadata["condition"][0]
        results_table.loc[i_track, "track_id"] = current_track
        results_table.loc[i_track, "total_tracked_time"] = len(current_list_x)  # number of tracked time steps
        results_table.loc[i_track, "raw_visits"] = str(raw_visit_timestamps)  # all visits of all patches
        results_table.loc[i_track, "better_raw_visits"] = str(
            sort_visits_chronologically(raw_visit_timestamps))  # all visits of all patches
        results_table.loc[i_track, "order_of_visits"] = str(order_of_visits)  # patch order of visits
        results_table.loc[i_track, "total_visit_time"] = duration_sum  # total duration of visits
        results_table.loc[i_track, "nb_of_visits"] = nb_of_visits  # total nb of visits
        results_table.loc[i_track, "list_of_visited_patches"] = str(list_of_visited_patches)  # index of patches visited
        results_table.loc[i_track, "first_recorded_position"] = str(
            first_pos)  # first position for the whole plate (used to check trajectories)
        results_table.loc[i_track, "first_frame"] = current_data["frame"][0]
        results_table.loc[i_track, "first_tracked_position_patch"] = current_data["patch"][
            0]  # patch where the worm is when tracking starts (-1 = outside): one value per id
        results_table.loc[i_track, "last_frame"] = current_data["frame"].iloc[-1]
        results_table.loc[i_track, "last_tracked_position"] = str([current_list_x.iloc[-1], current_list_y.iloc[
            -1]])  # last position for the current worm (used to check tracking)
        results_table.loc[i_track, "last_tracked_position_patch"] = current_data["patch"].iloc[
            -1]  # patch where the worm is when tracking stops (-1 = outside)
        results_table.loc[i_track, "furthest_patch_position"] = str(furthest_patch_position)
        results_table.loc[i_track, "adjusted_raw_visits"] = str(adjusted_raw_visits)
        results_table.loc[i_track, "adjusted_total_visit_time"] = adjusted_duration_sum
        results_table.loc[i_track, "adjusted_nb_of_visits"] = adjusted_nb_of_visits
        results_table.loc[i_track, "average_speed_inside"] = average_speed_in
        results_table.loc[i_track, "average_speed_outside"] = average_speed_out

    return results_table


def nb_bad_events(data):
    """
    Takes one plate of our results_per_id table, and returns the number of times that the tracking stopped in one place
    and restarted in a different place (stopped inside and restarted outside, etc.)
    """
    nb_of_holes = len(data) - 1
    nb_of_bad_events = 0
    for i_hole in range(nb_of_holes):
        position_start_hole = data["first_tracked_position_patch"][i_hole]
        position_end_hole = data["last_tracked_position_patch"][i_hole + 1]
        if position_start_hole != position_end_hole:
            nb_of_bad_events += 1
    return nb_of_bad_events


# TODO change visit structure in results_per_id already, and adapt analyse_patch_visits function
def sort_visits_chronologically(by_patch_list_of_visits):
    """
    So... it's a shame but to fill the holes it would be better to have lists in a chronological order, with the third
    element being the patch. So I'll transform our list of visits into that here even though it would be better to actually
    change that in the results_per_id.
    Takes a list_of_visits: [[[v0,v1],...],[[v0,v1],...]...] with one sub-list per patch containing the list of visits to this patch
    Returns a list [[v0,v1,p0],[v0,v1,p1],...] with all the visits' time stamps SORTED by starting time, and the patch to which they were made
    """
    # Add patch info to all the visits
    for i_patch in range(len(by_patch_list_of_visits)):
        for i_visit in range(len(by_patch_list_of_visits[i_patch])):
            if by_patch_list_of_visits[i_patch][i_visit]:  # if it's not empty
                by_patch_list_of_visits[i_patch][i_visit].insert(2, i_patch)  # add it in second position
                # (here we use insert instead of append because for aggregated_visits including transits, the last
                # element should not be the patch but the starting time of the transits that were included)

    # Concatenate all patch sublists
    chrono_list_of_visits = []
    for i_patch in range(len(by_patch_list_of_visits)):
        for i_visit in range(len(by_patch_list_of_visits[i_patch])):
            if by_patch_list_of_visits[i_patch][i_visit]:  # if it's not empty
                chrono_list_of_visits.append(by_patch_list_of_visits[i_patch][i_visit])

    # Sort it according to first element
    chrono_list_of_visits = sorted(chrono_list_of_visits, key=lambda x: x[0])

    return chrono_list_of_visits


def sort_visits_by_patch(chronological_list_of_visits, nb_of_patches):
    """
    Take a list of visits in the chronological format: [[t0, t1, p], [t0, ...], ...] (with t0 start of visit, t1 end of
    visit, and visits sorted based on t0 values.
    And return a list of visits in the sublist_by_patch format: [ [[t0,t1],[t0,t1]], [[t0,t1]], []] with one sub-list
    per patch, and for each of those sublists the beginning (t0) and end (t1).
    """
    # Create one sublist per patch
    bypatch_list_of_visits = [[] for _ in range(nb_of_patches)]
    for visit in chronological_list_of_visits:
        # Fill the right sublist with the start / end info
        if visit[2] > nb_of_patches - 1:
            print("ayayay")
        bypatch_list_of_visits[visit[2]].append([visit[0], visit[1]])
    for i_patch in range(len(bypatch_list_of_visits)):
        # For each patch, sort the visits chronologically based on visit start
        bypatch_list_of_visits[i_patch] = sorted(bypatch_list_of_visits[i_patch], key=lambda x: x[0])
    return bypatch_list_of_visits


def fill_holes(data_per_id):
    """
    Function that takes a list of visit time stamps for a plate [[[v0,v1, p0],[v2,v3, p1],...],[[v10,v11, p2],...],...],
    with one sublist of time stamps for each track of the same plate, third element being the patch of the visit
    Returns a new list in the format [[v0,v1, p0],[v2,v3, p1],...] where the last visit of a track and the first of the next
    were fused into one in the case where the tracking started and stopped in the same patch,
    and another new list in the same format with the same concept but for transit times
    """
    # Sort the tracks for them to be in the order in which they were tracked otherwise it's a mess
    data_per_id = data_per_id.sort_values(by=['first_frame']).reset_index()

    # The original dataset: each track sublist = [[t,t],[t,t],...],[[t,t],...],...] with one sublist per patch
    # We want to remove all empty visits: it means the worm was outside before and after the end of the visit, so it
    # should not affect aggregation of visits.
    # However, we do not want to remove tracks that do not have visits, because we need to know in which track we are
    # to access variables like the last frame or the position of the worm at the end of a track.
    list_of_visits = [json.loads(data_per_id["raw_visits"][i_track]) for i_track in
                      range(len(data_per_id["raw_visits"]))]
    better_list_of_visits = [sort_visits_chronologically(list_of_visits[i_track]) for i_track in
                             range(len(list_of_visits))]
    for i in range(len(better_list_of_visits)):
        list_of_visits[i] = [nonempty for nonempty in better_list_of_visits[i] if nonempty != []]

    # Initializing loop variables
    corrected_list_of_visits = []
    corrected_list_of_transits = []  # size minus 1 nb of transitions
    nb_of_tracks = len(list_of_visits)
    i_track = 0
    i_next_track = 1
    init_visit_at_one = False

    while i_track < nb_of_tracks:  # for each track
        # print("==== i_track = ", i_track, " / ", nb_of_tracks)
        nb_of_visits = len(list_of_visits[i_track])  # update number of visits for current track
        i_visit = 0
        skipped_empty_tracks = False  # True if empty tracks have been skipped
        if init_visit_at_one:  # this is the case where the first visit of the current track was already treated by being fused to previous visit
            i_visit = 1
            init_visit_at_one = False
        while i_visit < nb_of_visits and i_track < nb_of_tracks:  # for each visit of that track
            current_visit_start = list_of_visits[i_track][i_visit][0]
            current_visit_end = list_of_visits[i_track][i_visit][1]
            current_patch = list_of_visits[i_track][i_visit][2]

            next_visit_start = -1000000
            next_visit_end = -1000000
            next_next_visit_start = -1000000

            # If this visit is the last of a track, and not of the last track, then we might have to aggregate it to the next
            is_last_visit = i_visit == nb_of_visits - 1
            is_last_nonempty_track = (i_track == nb_of_tracks - 1) or (
                    (i_next_track == nb_of_tracks - 1) and len(list_of_visits[i_next_track]) == 0)

            # We look for the next visit start and end
            if is_last_visit and not is_last_nonempty_track:  # if this is the last visit of the track and not the last track
                while not list_of_visits[
                    i_next_track] and i_next_track < nb_of_tracks - 1:  # go to next non-empty track
                    i_next_track += 1
                    skipped_empty_tracks = True
                if list_of_visits[i_next_track]:  # if a non-empty track was found in the end
                    next_visit_start = list_of_visits[i_next_track][0][
                        0]  # next visit is first visit of next non-empty track
                    next_visit_end = list_of_visits[i_next_track][0][1]
                    # Find next next visit (for updating transit durations in case of visit aggregation)
                    if len(list_of_visits[i_next_track]) >= 2:  # if there is a next next visit in the next track
                        next_next_visit_start = list_of_visits[i_next_track][1][0]
                    elif i_next_track < nb_of_tracks - 2:  # otherwise look for a next next visit in the next next track
                        i_next_next_track = i_next_track + 1
                        while not list_of_visits[i_next_next_track] and i_next_next_track < nb_of_tracks - 1:
                            i_next_next_track += 1
                        if list_of_visits[i_next_next_track]:  # if a next next visit was found in the end
                            next_next_visit_start = list_of_visits[i_next_next_track][0][0]
            # Otherwise we won't be looking for a next next visit
            elif not is_last_visit:  # not the last visit (so if it's a middle visit and not the last of the last track)
                next_visit_start = list_of_visits[i_track][i_visit + 1][0]
                next_visit_end = list_of_visits[i_track][i_visit + 1][1]
            # Otherwise, it's the last visit of the last track we won't need a next_visit to be defined

            # Update this
            is_last_nonempty_track = (i_track == nb_of_tracks - 1) or (
                    (i_next_track == nb_of_tracks - 1) and len(list_of_visits[i_next_track]) == 0)

            # Fill the transits list
            if not (
                    is_last_visit and is_last_nonempty_track):  # if we're not in the last visit of the last non-empty track
                # If it's the last visit of not-the-last-track
                if is_last_visit:
                    # Case where the tracking stops when the worm is out (so after end of last patch visit), and the worm is still out when it restarts
                    if current_visit_end < data_per_id["last_frame"][i_track]:
                        # If the tracking restarts with the worm still out, it's counted as transit
                        if data_per_id["first_tracked_position_patch"][i_next_track] == -1:
                            corrected_list_of_transits.append([current_visit_end, next_visit_start, -1])
                        # BAD HOLES: IF THE TRACKING ENDS OUT IT SHOULD NOT START ELSEWHERE
                        # So if the tracking restarts elsewhere, just end current transit at last frame
                        else:
                            corrected_list_of_transits.append(
                                [current_visit_end, data_per_id["last_frame"][i_next_track - 1], -1])
                    # Case where the tracking stops when the worm is inside, and it's not the last hole
                    if current_visit_end == data_per_id["last_frame"][i_track]:
                        # Case where the next track starts while the worm is still in, and we didn't have sneaky empty tracks in the middle
                        # We also check that there is indeed a next next visit otherwise this line makes no sense
                        if data_per_id["first_tracked_position_patch"][
                            i_next_track] != -1 and not skipped_empty_tracks and next_next_visit_start > 0:
                            # In this case we take the transit for the next visit now because the visit list loop will skip
                            # this value for the next loop
                            corrected_list_of_transits.append([next_visit_end, next_next_visit_start, -1])
                        # BAD HOLES: IF THE TRACKING ENDS INSIDE IT SHOULD NOT RESTART OUT (empty tracks)
                        # Case where the next track starts outside even though this one ended inside, and there is a visit
                        # somewhere in the next tracks (otherwise it's already taken care of at the very end)
                        # In that case the skipped_empty_track is True
                        if skipped_empty_tracks:
                            corrected_list_of_transits.append([current_visit_end, next_visit_start, -1])
                # If it's not the last visit of any track then it's a piece of cake
                else:
                    corrected_list_of_transits.append([current_visit_end, next_visit_start, -1])

            # Fill the visits list
            # If this is the end of this track, and not the last track, and the tracking stops during the visit
            if is_last_visit and not is_last_nonempty_track and current_visit_end == data_per_id["last_frame"][i_track]:
                # Check if the hole in the tracking is valid (ends and then starts in the same patch)
                if data_per_id["last_tracked_position_patch"][i_track] == data_per_id["first_tracked_position_patch"][
                    i_track + 1]:
                    # We increase i_track by 2, to not look at next visit as it has already been counted
                    init_visit_at_one = True
                    # Then the current visit in fact ends at the end of the first visit of the next track
                    corrected_list_of_visits.append([current_visit_start, next_visit_end, current_patch])
                # Else if the hole is not valid, don't aggregate the visits
                else:
                    corrected_list_of_visits.append([current_visit_start, current_visit_end, current_patch])

            # Else if it's not the end of a track, or it's the end of the last track, or the tracking stops outside a patch
            else:  # then just copy the end of the current visit in the corresponding corrected table visit
                corrected_list_of_visits.append([current_visit_start, current_visit_end, current_patch])

            i_visit += 1

        i_track = i_next_track
        i_next_track = i_track + 1  # update next_track index

    # Make everything a transit if there are no visits in the track
    if not corrected_list_of_visits:
        corrected_list_of_transits = [[data_per_id["first_frame"][0], data_per_id["last_frame"].iloc[-1], -1]]
    # Add first and last transit if needed
    else:
        if data_per_id["first_tracked_position_patch"][0] == -1:  # if worm starts the first track outside
            # Initialize first transit at first frame and end it at the beginning of the first visit
            corrected_list_of_transits = [[data_per_id["first_frame"][0],
                                           corrected_list_of_visits[0][0], -1]] + corrected_list_of_transits
        if data_per_id["last_tracked_position_patch"].iloc[-1] == -1:  # if worm ends last track outside
            # Initialize last transit at the end of the last visit and end it at last frame
            corrected_list_of_transits = corrected_list_of_transits + [[corrected_list_of_visits[-1][1],
                                                                        data_per_id["last_frame"].iloc[-1], -1]]

    return corrected_list_of_visits, corrected_list_of_transits


def make_results_per_plate(data_per_id, trajectories):
    """
    Function that takes our results_per_id table as an input, and will output a table with info for each plate:
        - folder
        - condition
        - length of the video (last - first time step)
        - number of tracked time steps
        - number of holes in the tracking
        - proportion of the frame numbers that have double tracking (if high, probably two worms)
        - an invalid tracking event column, whose content is described below
    AND THEN similar columns as in results table but aggregating visits as such:
        - if the tracking stops and restarts in the same situation (stops and restarts outside, or in the same patch),
        then the time for which the tracking stopped is counted as having been spent in the situation (so if the track
        stops outside of a patch, and restarts outside too, the time is counted as exploration)
        - if the tracking stops and restarts in different situations (worm "disappears" outside of a patch and
        "reappears" inside), the time spent outside of the tracking area is not counted, and the "invalid tracking
        event" column is non-zero
    and it also returns transit times aggregated in the same way!
    """
    results_per_plate = pd.DataFrame()
    list_of_plates = np.unique(data_per_id["folder"])
    for i_plate in range(len(list_of_plates)):
        if i_plate % 10 == 0:
            print(i_plate, " / ", len(list_of_plates))
        # Import tables
        current_folder = list_of_plates[i_plate]
        current_trajectory = trajectories[trajectories["folder"] == current_folder]
        current_metadata = fd.folder_to_metadata(current_folder)
        current_data = data_per_id[data_per_id["folder"] == current_folder].reset_index()

        # Loading visit info from data_per_id
        patch_list = current_metadata["patch_centers"]

        # Aggregating visits when worm disappears and then reappears in the same place
        aggregated_visit_timestamps, aggregated_transit_timestamps = fill_holes(current_data)

        # Adjusting it for MVT analyses
        adjusted_raw_durations = new_mvt_patch_visits(aggregated_visit_timestamps, patch_list)

        # Computing average speed by doing a weighted average of the average speeds in each track (weight is relative tracking time)
        # Doing it like this allows us to avoid the "holes in tracking" issues because they are excluded from the "id" slicing
        average_speed_in = np.sum((current_data["average_speed_inside"] * current_data["total_tracked_time"]) / np.sum(
            current_data["total_tracked_time"]))
        average_speed_out = np.sum(
            (current_data["average_speed_outside"] * current_data["total_tracked_time"]) / np.sum(
                current_data["total_tracked_time"]))

        # Fill up the table
        results_per_plate.loc[i_plate, "folder"] = current_folder
        results_per_plate.loc[i_plate, "condition"] = current_data["condition"][0]
        results_per_plate.loc[i_plate, "total_video_time"] = np.max(current_data["last_frame"]) - np.min(
            current_data["first_frame"])
        results_per_plate.loc[i_plate, "total_tracked_time"] = np.sum(current_data["total_tracked_time"])
        results_per_plate.loc[i_plate, "nb_of_holes"] = len(current_data)
        results_per_plate.loc[i_plate, "nb_of_bad_events"] = nb_bad_events(current_data)
        results_per_plate.loc[i_plate, "avg_proportion_double_frames"] = (len(current_trajectory["frame"]) / len(
            np.unique(current_trajectory["frame"]))) - 1
        results_per_plate.loc[i_plate, "total_visit_time"] = np.sum(
            [pd.DataFrame(aggregated_visit_timestamps).apply(lambda t: t[1] - t[0], axis=1)])
        results_per_plate.loc[i_plate, "total_transit_time"] = np.sum(
            [pd.DataFrame(aggregated_transit_timestamps).apply(lambda t: t[1] - t[0], axis=1)])
        results_per_plate.loc[i_plate, "no_hole_visits"] = str(aggregated_visit_timestamps)
        results_per_plate.loc[i_plate, "aggregated_raw_transits"] = str(aggregated_transit_timestamps)
        results_per_plate.loc[i_plate, "nb_of_visits"] = len(aggregated_visit_timestamps)
        results_per_plate.loc[i_plate, "mvt_raw_visits"] = str(adjusted_raw_durations)
        results_per_plate.loc[i_plate, "mvt_nb_of_visits"] = len(adjusted_raw_durations)
        results_per_plate.loc[i_plate, "average_speed_inside"] = average_speed_in
        results_per_plate.loc[i_plate, "average_speed_outside"] = average_speed_out

    return results_per_plate


def add_aggregate_visit_info_to_results(results, threshold_list):
    """
    Will add columns in results for each threshold in threshold list, both including or excluding transits.
    Will create new "aggregated visits", where visits are fused if separated by less than a temporal threshold.
    See aggregate_visits function for examples.
    Created columns:
        - aggregated_visits_thresh_X: list of visit durations (one sublist per patch), visits separated by less than X are merged
                Structure: output of aggregate_visits
                           one sublist per patch, with one sublist per visit to this patch, and in each visit sublist:
                           visit start, visit end, list of time stamps for the transits [[t0,t1], [t0,t1], ...]
        - aggregated_visits_thresh_X_visit_durations: corresponding duration of each visit (so excluding transit durations)
        - aggregated_visits_thresh_X_nb_of_visits: corresponding number of visits for each patch
        - aggregated_visits_thresh_X_leaving_events: corresponding list of visit events (see architecture in ana.leaving_events_time_stamps)

    """
    # Create the columns
    for i_thresh in range(len(threshold_list)):
        # For visit list
        thresh = threshold_list[i_thresh]
        results["aggregated_visits_thresh_" + str(thresh)] = pd.DataFrame(np.zeros(len(results)))
        results["aggregated_visits_thresh_" + str(thresh) + "_visit_durations"] = pd.DataFrame(np.zeros(len(results)))
        results["aggregated_visits_thresh_" + str(thresh) + "_nb_of_visits"] = pd.DataFrame(np.zeros(len(results)))
        results["aggregated_visits_thresh_" + str(thresh) + "_total_visit_time"] = pd.DataFrame(np.zeros(len(results)))
        results["aggregated_visits_thresh_" + str(thresh) + "_leaving_events_time_stamps"] = pd.DataFrame(
            np.zeros(len(results)))

    # We run this for each plate separately to keep different patches separate
    for i_plate in range(len(results)):
        current_plate = results["folder"][i_plate]
        current_results = results[results["folder"] == current_plate].reset_index(drop=True)
        this_plate_visits = fd.load_list(current_results, "no_hole_visits")
        this_plate_condition = current_results["condition"][0]
        for i_thresh in range(len(threshold_list)):
            thresh = threshold_list[i_thresh]
            aggregated_visits_w_transit_info = ana.aggregate_visits(this_plate_visits, this_plate_condition, thresh,
                                                                    return_duration=False)
            aggregated_visits_durations = ana.aggregate_visits(this_plate_visits, this_plate_condition, thresh,
                                                               return_duration=True)
            leaving_events = ana.leaving_events_time_stamps(aggregated_visits_w_transit_info)

            # Sort them chronologically like other visits (works only with the ones that still have time stamps)
            aggregated_visits_w_transit_info = sort_visits_chronologically(aggregated_visits_w_transit_info)

            # Fill the columns
            results.loc[i_plate, "aggregated_visits_thresh_" + str(thresh)] = str(aggregated_visits_w_transit_info)
            results.loc[i_plate, "aggregated_visits_thresh_" + str(thresh) + "_visit_durations"] = str(
                aggregated_visits_durations)
            results.loc[i_plate, "aggregated_visits_thresh_" + str(thresh) + "_nb_of_visits"] = np.sum(
                [len(sublist) for sublist in aggregated_visits_durations])
            results.loc[i_plate, "aggregated_visits_thresh_" + str(thresh) + "_total_visit_time"] = np.sum(
                [np.sum(sublist) for sublist in aggregated_visits_durations])
            results.loc[i_plate, "aggregated_visits_thresh_" + str(thresh) + "_leaving_events_time_stamps"] = str(
                leaving_events)

    return results


def exclude_invalid_videos(trajectories, results_per_plate):
    cleaned_results = results_per_plate[results_per_plate["total_video_time"] >= 10000]
    cleaned_results = cleaned_results[cleaned_results["avg_proportion_double_frames"] <= 0.01]
    valid_folders = np.unique(cleaned_results["folder"])
    cleaned_traj = pd.DataFrame(columns=trajectories.columns)
    for plate in valid_folders:
        cleaned_traj = pd.concat([cleaned_traj, trajectories[trajectories["folder"] == plate]])
    return cleaned_traj, cleaned_results


def generate_trajectories(path):
    # Retrieve trajectories from the folder path and save them in one dataframe
    trajectories = fd.trajmat_to_dataframe(fd.path_finding_traj(path))
    print("Finished retrieving trajectories")
    # Add a column with
    print("Computing where the worm is...")
    trajectories["patch"] = in_patch_list(trajectories)
    print("Finished computing in which patch the worm is at each time step")
    print("Computing distances...")
    # Add a column with the distance the worm crawled since last time step, for each time step
    # It should put 0 for the first point of every folder, but record distance even when there's a tracking hole
    # Distances are computed now because they are used for average speed analysis in results_per_id.
    # Doing speed analysis later is tiring because of the tracking holes that are not in the results_per_plate
    # table anymore.
    trajectories["distances"] = trajectory_distances(trajectories)
    trajectories.to_csv(path + "trajectories.csv")
    print("Finished computing distance covered by the worm at each time step")
    return 0


def generate_results_per_id(path):
    print("Building results_per_id...")
    trajectories = pd.read_csv(path + "trajectories.csv")
    print("Starting to build results_per_id from trajectories...")
    results_per_id = make_results_per_id_table(trajectories)
    print("Finished!")
    results_per_id.to_csv(path + "results_per_id.csv")
    return 0


def generate_results_per_plate(path):
    print("Aggregating and preprocessing results per plate...")
    print("Retrieving results...")
    trajectories = pd.read_csv(path + "trajectories.csv",
                               index_col=0)  # index_col=0 is to prevent the addition of new index columns at each import
    results_per_id = pd.read_csv(path + "results_per_id.csv", index_col=0)
    print("Starting to build results_per_plate from results_per_id...")
    results_per_plate = make_results_per_plate(results_per_id, trajectories)
    results_per_plate.to_csv(path + "results_per_plate.csv")
    return 0


def generate_clean_tables_and_speed(path):
    print("Retrieving results and trajectories...")
    results_per_plate = pd.read_csv(path + "results_per_plate.csv", index_col=0)
    trajectories = pd.read_csv(path + "trajectories.csv", index_col=0)
    print("Cleaning results...")
    clean_trajectories, clean_results = exclude_invalid_videos(trajectories, results_per_plate)
    clean_results.to_csv(path + "clean_results.csv")
    print("Computing speeds...")
    # For faster execution, we only compute speeds here (and not at the same time as distances), when invalid
    # trajectories have been excluded
    clean_trajectories["speeds"] = trajectory_speeds(clean_trajectories)
    clean_trajectories.to_csv(path + "clean_trajectories.csv")
    return 0


def generate_aggregated_visits(path, threshold_list):
    print("Retrieving results and trajectories...")
    clean_results = pd.read_csv(path + "clean_results.csv", index_col=0).reset_index()
    print("Adding aggregated visits info for thresholds " + str(threshold_list) + "...")
    new_results = add_aggregate_visit_info_to_results(clean_results, threshold_list)
    new_results.to_csv(path + "clean_results.csv", index=False)
    print("Done!")
    return new_results  # return this because this function is also used dynamically


def generate(starting_from=""):
    """
    Will generate the data tables starting more or less from scratch.
    Argument = from which level to regenerate stuff.
    Returns path.
    """

    # Data path
    if fd.is_linux():  # Linux path
        path = "/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/"
    else:  # Windows path
        path = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/Results_minipatches_20221108_clean_fp_less/"

    if starting_from == "trajectories":
        generate_controls(path)
        generate_trajectories(path)
        generate_results_per_id(path)
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, parameters.threshold_list)

    elif starting_from == "results_per_id":
        generate_results_per_id(path)
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, parameters.threshold_list)

    elif starting_from == "results_per_plate":
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, parameters.threshold_list)

    elif starting_from == "clean":
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, parameters.threshold_list)

    return path
