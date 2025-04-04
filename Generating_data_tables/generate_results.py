from scipy.spatial import distance
import pandas as pd
from itertools import groupby
import copy
import json
import numpy as np
import datatable as dt

# My code
import analysis as ana
import find_data as fd
from Parameters import parameters as param


def single_traj_analysis(which_patch_list, list_of_time_stamps, patch_centers):
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

    teleportation_count = 0

    # In list_of_visit_times, we will have one list per patch, containing the beginning and end of successive visits to that patch
    # [0,0] means "the worm was not in that patch in the previous timestep"
    # As soon as the worm enters the patch, first zero is updated to be current frame
    # As soon as the worm leaves the patch, second zero becomes current frame, and a new [0,0] is added to this patch's list
    # These [0,0] are added for computational ease and will be removed in the end
    list_of_visit_times = [[list(i)] for i in np.zeros((len(patch_centers), 2), dtype='int')]
    # List with the right format [[[0,0]],[[0,0]],...,[[0,0]]]
    # => each [0,0] is an empty visit that begins and ends in 0
    # in the end we want

    # Order in which the patches were visited (should have as many elements as list_of_visit_times)
    # (function that removes consecutive duplicates, [1,2,2,3] => [1,2,3]
    # !!! contains -1 whenever the worm went outside !!!
    order_of_visits = [i[0] for i in groupby(which_patch_list)]

    # Reset index of frame table, otherwise frames are not found once it's not the first worm
    list_of_time_stamps = list_of_time_stamps.reset_index()

    # In order to compute when visits to patches start, end, and how long they last, we avoid looking at every line by
    # only looking at the indexes where the patch value changes:
    which_patch_array = np.array(which_patch_list)
    event_indexes = np.where(which_patch_array[:-1] != which_patch_array[1:])[0]
    # (this formula works by looking at differences between the list shifted by one to the left or to the right)
    # (for [1,1,2,2,6,6,6,6] it will return [1,3])

    # If the timeline starts with a visit, start it now!
    if which_patch_array[0] != -1:
        list_of_visit_times[which_patch_array[0]][-1][0] = np.round(list_of_time_stamps["time"][0], 2)

    # We go through every event
    patch_where_it_is = which_patch_array[0]
    for time in event_indexes:
        patch_where_it_was = patch_where_it_is  # memorize the patch where it was
        patch_where_it_is = which_patch_array[min(len(which_patch_array) - 1, time + 1)]  # update patch, max to not exceed size
        current_time = np.round(list_of_time_stamps["time"][time], 2)  # ONLY SLIGHT ROUNDING BECAUSE RISK OF DOUBLE FRAMES!!
        # Worm just exited a patch
        if patch_where_it_is == -1:  # worm currently out
            if patch_where_it_was != patch_where_it_is:  # was inside before
                list_of_visit_times[patch_where_it_was][-1][1] = current_time  # end the previous visit
                list_of_visit_times[patch_where_it_was].append([0, 0])  # add new visit to the previous patch
        # Worm just entered a patch
        if patch_where_it_is != -1:  # worm currently inside
            if patch_where_it_was == -1:  # it's a new visit (first patch or was outside before)
                if len(list_of_visit_times) < patch_where_it_is:
                    print('WARNING: In single_traj_analysis(), the patch index called was higher than the number of patches.')
                else:
                    list_of_visit_times[patch_where_it_is][-1][0] = current_time  # begin visit in current patch sublist
            # If the worm used to be in a different food patch, and transited without going through the outside
            # print a warning! this is not supposed to happen.
            else:
                print("WARNING: There's a worm teleporting from patch to patch!!! time in video: ", current_time)
                teleportation_count += 1
                list_of_visit_times[patch_where_it_was][-1][1] = current_time  # end the previous visit
                list_of_visit_times[patch_where_it_was].append([0, 0])  # add new visit to the previous patch
                list_of_visit_times[patch_where_it_is][-1][0] = current_time  # begin visit in current patch sublist

    # Close the last visit
    if which_patch_array[-1] != -1:  # if the worm is inside, last visit hasn't been closed (no exit event)
        list_of_visit_times[patch_where_it_is][-1][1] = np.round(list_of_time_stamps["time"].iloc[-1], 2)  # = last frame
    # Clean the table (remove the zeros because they're just here for the duration algorithm)
    for i_patch in range(len(list_of_visit_times)):
        list_of_visit_times[i_patch] = [nonzero for nonzero in list_of_visit_times[i_patch] if nonzero != [0, 0]]

    s = [l[i][1] - l[i][0] for l in list_of_visit_times for i in range(len(l))]
    if len(np.where(np.array(s) < 0)[0]) > 0:
        print("Some visits are negative in a folder! Starting times of those visits: ",
              np.array(list_of_time_stamps)[np.where(np.array(s) < 0)[0]])

    return list_of_visit_times, order_of_visits, teleportation_count


def analyse_patch_visits(list_of_timestamps, patch_centers, first_xy):
    """
    Function that takes a list of timestamps as spitted out by single_traj_analysis,
    and returns global variables on these visits.
    """

    duration_sum = 0  # this is to compute the avg duration of visits
    nb_of_visits = 0
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

        # Update list of visited patches and the furthest patch visited
        if current_nb_of_visits > 0:  # if the patch was visited at least once in this trajectory
            patch_distance_to_center = distance.euclidean(first_xy, patch_centers[i_patch])
            if patch_distance_to_center > furthest_patch_distance:
                furthest_patch_position = patch_centers[i_patch]
                furthest_patch_distance = distance.euclidean(first_xy, furthest_patch_position)
            list_of_visited_patches.append(i_patch)

    return duration_sum, nb_of_visits, list_of_visited_patches, furthest_patch_position


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
    time_bug = "nothing_wrong"
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

        # Only when the folder changes
        if current_folder != old_folder:
            # First recorded position of each plate is first position of the first worm of the plate
            first_pos = [current_list_x[0], current_list_y[0]]
            print(current_folder)
            # Update for the next folder change!
            old_folder = current_folder

        # Getting to the metadata through the folder name in the data
        current_metadata = fd.folder_to_metadata(current_folder)

        if "model" in current_folder:  # for modeled data, use centroid info because i did not model fake silhouettes
            which_patch_list = current_data.reset_index(drop=True)["patch_centroid"]
        else:
            which_patch_list = current_data.reset_index(drop=True)["patch_silhouette"]

        # Computing the list of visits
        patch_list = current_metadata["patch_centers"]
        raw_visit_timestamps, order_of_visits, nb_of_teleportations = single_traj_analysis(which_patch_list,
                                                                      current_data["time"],
                                                                      patch_list)
        # Computing global variables
        duration_sum, nb_of_visits, list_of_visited_patches, furthest_patch_position = analyse_patch_visits(raw_visit_timestamps, patch_list, first_pos)

        # Computing average speed
        average_speed_in, average_speed_out = avg_speed_analysis(which_patch_list,
                                                                 current_data["time"],
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
        results_table.loc[i_track, "first_frame"] = current_data["time"].iloc[0]
        results_table.loc[i_track, "first_tracked_position_patch"] = which_patch_list[
            0]  # patch where the worm is when tracking starts (-1 = outside): one value per id
        results_table.loc[i_track, "last_frame"] = current_data["time"].iloc[-1]
        results_table.loc[i_track, "last_tracked_position"] = str([current_list_x.iloc[-1], current_list_y.iloc[
            -1]])  # last position for the current worm (used to check tracking)
        results_table.loc[i_track, "last_tracked_position_patch"] = which_patch_list.iloc[
            -1]  # patch where the worm is when tracking stops (-1 = outside)
        results_table.loc[i_track, "furthest_patch_position"] = str(furthest_patch_position)
        results_table.loc[i_track, "average_speed_inside"] = average_speed_in
        results_table.loc[i_track, "average_speed_outside"] = average_speed_out
        results_table.loc[i_track, "nb_of_teleportations"] = nb_of_teleportations

    return results_table


def nb_bad_events(data):
    """
    Takes one plate of our results_per_id table, and returns the number of times that the tracking stopped in one place
    and restarted in a different place (stopped inside and restarted outside, etc.)
    """
    nb_of_holes = len(data) - 1
    nb_all_bad_events = 0
    length_all_bad_events = 0
    nb_long_bad_events = 0
    length_long_bad_events = 0
    for i_hole in range(nb_of_holes):
        position_start_hole = data["last_tracked_position_patch"][i_hole]
        position_end_hole = data["first_tracked_position_patch"][i_hole + 1]
        if position_start_hole != position_end_hole:
            nb_all_bad_events += 1
            length = data["first_frame"][i_hole + 1] - data["last_frame"][i_hole]
            length_all_bad_events += length
            if length >= param.hole_filling_threshold:
                nb_long_bad_events += 1
                length_long_bad_events += length
    return nb_all_bad_events, length_all_bad_events, nb_long_bad_events, length_long_bad_events


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
            return False
        bypatch_list_of_visits[int(visit[2])].append([visit[0], visit[1]])
    for i_patch in range(len(bypatch_list_of_visits)):
        # For each patch, sort the visits chronologically based on visit start
        bypatch_list_of_visits[i_patch] = sorted(bypatch_list_of_visits[i_patch], key=lambda x: x[0])
    return bypatch_list_of_visits


def avg_speed_analysis(which_patch_list, list_of_times, distance_list):
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
    list_of_times = list_of_times.reset_index()["time"]
    distance_list = distance_list.reset_index()["distances"]
    distance_inside_sum = 0
    time_inside_sum = 0
    distance_outside_sum = 0
    time_outside_sum = 0
    for i in range(1, len(which_patch_list)):
        if which_patch_list[i] == -1:
            time_outside_sum += list_of_times[i] - list_of_times[i - 1]
            distance_outside_sum += distance_list[i]
        else:
            time_inside_sum += list_of_times[i] - list_of_times[i - 1]
            distance_inside_sum += distance_list[i]
    if (distance_inside_sum == 0 and time_inside_sum != 0) or (distance_inside_sum != 0 and time_inside_sum == 0) or (
            distance_outside_sum == 0 and time_outside_sum != 0) or (
            distance_outside_sum != 0 and time_outside_sum == 0):
        print("There's an issue with the avg_speed_analysis function!")
    # Now if the worm has spent zero time inside or outside, change the 0 to a nan, so that it's not considered in averages
    # (this might happen often because this function is called on tracks, which are fractions of the trajectory)
    if distance_inside_sum == 0 and time_inside_sum == 0:
        distance_inside_sum = np.nan
    if distance_outside_sum == 0 and time_outside_sum == 0:
        distance_outside_sum = np.nan
    return distance_inside_sum / max(1, time_inside_sum), distance_outside_sum / max(1,
                                                                                     time_outside_sum)  # the max is to prevent division by zero


def fill_holes(data_per_id):
    # Sort the tracks for them to be in the order in which they were tracked otherwise it's a mess
    data_per_id = data_per_id.sort_values(by=['first_frame']).reset_index()

    # Remove "np.int64()" or "np.float64()" that might have appeared during the string conversion
    data_per_id["raw_visits"] = [data_per_id["raw_visits"][i].replace("np.float64(", "").replace(")", "") for i in
                                 range(len(data_per_id))]
    data_per_id["raw_visits"] = [data_per_id["raw_visits"][i].replace("np.int64(", "").replace(")", "") for i in
                                 range(len(data_per_id))]

    # Convert first/last frames to int
    first_times = [np.round(data_per_id["first_frame"][i], 2) for i in range(len(data_per_id["first_frame"]))]
    last_times = [np.round(data_per_id["last_frame"][i], 2) for i in range(len(data_per_id["last_frame"]))]

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

    # We first go through the tracks and convert the events to something more convenient
    # The goal is that after this loop, we have a list of events with the following format:
    #           [t0, t1, id]
    #   with t0: timestamp of the start of the event
    #        t1: timestamp of the end of the event
    #        id: type of event. -1: transit between patches
    #                           Integer >= 0: visit to patch number id.
    list_of_events = []
    for i_track in range(len(list_of_visits)):
        track_start = first_times[i_track]
        track_end = last_times[i_track]
        track_visits = list_of_visits[i_track]
        # If the track starts by a transit
        if data_per_id["first_tracked_position_patch"][i_track] == -1:
            if len(track_visits) > 0:  # if there are visits in the track
                list_of_events.append([track_start, track_visits[0][0], -1])
            else:  # else it's just one big transit
                list_of_events.append([track_start, track_end, -1])
        for i_visit in range(len(track_visits)):
            current_visit = track_visits[i_visit]
            list_of_events.append(current_visit)
            # Then for every visit except for the last, add the transit following them to the events
            if i_visit < len(track_visits) - 1:
                list_of_events.append([current_visit[1], track_visits[i_visit + 1][0], -1])
        # If the track ends with a transit
        if data_per_id["last_tracked_position_patch"][i_track] == -1:
            if len(track_visits) > 0:  # if there are visits in the track
                list_of_events.append([track_visits[-1][1], track_end, -1])
        # Then, if this is not the last track, handle the tracking hole following this track
        if i_track < len(list_of_visits) - 1:
            hole_start = last_times[i_track]
            hole_end = first_times[i_track + 1]
            hole_start_position = data_per_id["last_tracked_position_patch"][i_track]
            hole_end_position = data_per_id["first_tracked_position_patch"][i_track + 1]
            # Valid holes (in/in or out/out) get assigned their value directly
            if hole_start_position == hole_end_position:
                list_of_events.append([hole_start, hole_end, hole_start_position])
            # Invalid holes (in/out or out/in)
            else:
                # If the hole is short enough, share it between start and end positions
                if hole_end - hole_start <= param.hole_filling_threshold:
                    list_of_events.append([hole_start, (hole_end + hole_start)/2, hole_start_position])
                    list_of_events.append([(hole_end + hole_start)/2, hole_end, hole_end_position])

    # Theeennnn we have a chronological list of events and all that's left to do is to fuse events A and B if:
    # - A and B have the same event ID
    # - end of A is the start of B
    # Initialize loop variables
    event_start = -1  # This is -1 when no event is open, and if an event is open its value is the event start time
    list_of_aggregated_events = []
    for i_event in range(len(list_of_events) - 1):
        current_event = list_of_events[i_event]
        next_event = list_of_events[i_event + 1]
        if event_start == -1:  # No ongoing, open event
            if current_event[1] != next_event[0] or current_event[2] != next_event[2]:  # next event should not be fused
                list_of_aggregated_events.append(current_event)
            else:  # next event should be fused
                event_start = current_event[0]
        else:  # There's an open event from previous events, and we're looking until when to fuse it
            if current_event[1] != next_event[0] or current_event[2] != next_event[2]:  # next event should not be fused
                list_of_aggregated_events.append([event_start, current_event[1], current_event[2]])
                event_start = -1  # reset to no ongoing event
            # Else, next event should be fused too, so nothing to do!
    # At this point everything should have been handled except for the last events
    # Case 1: the penultimate event has to be fused with the last
    if event_start != -1:
        list_of_aggregated_events.append([event_start, list_of_events[-1][1], list_of_events[-1][2]])
    # Case 2: just add the last event to the list of events
    else:
        list_of_aggregated_events.append(list_of_events[-1])

    # Now we just need to separate visits and transits
    corrected_list_of_transits = []
    corrected_list_of_visits = []
    for i_event, event in enumerate(list_of_aggregated_events):
        if event[2] == -1:
            corrected_list_of_transits.append(event)
        else:
            corrected_list_of_visits.append(event)

    return corrected_list_of_visits, corrected_list_of_transits


def remove_censored_events(data_per_id, list_of_visits, list_of_transits):
    """
    Function that takes data from the results_per_id table for one given folder, and then
    two lists of events [[t0,t1, i],[t0,t1, i],...], with t0 start of event, t1 end of event, and i = index of the event
    (so patch number if it's a visit, and -1 if it's a transit).
    Returns:
        - two new lists in the same format, but removing transits and visits that were interrupted by a tracking hole.
        - one new list of visits in the same format, but removing all visits to a patch containing any censored visit.
    """
    # Sort the tracks for them to be in the order in which they were tracked otherwise it's a mess
    data_per_id = data_per_id.sort_values(by=['first_frame']).reset_index()

    # At this point in the pipeline, the censored events are those that do not have an adjacent event
    # So the uncensored events are those that start when another event ends, or end when another event starts.
    # We go through the events and only include those events.

    # Initialize lists
    uncensored_visits = []
    uncensored_transits = []
    patches_with_censored_visits = []

    # Extract uncensored transits
    visit_start_times = [visit[0] for visit in list_of_visits]
    visit_end_times = [visit[1] for visit in list_of_visits]
    for transit in list_of_transits:
        # We check that this transit starts/ends at the same time as a visit ends/starts
        if transit[0] in visit_end_times and transit[1] in visit_start_times:
            uncensored_transits.append(transit)

    # Extract uncensored visits
    transit_start_times = [transit[0] for transit in list_of_transits]
    transit_end_times = [transit[1] for transit in list_of_transits]
    for visit in list_of_visits:
        # We check that this transit starts/ends at the same time as a visit
        if visit[0] in transit_end_times and visit[1] in transit_start_times:
            uncensored_visits.append(visit)
        else:
            patches_with_censored_visits.append(visit[2])

    # Create a list of visits to patches containing no censored visit
    visits_to_uncensored_patches = []
    patches_with_censored_visits = np.unique(patches_with_censored_visits)
    for visit in uncensored_visits:
        if visit[2] not in patches_with_censored_visits:
            visits_to_uncensored_patches.append(visit)

    return uncensored_visits, uncensored_transits, visits_to_uncensored_patches


def make_results_per_plate(data_per_id, trajectories):
    """
    Function that takes our results_per_id table as an input, and will output a table with info for each plate:
        - folder
        - condition
        - length of the video (last - first TIME stamp) => could be shorter than nb of frames since 1 frame = 0.8 sec
        - number of tracked time steps (computed as number of frames)
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
        if i_plate % 1 == 0:
            print(i_plate, " / ", len(list_of_plates))
        # Import tables
        current_folder = list_of_plates[i_plate]
        current_trajectory = trajectories[trajectories["folder"] == current_folder]
        current_data = data_per_id[data_per_id["folder"] == current_folder].reset_index()

        # Aggregating visits when worm disappears and then reappears in the same place
        aggregated_visit_timestamps, aggregated_transit_timestamps = fill_holes(current_data)

        # Extracting uncensored events (not interrupted by tracking holes)
        uncensored_visit_timestamps, uncensored_transit_timestamps, visit_to_uncensored_patches_timestamps = remove_censored_events(
            current_data, aggregated_visit_timestamps, aggregated_transit_timestamps)

        # Computing average speed by doing a weighted average of the average speeds in each track
        # (weight is relative total time inside or outside)
        # Doing it like this allows us to avoid the "holes in tracking" issues because they are excluded from the "id" slicing
        average_speed_in = np.nansum((current_data["average_speed_inside"] * current_data["total_visit_time"]) /
                                     np.nansum(current_data["total_visit_time"]))
        # To get time out, do total tracked time minus time inside
        average_speed_out = np.nansum(
            (current_data["average_speed_outside"] * (
                    current_data["total_tracked_time"] - current_data["total_visit_time"])) /
            np.nansum((current_data["total_tracked_time"] - current_data["total_visit_time"])))

        # Compute the number and total length of bad events with length < or > parameters.hole_filling_threshold
        nb_all_bad, length_all_bad, nb_long_bad, length_long_bad = nb_bad_events(current_data)

        # Fill up the table
        results_per_plate.loc[i_plate, "folder"] = current_folder
        results_per_plate.loc[i_plate, "condition"] = current_data["condition"][0]
        results_per_plate.loc[i_plate, "total_video_time"] = np.max(current_data["last_frame"]) - np.min(
            current_data["first_frame"])
        results_per_plate.loc[i_plate, "total_tracked_time"] = np.sum(current_data["total_tracked_time"])
        results_per_plate.loc[i_plate, "nb_of_holes"] = len(current_data) - 1
        results_per_plate.loc[i_plate, "nb_all_bad_holes"] = nb_all_bad
        results_per_plate.loc[i_plate, "length_all_bad_holes"] = length_all_bad
        results_per_plate.loc[i_plate, "nb_long_bad_holes"] = nb_long_bad
        results_per_plate.loc[i_plate, "length_long_bad_holes"] = length_long_bad
        results_per_plate.loc[i_plate, "nb_of_teleportations"] = np.sum(current_data["nb_of_teleportations"])
        results_per_plate.loc[i_plate, "avg_proportion_double_frames"] = (len(current_trajectory["frame"]) / len(
            np.unique(current_trajectory["frame"]))) - 1
        results_per_plate.loc[i_plate, "total_visit_time"] = np.sum(
            [pd.DataFrame(aggregated_visit_timestamps).apply(lambda t: t[1] - t[0], axis=1)])
        results_per_plate.loc[i_plate, "total_transit_time"] = np.sum(
            [pd.DataFrame(aggregated_transit_timestamps).apply(lambda t: t[1] - t[0], axis=1)])
        results_per_plate.loc[i_plate, "no_hole_visits"] = str(aggregated_visit_timestamps)
        results_per_plate.loc[i_plate, "aggregated_raw_transits"] = str(aggregated_transit_timestamps)
        results_per_plate.loc[i_plate, "uncensored_visits"] = str(uncensored_visit_timestamps)
        results_per_plate.loc[i_plate, "uncensored_transits"] = str(uncensored_transit_timestamps)
        results_per_plate.loc[i_plate, "visits_to_uncensored_patches"] = str(visit_to_uncensored_patches_timestamps)
        results_per_plate.loc[i_plate, "nb_of_visits"] = len(aggregated_visit_timestamps)
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
        results["aggregated_visits_thresh_" + str(thresh)] = pd.DataFrame(np.zeros(len(results))).astype("str")
        results["aggregated_visits_thresh_" + str(thresh) + "_visit_durations"] = pd.DataFrame(
            np.zeros(len(results))).astype("str")
        results["aggregated_visits_thresh_" + str(thresh) + "_nb_of_visits"] = pd.DataFrame(
            np.zeros(len(results))).astype("str")
        results["aggregated_visits_thresh_" + str(thresh) + "_total_visit_time"] = pd.DataFrame(
            np.zeros(len(results))).astype("str")
        results["aggregated_visits_thresh_" + str(thresh) + "_leaving_events_time_stamps"] = pd.DataFrame(
            np.zeros(len(results))).astype("str")

    # We run this for each plate separately to keep different patches separate
    for i_plate in range(len(results)):
        current_plate = results["folder"][i_plate]
        current_results = results[results["folder"] == current_plate].reset_index(drop=True)
        this_plate_visits = fd.load_list(current_results, "visits_to_uncensored_patches")
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


def generate_pixelwise_visits(time_stamp_list, folder):
    """
    Function that takes a folder containing a time series of silhouettes, and returns a list of lists with the dimension
    of the plate in :folder:, and in each cell, a list with [start time, end time] of the successive visits to this pixel.
    (a visit starts when a pixel of the worm overlaps with the pixel, and ends when this overlap stops)
    When this function is called, it also saves this output under the name "pixelwise_visits.npy" in folder.
    Takes trajectory in argument only to access correspondence between index in video and frame number.
    """
    # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    pixels, intensities, frame_size = fd.load_silhouette(folder)
    pixels = fd.reindex_silhouette(pixels, frame_size)

    # If the trajectories should be shortened to some time point, also shorten the silhouettes
    if "shortened" in folder:
        pixels = pixels[:fd.find_closest(time_stamp_list[0], param.times_to_cut_videos)]

    # Create a table with a list containing, for each pixel in the image, a sublist with the [start, end] of the visits
    # to this pixel. In the following algorithm, when the last element of a sublist is -1, it means that the pixel
    # was not being visited at the previous time point.
    # We start by creating an array with one sublist per pixel, each sublist only containing -1 in the beginning
    visit_times_each_pixel = [[[[-1]] for _ in range(frame_size[0])] for _ in range(frame_size[1])]
    # For each time point, create visits in pixels that just started being visited, continue those that have already
    # started, and end those that are finished
    for j_time in range(len(pixels)):
        current_visited_pixels = pixels[j_time]
        for i_pixel in range(len(current_visited_pixels[0])):
            current_pixel = [current_visited_pixels[0][i_pixel], current_visited_pixels[1][i_pixel]]
            # If visit just started, start it
            if visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] == [-1]:
                visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] = [time_stamp_list[j_time],
                                                                                  time_stamp_list[j_time]]
            # If visit is continuing, update end time
            else:
                visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1][1] = time_stamp_list[j_time]
        # Then, close the visits of the previous time step that are not being continued
        if j_time > 0:
            previous_visited_pixels = pixels[j_time - 1]
            for i_pixel in range(len(previous_visited_pixels[0])):
                current_pixel = [previous_visited_pixels[0][i_pixel], previous_visited_pixels[1][i_pixel]]
                # If this pixel is not in the current visited pixels, then close the visit
                if True not in np.logical_and(np.array(current_visited_pixels[0]) == current_pixel[0],
                                              np.array(current_visited_pixels[1]) == current_pixel[1]):
                    visit_times_each_pixel[current_pixel[1]][current_pixel[0]].append([-1])

    # Remove the [-1] because they were only useful for the algorithm
    for j_line in range(len(visit_times_each_pixel)):
        for i_column in range(len(visit_times_each_pixel[j_line])):
            if visit_times_each_pixel[j_line][i_column][-1] == [-1]:
                visit_times_each_pixel[j_line][i_column] = visit_times_each_pixel[j_line][i_column][:-1]

    np.save(folder[:-len("traj.csv")] + "pixelwise_visits.npy", np.array(visit_times_each_pixel, dtype=object))

    return visit_times_each_pixel
