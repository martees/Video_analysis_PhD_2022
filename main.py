## Hi
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

import analysis
import param
# My code
import plots
import generate_results as gr
import find_data as fd


def plot_graphs(plot, raw_condition_list=None, include_control=True):
    # Input variable control
    # Default values
    if raw_condition_list is None:
        raw_condition_list = ["all"]
    # If plot is just one string, transform it into a one element list for the later loops to work
    # (otherwise it will do len() of a string and run many times x))
    if type(plot) == str:
        plot = [plot]
    # Same for condition list
    if type(raw_condition_list) == int or type(raw_condition_list) == str:
        raw_condition_list = [raw_condition_list]

    # Fork to fill condition names and densities depending on densities
    condition_pools = []
    condition_names = []
    condition_colors = []

    for condition in raw_condition_list:
        list_of_conditions = param.name_to_nb_list[condition]
        list_of_names = [param.nb_to_name[i] for i in list_of_conditions]
        condition_pools.append(list_of_conditions)
        condition_names.append(list_of_names)
        condition_colors.append(param.name_to_color[condition])

    # Add control to every condition sublist
    if include_control:
        for i_condition_sublist in range(len(condition_pools)):
            if 11 not in condition_pools[i_condition_sublist]:
                condition_pools[i_condition_sublist].append(11)
                condition_names[i_condition_sublist].append("control")

    for _ in range(len(plot)):
        for i_pool in range(len(condition_pools)):
            current_pool_name = raw_condition_list[i_pool]
            current_condition_pool = condition_pools[i_pool]
            current_condition_names = condition_names[i_pool]
            current_color = condition_colors[i_pool]

            # Data quality
            if "double_frames" in plot:
                plots.plot_selected_data(results,
                                         "Average proportion of double frames in " + current_pool_name + " densities",
                                         condition_pools,
                                         current_condition_names, "avg_proportion_double_frames", mycolor=current_color)
            if "bad_events" in plot:
                plots.plot_selected_data(results, "Average number of bad events in " + current_pool_name + " densities",
                                         condition_pools,
                                         current_condition_names, "nb_of_bad_events", mycolor=current_color)

            # Speed plots
            if "speed" in plot:
                plots.plot_selected_data(results, "Average speed in " + current_pool_name + " densities (inside)",
                                         condition_pools,
                                         current_condition_names,
                                         "average_speed_inside", divided_by="", mycolor=current_color)
                plots.plot_selected_data(results, "Average speed in " + current_pool_name + " densities (outside)",
                                         condition_pools,
                                         current_condition_names,
                                         "average_speed_outside", divided_by="", mycolor=current_color)

            # Visits plots
            if "visit_duration" in plot:
                plots.plot_selected_data(results, "Average duration of visits in " + current_pool_name + " densities",
                                         current_condition_pool,
                                         current_condition_names, "total_visit_time", divided_by="nb_of_visits", mycolor=current_color,
                                         plot_model=True)
            if "visit_duration_mvt" in plot:
                plots.plot_selected_data(results, "Average duration of MVT visits in " + current_pool_name + " densities",
                                         current_condition_pool,
                                         current_condition_names, "total_visit_time", divided_by="mvt_nb_of_visits",
                                         mycolor=current_color, plot_model=True)

            if "aggregated_visit_duration" in plot:
                for thresh in param.threshold_list:
                    plots.plot_selected_data(results, "Average duration visits in " + raw_condition_list[
                        i_pool] + " densities, aggregated with threshold " + str(thresh),
                                             current_condition_pool, current_condition_names,
                                             "aggregated_visits_thresh_" + str(thresh) + "_total_visit_time",
                                             divided_by="aggregated_visits_thresh_" + str(thresh) + "_nb_of_visits",
                                             mycolor=current_color, plot_model=True)

            # Transits plots
            if "transit_duration" in plot:
                plots.plot_selected_data(results, "Average duration of transits in low densities", [0, 1, 2, 11],
                                         ["close 0.2", "med 0.2", "far 0.2", "control"], "total_transit_time",
                                         divided_by="nb_of_visits", mycolor="brown")
                plots.plot_selected_data(results, "Average duration of transits in medium densities", [4, 5, 6, 11],
                                         ["close 0.5", "med 0.5", "far 0.5", "control"], "total_transit_time",
                                         divided_by="nb_of_visits", mycolor=current_color)

            if "visit_duration_vs_previous_transit" in plot:
                plots.plot_visit_time(results, trajectories,
                                      "Visit duration vs. previous transit in " + current_condition_pool + " densities",
                                      condition_pools, "Last travel time", current_condition_names, split_conditions=False)
                plots.plot_visit_time(results, trajectories, "Visit duration vs. previous transit in control",
                                      [11], "Last travel time", ["control"])

            if "visit_duration_vs_visit_start" in plot:
                plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in low densities",
                                      [0, 1, 2, 3, 11],
                                      "Visit start", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "control"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in medium densities",
                                      [4, 5, 6, 7, 11],
                                      "Visit start", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

            if "visit_duration_vs_entry_speed" in plot:
                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [4],
                                      "Speed when entering",
                                      ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [5],
                                      "Speed when entering", ["close 0.5"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [6],
                                      "Speed when entering", ["med 0.5"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [7],
                                      "Speed when entering", ["far 0.5"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [0],
                                      "Speed when entering", ["close 0.2"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [1],
                                      "Speed when entering", ["med 0.2"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [2],
                                      "Speed when entering", ["far 0.2"])

                plots.plot_visit_time(results, trajectories, "Visit duration vs. speed when entering the patch",
                                      [11],
                                      "Speed when entering", ["control"])

            # Visit rate plots
            if "visit_rate" in plot:
                plots.plot_selected_data(results, "Average visit rate in low densities", [0, 1, 2, 11],
                                         ["close 0.2", "med 0.2", "far 0.2", "control"], "nb_of_visits",
                                         divided_by="total_video_time", mycolor=current_color)
                plots.plot_selected_data(results, "Average visit rate in medium densities", [4, 5, 6, 11],
                                         ["close 0.5", "med 0.5", "far 0.5", "control"], "nb_of_visits",
                                         divided_by="total_video_time", mycolor=current_color)
                plots.plot_selected_data(results, "Average visit rate MVT in low densities", [0, 1, 2, 11],
                                         ["close 0.2", "med 0.2", "far 0.2", "control"], "mvt_nb_of_visits",
                                         divided_by="total_video_time", mycolor=current_color)
                plots.plot_selected_data(results, "Average visit rate MVT in medium densities", [4, 5, 6, 11],
                                         ["close 0.5", "med 0.5", "far 0.5", "control"], "mvt_nb_of_visits",
                                         divided_by="total_video_time", mycolor=current_color)

            # Proportion of visited patches plots
            if "proportion_of_time" in plot:
                plots.plot_selected_data(results, "Average proportion of time spent in patches in" + str(
                    current_condition_names) + "densities",
                                         current_condition_pool,
                                         current_condition_names, "total_visit_time",
                                         divided_by="total_video_time", mycolor=current_color)

                # plot_selected_data("Average number of visits in low densities", 0, 3, "nb_of_visits", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
                # plot_selected_data("Average furthest visited patch distance in low densities", 0, 3, "furthest_patch_distance", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
                # plot_selected_data("Average proportion of visited patches in low densities", 0, 3, "proportion_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
                # plot_selected_data("Average number of visited patches in low densities", 0, 3, "nb_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")

                # plot_selected_data("Average number of visits in medium densities", 4, 7, "nb_of_visits", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
                # plot_selected_data("Average furthest visited patch distance in medium densities", 4, 7, "furthest_patch_distance", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
                # plot_selected_data("Average proportion of visited patches in medium densities", 4, 7, "proportion_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
                # plot_selected_data("Average number of visited patches in medium densities", 4, 7, "nb_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")

            if "distribution" in plot:
                plots.plot_variable_distribution(results, condition_list=current_condition_pool, effect_of="nothing")
                plots.plot_variable_distribution(results, condition_list=current_condition_pool, effect_of="food")
                plots.plot_variable_distribution(results, condition_list=current_condition_pool, effect_of="density")
                plots.plot_variable_distribution(results, condition_list=current_condition_pool, effect_of="distance")

            if "distribution_aggregated" in plot:
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="nothing",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="food",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="distance",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="density",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])

            if "leaving_events" in plot:
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="nothing",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="food",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="distance",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_condition_pool, effect_of="density",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])

            if "leaving_events_delay_distribution" in plot:
                plots.plot_leaving_delays(results,
                                          "Frequency of leaving events as a function of in_patch time in " + current_pool_name + " conditions",
                                          current_condition_pool, bin_size=1000, color=current_color)

            if "transit_properties" in plot:
                revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch = analysis.transit_properties(
                    results, condition_pools, split_conditions=True)
                for i_cond in range(len(condition_pools)):
                    print("Transit properties for condition ", current_condition_names[i_cond])
                    print("Revisit probability: ", revisit_probability[i_cond])
                    print("Cross-patch probability: ", cross_transit_probability[i_cond])
                    print("Exponential leaving probability: ", exponential_leaving_probability[i_cond])
                    print("Minimal duration of visits: ", min_visit[i_cond])
                    print("Average duration of visits: ", average_visit[i_cond])
                    print("Average duration of same patch transits: ", average_same_patch[i_cond])
                    print("Average duration of cross patch transits: ", average_cross_patch[i_cond])
                    print("-----")

                revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch = analysis.transit_properties(
                    results, condition_pools, split_conditions=False)

                print("Transit properties for conditions ", current_condition_names)
                print("Revisit probability: ", revisit_probability)
                print("Cross-patch probability: ", cross_transit_probability)
                print("Exponential leaving probability: ", exponential_leaving_probability)
                print("Minimal duration of visits: ", min_visit)
                print("Average duration of visits: ", average_visit)
                print("Average duration of same patch transits: ", average_same_patch)
                print("Average duration of cross patch transits: ", average_cross_patch)
                print("-----")


#   Saves the results in a path that is returned:
# "trajectories" will generate everything starting here ->
#       "trajectories.csv": raw trajectories, one line per tracked point
# "results_per_id" will generate everything starting here ->
#       "results_per_id.csv": one line per id_conservative in the tracking, ie one line per continuous tracking track
# "results_per_plate" will generate everything starting here ->
#       "results_per_plate.csv": one line per plate in the tracking, so hopefully one line per worm :p
# "clean" will generate everything starting here ->
#       "clean_results.csv": same but removing some plates (see generate_results.exclude_invalid_videos)
#       "clean_trajectories.csv": trajectories csv but with only the tracks corresponding to the valid plates
# NOTE: lists are stored as strings in the csv, so we retrieve the values with json loads function
path = gr.generate(starting_from="")
# Only generate the results in the beginning of your analysis!

# Retrieve results from what generate_and_save has saved

# REMOVE THIS ONCE YOU HAVE THE CLEAN TRAJECTORIES ON YOUR PC YOU AIRHEAD
# if fd.is_linux():  # Linux path
#    trajectories = pd.read_csv(path + "clean_trajectories.csv")
if not fd.is_linux():  # Windows path
    trajectories = pd.read_csv(path + "trajectories.csv")
results = pd.read_csv(path + "clean_results.csv")

print("Finished retrieving stuff")

# Possible arguments for plot_graphs:
#               - "double_frames"
#               - "bad_events"
#               - "speed"
#               - "visit_duration"
#               - "visit_duration_mvt"
#               - "aggregated_visit_duration"
#               - "transit_duration"
#               - "visit_duration_vs_previous_transit"
#               - "visit_duration_vs_visit_start"
#               - "visit_duration_vs_entry_speed"
#               - "visit_rate"
#               - "proportion_of_time"
#               - "distribution"
#               - "distribution_aggregated"
#               - "transit_properties"
#               - "aggregated_visits"
#               - "leaving_events"
#               - "leaving_events_delay_distribution"


## Plot
plot_graphs("leaving_events_delay_distribution", ["0.5", "med 0.5"])

# TODO function find frame that returns index of a frame in a traj with two options: either approach from below, or approach from top => for speed analysis
# TODO function that shows speed as a function of time since patch has been entered (ideally, concatenate all visits)
# TODO function that shows length of (first) visit to a patch as a function of last travel time / average feeding rate in window

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way

##

