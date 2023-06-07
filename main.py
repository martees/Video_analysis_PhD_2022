# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

import analysis
# My code
import plots
import generate_results as gr
import find_data as fd


def plot_graphs(plot, densities_list=None, include_control=True):
    if densities_list is None:
        densities_list = ["all"]

    # Fork to fill condition names and densities depending on densities
    condition_list = []
    condition_names = []
    if "low" in densities_list:
        condition_list += [0, 1, 2, 3]
        condition_names += ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"]
        color = "brown"
    if "high" in densities_list:
        condition_list += [4, 5, 6, 7]
        condition_names += ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"]
        color = "orange"
    if "all" in densities_list:
        condition_list += [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        condition_names += ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25"]
        color = "green"

    if len(densities_list) > 1:  # for low+high or other combinations, put another color
        color = "blue"

    if include_control:
        condition_list.append(11)
        condition_names.append("control")

    for densities in densities_list:
        # Data quality
        if "quality" in plot:
            plots.plot_selected_data(results, "Average proportion of double frames in " + densities + " densities",
                                     condition_list,
                                     condition_names, "avg_proportion_double_frames", mycolor=color)
            plots.plot_selected_data(results, "Average number of bad events in " + densities + " densities", condition_list,
                                     condition_names, "nb_of_bad_events", mycolor=color)
        # Speed plots
        if "speed" in plot:
            plots.plot_selected_data(results, "Average speed in " + densities + " densities (inside)", condition_list,
                                     condition_names,
                                     "average_speed_inside", divided_by="", mycolor=color)
            plots.plot_selected_data(results, "Average speed in " + densities + " densities (outside)", condition_list,
                                     condition_names,
                                     "average_speed_outside", divided_by="", mycolor=color)

        # Visits plots
        if "visit_duration" in plot:
            plots.plot_selected_data(results, "Average duration of visits in " + densities + " densities", condition_list,
                                     condition_names, "total_visit_time", divided_by="nb_of_visits", mycolor=color, plot_model=True)
            plots.plot_selected_data(results, "Average duration of MVT visits in " + densities + " densities",
                                     condition_list,
                                     condition_names, "total_visit_time", divided_by="mvt_nb_of_visits", mycolor=color, plot_model=True)

        # Transits plots
        if "transit_duration" in plot:
            plots.plot_selected_data(results, "Average duration of transits in low densities", [0, 1, 2, 11],
                                     ["close 0.2", "med 0.2", "far 0.2", "control"], "total_transit_time",
                                     divided_by="nb_of_visits", mycolor="brown")
            plots.plot_selected_data(results, "Average duration of transits in medium densities", [4, 5, 6, 11],
                                     ["close 0.5", "med 0.5", "far 0.5", "control"], "total_transit_time",
                                     divided_by="nb_of_visits", mycolor="orange")

        if "visit_duration_vs_previous_transit" in plot:
            plots.plot_visit_time(results, trajectories,
                                  "Visit duration vs. previous transit in " + densities + " densities",
                                  condition_list, "Last travel time", condition_names, split_conditions=False)
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
                                  "Speed when entering", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

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
                                     divided_by="total_video_time", mycolor="brown")
            plots.plot_selected_data(results, "Average visit rate in medium densities", [4, 5, 6, 11],
                                     ["close 0.5", "med 0.5", "far 0.5", "control"], "nb_of_visits",
                                     divided_by="total_video_time", mycolor="orange")
            plots.plot_selected_data(results, "Average visit rate MVT in low densities", [0, 1, 2, 11],
                                     ["close 0.2", "med 0.2", "far 0.2", "control"], "mvt_nb_of_visits",
                                     divided_by="total_video_time", mycolor="brown")
            plots.plot_selected_data(results, "Average visit rate MVT in medium densities", [4, 5, 6, 11],
                                     ["close 0.5", "med 0.5", "far 0.5", "control"], "mvt_nb_of_visits",
                                     divided_by="total_video_time", mycolor="orange")

        # Proportion of visited patches plots
        if "proportion_of_time" in plot:
            plots.plot_selected_data(results, "Average proportion of time spent in patches in low densities", [0, 1, 2, 11],
                                     ["close 0.2", "med 0.2", "far 0.2", "control"], "total_visit_time",
                                     divided_by="total_video_time", mycolor="brown")
            plots.plot_selected_data(results, "Average proportion of time spent in patches in medium densities",
                                     [4, 5, 6, 11], ["close 0.5", "med 0.5", "far 0.5", "control"], "total_visit_time",
                                     divided_by="total_video_time", mycolor="orange")

            # plot_selected_data("Average number of visits in low densities", 0, 3, "nb_of_visits", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
            # plot_selected_data("Average furthest visited patch distance in low densities", 0, 3, "furthest_patch_distance", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
            # plot_selected_data("Average proportion of visited patches in low densities", 0, 3, "proportion_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")
            # plot_selected_data("Average number of visited patches in low densities", 0, 3, "nb_of_visited_patches", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], mycolor = "brown")

            # plot_selected_data("Average number of visits in medium densities", 4, 7, "nb_of_visits", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
            # plot_selected_data("Average furthest visited patch distance in medium densities", 4, 7, "furthest_patch_distance", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
            # plot_selected_data("Average proportion of visited patches in medium densities", 4, 7, "proportion_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")
            # plot_selected_data("Average number of visited patches in medium densities", 4, 7, "nb_of_visited_patches", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], mycolor = "orange")

        if "distribution" in plot:
            plots.plot_variable_distribution(results, condition_list=condition_list, condition_names=condition_names,
                                             pool_by="density", plot_cumulative=False)
            plots.plot_variable_distribution(results, condition_list=condition_list, condition_names=condition_names,
                                             pool_by="density")
            plots.plot_variable_distribution(results, condition_list=condition_list, condition_names=condition_names,
                                             pool_by="distance", plot_cumulative=False)
            plots.plot_variable_distribution(results, condition_list=condition_list, condition_names=condition_names,
                                             pool_by="distance")

        if "transit_properties" in plot:
            revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch = analysis.transit_properties(
                                                                                                results, condition_list, split_conditions=True)
            for i_cond in range(len(condition_list)):
                print("Transit properties for condition ", condition_names[i_cond])
                print("Revisit probability: ", revisit_probability[i_cond])
                print("Cross-patch probability: ", cross_transit_probability[i_cond])
                print("Exponential leaving probability: ", exponential_leaving_probability[i_cond])
                print("Minimal duration of visits: ", min_visit[i_cond])
                print("Average duration of visits: ", average_visit[i_cond])
                print("Average duration of same patch transits: ", average_same_patch[i_cond])
                print("Average duration of cross patch transits: ", average_cross_patch[i_cond])
                print("-----")

            revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch = analysis.transit_properties(
                results, condition_list, split_conditions=False)

            print("Transit properties for conditions ", condition_names)
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
# "results_per id" will generate everything starting here ->
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
if fd.is_linux():  # Linux path
    trajectories = pd.read_csv(path + "clean_trajectories.csv")
else:  # Windows path
    trajectories = pd.read_csv(path + "trajectories.csv")
results = pd.read_csv(path + "clean_results.csv")

print("Finished retrieving stuff")

# plot_patches(fd.path_finding_traj(path))
# plot_avg_furthest_patch()
# plot_data_coverage(trajectories)
# plots.plot_traj(trajectories, 11, n_max=4, is_plot_patches=True, show_composite=False, plot_in_patch=True, plot_continuity=True, plot_speed=True, plot_time=False)
# plot_graphs(plot_visit_duration_vs_visit_start=True)
# plot_graphs(plot_visit_duration_vs_previous_transit=True)
# plot_graphs(plot_visit_duration_vs_entry_speed=True)
# plot_speed_time_window_list(trajectories, [1, 100, 1000], 1, out_patch=True)
# plot_speed_time_window_continuous(trajectories, 1, 120, 1, 100, current_speed=False, speed_history=False, past_speed=True)
# binned_speed_as_a_function_of_time_window(trajectories, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 100, 1000], [0, 1], 1)

plot_graphs(plot="visit_duration", densities_list=["low"])

# Variable distributions
# plots.plot_variable_distribution(results, "transit_duration", condition_list=[0, 4], only_same_patch_transits=True)
# plots.plot_test(results)


# TODO function find frame that returns index of a frame in a traj with two options: either approach from below, or approach from top => for speed analysis
# TODO function that shows speed as a function of time since patch has been entered (ideally, concatenate all visits)
# TODO function that shows length of (first) visit to a patch as a function of last travel time / average feeding rate in window

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
