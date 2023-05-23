# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

# My code
import plots
import generate_results as gr
import find_data as fd


def plot_graphs(plot_quality=False, plot_speed=False, plot_visit_duration=False, plot_transit_duration=False,
                plot_visit_rate=False, plot_proportion=False, plot_full=False,
                plot_visit_duration_vs_visit_start=False, plot_visit_duration_vs_previous_transit=False,
                plot_visit_duration_vs_entry_speed=False):
    #TODO diversify variables of this function: eg add density=low/high and it modifies the condition list and names automatically?

    # Data quality
    if plot_quality:
        plots.plot_selected_data(results, "Average proportion of double frames in all densities", 0,
                                 "avg_proportion_double_frames", 11,
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], mycolor="green")
        plots.plot_selected_data(results, "Average number of bad events in all densities", 0, "nb_of_bad_events", 11,
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], mycolor="green")

    # Speed plots
    if plot_speed:
        plots.plot_selected_data(results, "Average speed in all densities (inside)", range(12),
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                                 "average_speed_inside", divided_by="", mycolor="green")
        plots.plot_selected_data(results, "Average speed in all densities (outside)", range(12),
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                                 "average_speed_outside", divided_by="", mycolor="green")

        plots.plot_selected_data(results, "Average speed in low densities (inside)", [0, 1, 2, 11],
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], "average_speed_inside", divided_by="",
                                 mycolor="brown")
        plots.plot_selected_data(results, "Average speed in low densities (outside)", [0, 1, 2, 11],
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], "average_speed_outside", divided_by="",
                                 mycolor="brown")

        plots.plot_selected_data(results, "Average speed in medium densities (inside)", [4, 5, 6, 11],
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], "average_speed_inside", divided_by="",
                                 mycolor="orange")
        plots.plot_selected_data(results, "Average speed in medium densities (outside)", [4, 5, 6, 11],
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], "average_speed_outside", divided_by="",
                                 mycolor="orange")

    # Visits plots
    if plot_visit_duration:
        plots.plot_selected_data(results, "Average duration of visits in low densities", [0, 1, 2, 11],
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], "total_visit_time",
                                 divided_by="nb_of_visits", mycolor="brown")
        plots.plot_selected_data(results, "Average duration of visits in medium densities", [4, 5, 6, 11],
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], "total_visit_time",
                                 divided_by="nb_of_visits", mycolor="orange")
        plots.plot_selected_data(results, "Average duration of MVT visits in low densities", [0, 1, 2, 11],
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], "total_visit_time",
                                 divided_by="mvt_nb_of_visits", mycolor="brown")
        plots.plot_selected_data(results, "Average duration of MVT visits in medium densities", [4, 5, 6, 11],
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], "total_visit_time",
                                 divided_by="mvt_nb_of_visits", mycolor="orange")

    # Transits plots
    if plot_transit_duration:
        plots.plot_selected_data(results, "Average duration of transits in low densities", [0, 1, 2, 11],
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], "total_transit_time",
                                 divided_by="nb_of_visits", mycolor="brown")
        plots.plot_selected_data(results, "Average duration of transits in medium densities", [4, 5, 6, 11],
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], "total_transit_time",
                                 divided_by="nb_of_visits", mycolor="orange")

    if plot_visit_duration_vs_previous_transit:
        plots.plot_visit_time(results, trajectories, "Visit duration vs. previous transit in low densities",
                              [0, 1, 2, 3, 11],
                              "Last travel time", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "control"])

        plots.plot_visit_time(results, trajectories, "Visit duration vs. previous transit in medium densities",
                              [4, 5, 6, 7, 11],
                              "Last travel time", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

    if plot_visit_duration_vs_visit_start:
        plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in low densities",
                              [0, 1, 2, 3, 11],
                              "Visit start", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "control"])

        plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in medium densities",
                              [4, 5, 6, 7, 11],
                              "Visit start", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

    if plot_visit_duration_vs_entry_speed:
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
    if plot_visit_rate:
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
    if plot_proportion:
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

    # Full plots
    if plot_full:
        plots.plot_selected_data(results, "Average duration of visits in all densities", 0, "total_visit_time", 11,
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], mycolor="brown")
        plots.plot_selected_data(results, "Average duration of MVT visits in all densities", 0, "total_visit_time", 11,
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], mycolor="brown")


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
#plot_graphs(plot_visit_duration_vs_visit_start=True)
#plot_graphs(plot_visit_duration_vs_previous_transit=True)
plot_graphs(plot_visit_duration_vs_entry_speed=True)
# plot_speed_time_window_list(trajectories, [1, 100, 1000], 1, out_patch=True)
# plot_speed_time_window_continuous(trajectories, 1, 120, 1, 100, current_speed=False, speed_history=False, past_speed=True)
# plots.plot_variable_distribution(results, "transit_duration")
# binned_speed_as_a_function_of_time_window(trajectories, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 100, 1000], [0, 1], 1)

# plots.plot_test(results)


# TODO function find frame that returns index of a frame in a traj with two options: either approach from below, or approach from top => for speed analysis
# TODO function that shows speed as a function of time since patch has been entered (ideally, concatenate all visits)
# TODO function that shows length of (first) visit to a patch as a function of last travel time / average feeding rate in window

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
