# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

# My code
import plots
import generate_results as gr
import find_data as fd

def plot_graphs(plot_quality=False, plot_speed=False, plot_visit_duration=False, plot_transit_duration=False,
                plot_visit_duration_analysis=False, plot_visit_rate=False, plot_proportion=False, plot_full=False):
    # Data quality
    if plot_quality:
        plots.plot_selected_data(results, "Average proportion of double frames in all densities", 0, 11,
                                 "avg_proportion_double_frames",
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                                 mycolor="green")
        plots.plot_selected_data(results, "Average number of bad events in all densities", 0, 11, "nb_of_bad_events",
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                                 mycolor="green")

    # Speed plots
    if plot_speed:
        plots.plot_selected_data(results, "Average speed in all densities (inside)", range(12), "average_speed_inside",
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="",
                                 mycolor="green")
        plots.plot_selected_data(results, "Average speed in all densities (outside)", range(12),
                                 "average_speed_outside",
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                                  "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"], divided_by="",
                                 mycolor="green")

        plots.plot_selected_data(results, "Average speed in low densities (inside)", [0, 1, 2, 11],
                                 "average_speed_inside",
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by="",
                                 mycolor="brown")
        plots.plot_selected_data(results, "Average speed in low densities (outside)", [0, 1, 2, 11],
                                 "average_speed_outside",
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by="",
                                 mycolor="brown")

        plots.plot_selected_data(results, "Average speed in medium densities (inside)", [4, 5, 6, 11],
                                 "average_speed_inside",
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by="",
                                 mycolor="orange")
        plots.plot_selected_data(results, "Average speed in medium densities (outside)", [4, 5, 6, 11],
                                 "average_speed_outside",
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by="",
                                 mycolor="orange")

    # Visits plots
    if plot_visit_duration:
        plots.plot_selected_data(results, "Average duration of visits in low densities", [0, 1, 2, 11],
                                 "total_visit_time",
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by="nb_of_visits",
                                 mycolor="brown")
        plots.plot_selected_data(results, "Average duration of visits in medium densities", [4, 5, 6, 11],
                                 "total_visit_time",
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by="nb_of_visits",
                                 mycolor="orange")
        plots.plot_selected_data(results, "Average duration of MVT visits in low densities", [0, 1, 2, 11],
                                 "total_visit_time",
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by="mvt_nb_of_visits",
                                 mycolor="brown")
        plots.plot_selected_data(results, "Average duration of MVT visits in medium densities", [4, 5, 6, 11],
                                 "total_visit_time",
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by="mvt_nb_of_visits",
                                 mycolor="orange")

    # Transits plots
    if plot_transit_duration:
        plots.plot_selected_data(results, "Average duration of transits in low densities", [0, 1, 2, 11],
                                 "total_transit_time",
                                 ["close 0.2", "med 0.2", "far 0.2", "control"], divided_by="nb_of_visits",
                                 mycolor="brown")
        plots.plot_selected_data(results, "Average duration of transits in medium densities", [4, 5, 6, 11],
                                 "total_transit_time",
                                 ["close 0.5", "med 0.5", "far 0.5", "control"], divided_by="nb_of_visits",
                                 mycolor="orange")

    if plot_visit_duration_analysis:
        plots.plot_visit_time(results, trajectories, "Visit duration vs. previous transit in low densities", [0, 1, 2, 3, 11],
                              "last_travel_time", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "control"])

        plots.plot_visit_time(results, trajectories, "Visit duration vs. previous transit in medium densities", [4, 5, 6, 7, 11],
                              "last_travel_time", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

        plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in low densities", [0, 1, 2, 3, 11],
                              "visit_start", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "control"])

        plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in medium densities", [4, 5, 6, 7, 11],
                              "visit_start", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

        plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in low densities", [0, 1, 2, 3, 11],
                              "speed_when_entering", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "control"])

        plots.plot_visit_time(results, trajectories, "Visit duration vs. visit start in medium densities", [4, 5, 6, 7, 11],
                              "speed_when_entering", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5", "control"])

    # Visit rate plots
    if plot_visit_rate:
        plots.plot_selected_data(results, "Average visit rate in low densities", [0, 1, 2, 11], "nb_of_visits",
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], divided_by="total_video_time",
                                 mycolor="brown")
        plots.plot_selected_data(results, "Average visit rate in medium densities", [4, 5, 6, 11], "nb_of_visits",
                                 ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], divided_by="total_video_time",
                                 mycolor="orange")
        plots.plot_selected_data(results, "Average visit rate MVT in low densities", [0, 1, 2, 11], "mvt_nb_of_visits",
                                 ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"], divided_by="total_video_time",
                                 mycolor="brown")
        plots.plot_selected_data(results, "Average visit rate MVT in medium densities", [4, 5, 6, 11],
                                 "mvt_nb_of_visits",
                                 ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"], divided_by="total_video_time",
                                 mycolor="orange")

    # Proportion of visited patches plots
    if plot_proportion:
        plots.plot_selected_data(results, "Average proportion of time spent in patches in low densities", [0, 1, 2, 11],
                                 "total_visit_time", ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"],
                                 divided_by="total_video_time", mycolor="brown")
        plots.plot_selected_data(results, "Average proportion of time spent in patches in medium densities", [4, 5, 6, 11],
                                 "total_visit_time", ["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"],
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
        plots.plot_selected_data(results, "Average duration of visits in all densities", 0, 11, "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                           divided_by="nb_of_visits", mycolor="brown")
        plots.plot_selected_data(results, "Average duration of MVT visits in all densities", 0, 11, "total_visit_time",
                           ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                            "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "buffer"],
                           divided_by="mvt_nb_of_visits", mycolor="brown")


# Data path
if fd.is_linux():  # Linux path
    path = "/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/"
else:  # Windows path
    path = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/Results_minipatches_20221108_clean_fp_less/"

# Extracting data, the function looks for all "traj.csv" files in the indicated path (will look into subfolders)
# It will then generate a "results" table, with one line per worm, and these info:
# NOTE: lists are stored as strings in the csv so we retrieve the values with json loads function

# Only generate the results in the beginning of your analysis!
### Saves the results in path:
####### "trajectories.csv": raw trajectories, one line per tracked point
####### "results_per_id.csv":
####### "results_per_plate.csv":
####### "clean_results.csv":
# Will regenerate the dataset from the first True boolean
regenerate_trajectories = False
regenerate_results_per_id = False
regenerate_results_per_plate = False
regenerate_clean_results = False

if regenerate_trajectories:
    gr.generate_trajectories(path)
    gr.generate_results_per_id(path)
    gr.generate_results_per_plate(path)
    gr.generate_clean_tables_and_speed(path)

elif regenerate_results_per_id:
    gr.generate_results_per_id(path)
    gr.generate_results_per_plate(path)
    gr.generate_clean_tables_and_speed(path)

elif regenerate_results_per_plate:
    gr.generate_results_per_plate(path)
    gr.generate_clean_tables_and_speed(path)

elif regenerate_clean_results:
    gr.generate_clean_tables_and_speed(path)

# Retrieve results from what generate_and_save has saved
if fd.is_linux():  # Linux path
    trajectories = pd.read_csv(path + " clean_trajectories.csv")
else:  # Windows path
    trajectories = pd.read_csv(path + "trajectories.csv")
results = pd.read_csv(path + "clean_results.csv")

print("finished retrieving stuff")

# plot_patches(fd.path_finding_traj(path))
# plot_avg_furthest_patch()
# plot_data_coverage(trajectories)
# plot_traj(trajectories, 11, n_max=4, is_plot_patches=True, show_composite=False, plot_in_patch=True, plot_continuity=False, plot_speed=False, plot_time=False)
plot_graphs(plot_visit_duration_analysis=True)
# plot_speed_time_window_list(trajectories, [1, 100, 1000], 1, out_patch=True)
# plot_speed_time_window_continuous(trajectories, 1, 120, 1, 100, current_speed=False, speed_history=False, past_speed=True)

# binned_speed_as_a_function_of_time_window(trajectories, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 100, 1000], [0, 1], 1)

# plots.plot_test(results)


# TODO function find frame that returns index of a frame in a traj with two options: either approach from below, or approach from top
# TODO function that shows speed as a function of time since patch has been entered (ideally, concatenate all visits)
# TODO function that shows length of (first) visit to a patch as a function of last travel time / average feeding rate in window

# TODO review fill_holes function to double-check what it does to bad holes...

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
