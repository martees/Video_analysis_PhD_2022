## Hi
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

import analysis as ana
from Parameters import parameters as param
# My code
import plots
from Generating_data_tables import main as gr


def plot_graphs(result_table, what_to_plot, curve_list=None):
    """
    Global plotting function.
    :what_to_plot: list that contains strings that indicate what to plot xD see big loop of the function for the options.
    :curve_list: should have one element per curve you want to make. The elements in each curve sublist should
                         be keys from the param.name_to_nb_list dictionary in parameters.py.
    Syntax:
    If curve_list = [["0.2"], ["0.5"]]:
        for bar plots it will plot all 0.2 and 0.5 conditions on top of each others xD (to fix one day)
        for other curves, it should plot one curve with all 0.2 densities together, and one with all 0.5 conditions
    If curve_list = ["0.2", "0.5"]:
        for bar plots it will plot bars for all 0.2 and 0.5 conditions, so one bar for med 0.2, one for close 0.2, etc.
        for other curves, should do the same but with curves
    """
    # Input variable control
    # Default values
    if curve_list is None:
        curve_list = [["all"]]
    # If plot is just one string, transform it into a one element list for the following loops to work
    # (otherwise it will do len() of a string and run many times x))
    if type(what_to_plot) == str:
        what_to_plot = [what_to_plot]
    # Same for curve list
    if type(curve_list) == int or type(curve_list) == str:
        curve_list = [curve_list]

    # If curve elements are int or strings (so curve_list = ["0.2", "0.5"], then convert it to [["close 0.2"], ["med 0.2"], ...]
    if type(curve_list[0]) == int or type(curve_list[0]) == str:
        list_of_conditions = [param.name_to_nb_list[curve_list[i]] for i in range(len(curve_list))][0]
        curve_list = [[param.nb_to_name[list_of_conditions[i]]] for i in range(len(list_of_conditions))]

    # Fork to fill condition names and densities depending on densities
    conditions_each_curve = [[] for _ in range(len(curve_list))]
    condition_names = [[] for _ in range(len(curve_list))]
    condition_colors = [[] for _ in range(len(curve_list))]

    # For example if my curve_list is: [["0.2", "0.5"], ["0"]]
    for i_curve in range(len(curve_list)):
        curve = curve_list[i_curve]  # curve[0] is ["0.2", "0.5"]
        list_of_conditions = []
        for condition_pool in curve:  # for each element, for example "0.2"
            list_of_conditions += param.name_to_nb_list[condition_pool]  # add numbers for close 0.2, med 0.2, etc.
        conditions_each_curve[i_curve] = list_of_conditions  # add to first curve all the condition numbers
        condition_names[i_curve] = [param.nb_to_name[i] for i in
                                    list_of_conditions]  # add to first curve all the different
        if curve[0] != "all":
            condition_colors[i_curve] = param.name_to_color[param.nb_to_density[
                param.name_to_nb[curve[0]]]]  # color of first curve element density prevails, im tired xD
        else:
            condition_colors[i_curve] = param.name_to_color["all"]

    # TODO remove this loop, since the fork happens with if statements no need to run it once per element in what_to_plot
    for _ in range(len(what_to_plot)):
        # Transform "[["0.2"], ["med 0"]]" into '0.2 & med 0'
        plot_name = str(curve_list).replace("], [", " & ").replace("[[", "").replace("]]", "").replace("''", "")
        if "model" in result_table["folder"][0]:
            plot_name = "modelled " + plot_name
        # And then DRAAAAWWWW
        for i_curve in range(len(conditions_each_curve)):
            curve_name = str(curve_list[i_curve]).replace("[", "").replace("]", "")
            current_conditions = conditions_each_curve[i_curve]
            current_condition_names = condition_names[i_curve]
            current_color = condition_colors[i_curve]

            is_plot = False  # if False, plot functions will not run "plt.show()", so that we can display the next curve
            if i_curve == len(conditions_each_curve) - 1:  # If this is the last curve to plot
                is_plot = True

            if "aggregated_visit_duration" in what_to_plot:
                for thresh in param.threshold_list:
                    plots.plot_selected_data(result_table, "Average duration visits in " + plot_name +
                                             " conditions, aggregated with threshold " + str(thresh),
                                             current_conditions,
                                             "aggregated_visits_thresh_" + str(thresh) + "_total_visit_time",
                                             divided_by="aggregated_visits_thresh_" + str(thresh) + "_nb_of_visits",
                                             plot_model=False, is_plot=is_plot)

            # Speed plots
            if "average_speed" in what_to_plot:
                plots.plot_selected_data(result_table, "OD=" + param.nb_to_density[current_conditions[0]]+", inside",
                                         current_conditions, "average_speed_inside",
                                         divided_by="", is_plot=is_plot)
                plots.plot_selected_data(result_table, "OD=" + param.nb_to_density[current_conditions[0]]+", outside",
                                         current_conditions, "average_speed_outside",
                                         divided_by="", is_plot=is_plot)

            # Data quality
            if "bad_events" in what_to_plot:
                plots.plot_selected_data(result_table, "Average number of bad events in " + plot_name + " conditions",
                                         current_conditions, "nb_of_bad_events",
                                         is_plot=is_plot)

            # if "distribution" in what_to_plot:
            # plots.plot_variable_distribution(results, curve_list=current_conditions, curveeffect_of="nothing", only_first=True)
            # plots.plot_variable_distribution(results, condition_list=current_conditions, effect_of="food")
            # plots.plot_variable_distribution(results, condition_list=current_conditions, effect_of="density")
            # plots.plot_variable_distribution(results, condition_list=current_conditions, effect_of="distance")

            if "distribution_aggregated" in what_to_plot:
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="nothing",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="food",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="distance",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="density",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])

            # Number of double frames (which are a sign of two worms being on the plate)
            if "double_frames" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Average proportion of double frames in " + plot_name + " conditions",
                                         conditions_each_curve, "avg_proportion_double_frames",
                                         is_plot=is_plot)

            # Furthest distance
            if "furthest_distance" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Average number of visits in " + plot_name + "conditions",
                                         current_conditions, "furthest_patch_distance",
                                         divided_by="", is_plot=is_plot)

            if "leaving_events" in what_to_plot:
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="nothing",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="food",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="distance",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(result_table, current_conditions, effect_of="density",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])

            if "leaving_events_delay_distribution" in what_to_plot:
                plots.plot_leaving_delays(result_table,
                                          "Delays before leaving as a function of in_patch time in " + plot_name + " conditions",
                                          current_conditions, bin_size=1000, color=current_color)

            if "leaving_probability" in what_to_plot:
                plots.plot_leaving_probability(result_table,
                                               "Probability of leaving as a function of in_patch time (" + plot_name + ")",
                                               current_conditions, custom_bins=[0, 100, 200, 400, 600, 800, 1000, 1400, 2000, 4000, 6000, 8000, 12000],
                                               worm_limit=10, color=current_color,
                                               label=curve_name,
                                               split_conditions=True, is_plot=is_plot)

            # Number of visits
            if "number_of_visits" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Number of visits",
                                         current_conditions, "nb_of_visits",
                                         divided_by="", is_plot=is_plot,
                                         full_list_of_conditions = curve_list)

            # Number of visits per patch
            if "number_of_visits_per_patch" in what_to_plot:
                print("Pay attention, the title has the density of the first condition!")
                plots.plot_selected_data(result_table,
                                         "Number of visits per total number of patch",
                                         current_conditions, "nb_of_visits",
                                         divided_by="nb_of_patches", is_plot=is_plot,
                                         full_list_of_conditions = curve_list)

            # Number of visits per VISITED patch
            if "number_of_visits_per_visited_patch" in what_to_plot:
                print("Pay attention, the title has the density of the first condition!")
                plots.plot_selected_data(result_table,
                                         "Number of visits per visited patch",
                                         current_conditions, "nb_of_visits",
                                         hard_cut=False,
                                         divided_by="nb_of_visited_patches", is_plot=is_plot,
                                         full_list_of_conditions = curve_list)

            # Average total time spent in each patch
            if "total_visit_time" in what_to_plot:
                plots.plot_selected_data(result_table, "Total time spent in each patch",
                                         current_conditions, "total_visit_time",
                                         divided_by="nb_of_visited_patches", plot_model=False,
                                         is_plot=is_plot, hard_cut=False, remove_censored_events=False,
                                         full_list_of_conditions = curve_list)

            # Visits plots
            if "visit_duration" in what_to_plot:
                plots.plot_selected_data(result_table, "Average duration of visits",
                                         current_conditions, "total_visit_time",
                                         divided_by="nb_of_visits", plot_model=False,
                                         is_plot=is_plot, remove_censored_events=True, soft_cut=False,
                                         full_list_of_conditions = curve_list)

            if "long_enough_visit_duration" in what_to_plot:
                plots.plot_selected_data(result_table, "Visits longer than 4 minutes",
                                         current_conditions, "total_visit_time",
                                         divided_by="nb_of_visits", plot_model=False,
                                         is_plot=is_plot, remove_censored_events=True, soft_cut=True,
                                         visits_longer_than=240,
                                         full_list_of_conditions = curve_list)

            if "first_visit_duration_patch" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "First visit to each patch",
                                         current_conditions, "first_visit_duration",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot, remove_censored_patches=True,
                                         only_first_visited_patch=False,
                                         full_list_of_conditions = curve_list)

            if "first_visit_duration_video" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "First visit of each video",
                                         current_conditions, "first_visit_duration",
                                         divided_by="", plot_model=False, is_plot=is_plot,
                                         remove_censored_patches=True, only_first_visited_patch=True,
                                         full_list_of_conditions = curve_list)

            if "total_visit_time_first_patch" in what_to_plot:
                print("Pay attention, the title has the density of the first condition!")
                print("For now, this codes takes the total time in FIRST UNCENSORED PATCH.")
                print("As opposed to excluding plates with a censored first patch.")
                plots.plot_selected_data(result_table,
                                         "OD=" + param.nb_to_density[current_conditions[0]],
                                         current_conditions, "total_visit_time",
                                         divided_by="", plot_model=False, soft_cut=True,
                                         is_plot=is_plot, remove_censored_patches=True, only_first_visited_patch=True,
                                         full_list_of_conditions = curve_list)

            if "visit_duration_mvt" in what_to_plot:
                plots.plot_selected_data(result_table, "Average duration of MVT visits in " + plot_name + " conditions",
                                         current_conditions, "total_visit_time",
                                         divided_by="mvt_nb_of_visits", plot_model=False,
                                         is_plot=is_plot)

            if "visit_duration_dist_speeds" in what_to_plot:
                # This should only be called from the 20240117-visitdistanceandspeed script
                plots.plot_selected_data(result_table, "Average distance of visits in " + plot_name + " conditions",
                                         current_conditions, "average_distance_each_visit",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot)
                plots.plot_selected_data(result_table, "Average speed during visits in " + plot_name + " conditions",
                                         current_conditions, "average_speed_each_visit",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot)
                plots.plot_selected_data(result_table,
                                         "Average of inverse of speed during visits in " + plot_name + " conditions",
                                         current_conditions, "average_speed_each_visit_inv",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot)

            # Rate of visits (number of visits per transit time unit)
            if "visit_rate" in what_to_plot:
                plots.plot_selected_data(result_table, "Average visit rate in " + plot_name + " conditions",
                                         current_conditions, "nb_of_visits",
                                         divided_by="total_transit_time", is_plot=is_plot)

            if "visit_duration_vs_previous_transit" in what_to_plot:
                plots.plot_visit_time(result_table, trajectories,
                                      "Visit duration vs. previous transit in " + plot_name + " conditions",
                                      current_conditions, "last_travel_time", current_condition_names,
                                      split_conditions=False, is_plot=is_plot, patch_or_pixel="patch", only_first=True)

            if "visit_duration_vs_visit_start" in what_to_plot:
                plots.plot_visit_time(result_table, trajectories,
                                      "Visit duration vs. visit start in " + plot_name + " conditions",
                                      current_conditions, "visit_start", current_condition_names,
                                      split_conditions=True, is_plot=is_plot, patch_or_pixel="patch", only_first=False)

            if "visit_duration_vs_entry_speed" in what_to_plot:
                plots.plot_visit_time(result_table, trajectories,
                                      "Visit duration vs. speed when entering in " + plot_name + " conditions",
                                      current_conditions, "speed_when_entering", current_condition_names,
                                      split_conditions=True, is_plot=is_plot, patch_or_pixel="pixel", only_first=False)

            if "pixels_avg_visit_duration" in what_to_plot:
                # This should be called from the s20240313_pixelwiseleavingprob.py script
                plots.plot_selected_data(result_table,
                                         "Average duration of visits to pixels inside food patches in " + plot_name + " conditions",
                                         current_conditions,
                                         "avg_visit_duration_to_pixels_inside_patches",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot)
                plots.plot_selected_data(result_table,
                                         "Average duration of visits to pixels outside food patches in " + plot_name + " conditions",
                                         current_conditions,
                                         "avg_visit_duration_to_pixels_outside_patches",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot)

            # Proportion of time spent in patches
            if "proportion_of_time" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Average proportion of time spent in patches in" + plot_name + "conditions",
                                         current_conditions, "total_visit_time",
                                         divided_by="total_video_time", is_plot=is_plot)
            # Number of visited patches
            if "number_of_visited_patches" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Average proportion of visited patches in patches in" + plot_name + "conditions",
                                         current_conditions, "nb_of_visited_patches",
                                         divided_by="", is_plot=is_plot)

            # Proportion of visited patches
            if "proportion_of_visited_patches" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Average proportion of visited patches in patches in" + plot_name + "conditions",
                                         current_conditions, "proportion_of_visited_patches",
                                         divided_by="", is_plot=is_plot)

            if "print_parameters_for_model" in what_to_plot:
                revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch = ana.transit_properties(
                    result_table, conditions_each_curve[0], split_conditions=True, is_print=True)

            # Duration between last and first tracked frame
            if "total_video_time" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Total video time in " + plot_name + "conditions",
                                         current_conditions, "total_video_time",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot, hard_cut=False, remove_censored_events=False,
                                         full_list_of_conditions = curve_list)

            # Sum of transit + visit events
            if "total_event_time" in what_to_plot:
                plots.plot_selected_data(result_table,
                                         "Total event time in " + plot_name + "conditions",
                                         current_conditions, "total_event_time",
                                         divided_by="", plot_model=False,
                                         is_plot=is_plot, hard_cut=False, remove_censored_events=False,
                                         full_list_of_conditions = curve_list)

            # Transits plot
            if "transit_duration" in what_to_plot:
                plots.plot_selected_data(result_table, "Average duration of transits in " + plot_name + "conditions",
                                         current_conditions, "total_transit_time",
                                         divided_by="nb_of_visits", is_plot=is_plot)


if __name__ == "__main__":
    # Returns the path where the results are on my machines, for a custom path use the custom_path argument
    # Depending on the starting_from argument, can also generate result tables
    # path = gr.generate(starting_from="results_per_id", custom_path="E:/Results_minipatches_retracked_test/")
    # path = gr.generate(starting_from="controls", custom_path="E:/Alid_hard_drive_20241122/Results_minipatches_retracked/") # This works on my desktop at the lab
    path = gr.generate(starting_from="", test_pipeline=False)
    # starting_from determines where to start generating results:
    # "controls" will generate everything starting here ->
    #       will generate control subfolders with fake patches of each distance in the control folders
    # "smoothing" will generate everything starting here ->
    #       "trajectories.csv": will put all traj.csv together in a big .csv, and smooth it (see smooth_trajectory())
    # "trajectories" will generate everything starting here ->
    #       "trajectories.csv": one line per time point, and information about the patch where the worm is, and speed
    # "results_per_id" will generate everything starting here ->
    #       "results_per_id.csv": one line per id_conservative in the tracking, ie one line per continuous tracking track
    # "results_per_plate" will generate everything starting here ->
    #       "results_per_plate.csv": one line per plate in the tracking, so hopefully one line per worm :p
    # "clean" will generate everything starting here ->
    #       "clean_results.csv": same but removing some plates (see generate_results.exclude_invalid_videos)
    #       "clean_trajectories.csv": trajectories csv but with only the tracks corresponding to the valid plates
    # "" will simply return the working directory path, adapted for Linux or Windows
    # NOTE: lists are stored as strings in the csv, so we retrieve the values with json loads function

    # Retrieve results from what generate_and_save has saved
    trajectories = pd.read_csv(path + "clean_trajectories.csv")
    results = pd.read_csv(path + "clean_results.csv")
    print("Finished retrieving stuff")

    # NOTE: if you are using results generated on a different machines, run this line to switch the "folder" column
    #       of clean_results.csv, which contains the full path to the results of a plate
    # from Generating_data_tables import converting_folder_names
    # converting_folder_names.switch_path(custom_path="E:/Results_minipatches_retracked/")

    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #            [["close 0.2"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #            [["med 0"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #            [["far 0"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #             [["close 0.2"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #             [["med 0.2"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #             [["far 0.2"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #             [["close 0.5"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",S
    #             [["med 0.5"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed",
    #             [["far 0.5"]])

    # plots.trajectories_1condition(trajectories, 12)

    # plots.trajectories_1condition(path, trajectories, [], plate_list=["/media/admin/T7 Shield/Results_minipatches_retracked_shortened/20221126T114423_SmallPatches_C5-CAM1/traj.csv"], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=True, save_fig=False)
    #
    # plots.trajectories_1condition(path, trajectories, [17], show_composite=False, is_plot_patches=True, plot_continuity=False, plot_in_patch=True, plot_speed=False, is_plot=True, save_fig=False, plot_lines=True)
    # plots.trajectories_1condition(path, trajectories, [1], show_composite=False, is_plot_patches=True, plot_continuity=True,  plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [2], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [3], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [4], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [5], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=True, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [6], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [7], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [8], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [9], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [10], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [12], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [13], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [14], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [15], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [16], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [17], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [17, 18, 19, 20, 21], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=True, save_fig=False)
    # plots.trajectories_1condition(path, trajectories, [19], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [20], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)
    # plots.trajectories_1condition(path, trajectories, [21], show_composite=False, is_plot_patches=True, plot_continuity=True, plot_in_patch=False, plot_speed=True, is_plot=False, save_fig=True)

    # plot_graphs(results, "total_visit_time_first_patch", [["close 0", "med 0", "far 0", "superfar 0"]])
    # plot_graphs(results, "total_visit_time_first_patch", [["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]])
    # plot_graphs(results, "total_visit_time_first_patch", [["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]])
    # plot_graphs(results, "total_visit_time_first_patch", [["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    # plot_graphs(results, "total_visit_time", [["close 0", "med 0", "far 0", "superfar 0"],
    #                                                                               ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
    #                                                                               ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
    #                                                                               ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    plot_graphs(results, "total_event_time", [["close 0", "med 0", "far 0", "superfar 0"],
                                                                  ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
                                                                  ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
                                                                  ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    #
    # plot_graphs(results, "visit_duration", [["close 0", "med 0", "far 0", "superfar 0"],
    #                                                                               ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
    #                                                                               ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
    #                                                                               ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])
    #
    #
    # plot_graphs(results, "first_visit_duration_video", [["close 0", "med 0", "far 0", "superfar 0"],
    #                                                                               ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
    #                                                                               ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
    #                                                                               ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])
    #
    #
    # plot_graphs(results, "first_visit_duration_patch", [["close 0", "med 0", "far 0", "superfar 0"],
    #                                                                               ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
    #                                                                               ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
    #                                                                               ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])
    #
    #
    # plot_graphs(results, "number_of_visits_per_visited_patch", [["close 0", "med 0", "far 0", "superfar 0"],
    #                                                                               ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
    #                                                                               ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
    #                                                                               ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])
    #

    # plot_graphs(results, "leaving_probability", [["close 0", "med 0", "far 0", "superfar 0"]])
    # plot_graphs(results, "leaving_probability", [["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]])
    # plot_graphs(results, "leaving_probability", [["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]])
    # plot_graphs(results, "leaving_probability", [["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    # plot_graphs(results, "average_speed", [["close 0", "med 0", "far 0", "superfar 0"]])
    # plot_graphs(results, "average_speed", [["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]])
    # plot_graphs(results, "average_speed", [["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]])
    # plot_graphs(results, "average_speed", [["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    # plots.plot_variable_distribution(results, curve_list=[[17], [18], [19], [20]], variable_list=["cross transits"], only_first=False, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[0], [1], [2], [14]], variable_list=["cross transits"], only_first=False, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[4], [5], [6], [15]], variable_list=["cross transits"], only_first=False, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[12], [8], [13], [16]], variable_list=["cross transits"], only_first=False, plot_cumulative=True)
    #
    # plots.plot_variable_distribution(results, curve_list=[[17], [18], [19], [20]], variable_list=["same transits"], only_first=False, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[0], [1], [2], [14]], variable_list=["same transits"], only_first=False, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[4], [5], [6], [15]], variable_list=["same transits"], only_first=False, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[12], [8], [13], [16]], variable_list=["same transits"], only_first=False, plot_cumulative=True)

    # plots.plot_variable_distribution(results, curve_list=[[17], [18], [19], [20]], variable_list=["visits"], only_first=True, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[0], [1], [2], [14]], variable_list=["visits"], only_first=True, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[4], [5], [6], [15]], variable_list=["visits"], only_first=True, plot_cumulative=True)
    # plots.plot_variable_distribution(results, curve_list=[[12], [8], [13], [16]], variable_list=["visits"], only_first=True, plot_cumulative=True)

    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0.2", "med 0.2", "far 0.2"]])
    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0.5", "med 0.5", "far 0.5"]])

    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0", "med 0", "far 0"]])
    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0.2", "med 0.2", "far 0.2"]])
    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0.5", "med 0.5", "far 0.5"]])

    # plot_graphs(results, "visit_duration_vs_entry_speed", [["close 0", "med 0", "far 0"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed", [["close 0.2", "med 0.2", "far 0.2"]])
    # plot_graphs(results, "visit_duration_vs_entry_speed", [["close 0.5", "med 0.5", "far 0.5"]])

    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0.2", "med 0.2", "far 0.2"]])
    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0.5", "med 0.5", "far 0.5"]])
    # plot_graphs(results, "visit_duration_vs_visit_start", [["close 0", "med 0", "far 0"]])

    # plot_graphs(results, "visit_duration_vs_previous_transit", [["close 0", "med 0", "far 0", "superfar 0"]])
    # plot_graphs(results, "visit_duration_vs_previous_transit", [["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]])
    # plot_graphs(results, "visit_duration_vs_previous_transit", [["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]])
    # plot_graphs(results, "visit_duration_vs_previous_transit", [["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    # plot_graphs(results, "number_of_visits_per_patch", [["close 0", "med 0", "far 0", "superfar 0"]])
    # plot_graphs(results, "number_of_visits_per_patch", [["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]])
    # plot_graphs(results, "number_of_visits_per_patch", [["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]])
    # plot_graphs(results, "number_of_visits_per_patch", [["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    # plot_graphs(results, "number_of_visits_per_visited_patch", [["close 0", "med 0", "far 0", "superfar 0"]])
    # plot_graphs(results, "number_of_visits_per_visited_patch", [["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"]])
    # plot_graphs(results, "number_of_visits_per_visited_patch", [["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"]])
    # plot_graphs(results, "number_of_visits_per_visited_patch", [["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])

    # plot_graphs(results, "leaving_probability", [["close 0", "med 0", "far 0"], ["close 0.2", "med 0.2", "far 0.2"],
    #                                               ["close 0.5", "med 0.5", "far 0.5"], ["med 1.25"]])
    # plot_graphs(results, "leaving_probability", [["close 0"], ["close 0.2"], ["close 0.5"]])
    # plot_graphs(results, "leaving_probability", [["med 0.2"]])
    # plot_graphs(results, "leaving_probability", [["med 0", "med 0.2", "med 0.5", "med 1.25"]])
    # plot_graphs(results, "leaving_probability", [["far 0"], ["far 0.2"], ["far 0.5"]])
    # plot_graphs(results, "leaving_probability", [["cluster 0"], ["cluster 0.2"], ["cluster 0.5"]])

    # plot_graphs(results, "", [["close 0", "med 0", "far 0", "cluster 0"]])
    # plot_graphs(results, "proportion_of_visited_patches", [["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"]])
    # plot_graphs(results, "proportion_of_visited_patches", [["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"]])

    # Possible arguments for plot_graphs:
    #               - "aggregated_visit_duration"
    #               - "average_speed"
    #               - "bad_events"
    #               - "distribution"
    #               - "distribution_aggregated"
    #               - "double_frames"
    #               - "leaving_events"
    #               - "leaving_events_delay_distribution"
    #               - "leaving_probability"
    #               - "visit_duration"
    #               - "visit_duration_dist_speeds"
    #               - "visit_duration_mvt"
    #               - "visit_rate"
    #               - "visit_duration_vs_previous_transit"
    #               - "visit_duration_vs_visit_start"
    #               - "visit_duration_vs_entry_speed"
    #               - "proportion_of_time"
    #               - "number_of_visited_patches"
    #               - "proportion_of_visited_patches"
    #               - "print_parameters_for_model"
    #               - "transit_duration"
    #               - "total_visit_time"

    # video.show_frames("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221015T201543_SmallPatches_C5-CAM4/traj.csv", 11771)
    # plots.patches(["/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221013T114735_SmallPatches_C1-CAM2/traj.csv"])
    # video.show_frames("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111213_SmallPatches_C1-CAM1/traj.csv", 1466)
    # video.show_frames("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111318_SmallPatches_C2-CAM7/traj.csv", 15892)

# TODO function that shows speed as a function of time since patch has been entered (ideally, concatenate all visits)
# TODO function that shows length of (first) visit to a patch as a function of last travel time / average feeding rate in window

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
