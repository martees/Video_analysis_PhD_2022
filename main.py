## Hi
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

import analysis as ana
import parameters as param
# My code
import plots
import generate_results as gr
import find_data as fd
import video


def plot_graphs(what_to_plot, curve_list=None):
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
        condition_colors[i_curve] = param.name_to_color[curve[0]]  # color of first curve element prevails, im tired xD

    for _ in range(len(what_to_plot)):
        # Transform "[["0.2"], ["med 0"]]" into '0.2 & med 0'
        plot_name = str(curve_list).replace("], [", " & ").replace("[[", "").replace("]]", "").replace("''", "")
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
                    plots.plot_selected_data(results, "Average duration visits in " + plot_name +
                                             " conditions, aggregated with threshold " + str(thresh),
                                             current_conditions, current_condition_names,
                                             "aggregated_visits_thresh_" + str(thresh) + "_total_visit_time",
                                             divided_by="aggregated_visits_thresh_" + str(thresh) + "_nb_of_visits",
                                             mycolor=current_color, plot_model=True, is_plot=is_plot)

            # Speed plots
            if "average_speed" in what_to_plot:
                plots.plot_selected_data(results, "Average speed in " + plot_name + " conditions (inside)",
                                         current_conditions, current_condition_names, "average_speed_inside",
                                         divided_by="", mycolor=current_color, is_plot=is_plot)
                plots.plot_selected_data(results, "Average speed in " + plot_name + " conditions (outside)",
                                         current_conditions, current_condition_names, "average_speed_outside",
                                         divided_by="", mycolor=current_color, is_plot=is_plot)

            # Data quality
            if "bad_events" in what_to_plot:
                plots.plot_selected_data(results, "Average number of bad events in " + plot_name + " conditions",
                                         current_conditions, current_condition_names, "nb_of_bad_events",
                                         mycolor=current_color, is_plot=is_plot)

            if "distribution" in what_to_plot:
                plots.plot_variable_distribution(results, condition_list=current_conditions, effect_of="nothing")
                plots.plot_variable_distribution(results, condition_list=current_conditions, effect_of="food")
                plots.plot_variable_distribution(results, condition_list=current_conditions, effect_of="density")
                plots.plot_variable_distribution(results, condition_list=current_conditions, effect_of="distance")

            if "distribution_aggregated" in what_to_plot:
                plots.plot_variable_distribution(results, current_conditions, effect_of="nothing",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_conditions, effect_of="food",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_conditions, effect_of="distance",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_conditions, effect_of="density",
                                                 variable_list=["aggregated_visits"],
                                                 threshold_list=[0, 10, 100, 100000])

            # Number of double frames (which are a sign of two worms being on the plate)
            if "double_frames" in what_to_plot:
                plots.plot_selected_data(results,
                                         "Average proportion of double frames in " + plot_name + " conditions",
                                         conditions_each_curve, current_condition_names, "avg_proportion_double_frames",
                                         mycolor=current_color, is_plot=is_plot)

            # Furthest distance
            if "furthest_distance" in what_to_plot:
                plots.plot_selected_data(results,
                                         "Average number of visits in " + plot_name + "conditions",
                                         current_conditions, current_condition_names, "furthest_patch_distance",
                                         divided_by="", mycolor=current_color, is_plot=is_plot)

            if "leaving_events" in what_to_plot:
                plots.plot_variable_distribution(results, current_conditions, effect_of="nothing",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_conditions, effect_of="food",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_conditions, effect_of="distance",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])
                plots.plot_variable_distribution(results, current_conditions, effect_of="density",
                                                 variable_list=["aggregated_leaving_events"],
                                                 threshold_list=[0, 10, 100, 100000])

            if "leaving_events_delay_distribution" in what_to_plot:
                plots.plot_leaving_delays(results,
                                          "Delays before leaving as a function of in_patch time in " + plot_name + " conditions",
                                          current_conditions, bin_size=1000, color=current_color)

            if "leaving_probability" in what_to_plot:
                plots.plot_leaving_probability(results,
                                               "Probability of leaving as a function of in_patch time (" + plot_name + ")",
                                               current_conditions, bin_size=1000, worm_limit=10, color=current_color, label=curve_name,
                                               split_conditions=False, is_plot=is_plot)

            # Number of visits
            if "number_of_visits" in what_to_plot:
                plots.plot_selected_data(results,
                                         "Average number of visits in " + plot_name + "conditions",
                                         current_conditions, current_condition_names, "nb_of_visits",
                                         divided_by="", mycolor=current_color, is_plot=is_plot)
            # Visits plots
            if "visit_duration" in what_to_plot:
                plots.plot_selected_data(results, "Average duration of visits in " + plot_name + " conditions",
                                         current_conditions, current_condition_names, "total_visit_time",
                                         divided_by="nb_of_visits", mycolor=current_color, plot_model=True,
                                         is_plot=is_plot)

            if "visit_duration_mvt" in what_to_plot:
                plots.plot_selected_data(results, "Average duration of MVT visits in " + plot_name + " conditions",
                                         current_conditions, current_condition_names, "total_visit_time",
                                         divided_by="mvt_nb_of_visits", mycolor=current_color, plot_model=True,
                                         is_plot=is_plot)

            # Rate of visits (number of visits per time unit)
            if "visit_rate" in what_to_plot:
                plots.plot_selected_data(results, "Average visit rate in " + plot_name + " conditions",
                                         current_conditions, current_condition_names, "nb_of_visits",
                                         divided_by="total_video_time", mycolor=current_color, is_plot=is_plot)

            if "visit_duration_vs_previous_transit" in what_to_plot:
                plots.plot_visit_time(results, trajectories,
                                      "Visit duration vs. previous transit in " + plot_name + " conditions",
                                      current_conditions, "Last travel time", current_condition_names,
                                      split_conditions=True, is_plot=is_plot)

            if "visit_duration_vs_visit_start" in what_to_plot:
                plots.plot_visit_time(results, trajectories,
                                      "Visit duration vs. visit start in " + plot_name + " conditions",
                                      current_conditions, "Visit start", current_condition_names, is_plot=is_plot)

            if "visit_duration_vs_entry_speed" in what_to_plot:
                plots.plot_visit_time(results, trajectories,
                                      "Visit duration vs. speed when entering the patch in " + plot_name + " conditions",
                                      current_conditions, "Speed when entering", current_condition_names,
                                      is_plot=is_plot)

            # Proportion of time spent in patches
            if "proportion_of_time" in what_to_plot:
                plots.plot_selected_data(results,
                                         "Average proportion of time spent in patches in" + plot_name + "conditions",
                                         current_conditions, current_condition_names, "total_visit_time",
                                         divided_by="total_video_time", mycolor=current_color, is_plot=is_plot)

            # Proportion of visited patches
            if "proportion_of_visited_patches" in what_to_plot:
                plots.plot_selected_data(results,
                                         "Average proportion of visited patches in patches in" + plot_name + "conditions",
                                         current_conditions, current_condition_names, "proportion_of_visited_patches",
                                         divided_by="total_video_time", mycolor=current_color, is_plot=is_plot)

            if "print_parameters_for_model" in what_to_plot:
                revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch = ana.transit_properties(
                    results, conditions_each_curve, split_conditions=True, is_print=True)

                revisit_probability, cross_transit_probability, exponential_leaving_probability, min_visit, average_visit, average_same_patch, average_cross_patch = ana.transit_properties(
                    results, conditions_each_curve, split_conditions=False)

                print("Transit properties for conditions ", current_condition_names)
                print("Revisit probability: ", revisit_probability)
                print("Cross-patch probability: ", cross_transit_probability)
                print("Exponential leaving probability: ", exponential_leaving_probability)
                print("Minimal duration of visits: ", min_visit)
                print("Average duration of visits: ", average_visit)
                print("Average duration of same patch transits: ", average_same_patch)
                print("Average duration of cross patch transits: ", average_cross_patch)
                print("-----")

            # Transits plot
            if "transit_duration" in what_to_plot:
                plots.plot_selected_data(results, "Average duration of transits in " + plot_name + "conditions",
                                         current_conditions, current_condition_names, "total_transit_time",
                                         divided_by="nb_of_visits", mycolor=current_color, is_plot=is_plot)


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
# "" will simply return the working directory path, adapted for Linux or Windows
# NOTE: lists are stored as strings in the csv, so we retrieve the values with json loads function
path = gr.generate(starting_from="")
# Only generate the results in the beginning of your analysis!

# Retrieve results from what generate_and_save has saved

#trajectories = pd.read_csv(path + "clean_trajectories.csv")
results = pd.read_csv(path + "clean_results.csv")
print("Finished retrieving stuff")

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
#               - "visit_duration_mvt"
#               - "visit_rate"
#               - "visit_duration_vs_previous_transit"
#               - "visit_duration_vs_visit_start"
#               - "visit_duration_vs_entry_speed"
#               - "proportion_of_time"
#               - "print_parameters_for_model"
#               - "transit_duration"

#plot_graphs(["visit_duration"], ["all"])


# TODO function that shows speed as a function of time since patch has been entered (ideally, concatenate all visits)
# TODO function that shows length of (first) visit to a patch as a function of last travel time / average feeding rate in window

# TODO movement stuff between patches: speed, turning rate, MSD over time
# TODO radial_tolerance in a useful way
