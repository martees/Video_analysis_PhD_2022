import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import random

from Parameters import parameters as param
from Parameters import custom_legends
from Generating_data_tables import main as gen
import plots

def visit_versus_previous_travel(results, curve_list, travel_time_bins, only_first, min_nb_points,
                                 show_counts=True, show_dots=False):
    for i_curve, curve in enumerate(curve_list):
        current_curve_name_list = [param.nb_to_name[nb] for nb in curve]
        current_curve_name = param.nb_list_to_name[str(curve)]
        current_curve_color = param.name_to_color[current_curve_name]
        # If it's just one condition, just take the values and put them in bags
        if len(curve) == 1:
            transit_bins, average_visit_duration, [errors_inf, errors_sup], counts, binned_visits = plots.plot_visit_time(results, [],
                                                                                                          "",
                                                                                                          curve,
                                                                                                          "last_travel_time",
                                                                                                          current_curve_name_list,
                                                                                                          split_conditions=False,
                                                                                                          is_plot=False,
                                                                                                          patch_or_pixel="patch",
                                                                                                          only_first=only_first,
                                                                                                          custom_bins=travel_time_bins,
                                                                                                          min_nb_data_points=min_nb_points)

            # Convert visit length to minutes and transit length to hours
            average_visit_duration = [avg / 60 for avg in average_visit_duration]
            errors_inf = [err / 60 for err in errors_inf]
            errors_sup = [err / 60 for err in errors_sup]
            transit_bins = [value / 3600 for value in transit_bins]

            # Show error bars as area around curve
            plt.fill_between(transit_bins,
                             np.array(average_visit_duration) - np.array(errors_inf),
                             np.array(average_visit_duration) + np.array(errors_sup),
                             alpha=0.2, facecolor=current_curve_color, antialiased=True)

            # Show visit distribution
            if show_dots:
                for i_bin in range(len(binned_visits)):
                    plt.scatter([transit_bins[i_bin] + 0.00001 * random.randint(-100, 100)
                                 for _ in range(len(binned_visits[i_bin]))], binned_visits[i_bin],
                                color=current_curve_color, alpha=0.4)

            # Show the counts above each point
            if show_counts:
                for i_bin in range(len(transit_bins)):
                    plt.text(transit_bins[i_bin], 1.01*average_visit_duration[i_bin], str(counts[i_bin]))

        # If it's more than one condition, do it so that each condition in the curve has the same weight
        else:
            print("I haven't updated the multi condition per curve feature for this script. Please check it.")
            # Tables with one line per condition, one column per bin
            bins_each_cond = np.empty((len(curve), len(travel_time_bins)))
            visit_durations_each_cond_each_bin = np.empty((len(curve), len(travel_time_bins)))
            errors_inf_each_cond_each_bin = np.empty((len(curve), len(travel_time_bins)))
            errors_sup_each_cond_each_bin = np.empty((len(curve), len(travel_time_bins)))
            # Set them up with nans, that won't be replaced when there are no values
            bins_each_cond[:] = np.nan
            visit_durations_each_cond_each_bin[:] = np.nan
            errors_inf_each_cond_each_bin[:] = np.nan
            errors_sup_each_cond_each_bin[:] = np.nan
            for i_cond, cond in enumerate(curve):
                current_bins, current_visits, [current_err_inf, current_err_sup] = plots.plot_visit_time(results,
                                                                                                         [],
                                                                                                         "",
                                                                                                         [cond],
                                                                                                         "last_travel_time",
                                                                                                         current_curve_name_list,
                                                                                                         split_conditions=False,
                                                                                                         is_plot=False,
                                                                                                         patch_or_pixel="patch",
                                                                                                         only_first=only_first,
                                                                                                         custom_bins=travel_time_bins,
                                                                                                         min_nb_data_points=min_nb_points)
                for i_bin in range(len(current_bins)):
                    bin_index = np.where(np.array(travel_time_bins) == current_bins[i_bin])[0][0]
                    bins_each_cond[i_cond][bin_index] = current_bins[i_bin]
                    visit_durations_each_cond_each_bin[i_cond][bin_index] = current_visits[i_bin]
                    errors_inf_each_cond_each_bin[i_cond][bin_index] = current_err_inf[i_bin]
                    errors_sup_each_cond_each_bin[i_cond][bin_index] = current_err_sup[i_bin]

            # At this point we have big tables with one line per condition, one column per bin, and NaN's when the value
            # does not exist. Now we want to make the final averages by averaging over the condition averages in each bin
            average_visit_duration = np.nanmean(visit_durations_each_cond_each_bin, axis=0)
            # Remove any average that's just on one point!
            for i in range(len(average_visit_duration)):
                if np.sum(~np.isnan(visit_durations_each_cond_each_bin[:, i]), axis=0) == 1:
                    average_visit_duration[i] = np.nan
            # Take the maximal y-extent of the error bars, so sum them with the average and get the max
            coordinates_error_top = np.nansum(np.stack((visit_durations_each_cond_each_bin, errors_sup_each_cond_each_bin)), axis=0)
            coordinates_error_bottom = np.nansum(np.stack((visit_durations_each_cond_each_bin, -1 * errors_inf_each_cond_each_bin)), axis=0)
            indices_of_max = np.argmax(coordinates_error_top, axis=0)
            indices_of_min = np.argmin(coordinates_error_bottom, axis=0)
            errors_inf = [errors_inf_each_cond_each_bin[indices_of_max[i]][i] for i in range(len(travel_time_bins))]
            errors_sup = [errors_sup_each_cond_each_bin[indices_of_max[i]][i] for i in range(len(travel_time_bins))]
            # And then keep the bins
            transit_bins = travel_time_bins
            # And remove any values where there's no average
            errors_inf = np.array([errors_inf[i] for i in range(len(errors_inf)) if not np.isnan(average_visit_duration[i])])
            errors_sup = np.array([errors_sup[i] for i in range(len(errors_sup)) if not np.isnan(average_visit_duration[i])])
            transit_bins = np.array([transit_bins[i] for i in range(len(transit_bins)) if not np.isnan(average_visit_duration[i])])
            average_visit_duration = np.array([average_visit_duration[i] for i in range(len(average_visit_duration)) if not np.isnan(average_visit_duration[i])])

            # Convert visit length to minutes and transit length to hours
            average_visit_duration = [avg/60 for avg in average_visit_duration]
            errors_inf = [err/60 for err in errors_inf]
            errors_sup = [err/60 for err in errors_sup]
            transit_bins = [value/3600 for value in transit_bins]

            # Show errorbars as area around curve
            plt.fill_between([x_bin * (1 + 0.1 * i_curve) for x_bin in transit_bins], average_visit_duration - errors_inf,
                             average_visit_duration + errors_sup,
                             alpha=0.2, facecolor=current_curve_color, antialiased=True)

        # In any case plot the lines all the same
        plt.plot(transit_bins, average_visit_duration,
                 color=current_curve_color, linewidth=4,
                 label=current_curve_name, marker=param.distance_to_marker[param.nb_to_distance[curve[0]]], markersize=16)

    print("Pay attention, title has density of first condition!!")
    plt.title("OD=" + param.nb_to_density[curve_list[0][0]], fontsize=20)
    if only_first == 1:
        plt.ylabel("Average duration of first visit to a patch (minutes)", fontsize=16)
    elif only_first is not False:
        plt.ylabel("Average duration of first " + str(only_first) + " visits to each patch (minutes)", fontsize=16)
    else:
        plt.ylabel("Average duration of all visits (minutes)", fontsize=16)

    # Fancy legend
    # Only works when all the conditions of one curve have the same distance! (because the fancy legend has distance icons)
    if not np.any([len(np.unique([param.nb_to_distance[cond] for cond in curve])) != 1 for curve in curve_list]):
        # Plot empty lines to make the custom legend
        lines = []
        for curve in curve_list:
            line, = plt.plot([], [], color=param.name_to_color[param.nb_list_to_name[str(curve)]], linewidth=6,
                             marker=param.distance_to_marker[param.nb_to_distance[curve[0]]], markersize=4,
                             path_effects=[pe.Stroke(offset=(-0.2, 0.2), linewidth=8,
                                                     foreground=param.name_to_color[param.nb_to_distance[curve[0]]]),
                                           pe.Normal()])
            lines.append(line)
        plt.legend(lines, ["" for _ in range(len(lines))],
                   handler_map={lines[i]: custom_legends.HandlerLineImage(
                       "icon_" + str(param.nb_to_distance[curve_list[i][0]]) + ".png") for i in
                       range(len(lines))},
                   handlelength=1.6, labelspacing=0.0, fontsize=40, borderpad=0.10, loc=2,
                   handletextpad=0.1, borderaxespad=0.3)
        # borderpad is the spacing between the legend and the bottom line of the rectangle around the legend
        # handletextpad is spacing between the legend and the right line of the rectangle around the legend
        # borderaxespad is the spacing between the legend rectangle and the axes of the figure
    else:
        plt.legend()

    plt.gcf().set_size_inches(6, 7)
    plt.xlabel("Duration of the previous travel (hours)", fontsize=16)
    plt.xscale("log")
    plt.yscale("linear")
    # plt.ylim(0, 4)
    plt.show()

path = gen.generate("", test_pipeline=False, shorten_traj=False)
clean_results = pd.read_csv(path + "clean_results.csv")
bin_list = [0, 200, 600, 1200, 2000, 3600, 7200, 10800, 14400, 18000]
only_first_visit = False
#
# visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 0"]], [param.name_to_nb["med 0"]], [param.name_to_nb["far 0"]],
#             [param.name_to_nb["superfar 0"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=10, show_counts=False)
# visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 0.2"]], [param.name_to_nb["med 0.2"]], [param.name_to_nb["far 0.2"]],
#              [param.name_to_nb["superfar 0.2"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=10, show_counts=False)
# visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 0.5"]], [param.name_to_nb["med 0.5"]], [param.name_to_nb["far 0.5"]],
#              [param.name_to_nb["superfar 0.5"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=10, show_counts=False)
# visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 1.25"]], [param.name_to_nb["med 1.25"]], [param.name_to_nb["far 1.25"]],
#              [param.name_to_nb["superfar 1.25"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=10, show_counts=False)

only_first_visit = True

visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 0"]], [param.name_to_nb["med 0"]], [param.name_to_nb["far 0"]],
            [param.name_to_nb["superfar 0"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=3, show_counts=False)
visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 0.2"]], [param.name_to_nb["med 0.2"]], [param.name_to_nb["far 0.2"]],
             [param.name_to_nb["superfar 0.2"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=3, show_counts=False)
visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 0.5"]], [param.name_to_nb["med 0.5"]], [param.name_to_nb["far 0.5"]],
             [param.name_to_nb["superfar 0.5"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=3, show_counts=False)
visit_versus_previous_travel(clean_results, [[param.name_to_nb["close 1.25"]], [param.name_to_nb["med 1.25"]], [param.name_to_nb["far 1.25"]],
             [param.name_to_nb["superfar 1.25"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=3, show_counts=False)

# visit_versus_previous_travel(clean_results, [[param.name_to_nb["far 0.2"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=1)
# visit_versus_previous_travel(clean_results, [[param.name_to_nb["far 1.25"]], [param.name_to_nb["superfar 1.25"]]], travel_time_bins=bin_list, only_first=only_first_visit, min_nb_points=1)

