import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from Parameters import parameters as param
from Parameters import custom_legends
from Generating_data_tables import main as gen
import plots

path = gen.generate("", test_pipeline=False, shorten_traj=False)
clean_results = pd.read_csv(path + "clean_results.csv")
travel_time_bins = [1, 10, 100, 1000, 10000]
# curve_list = [param.name_to_nb_list["close"], param.name_to_nb_list["med"], param.name_to_nb_list["far"],
#               param.name_to_nb_list["superfar"]]
# curve_list = [param.name_to_nb_list["0"], param.name_to_nb_list["0.2"], param.name_to_nb_list["0.5"],
#               param.name_to_nb_list["1.25"]]

curve_list = [[param.name_to_nb["close 0"]], [param.name_to_nb["med 0"]], [param.name_to_nb["far 0"]],
              [param.name_to_nb["superfar 0"]]]
#curve_list = [[param.name_to_nb["close 0.2"]], [param.name_to_nb["med 0.2"]], [param.name_to_nb["far 0.2"]],
#              [param.name_to_nb["superfar 0.2"]]]
#curve_list = [[param.name_to_nb["close 0.5"]], [param.name_to_nb["med 0.5"]], [param.name_to_nb["far 0.5"]],
#              [param.name_to_nb["superfar 0.5"]]]
#curve_list = [[param.name_to_nb["close 1.25"]], [param.name_to_nb["med 1.25"]], [param.name_to_nb["far 1.25"]],
#              [param.name_to_nb["superfar 1.25"]]]

only_first = 1

for i_curve, curve in enumerate(curve_list):
    current_curve_name_list = [param.nb_to_name[nb] for nb in curve]
    current_curve_name = param.nb_list_to_name[str(curve)]
    current_curve_color = param.name_to_color[current_curve_name]
    # If it's just one condition, just take the values and put them in bags
    if len(curve) == 1:
        variable_value_bins, average_visit_duration, [errors_inf, errors_sup] = plots.plot_visit_time(clean_results, [],
                                                                                                      "",
                                                                                                      curve,
                                                                                                      "last_travel_time",
                                                                                                      current_curve_name_list,
                                                                                                      split_conditions=False,
                                                                                                      is_plot=False,
                                                                                                      patch_or_pixel="patch",
                                                                                                      only_first=only_first,
                                                                                                      custom_bins=travel_time_bins,
                                                                                                      min_nb_data_points=20)
        # Classical error bars
        # plt.errorbar([x_bin * (1 + 0.1 * i_curve) for x_bin in variable_value_bins], average_visit_duration,
        #              [errors_inf, errors_sup], fmt='.k', capsize=5)
        # Show error bars as area around curve
        plt.fill_between([x_bin * (1 + 0.1 * i_curve) for x_bin in variable_value_bins],
                         np.array(average_visit_duration) - np.array(errors_inf),
                         np.array(average_visit_duration) + np.array(errors_sup),
                         alpha=0.2, facecolor=current_curve_color, antialiased=True)
    # If it's more than one condition, do it so that each condition in the curve has the same weight
    else:
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
            current_bins, current_visits, [current_err_inf, current_err_sup] = plots.plot_visit_time(clean_results,
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
                                                                                                     min_nb_data_points=10)
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
        coordinates_error_top = np.nansum(np.stack((visit_durations_each_cond_each_bin,errors_sup_each_cond_each_bin)), axis=0)
        coordinates_error_bottom = np.nansum(np.stack((visit_durations_each_cond_each_bin, -1 * errors_inf_each_cond_each_bin)), axis=0)
        indices_of_max = np.argmax(coordinates_error_top, axis=0)
        indices_of_min = np.argmin(coordinates_error_bottom, axis=0)
        errors_inf = [errors_inf_each_cond_each_bin[indices_of_max[i]][i] for i in range(len(travel_time_bins))]
        errors_sup = [errors_sup_each_cond_each_bin[indices_of_max[i]][i] for i in range(len(travel_time_bins))]
        # And then keep the bins
        variable_value_bins = travel_time_bins
        # And remove any values where there's no average
        errors_inf = np.array([errors_inf[i] for i in range(len(errors_inf)) if not np.isnan(average_visit_duration[i])])
        errors_sup = np.array([errors_sup[i] for i in range(len(errors_sup)) if not np.isnan(average_visit_duration[i])])
        variable_value_bins = np.array([variable_value_bins[i] for i in range(len(variable_value_bins)) if not np.isnan(average_visit_duration[i])])
        average_visit_duration = np.array([average_visit_duration[i] for i in range(len(average_visit_duration)) if not np.isnan(average_visit_duration[i])])

        # Show errorbars as area around curve
        plt.fill_between([x_bin * (1 + 0.1 * i_curve) for x_bin in variable_value_bins], average_visit_duration - errors_inf,
                         average_visit_duration + errors_sup,
                         alpha=0.2, facecolor=current_curve_color, antialiased=True)

    # In any case plot the lines all the same
    plt.plot([x_bin * (1 + 0.1 * i_curve) for x_bin in variable_value_bins], average_visit_duration,
             color=current_curve_color, linewidth=4,
             label=current_curve_name)


if only_first == 1:
    plt.title("OD=" + param.nb_to_density[curve[0]], fontsize=20)
    plt.ylabel("Average duration of first visit to a patch (seconds)", fontsize=12)
elif only_first is not False:
    plt.title("Average of first " + str(only_first) + " visit length, OD=" + param.nb_to_density[curve[0]])
    plt.ylabel("Average duration of first " + str(only_first) + " visits to a patch (sec)", fontsize=12)
else:
    plt.title("Average of all visits as a function of the duration of the previous transit")
    plt.ylabel("Average duration of all visits (seconds)", fontsize=12)

# Fancy legend
# Only works when all the conditions of one curve have the same distance! (because the fancy legend has distance icons)
if not np.any([len(np.unique([param.nb_to_distance[cond] for cond in curve])) != 1 for curve in curve_list]):
    # Plot empty lines to make the custom legend
    lines = []
    for curve in curve_list:
        line, = plt.plot([], [], color=param.name_to_color[param.nb_list_to_name[str(curve)]], linewidth=12,
                         path_effects=[pe.Stroke(offset=(-0.2, 0.2), linewidth=15,
                                                 foreground=param.name_to_color[param.nb_to_distance[curve[0]]]),
                                       pe.Normal()])
        lines.append(line)
    plt.legend(lines, ["" for _ in range(len(lines))],
               handler_map={lines[i]: custom_legends.HandlerLineImage(
                   "icon_" + str(param.nb_to_distance[curve_list[i][0]]) + ".png") for i in
                   range(len(lines))},
               handlelength=1.6, labelspacing=0.0, fontsize=36, borderpad=0.10, loc=2,
               handletextpad=0.3, borderaxespad=0.3)
    # borderpad is the spacing between the legend and the bottom line of the rectangle around the legend
    # handletextpad is spacing between the legend and the right line of the rectangle around the legend
    # borderaxespad is the spacing between the legend rectangle and the axes of the figure
else:
    plt.legend()

plt.gcf().set_size_inches(6, 7)
plt.xlabel("Duration of the previous travel (seconds, log scale)", fontsize=12)
plt.xscale("log")
plt.ylim(0, 3000)
plt.show()
