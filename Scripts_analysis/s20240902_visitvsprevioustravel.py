import pandas as pd
import matplotlib.pyplot as plt
from Parameters import parameters as param
from Generating_data_tables import main as gen
import plots

path = gen.generate("", test_pipeline=False)
clean_results = pd.read_csv(path + "clean_results.csv")
#curve_list = [param.name_to_nb_list["close"], param.name_to_nb_list["med"], param.name_to_nb_list["far"],
#             param.name_to_nb_list["superfar"]]
#curve_list = [param.name_to_nb_list["0"], param.name_to_nb_list["0.2"], param.name_to_nb_list["0.5"],
#              param.name_to_nb_list["1.25"]]

#curve_list = [[param.name_to_nb["close 0"]], [param.name_to_nb["med 0"]], [param.name_to_nb["far 0"]],
#              [param.name_to_nb["superfar 0"]]]
#curve_list = [[param.name_to_nb["close 0.2"]], [param.name_to_nb["med 0.2"]], [param.name_to_nb["far 0.2"]],
#              [param.name_to_nb["superfar 0.2"]]]
#curve_list = [[param.name_to_nb["close 0.5"]], [param.name_to_nb["med 0.5"]], [param.name_to_nb["far 0.5"]],
#              [param.name_to_nb["superfar 0.5"]]]
#curve_list = [[param.name_to_nb["close 1.25"]], [param.name_to_nb["med 1.25"]], [param.name_to_nb["far 1.25"]],
#              [param.name_to_nb["superfar 1.25"]]]

curve_list = [[param.name_to_nb["med 0"]], [param.name_to_nb["med 0.2"]], [param.name_to_nb["med 0.5"]],
              [param.name_to_nb["med 1.25"]]]

only_first = 1

for i_curve, curve in enumerate(curve_list):
    current_curve_name_list = [param.nb_to_name[nb] for nb in curve]
    current_curve_name = param.nb_list_to_name[str(curve)]
    current_curve_color = param.name_to_color[current_curve_name]
    variable_value_bins, average_visit_duration, [errors_inf, errors_sup] = plots.plot_visit_time(clean_results, [],
                                                                                                  "",
                                                                                                  curve,
                                                                                                  "last_travel_time",
                                                                                                  current_curve_name_list,
                                                                                                  split_conditions=False,
                                                                                                  is_plot=False,
                                                                                                  patch_or_pixel="patch",
                                                                                                  only_first=only_first)
    plt.plot([x_bin * (1 + 0.1*i_curve) for x_bin in variable_value_bins], average_visit_duration, color=current_curve_color, linewidth=4,
             label=current_curve_name)
    plt.errorbar([x_bin * (1 + 0.1*i_curve) for x_bin in variable_value_bins], average_visit_duration, [errors_inf, errors_sup], fmt='.k', capsize=5)

if only_first != False:
    plt.title("Average of first "+str(only_first)+" visit lengths as a function of the duration of the previous transit")
    plt.ylabel("Average duration of first "+str(only_first)+" visits to a patch (sec)")
else:
    plt.title("Average of all visits as a function of the duration of the previous transit")
    plt.ylabel("Average duration of all visits (sec)")
plt.xlabel("Previous travel time")
plt.xscale("log")
plt.legend()
plt.show()
