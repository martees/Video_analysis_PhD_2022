import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

from Parameters import parameters as param
from Generating_data_tables import main as gr
import plots


def nb_of_visits_model_xp(xp_results, model_results, condition_names):
    density = param.nb_to_density[param.name_to_nb[condition_names[0]]]
    condition_list = []
    condition_list = [param.name_to_nb[c] for c in condition_names]
    # Plot experimental data
    xp_averages, xp_full_values, xp_error_bars = plots.plot_selected_data(xp_results, "",
                                                                                  condition_list,
                                                                                  "nb_of_visits",
                                                                                  divided_by="nb_of_visited_patches",
                                                                                  is_plot=False, show_stats=False,
                                                                                  remove_censored_events=False,
                                                                                  hard_cut=True)

    plt.gcf().clf()
    plt.scatter(range(len(xp_averages)), xp_averages, marker="x", color=colors_xp[density], s=100,
                linewidth=3, zorder=10,
                 label="Experimental data")
    plt.errorbar(range(len(xp_averages)), xp_averages, xp_error_bars,
                 color=colors_xp[density], capsize=5, linewidth=0, elinewidth=3, markeredgewidth=3, zorder=10)

    # Plot model data
    model_condition_names = np.array([model_results[i, "dist"]+" "+str(model_results[i, "dens"]) for i in range(model_results.shape[0])])
    averages_finite_lanscape = []
    averages_infinite_lanscape = []
    for i_cond, condition in enumerate(condition_names):
        indices = np.where(model_condition_names == condition)
        current_model_results = model_results[indices, :]
        avg_finite = current_model_results[dt.f.source == "finite", dt.f.nvis][0,0]
        avg_infinite = current_model_results[dt.f.source == "infinite", dt.f.nvis][0,0]

        averages_finite_lanscape.append(avg_finite)
        averages_infinite_lanscape.append(avg_infinite)

        plt.scatter(i_cond, avg_finite, color=colors_model_finite[density],
                    marker = param.distance_to_marker[param.nb_to_distance[param.name_to_nb[condition]]],
                    s=114)
        plt.scatter(i_cond, avg_infinite, color=colors_model_infinite[density],
                    marker = param.distance_to_marker[param.nb_to_distance[param.name_to_nb[condition]]],
                    s=114)

    plt.plot(range(len(averages_finite_lanscape)), averages_finite_lanscape,
                 color=colors_model_finite[density], linewidth=3, label="Model, finite environment")
    plt.plot(range(len(averages_infinite_lanscape)), averages_infinite_lanscape,
                 color=colors_model_infinite[density], linewidth=3, label="Model, infinite environment")

    # Set the x labels to the distance icons!
    # Stolen from https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings
    for i in range(len(condition_list)):
        ax = plt.gcf().gca()
        ax.set_xticks([])

        # Image to use
        arr_img = plt.imread(
            os.getcwd().replace("\\", "/") + "/icon_" +
            param.nb_to_distance[condition_list[i]] + '.png')

        # Image box to draw it!
        imagebox = OffsetImage(arr_img, zoom=0.68)
        imagebox.image.axes = ax

        x_annotation_box = AnnotationBbox(imagebox, (i, 0),
                                          xybox=(0, -8),
                                          # that's the shift that the image will have compared to (i, 0)
                                          xycoords=("data", "axes fraction"),
                                          boxcoords="offset points",
                                          box_alignment=(.5, 1),
                                          bboxprops={"edgecolor": "none", "alpha": 0})

        ax.add_artist(x_annotation_box)


    plt.gcf().set_size_inches(5,4.8)

    # plt.legend(frameon=False, bbox_to_anchor=(0, 1.04), loc="upper left", framealpha=0.4, fontsize=14)
    plt.legend(frameon=False, fontsize=16, draggable=True)
    plt.yticks(fontsize=16)
    plt.ylim(0, 43.7)

    color_first_density = param.name_to_color[param.nb_to_density[param.name_to_nb[condition_names[0]]]]

    plt.tight_layout(pad=1)

    if len(plt.gca().get_yticks()) > 5:
        plt.gca().set_yticks(plt.gca().get_yticks()[::2])
    plt.show()


# I create this script to plot the results of Roger's model
path = gr.generate("")
results_of_model = dt.fread(path + "random_walk_results.csv")
results_of_xp = pd.read_csv(path + "clean_results.csv")

colors_xp = {"0.2": (0.969, 0.871, 0.624), "0.5": (0.945, 0.776, 0.353), "1.25": (0.878, 0.737, 0.361)}
colors_model_finite = {"0.2": (0.851, 0.722, 0.890), "0.5": (0.682, 0.451, 0.757), "1.25": (0.518, 0.204, 0.612)}
colors_model_infinite = {"0.2": (0.824, 0.941, 0.910), "0.5": (0.592, 0.851, 0.784), "1.25": (0.369, 0.761, 0.627)}

nb_of_visits_model_xp(results_of_xp, results_of_model, ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"])
nb_of_visits_model_xp(results_of_xp, results_of_model, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"])
nb_of_visits_model_xp(results_of_xp, results_of_model, ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"])