# I'm writing this script to determine the influence of a given experimental parameter on the average feeding rate of
# the worm in a given environment.
# We work with a model that has 5 parameters:
# t1: average time it takes to travel between two different patches
# t2: average time it takes to exit a food patch and come back to it
# d: average duration of a visit to a food patch
# constant: temporal constant of the exponential decay of food intake when foraging on a food patch
import random

import Parameters.parameters
import analysis as ana
import pandas as pd
import numpy as np
from itertools import groupby
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import (DrawingArea, OffsetImage, AnnotationBbox)

from Generating_data_tables import main as gen
from Parameters import parameters as param


def plot_parameter_distribution(condition_list, values_dictionary):
    fig, axs = plt.subplots(1, len(values_dictionary))

    condition_names = [Parameters.parameters.nb_to_name[condition_list[i]] for i in range(len(condition_list))]
    condition_color = [Parameters.parameters.name_to_color[condition_names[i]] for i in range(len(condition_names))]
    bins = np.logspace(0, 4, 20)
    for i_parameter in range(len(values_dictionary)):
        current_axis = axs[i_parameter]
        current_parameter = list(values_dictionary.keys())[i_parameter]
        current_values = values_dictionary[current_parameter]

        #boxplot = current_axis.boxplot(current_values, patch_artist=True)
        # # Fill with colors
        #for patch, color in zip(boxplot['boxes'], condition_color):
        #    patch.set_facecolor(color)

        for i_condition in range(len(condition_list)):
            current_axis.hist(current_values[i_condition], color=condition_color[i_condition], linewidth=3,
                              label=condition_names[i_condition], histtype="step", density=True, bins=bins)

        # current_axis.set_xticks(range(1, len(current_values) + 1), condition_names)
        current_axis.set_title(current_parameter)
        current_axis.set_yscale("log")

    plt.legend()
    plt.show()


def plot_matrix(condition_list, value_matrix, parameter_to_exchange, nb_of_draws):
    condition_names = [Parameters.parameters.nb_to_name[condition_list[i]] for i in range(len(condition_list))]
    #value_matrix /= np.max(np.abs(value_matrix))

    fig, ax = plt.subplots()
    ax.imshow(value_matrix, cmap="coolwarm")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(condition_names)), labels=condition_names)
    ax.set_yticks(np.arange(len(condition_names)), labels=condition_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(condition_list)):
        for j in range(len(condition_list)):
            ax.text(j, i, int(np.rint(value_matrix[i, j])), ha="center", va="center", color="w")

    fig.set_size_inches(len(condition_list) * 0.9, (7.5 / (9 * 0.82)) * len(condition_list))  # don't mind me xD
    plt.tight_layout(pad=2)
    fig.suptitle("Lends their " + parameter_to_exchange, fontsize=12)  # that's actually the x label (on top of figure)
    ax.set_xlabel("OD="+param.nb_to_density[condition_list[0]], labelpad=10, fontsize=20)  # That's actually the title (under the figure)
    ax.set_ylabel("Steals their " + parameter_to_exchange, labelpad=45, fontsize=12)
    # Move the x ticks from bottom to top, and no labels (will be images)
    plt.tick_params(axis='x', top=True, labeltop=False, bottom=False, labelbottom=False, labelsize=12)
    plt.tick_params(axis='y', left=False, labelleft=False, labelsize=12)

    # Set the x and y labels to the distance icons!
    # Stolen from https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings
    for i in range(len(condition_list)):
        # Image to use
        arr_img = plt.imread(os.getcwd().replace("\\", "/")[:-len("Scripts_models/")] +
                             "/Parameters/icon_" + param.nb_to_distance[condition_list[i]] + '.png')

        # Image box to draw it!
        imagebox = OffsetImage(arr_img, zoom=0.5)
        imagebox.image.axes = ax

        x_annotation_box = AnnotationBbox(imagebox, (i, 0),
                                          xybox=(0, 180),  # that's the shift that the image will have compared to (i, 0)
                                          xycoords=("data", "axes fraction"),
                                          boxcoords="offset points",
                                          box_alignment=(.5, 1),
                                          bboxprops={"edgecolor": "none"})

        y_annotation_box = AnnotationBbox(imagebox, (0, 0.85 - i/4),
                                          xybox=(-30, 0),
                                          xycoords=("data", "axes fraction"),
                                          boxcoords="offset points",
                                          box_alignment=(1, .5),
                                          bboxprops={"edgecolor": "none"})

        ax.add_artist(x_annotation_box)
        ax.add_artist(y_annotation_box)

    plt.show()


def matrix_of_total_time_in_patch(condition_list, parameter_to_exchange, nb_of_draws, plot_distribution=False):
    """
    Function that takes a list of condition numbers, and will show a matrix of those conditions, with in each cell of
    the matrix, the value of the average feeding rate predicted by a mvt-like model with the feeding rate in a patch
    after t time steps: f(t) = exp(-t/constant), and every time the agent leaves a food patch, it has some probability
    of coming back. For now, "mvt with null revisits": revisits are of length 0.
    For each cell i, j of the output matrix, the algorithm computes the feeding rate for the condition i, but using
    the "parameter_to_exchange" of condition j.
    """
    # Load the results because experimental analysis functions use them
    results_path = gen.generate(test_pipeline=False)
    results = pd.read_csv(results_path + "clean_results.csv")
    # First, load the parameters / distributions for each condition in condition_list
    t1_list = [[] for _ in range(len(condition_list))]
    t2_list = [[] for _ in range(len(condition_list))]
    d_list = [[] for _ in range(len(condition_list))]
    patch_list_list = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        t1_list[i_condition] = ana.return_value_list(results, "cross transits", [condition], True)
        t2_list[i_condition] = ana.return_value_list(results, "same transits", [condition], True)
        d_list[i_condition] = ana.return_value_list(results, "visits", [condition], True)
        patch_list_list[i_condition] = ana.return_value_list(results, "patch_sequence", [condition], False)

    if plot_distribution:
        name_to_values = {"t1": t1_list, "t2": t2_list, "d": d_list}
        plot_parameter_distribution(condition_list, name_to_values)

    #nb_of_draws = np.max([len(name_to_values[parameter_to_exchange][i]) for i in range(len(condition_list))])
    total_time_in_patch_matrix = np.zeros((len(condition_list), len(condition_list)))
    total_time_matrix = np.zeros((len(condition_list), len(condition_list)))
    for i_line in range(len(total_time_in_patch_matrix)):
        print("Condition ", i_line + 1, " / ", len(condition_list))
        for i_col in range(len(total_time_in_patch_matrix)):
            total_time_in_patch = np.zeros((nb_of_draws, 1))
            total_time = np.zeros((nb_of_draws, 1))

            for i_repetition in range(nb_of_draws):

                where_to_take_each_parameter_from = {"t1": i_line, "t2": i_line, "d": i_line, "p": i_line, "n": i_line,
                                                     "patch_list": i_line, parameter_to_exchange: i_col}

                # Choose a patch sequence and compute its characteristics
                patch_list = random.choice(patch_list_list[where_to_take_each_parameter_from["patch_list"]])
                # Remove double values from patch list ([1, 2, 2, 2, 1] => [1, 2, 1])
                patch_sequence = [i[0] for i in groupby(patch_list)]
                nb_of_visits = len(patch_list)
                total_nb_of_transits = len(patch_list) + 1
                nb_of_revisits = len(patch_list) - len(patch_sequence)
                nb_of_non_revisits = total_nb_of_transits - nb_of_revisits

                # Choose values from the experimental data
                if len(t1_list) >= nb_of_non_revisits:
                    t1 = random.sample(t1_list[where_to_take_each_parameter_from["t1"]], nb_of_non_revisits)
                else:
                    t1 = random.choices(t1_list[where_to_take_each_parameter_from["t1"]], k=nb_of_non_revisits)

                if len(t2_list) >= nb_of_revisits:
                    t2 = random.sample(t2_list[where_to_take_each_parameter_from["t2"]], nb_of_revisits)
                else:
                    t2 = random.choices(t2_list[where_to_take_each_parameter_from["t2"]], k=nb_of_revisits)

                if len(d_list) >= nb_of_visits:
                    d = random.sample(d_list[where_to_take_each_parameter_from["d"]], nb_of_visits)
                else:
                    d = random.choices(d_list[where_to_take_each_parameter_from["d"]], k=nb_of_visits)

                # Compute total time spent in each food patch
                list_of_times = [0 for _ in range(np.max(patch_list) + 1)]
                #duration_to_this_point = 0
                #previous_patch = -1
                for i_visit in range(nb_of_visits):
                    current_patch = patch_list[i_visit]
                    list_of_times[current_patch] += d[i_visit]
                    #duration_to_this_point += d[i_visit]
                    #if current_patch == previous_patch:
                    #    duration_to_this_point += t2.pop()
                    #else:
                    #    duration_to_this_point += t1.pop()
                    #previous_patch = current_patch

                #total_time_in_patch[i_repetition] = np.sum(list_of_times)
                total_time_in_patch[i_repetition] = np.sum(list_of_times) / np.sum(np.sum([t1, t2, d]))
                total_time[i_repetition] = np.sum(np.sum([t1, t2, d]))

            total_time_in_patch_matrix[i_line][i_col] = np.mean(total_time_in_patch)
            total_time_matrix[i_line][i_col] = np.mean(total_time)

    plot_matrix(condition_list, total_time_in_patch_matrix, parameter_to_exchange, nb_of_draws)
    # plot_matrix(condition_list, total_time_matrix, parameter_to_exchange, nb_of_draws)


if __name__ == "main":
    matrix_of_total_time_in_patch([0, 1, 2], parameter_to_exchange="d", nb_of_draws=10000, plot_distribution=False)
    matrix_of_total_time_in_patch([0, 1, 2], parameter_to_exchange="t1", nb_of_draws=10000)
    matrix_of_total_time_in_patch([0, 1, 2], parameter_to_exchange="t2", nb_of_draws=10000)
    matrix_of_total_time_in_patch([0, 1, 2], parameter_to_exchange="patch_list", nb_of_draws=10000)
    matrix_of_total_time_in_patch([4, 5, 6], parameter_to_exchange="d", nb_of_draws=10000, plot_distribution=False)
    matrix_of_total_time_in_patch([4, 5, 6], parameter_to_exchange="t1", nb_of_draws=10000)
    matrix_of_total_time_in_patch([4, 5, 6], parameter_to_exchange="t2", nb_of_draws=10000)
    matrix_of_total_time_in_patch([4, 5, 6], parameter_to_exchange="patch_list", nb_of_draws=10000)
