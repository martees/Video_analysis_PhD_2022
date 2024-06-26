# I'm writing this script to determine the influence of a given experimental parameter on the average feeding rate of
# the worm in a given environment.
# We work with a model that has 5 parameters:
# t1: average time it takes to travel between two different patches
# t2: average time it takes to exit a food patch and come back to it
# d (or p_leave): average duration of a visit to a food patch (inverse of probability of leaving p_leave)
# p_rev: probability, once the agent left a given food patch, that it comes back to the same patch
# constant: temporal constant of the exponential decay of food intake when foraging on a food patch
import random

import Parameters.parameters
import analysis as ana
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Generating_data_tables import main as gen
from Parameters import parameters as param


def exponential_average_feeding_rate(visit_time, travel_time, revisit_time, p_revisit, n_visits, constant):
    average_amount_of_food_1patch = 1 - constant*(np.exp(-(visit_time*n_visits)/constant))
    average_travel_and_feeding_1visit = (1-p_revisit) * travel_time + p_revisit * revisit_time + visit_time
    return average_amount_of_food_1patch / (n_visits * average_travel_and_feeding_1visit)


def plot_parameter_distribution(condition_list, values_dictionary):
    fig, axs = plt.subplots(1, len(values_dictionary))
    condition_names = [Parameters.parameters.nb_to_name[condition_list[i]] for i in range(len(condition_list))]
    condition_color = [Parameters.parameters.name_to_color[condition_names[i]] for i in range(len(condition_names))]

    for i_parameter in range(len(values_dictionary)):
        current_axis = axs[i_parameter]
        current_parameter = list(values_dictionary.keys())[i_parameter]
        current_values = values_dictionary[current_parameter]

        boxplot = current_axis.boxplot(current_values, patch_artist=True)
        # Fill with colors
        for patch, color in zip(boxplot['boxes'], condition_color):
            patch.set_facecolor(color)

        current_axis.set_xticks(range(1, len(current_values) + 1), condition_names)
        current_axis.set_title(current_parameter)

    plt.show()


def plot_matrix(condition_list, value_matrix, parameter_to_exchange, time_constant, nb_of_draws):
    condition_names = [Parameters.parameters.nb_to_name[condition_list[i]] for i in range(len(condition_list))]
    value_matrix /= np.max(np.abs(value_matrix))

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
            ax.text(j, i, np.round(value_matrix[i, j], 2), ha="center", va="center", color="w")

    fig.set_size_inches(len(condition_list)*0.9, (7.5/(9*0.82)) * len(condition_list))  # don't mind me xD
    plt.tight_layout(pad=2)
    fig.suptitle("Exchanging "+str(parameter_to_exchange)+", time constant = "+str(time_constant)+", nb_of_draws = "+str(nb_of_draws))
    ax.set_xlabel("Lends their "+parameter_to_exchange)
    ax.set_ylabel("Steals their "+parameter_to_exchange)
    plt.show()


def matrix_of_feeding_rates(condition_list, parameter_to_exchange, time_constant, nb_of_draws):
    """
    Function that takes a list of condition numbers, and will show a matrix of those conditions, with in each cell of
    the matrix, the value of the average feeding rate predicted by a mvt-like model with the feeding rate in a patch
    after t time steps: f(t) = exp(-t/constant), and every time the agent leaves a food patch, it has some probability
    of coming back. For now, "mvt with null revisits": revisits are of length 0.
    For each cell i, j of the output matrix, the algorithm computes the feeding rate for the condition i, but using
    the "parameter_to_exchange" of condition j.
    @param condition_list:
    @param parameter_to_exchange:
    @param time_constant:
    @return:
    """
    # Load the results because experimental analysis functions use them
    results_path = gen.generate(test_pipeline=False)
    results = pd.read_csv(results_path + "clean_results.csv")
    # First, load the parameters / distributions for each condition in condition_list
    t1_list = [[] for _ in range(len(condition_list))]
    t2_list = [[] for _ in range(len(condition_list))]
    d_list = [[] for _ in range(len(condition_list))]
    p_rev_list = [[] for _ in range(len(condition_list))]
    n_list = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        t1_list[i_condition] = ana.return_value_list(results, "cross transits", [condition], True)
        t2_list[i_condition] = ana.return_value_list(results, "same transits", [condition], True)
        d_list[i_condition] = ana.return_value_list(results, "visits", [condition], True)
        p_rev_list[i_condition] = len(t2_list[i_condition]) / (len(t1_list[i_condition]) + len(t2_list[i_condition]))
        n_list[i_condition] = len(d_list[i_condition]) / len(param.distance_to_xy[param.nb_to_distance[condition]])

    name_to_values = {"t1": t1_list, "t2": t2_list, "d": d_list, "p": p_rev_list, "n": n_list}
    plot_parameter_distribution(condition_list, name_to_values)

    #nb_of_draws = np.max([len(name_to_values[parameter_to_exchange][i]) for i in range(len(condition_list))])
    feeding_rate_matrix = np.zeros((len(condition_list), len(condition_list)))
    for i_line in range(len(feeding_rate_matrix)):
        for i_col in range(len(feeding_rate_matrix)):
            feeding_rate_list = np.zeros((nb_of_draws, 1))

            for i_repetition in range(nb_of_draws):
                where_to_take_each_parameter_from = {"t1": i_line, "t2": i_line, "d": i_line, "p": i_line, "n": i_line,
                                                     parameter_to_exchange: i_col}
                t1 = random.choice(t1_list[where_to_take_each_parameter_from["t1"]])
                t2 = random.choice(t2_list[where_to_take_each_parameter_from["t2"]])
                d = random.choice(d_list[where_to_take_each_parameter_from["d"]])
                p = p_rev_list[where_to_take_each_parameter_from["p"]]
                n = n_list[where_to_take_each_parameter_from["n"]]
                feeding_rate_list[i_repetition] = exponential_average_feeding_rate(d, t1, t2, p, n,
                                                                                   time_constant)

            feeding_rate_matrix[i_line][i_col] = np.mean(feeding_rate_list)

    #plot_matrix(condition_list, feeding_rate_matrix, parameter_to_exchange, time_constant, nb_of_draws)


matrix_of_feeding_rates([0, 1, 2], parameter_to_exchange="d", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([0, 1, 2], parameter_to_exchange="t1", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([0, 1, 2], parameter_to_exchange="t2", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([0, 1, 2], parameter_to_exchange="p", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([0, 1, 2], parameter_to_exchange="n", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([4, 5, 6], parameter_to_exchange="d", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([4, 5, 6], parameter_to_exchange="t1", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([4, 5, 6], parameter_to_exchange="t2", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([4, 5, 6], parameter_to_exchange="p", time_constant=1, nb_of_draws=10000)
matrix_of_feeding_rates([4, 5, 6], parameter_to_exchange="n", time_constant=1, nb_of_draws=10000)
