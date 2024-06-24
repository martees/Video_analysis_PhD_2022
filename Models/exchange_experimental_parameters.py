# I'm writing this script to determine the influence of a given experimental parameter on the average feeding rate of
# the worm in a given environment.
# We work with a model that has 5 parameters:
# t1: average time it takes to travel between two different patches
# t2: average time it takes to exit a food patch and come back to it
# d (or p_leave): average duration of a visit to a food patch (inverse of probability of leaving p_leave)
# p_rev: probability, once the agent left a given food patch, that it comes back to the same patch
# constant: temporal constant of the exponential decay of food intake when foraging on a food patch
import Parameters.parameters
import analysis as ana
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Generating_data_tables import main as gen


def plot_matrix(condition_list, value_matrix):
    condition_names = [Parameters.parameters.nb_to_name[condition_list[i]] for i in range(len(condition_list))]

    fig, ax = plt.subplots()
    im = ax.imshow(value_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(condition_names)), labels=condition_names)
    ax.set_yticks(np.arange(len(condition_names)), labels=condition_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()


def matrix_of_feeding_rates(condition_list, parameter_to_exchange):
    """
    Function that takes a list of condition numbers, and will show a matrix of those conditions, with in each cell of
    the matrix, the value of the average feeding rate predicted by a mvt-like model with the feeding rate in a patch
    after t time steps: f(t) = exp(-t/constant), and every time the agent leaves a food patch, it has some probability
    of coming back. For now, "mvt with null revisits": revisits are of length 0.
    For each cell i, j of the output matrix, the algorithm computes the feeding rate for the condition i, but using
    the "parameter_to_exchange" of condition j.
    @param condition_list:
    @param parameter_to_exchange:
    @return:
    """
    # Load the results because experimental analysis functions use them
    results_path = gen.generate(test_pipeline=True)
    results = pd.read_csv(results_path + "clean_results.csv")
    # First, load the parameters / distributions for each condition in condition_list
    t1_list = [[] for _ in range(len(condition_list))]
    t2_list = [[] for _ in range(len(condition_list))]
    d_list = [[] for _ in range(len(condition_list))]
    p_rev = [[] for _ in range(len(condition_list))]

    for i_condition, condition in enumerate(condition_list):
        t1_list[i_condition] = ana.return_value_list(results, "cross_transits", [condition], True)
        t2_list[i_condition] = ana.return_value_list(results, "same_transits", [condition], True)
        d_list[i_condition] = ana.return_value_list(results, "visits", [condition], True)
        p_rev[i_condition] = len(t2_list[i_condition]) / len(t1_list[i_condition])



