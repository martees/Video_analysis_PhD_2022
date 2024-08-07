# Script where for each condition, I create two NxN matrices, with N = number of patches in that condition, and save it
# in the results path, in a subfolder called "transition_matrices".
# "transition_probability_0.npy":
#       In each cell [i, j]: p_ij with p_ij being the probability that a transit leaving patch i reaches patch j
#       (so p_ii is the revisit probability), for condition 0
# "transition_duration_0.npy":
#       In each cell [i, j]: a list of transit durations, going from patch i to patch j.

from scipy import ndimage
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from itertools import repeat

from Generating_data_tables import main as gen
from Parameters import parameters as param
import analysis as ana
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
from Scripts_models import s202406_exchange_experimental_parameters as exchange_script


def generate_transition_matrices(condition_list, plot_everything=False, plot_transition_matrix=False):
    print("Generating transition matrices...")
    # First, load the visits, number of patches and patch positions for each condition in condition_list
    visit_list = [[] for _ in range(len(condition_list))]
    patch_positions = [[] for _ in range(len(condition_list))]
    nb_of_patches_list = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        visit_list[i_condition] = np.array(
            ana.return_value_list(results, "visits", [condition], convert_to_duration=False))
        patch_positions[i_condition] = param.distance_to_xy[param.nb_to_distance[condition]]
        nb_of_patches_list[i_condition] = len(patch_positions[i_condition])

    # Then, for each condition, create the matrices and save them
    for i_condition, condition in enumerate(condition_list):
        print("Condition ", condition, "...")
        nb_of_patches = nb_of_patches_list[i_condition]
        current_visits = visit_list[i_condition]
        transition_probability_matrix = np.zeros((nb_of_patches, nb_of_patches))
        transition_times_matrix = [[[] for _ in range(nb_of_patches)] for _ in range(nb_of_patches)]
        distance_matrix = [[[] for _ in range(nb_of_patches)] for _ in range(nb_of_patches)]
        for i_patch in range(nb_of_patches):
            # print(">>> Patch ", i_patch, " / ", nb_of_patches)
            # Look for the outgoing transits, and then look at next visits to see which patch they went to
            outgoing_transits_indices = np.where(current_visits[:, 2] == i_patch)
            next_patch_indices = outgoing_transits_indices[0] + 1
            next_patch_indices = next_patch_indices[next_patch_indices < len(current_visits)]
            next_patches = current_visits[next_patch_indices, 2]
            # Initialize list with in each cell i, the nb of travels going from patch i_patch to patch i
            counts_each_patch = np.zeros(nb_of_patches)
            for i_transit in range(len(next_patch_indices)):
                next_patch = next_patches[i_transit]
                next_patch_index = next_patch_indices[i_transit]
                previous_visit = current_visits[next_patch_index - 1]
                next_visit = current_visits[next_patch_index]
                travel_time = next_visit[0] - previous_visit[1] + 1
                #TODO check that I don't create fake transits??? if last visit in plate i is [100, 200], and
                # first visit in plate i+1 is [300, 400], won't it create a transit [200, 300]??
                # If travel time is negative, it means we're at the junction between two plates
                # => only look at positive travel times
                if travel_time >= 0:
                    transition_times_matrix[i_patch][next_patch].append(travel_time)
                    counts_each_patch[next_patch] += 1
            # Count how many outgoing transits go to each of the patches
            probability_each_patch = ana.array_division_ignoring_zeros(counts_each_patch, np.sum(counts_each_patch))
            transition_probability_matrix[i_patch] = probability_each_patch
            # Compute distance to all patches
            curr_patches = patch_positions[i_condition]
            distance_matrix[i_patch] = [np.sqrt((curr_patches[i_patch][0] - curr_patches[i][0]) ** 2 + (
                        curr_patches[i_patch][1] - curr_patches[i][1]) ** 2) for i in range(nb_of_patches)]

        #if not os.path.isdir(path_to_save + "transition_matrices"):
        #    os.mkdir(path_to_save + "transition_matrices")
        #np.save(path_to_save + "transition_matrices/transition_probability_" + str(condition) + ".npy", transition_probability_matrix)
        #np.save(path_to_save + "transition_matrices/transition_probability_" + str(condition) + ".npy", transition_probability_matrix)

        if plot_everything:
            # Plottttt
            fig, [ax0, ax1, ax2] = plt.subplots(1, 3)
            fig.set_size_inches(10, 3)
            fig.suptitle(param.nb_to_name[condition])
            # Distances
            im = ax0.imshow(distance_matrix)
            ax0.set_title("Euclidian distance")
            fig.colorbar(im, ax=ax0)
            # Transition probability
            im = ax1.imshow(transition_probability_matrix, vmax=0.5)
            ax1.set_title("Transition probabilities")
            fig.colorbar(im, ax=ax1)
            # Transition time
            im = ax2.imshow([list(map(np.nanmean, line)) for line in transition_times_matrix], vmax=500)
            ax2.set_title("Transition durations")
            #cax = fig.add_axes([ax1.get_position().x1 + 0.01, ax1.get_position().y0, 0.02, ax1.get_position().height])
            fig.colorbar(im, ax=ax2)
            plt.show()

        if plot_transition_matrix:
            # Transition probability
            plt.imshow(transition_probability_matrix, vmax=0.1)
            plt.title("Transition probability " + param.nb_to_name[condition] + " , vmax=0.1")
            plt.colorbar()
            plt.show()

        return transition_probability_matrix, transition_times_matrix


def plot_transition_matrix_graph(condition_list):
    fig, axs = plt.subplots(1, len(condition_list))

    # First, load the matrices for each condition in condition_list
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        transition_probability_matrices[i_condition], transition_duration_matrices[
            i_condition] = generate_transition_matrices([condition])
    # Then, load the patch positions for each distance
    patch_positions = heatmap_script.idealized_patch_centers_mm(1847)

    colors = plt.cm.viridis(np.linspace(0, 1, 101))
    for i_condition, condition in enumerate(condition_list):
        axs[i_condition].set_title(param.nb_to_name[condition])
        current_transition_matrix = transition_probability_matrices[i_condition]
        current_patch_positions = patch_positions[condition]
        # Find the maximal non-diagonal value to normalize edges with that
        mask = np.ones(np.array(current_transition_matrix).shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        max_value = current_transition_matrix[mask].max()
        # Plot lines between the patches
        for i_patch in range(len(current_transition_matrix)):
            for j_patch in range(len(current_transition_matrix[i_patch])):
                current_probability = current_transition_matrix[i_patch][j_patch]
                if current_probability != 0 and i_patch != j_patch:
                    axs[i_condition].axis('scaled')
                    x_origin = current_patch_positions[i_patch][0]
                    y_origin = current_patch_positions[i_patch][1]
                    x_target = current_patch_positions[j_patch][0]
                    y_target = current_patch_positions[j_patch][1]
                    # If there is no arrow the other way around, just plot half of the arrow (on the target side)
                    if current_transition_matrix[j_patch][i_patch] != 0:
                        x_origin = (x_origin + x_target) / 2
                        y_origin = (y_origin + y_target) / 2
                    axs[i_condition].arrow(x_origin, y_origin, x_target - x_origin, y_target - y_origin,
                                           color=colors[np.clip(int(current_probability/max_value * 100), 0, 100)],
                                           zorder=-10, width=4, head_length=40, length_includes_head=True)

        #print(np.clip(np.array(current_transition_matrix*100).astype(int), 0, 100))

        # Then, plot a circle around each of the centers + display patch number in the center
        # Color of the circle depends on probability of revisit of the patch
        for i_patch in range(len(current_patch_positions)):
            x = current_patch_positions[i_patch][0]
            y = current_patch_positions[i_patch][1]
            p_revisit = current_transition_matrix[i_patch][i_patch]
            circle = plt.Circle((x, y), radius=20, color=colors[np.clip(int(p_revisit * 100), 0, 100)])
            circle_outline = plt.Circle((x, y), radius=20, color=colors[0], fill=False)
            axs[i_condition].text(x, y, str(i_patch), horizontalalignment='center', verticalalignment='center',
                                  color=colors[0])
            axs[i_condition].add_artist(circle)
            axs[i_condition].add_artist(circle_outline)

        axs[i_condition].axis('scaled')

    plt.show()


def plot_nearest_neighbor_transition_probability(condition_list, plot_neighbors=False):
    patch_positions = heatmap_script.idealized_patch_centers_mm(1847)
    # First, find nearest neighbors for each patch in each condition
    nearest_neighbors = {}
    for i_condition, condition in enumerate(condition_list):
        current_patch_positions = patch_positions[condition]
        nb_of_patches = len(current_patch_positions)
        nearest_neighbors[condition] = np.zeros((nb_of_patches, nb_of_patches))
        for i_patch in range(nb_of_patches):
            curr_x = current_patch_positions[i_patch][0]
            curr_y = current_patch_positions[i_patch][1]
            distance_to_others = []
            for j_patch in range(nb_of_patches):
                target_x = current_patch_positions[j_patch][0]
                target_y = current_patch_positions[j_patch][1]
                distance_to_others.append(np.sqrt((curr_x - target_x)**2 + (curr_y - target_y)**2))
            neighbors_sorted_by_distance = np.argsort(distance_to_others)
            counter = 1  # start at 1 cause 0 is the distance of patch to itself
            i_neighbor = neighbors_sorted_by_distance[counter]
            while counter < nb_of_patches and distance_to_others[i_neighbor] <= distance_to_others[neighbors_sorted_by_distance[1]] * 1.3:
                nearest_neighbors[condition][i_patch][i_neighbor] = 1
                counter += 1
                if counter < nb_of_patches:
                    i_neighbor = neighbors_sorted_by_distance[counter]

    if plot_neighbors:
        # Then just plot them cause i'm scared
        for i_condition, condition in enumerate(condition_list):
            current_patch_positions = patch_positions[condition]
            current_nearest_neighbors = nearest_neighbors[condition]
            # Initialize plot
            fig = plt.gcf()
            ax = fig.gca()
            fig.set_size_inches(10, 10)
            ax.set_xlim(np.min(current_patch_positions[:, 0]) - 100, np.max(current_patch_positions[:, 0]) + 100)
            ax.set_ylim(np.min(current_patch_positions[:, 1]) - 100, np.max(current_patch_positions[:, 1]) + 100)
            ax.set_facecolor('xkcd:darkish blue')
            plt.axis('scaled')
            # Then, plot a circle around each of the centers + display patch number in the center
            for i_patch in range(len(current_patch_positions)):
                x = current_patch_positions[i_patch][0]
                y = current_patch_positions[i_patch][1]
                circle = plt.Circle((x, y), radius=100, color="xkcd:ochre")
                plt.text(x, y, str(np.where(current_nearest_neighbors[i_patch] == 1)[0]), horizontalalignment='center', verticalalignment='center', color="white")
                ax.add_artist(circle)
            plt.title(param.nb_to_distance[condition])
            plt.show()

    values_each_condition = [[] for _ in range(len(condition_list))]
    colors = []
    names = []
    for i_condition, condition in enumerate(condition_list):
        condition_name = param.nb_to_name[condition]
        current_transition_matrix, _ = generate_transition_matrices([condition])
        current_nearest_neighbors = nearest_neighbors[condition]
        nb_of_patches = len(current_nearest_neighbors)
        nearest_prob_each_patch = [0 for _ in range(nb_of_patches)]
        for i_patch in range(nb_of_patches):
            nearest_neighbors_this_patch = np.where(current_nearest_neighbors[i_patch] == 1)[0]
            total_prob = np.nansum(current_transition_matrix[i_patch])
            nearest_prob = np.nansum(current_transition_matrix[i_patch, nearest_neighbors_this_patch])
            revisit_prob = current_transition_matrix[i_patch, i_patch]
            if total_prob != revisit_prob:
                nearest_prob_each_patch[i_patch] = nearest_prob / (total_prob - revisit_prob)
            else:
                nearest_prob_each_patch[i_patch] = np.nan
        # Add it to the main list and remember names+colors for last plot
        values_each_condition[i_condition] += [nearest_prob_each_patch[i] for i in range(len(nearest_prob_each_patch)) if not np.isnan(nearest_prob_each_patch[i])]
        names.append(condition_name)
        colors.append(param.name_to_color[condition_name])
    boxplot = plt.boxplot(values_each_condition, patch_artist=True, labels=names, bootstrap=1000)
    # Fill with colors
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("Probability of transiting to one of your closest neighbors")
    plt.xticks(rotation=45)
    plt.show()


def show_patch_numbers(results_path, full_plate_list, condition_list):
    # First, load patch positions
    patch_positions = heatmap_script.idealized_patch_centers_mm(results_path, full_plate_list, 1847)
    for i_condition, condition in enumerate(condition_list):
        current_patch_positions = patch_positions[condition]
        current_initial_patch_indices = central_patches(param.nb_to_distance[condition])
        # Initialize plot
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_size_inches(5.2, 4)
        ax.set_xlim(np.min(current_patch_positions[:, 0]) - 100, np.max(current_patch_positions[:, 0]) + 100)
        ax.set_ylim(np.min(current_patch_positions[:, 1]) - 100, np.max(current_patch_positions[:, 1]) + 100)
        ax.set_facecolor('xkcd:darkish blue')
        plt.axis('scaled')
        # Then, plot a circle around each of the centers + display patch number in the center
        for i_patch in range(len(current_patch_positions)):
            x = current_patch_positions[i_patch][0]
            y = current_patch_positions[i_patch][1]
            if i_patch not in current_initial_patch_indices:
                circle = plt.Circle((x, y), radius=45, color="xkcd:ochre")
            else:
                circle = plt.Circle((x, y), radius=45, color="xkcd:rust")
            plt.text(x, y, str(i_patch), horizontalalignment='center', verticalalignment='center', color="white")
            ax.add_artist(circle)
        plt.title(param.nb_to_distance[condition])
        plt.show()


def central_patches(distance):
    if distance == "close":
        return [18, 19, 25, 26, 32, 33]
    if distance == "med":
        return [6, 11, 12, 17]
    if distance == "far":
        return [3]
    if distance == "cluster":
        return [1, 3, 6, 11, 13, 18]


def simulate_visit_list(condition, transition_probability, transition_durations, xp_visit_list, nb_of_exp,
                        xp_length=5000):
    """
    Will output a list with, for each experiment, a list of visits in the format [start time, end time, patch index].
    """
    initial_positions = central_patches(param.nb_to_distance[condition])
    visit_list = [[] for _ in range(nb_of_exp)]
    for i_exp in range(nb_of_exp):
        i_time = 0
        current_patch = random.choice(initial_positions)
        visit_list[i_exp].append([i_time, i_time + np.mean(xp_visit_list), current_patch])
        while i_time < xp_length:
            # Memorize where we came from
            previous_patch = current_patch
            # Pick a new place to go to
            current_patch = random.choices(range(len(transition_probability[current_patch])),
                                           weights=transition_probability[current_patch])[0]
            # Add a transit between the two
            visit_start = i_time + random.choice(transition_durations[previous_patch][current_patch])
            if visit_start < xp_length:
                # Randomly choose a visit duration
                visit_end = visit_start + np.mean(xp_visit_list)
                # Add it to the list and update timer
                visit_list[i_exp].append([visit_start, visit_end, current_patch])
                i_time = visit_end
            else:
                i_time = xp_length
        # If end time is > xp_length, cut the last visit
        if visit_list[i_exp][-1][1] > xp_length:
            visit_list[i_exp][-1][1] = xp_length

    return visit_list


def simulate_total_visit_time(results_table, condition_list, nb_of_exp):
    # Model
    average_per_condition = np.zeros(len(condition_list))
    errors_inf = np.zeros(len(condition_list))
    errors_sup = np.zeros(len(condition_list))
    average_per_plate_per_condition = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        print("Loading visits from results...")
        xp_visit_list = ana.return_value_list(results_table, "visits", [condition], True)
        print("Generating transition matrix...")
        transition_probability, transition_durations = generate_transition_matrices([condition])
        print("Running simulations...")
        visit_list = simulate_visit_list(condition, transition_probability, transition_durations, xp_visit_list,
                                         nb_of_exp)
        average_per_plate_per_condition[i_condition] += [
            np.sum(ana.convert_to_durations(visits)) / len(np.unique(np.array(visits)[:, 2])) for visits in visit_list]
        average_per_condition[i_condition] = np.mean(average_per_plate_per_condition[i_condition])
        errors = ana.bottestrop_ci(average_per_plate_per_condition[i_condition], 1000)
        errors_inf[i_condition], errors_sup[i_condition] = [average_per_condition[i_condition] - errors[0],
                                                            errors[1] - average_per_condition[i_condition]]

    # Experiments
    xp_average_per_plate_per_condition, xp_average_per_condition, xp_errorbars = ana.results_per_condition(
        results_table, condition_list, "total_visit_time", "nb_of_visited_patches")

    # Plot experiments vs model
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(10, 6)
    condition_names = [param.nb_to_name[cond] for cond in condition_list]

    # DATA
    axs[0].set_title("Experimental results")
    axs[0].set_ylabel("Total time per patch")
    # Bar plot
    axs[0].bar(range(len(condition_list)), xp_average_per_condition,
               color=[param.name_to_color[condition_names[i]] for i in range(len(condition_names))])
    axs[0].set_xticks(range(len(condition_list)))
    axs[0].set_xticklabels(condition_names, rotation=45)
    axs[0].set(xlabel="Condition number")
    # Plate averages as scatter on top
    for i in range(len(condition_list)):
        axs[0].scatter([range(len(condition_list))[i] for _ in range(len(xp_average_per_plate_per_condition[i]))],
                       xp_average_per_plate_per_condition[i], color="red", zorder=2)
    # Error bars
    axs[0].errorbar(range(len(condition_list)), xp_average_per_condition, xp_errorbars, fmt='.k', capsize=5)

    # MODEL
    axs[1].set_title("Simulated results")
    # Bar plot
    axs[1].bar(range(len(condition_list)), average_per_condition,
               color=[param.name_to_color[condition_names[i]] for i in range(len(condition_names))])
    axs[1].set_xticks(range(len(condition_list)))
    axs[1].set_xticklabels(condition_names, rotation=45)
    axs[1].set(xlabel="Condition number")
    # Plate averages as scatter on top
    for i in range(len(condition_list)):
        axs[1].scatter([range(len(condition_list))[i] for _ in range(len(average_per_plate_per_condition[i]))],
                       average_per_plate_per_condition[i], color="red", zorder=2)
    # Error bars
    axs[1].errorbar(range(len(condition_list)), average_per_condition, [errors_inf, errors_sup], fmt='.k', capsize=5)

    plt.show()


def simulate_nb_of_visits(results_table, condition_list, nb_of_exp):
    # Model
    average_per_condition = np.zeros(len(condition_list))
    errors_inf = np.zeros(len(condition_list))
    errors_sup = np.zeros(len(condition_list))
    average_per_plate_per_condition = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        print("Loading visits from results...")
        xp_visit_list = ana.return_value_list(results_table, "visits", [condition], True)
        print("Generating transition matrix...")
        transition_probability, transition_durations = generate_transition_matrices([condition])
        print("Running simulations...")
        visit_list = simulate_visit_list(condition, transition_probability, transition_durations, xp_visit_list,
                                         nb_of_exp)
        average_per_plate_per_condition[i_condition] += [len(visits) for visits in visit_list]
        average_per_condition[i_condition] = np.mean(average_per_plate_per_condition[i_condition])
        errors = ana.bottestrop_ci(average_per_plate_per_condition[i_condition], 1000)
        errors_inf[i_condition], errors_sup[i_condition] = [average_per_condition[i_condition] - errors[0],
                                                            errors[1] - average_per_condition[i_condition]]

    # Experiments
    xp_average_per_plate_per_condition, xp_average_per_condition, xp_errorbars = ana.results_per_condition(
        results_table, condition_list, "nb_of_visits")

    # Plot experiments vs model
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(10, 6)
    condition_names = [param.nb_to_name[cond] for cond in condition_list]

    # DATA
    axs[0].set_title("Experimental results")
    axs[0].set_ylabel("Number of visits")
    # Bar plot
    axs[0].bar(range(len(condition_list)), xp_average_per_condition,
               color=[param.name_to_color[condition_names[i]] for i in range(len(condition_names))])
    axs[0].set_xticks(range(len(condition_list)))
    axs[0].set_xticklabels(condition_names, rotation=45)
    axs[0].set(xlabel="Condition number")
    # Plate averages as scatter on top
    for i in range(len(condition_list)):
        axs[0].scatter([range(len(condition_list))[i] for _ in range(len(xp_average_per_plate_per_condition[i]))],
                       xp_average_per_plate_per_condition[i], color="red", zorder=2)
    # Error bars
    axs[0].errorbar(range(len(condition_list)), xp_average_per_condition, xp_errorbars, fmt='.k', capsize=5)

    # MODEL
    axs[1].set_title("Simulated results")
    # Bar plot
    axs[1].bar(range(len(condition_list)), average_per_condition,
               color=[param.name_to_color[condition_names[i]] for i in range(len(condition_names))])
    axs[1].set_xticks(range(len(condition_list)))
    axs[1].set_xticklabels(condition_names, rotation=45)
    axs[1].set(xlabel="Condition number")
    # Plate averages as scatter on top
    for i in range(len(condition_list)):
        axs[1].scatter([range(len(condition_list))[i] for _ in range(len(average_per_plate_per_condition[i]))],
                       average_per_plate_per_condition[i], color="red", zorder=2)
    # Error bars
    axs[1].errorbar(range(len(condition_list)), average_per_condition, [errors_inf, errors_sup], fmt='.k', capsize=5)

    plt.show()


def parameter_exchange_matrix(results_table, condition_list, variable_to_exchange, what_to_plot, nb_of_exp):
    # First, load the parameters / distributions for each condition in condition_list
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    xp_visit_duration = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        transition_probability_matrices[i_condition], transition_duration_matrices[
            i_condition] = generate_transition_matrices([condition])
        xp_visit_duration[i_condition] = ana.return_value_list(results_table, "visits", [condition],
                                                               convert_to_duration=True)

    total_time_in_patch_matrix = np.zeros((len(condition_list), len(condition_list)))
    for i_line in range(len(condition_list)):
        for i_col in range(len(condition_list)):
            where_to_take_each_parameter_from = {"transit_prob": i_line, "transit_times": i_line, "visit_times": i_line,
                                                 "revisit_probability": i_line, variable_to_exchange: i_col}
            transition_probability = transition_probability_matrices[where_to_take_each_parameter_from["transit_prob"]]
            revisit_probability_matrix = transition_probability_matrices[where_to_take_each_parameter_from["revisit_probability"]]
            # Transform the matrix to match the correct revisit_probability
            # (same-patch transitions are set to be the average revisit probability of the matrix we steal from, and
            # cross-patch transitions are normalized so that the sum of a line is still 1)
            original_revisit_probability = np.mean([transition_probability[i][i] for i in range(len(transition_probability))])
            target_revisit_probability = np.mean([revisit_probability_matrix[i][i] for i in range(len(revisit_probability_matrix))])
            for i_patch in range(len(transition_probability)):
                for j_patch in range(len(transition_probability[i_patch])):
                    if i_patch == j_patch and transition_probability[i_patch][j_patch] != 0:
                        transition_probability[i_patch][j_patch] = target_revisit_probability
                    else:
                        transition_probability[i_patch][j_patch] *= (1-target_revisit_probability)/(1-original_revisit_probability)

            transition_durations = transition_duration_matrices[where_to_take_each_parameter_from["transit_times"]]
            visit_durations = xp_visit_duration[where_to_take_each_parameter_from["visit_times"]]
            list_of_visits = simulate_visit_list(condition_list[i_line], transition_probability, transition_durations,
                                                 visit_durations, nb_of_exp)
            if what_to_plot == "total_visit_time":
                # Total visit time / nb of visited patches
                total_time_in_patch_matrix[i_line][i_col] = np.mean([np.sum(ana.convert_to_durations(visits))/len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits])
            if what_to_plot == "avg_visit_duration":
                # Average visit length
                total_time_in_patch_matrix[i_line][i_col] = np.mean([np.mean(ana.convert_to_durations(visits)) for visits in list_of_visits])
            if what_to_plot == "avg_nb_of_visits":
                # Average number of visits
                total_time_in_patch_matrix[i_line][i_col] = np.mean([len(visits) for visits in list_of_visits])
            if what_to_plot == "total_xp_time":
                # Average total experimental time
                total_time_in_patch_matrix[i_line][i_col] = np.mean([visits[-1][1] for visits in list_of_visits])
            if what_to_plot == "nb_of_explored_patches":
                total_time_in_patch_matrix[i_line][i_col] = np.mean([len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits])

    exchange_script.plot_matrix(condition_list, total_time_in_patch_matrix, variable_to_exchange, nb_of_exp)


# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=False, old_dataset=False)
results = pd.read_csv(path + "clean_results.csv")
trajectories = pd.read_csv(path + "clean_trajectories.csv")
full_list_of_folders = list(results["folder"])
if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
    full_list_of_folders.remove(
        "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")

#list_of_conditions = list(param.nb_to_name.keys())
#show_patch_numbers(list_of_conditions)
#simulate_total_visit_time(results, [0, 1, 2], 30)
#simulate_nb_of_visits(results, [0, 1, 2], 30)

# parameter_exchange_matrix(results, [0, 1, 2], "revisit_probability", "total_visit_time", 1000)
# parameter_exchange_matrix(results, [4, 5, 6], "revisit_probability", "total_visit_time", 1000)
# parameter_exchange_matrix(results, [12, 13, 14], "revisit_probability", "total_visit_time", 1000)
#
# parameter_exchange_matrix(results, [0, 1, 2], "revisit_probability", "avg_visit_duration", 1000)
# parameter_exchange_matrix(results, [4, 5, 6], "revisit_probability", "avg_visit_duration", 1000)
# parameter_exchange_matrix(results, [12, 13, 14], "revisit_probability", "avg_visit_duration", 1000)
#
# parameter_exchange_matrix(results, [0, 1, 2], "revisit_probability", "avg_nb_of_visits", 1000)
# parameter_exchange_matrix(results, [4, 5, 6], "revisit_probability", "avg_nb_of_visits", 1000)
# parameter_exchange_matrix(results, [12, 13, 14], "revisit_probability", "avg_nb_of_visits", 1000)
#
# parameter_exchange_matrix(results, [0, 1, 2], "revisit_probability", "total_xp_time", 1000)
# parameter_exchange_matrix(results, [4, 5, 6], "revisit_probability", "total_xp_time", 1000)
# parameter_exchange_matrix(results, [12, 13, 14], "revisit_probability", "total_xp_time", 1000)
#
# parameter_exchange_matrix(results, [0, 1, 2], "revisit_probability", "nb_of_explored_patches", 1000)
# parameter_exchange_matrix(results, [4, 5, 6], "revisit_probability", "nb_of_explored_patches", 1000)
# parameter_exchange_matrix(results, [12, 13, 14], "revisit_probability", "nb_of_explored_patches", 1000)


plot_nearest_neighbor_transition_probability(param.list_by_density)

#plot_transition_matrix_graph([0, 1, 2])
#plot_transition_matrix_graph([4, 5, 6])
#plot_transition_matrix_graph([12, 13, 14])
#plot_transition_matrix_graph([8, 9, 10])
#plot_transition_matrix_graph([10, 1])
#plot_transition_matrix_graph([15, 3, 7])
