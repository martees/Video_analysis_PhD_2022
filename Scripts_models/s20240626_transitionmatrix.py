# Script where for each condition, I create two NxN matrices, with N = number of patches in that condition, and save it
# in the results path, in a subfolder called "transition_matrices".
# "transition_probability_0.npy":
#       In each cell [i, j]: p_ij with p_ij being the probability that a transit leaving patch i reaches patch j
#       (so p_ii is the revisit probability), for condition 0
# "transition_duration_0.npy":
#       In each cell [i, j]: a list of transit durations, going from patch i to patch j.
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import plots
from Generating_data_tables import main as gen
from Parameters import parameters as param
import analysis as ana
import find_data as fd
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
from Scripts_models import s202406_exchange_experimental_parameters as exchange_script


def generate_transition_matrices(results_path, results_table, condition_list, plot_everything=False,
                                 plot_transition_matrix=False,
                                 is_recompute=False):
    if not os.path.isdir(results_path + "transition_matrices"):
        os.mkdir(results_path + "transition_matrices")

    print("Generating transition matrices...")
    folder_list = results_table["folder"]
    folder_list_each_cond = [[] for _ in range(len(condition_list))]
    for i_folder, folder in enumerate(folder_list):
        for i_condition, condition in enumerate(condition_list):
            if results_table[results_table["folder"] == folder]["condition"].iloc[0] == condition:
                folder_list_each_cond[i_condition].append(folder)

    # For each condition, load the matrices, or create them, save them and load them
    for i_condition, condition in enumerate(condition_list):
        print("Condition ", condition, "...")
        if not os.path.isfile(
                results_path + "transition_matrices/transition_probability_" + str(condition) + ".npy") or is_recompute or plot_everything:
            # Extract some useful info
            plates = folder_list_each_cond[i_condition]
            patch_positions = param.distance_to_xy[param.nb_to_distance[condition]]
            nb_of_patches = len(patch_positions)
            # Initialize the matrices
            transition_probability_matrix = np.zeros((nb_of_patches, nb_of_patches))
            transition_counts_matrix = np.zeros((nb_of_patches, nb_of_patches))
            transition_times_matrix = [[[] for _ in range(nb_of_patches)] for _ in range(nb_of_patches)]
            distance_matrix = [[[] for _ in range(nb_of_patches)] for _ in range(nb_of_patches)]
            for i_plate, plate in enumerate(plates):
                current_results = results_table[results["folder"] == plate]
                current_visits = np.array(ana.return_value_list(current_results, "visits",
                                                                         [condition], convert_to_duration=False))
                if np.max(current_visits[:, 2]) > nb_of_patches:
                    print("There's a plate with bad patch numbers!!!")
                    nb_of_patches = 0
                for i_patch in range(nb_of_patches):
                    # print(">>> Patch ", i_patch, " / ", nb_of_patches)
                    # Look for the outgoing transits, and then look at next visits to see which patch they went to
                    outgoing_transits_indices = np.where(current_visits[:, 2] == i_patch)
                    next_patch_indices = outgoing_transits_indices[0] + 1
                    next_patch_indices = next_patch_indices[next_patch_indices < len(current_visits)]
                    next_patches = current_visits[next_patch_indices, 2]
                    # Fill in the matrices for this patch
                    for i_transit in range(len(next_patch_indices)):
                        next_patch = next_patches[i_transit]
                        next_patch_index = next_patch_indices[i_transit]
                        previous_visit = current_visits[next_patch_index - 1]
                        next_visit = current_visits[next_patch_index]
                        travel_time = next_visit[0] - previous_visit[1] + 1
                        transition_times_matrix[i_patch][int(next_patch)].append(travel_time)
                        transition_counts_matrix[int(i_patch), int(next_patch)] += 1
            # Then after going through all the plates, one more loop on patches
            for i_patch in range(nb_of_patches):
                # Divide counts by sum of the row to get the probability of reaching patch j from patch i in [i, j]
                transition_probability_matrix[i_patch] = transition_counts_matrix[i_patch] / np.sum(transition_counts_matrix[i_patch])
                # Compute distance to all patches
                distance_matrix[i_patch] = [np.sqrt((patch_positions[i_patch][0] - patch_positions[i][0]) ** 2 + (
                        patch_positions[i_patch][1] - patch_positions[i][1]) ** 2) for i in range(nb_of_patches)]

            np.save(results_path + "transition_matrices/transition_probability_" + str(condition) + ".npy",
                    transition_probability_matrix)
            np.save(results_path + "transition_matrices/transition_times_" + str(condition) + ".npy",
                    np.array(transition_times_matrix, dtype=object))
        else:
            transition_probability_matrix = np.load(
                results_path + "transition_matrices/transition_probability_" + str(condition) + ".npy")
            transition_times_matrix = np.load(
                results_path + "transition_matrices/transition_times_" + str(condition) + ".npy", allow_pickle=True)

        if plot_everything:
            # Plottttt
            fig, [ax0, ax1, ax2] = plt.subplots(1, 3)
            fig.set_size_inches(14, 4)
            fig.suptitle(param.nb_to_name[condition], x=0.1)  # x value is to put the title on the left
            # Distances
            im = ax0.imshow(distance_matrix, cmap="Greys")
            ax0.set_title("Euclidian distance", fontsize=18)
            fig.colorbar(im, ax=ax0)
            # Transition probability
            # Sum the non-diagonal values to normalize edges with that
            mask = np.zeros(np.array(transition_probability_matrix).shape, dtype=bool)
            np.fill_diagonal(mask, 1)
            non_diagonal = np.ma.masked_array(transition_probability_matrix, mask)
            sum_each_line = np.sum(non_diagonal, axis=1)
            cmap = plt.get_cmap('plasma')
            cmap.set_bad('white', 1.)
            im = ax1.imshow(non_diagonal/np.transpose(np.atleast_2d(sum_each_line)), cmap=cmap)
            ax1.set_title("Transition probabilities", fontsize=18)
            #ax1.set_xticks([0, 1, 2], labels=["0", "1", "2"])
            #ax1.set_yticks([0, 1, 2], labels=["0", "1", "2"])
            #ax1.tick_params(axis='both', labelsize=18, top=True, labeltop=True, bottom=False, labelbottom=False)
            fig.colorbar(im, ax=ax1, fraction=0.046)
            # Transition time
            transition_matrix_avg = [list(map(np.nanmean, line)) for line in transition_times_matrix]
            im = ax2.imshow(transition_matrix_avg, vmax=np.nanquantile(transition_matrix_avg, 0.9))
            ax2.set_title("Transition durations (s)", fontsize=18)
            #ax2.set_xticks([0, 1, 2], labels=["0", "1", "2"])
            #ax2.set_yticks([0, 1, 2], labels=["0", "1", "2"])
            #ax2.tick_params(axis='both', labelsize=18, top=True, labeltop=True, bottom=False, labelbottom=False)
            fig.colorbar(im, ax=ax2, fraction=0.046, extend="max")
            plt.tight_layout()
            plt.show()

        if plot_transition_matrix:
            # Transition probability
            # Sum the non-diagonal values to normalize the line with that
            mask = np.ones(np.array(transition_probability_matrix).shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            line_sum = transition_probability_matrix[mask].sum()
            transition_probability_matrix = np.clip(np.rint(transition_probability_matrix / line_sum * 100) / 100, 0, 1)
            # Plot
            plt.imshow(transition_probability_matrix)
            plt.title("Transition probability " + param.nb_to_name[condition])
            plt.colorbar()
            # Custom ticks (patch numbers)
            plt.xticks([0, 1, 2], label=["0", "1", "2"])
            plt.yticks([0, 1, 2], label=["0", "1", "2"])
            plt.tick_params(axis='both', labelsize=18, top=True, labeltop=True, bottom=False, labelbottom=False)
            plt.show()

    return transition_probability_matrix, transition_times_matrix


def draw_circular_arrow(x0, y0, radius=1, aspect=1, direction=270, closing_angle=-330,
                        arrowhead_relative_size=0.2, arrowhead_open_angle=20, color="black", line_width=2, z_order=-10):
    """
    Stolen from:
    https://stackoverflow.com/questions/37512502/how-to-make-arrow-that-loops-in-matplotlib
    Circular arrow drawing. x0 and y0 are the anchor points.
    direction gives the angle of the circle center relative to the anchor
    in degrees. closingangle indicates how much of the circle is drawn
    in degrees with positive being counterclockwise and negative being
    clockwise. aspect is important to make the aspect of the arrow
    fit the current figure.
    """

    xc = x0 + radius * np.cos(direction * np.pi / 180)
    yc = y0 + aspect * radius * np.sin(direction * np.pi / 180)

    head_correction_angle = 5

    if closing_angle < 0:
        step = -1
    else:
        step = 1
    x = [xc + radius * np.cos((ang + 180 + direction) * np.pi / 180)
         for ang in np.arange(0, closing_angle, step)]
    y = [yc + aspect * radius * np.sin((ang + 180 + direction) * np.pi / 180)
         for ang in np.arange(0, closing_angle, step)]

    plt.plot(x, y, color=color, linewidth=line_width, zorder=z_order,
             path_effects=[pe.Stroke(linewidth=1.5 * line_width, foreground="darkgrey"), pe.Normal()])

    x_last = x[-1]
    y_last = y[-1]

    l = radius * arrowhead_relative_size

    head_angle = (direction + closing_angle + (90 - head_correction_angle) *
                  np.sign(closing_angle))

    x = [x_last +
         l * np.cos((head_angle + arrowhead_open_angle) * np.pi / 180),
         x_last,
         x_last +
         l * np.cos((head_angle - arrowhead_open_angle) * np.pi / 180)]
    y = [y_last +
         aspect * l * np.sin((head_angle + arrowhead_open_angle) * np.pi / 180),
         y_last,
         y_last +
         aspect * l * np.sin((head_angle - arrowhead_open_angle) * np.pi / 180)]

    plt.plot(x, y, color=color, linewidth=line_width, zorder=z_order,
             path_effects=[pe.Stroke(linewidth=1.5 * line_width, foreground="darkgrey"), pe.Normal()])


def plot_transition_matrix_graph(results_path, full_plate_list, condition_list, probability_or_time="probability"):
    # First, load the matrices for each condition in condition_list
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        transition_probability_matrices[i_condition], transition_duration_matrices[
            i_condition] = generate_transition_matrices(results_path, [condition], is_recompute=False)
    # Then, load the patch positions for each distance
    patch_positions = heatmap_script.idealized_patch_centers_mm(results_path, full_plate_list, 1847)

    if probability_or_time == "probability":
        value_matrices = transition_probability_matrices
        colors = plt.cm.plasma(np.linspace(0, 1, 101))
    if probability_or_time == "time":
        value_matrices = transition_duration_matrices
        colors = plt.cm.viridis(np.linspace(0, 1, 101))
    print(value_matrices)

    for i_condition, condition in enumerate(condition_list):
        plt.title(param.nb_to_name[condition])
        current_value_matrix = value_matrices[i_condition]
        if probability_or_time == "time":
            current_value_matrix = np.array([list(map(np.nanmean, y)) for y in current_value_matrix])
        current_patch_positions = patch_positions[condition]
        # Find the maximal non-diagonal value to normalize edges with that
        mask = np.ones(np.array(current_value_matrix).shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        max_value = current_value_matrix[mask].max()
        patch_radius_for_plot = 100
        # Plot lines between the patches
        for i_patch in range(len(current_value_matrix)):
            for j_patch in range(len(current_value_matrix[i_patch])):
                current_value = current_value_matrix[i_patch][j_patch]
                # if current_value != 0 and i_patch != j_patch:  # if you don't want the circular arrows
                if current_value != 0:
                    x_origin = current_patch_positions[i_patch][0]
                    y_origin = current_patch_positions[i_patch][1]
                    x_target = current_patch_positions[j_patch][0]
                    y_target = current_patch_positions[j_patch][1]
                    # Adjust x and y to remove patch radius!!!
                    origin_target_distance = ana.distance([x_origin, y_origin], [x_target, y_target])
                    if i_patch != j_patch:
                        adjusted_x_target = x_target - patch_radius_for_plot * (
                                (x_target - x_origin) / origin_target_distance)
                        adjusted_y_target = y_target - patch_radius_for_plot * (
                                (y_target - y_origin) / origin_target_distance)
                        arrow = patches.FancyArrowPatch((x_origin, y_origin), (adjusted_x_target, adjusted_y_target),
                                                        connectionstyle=patches.ConnectionStyle("Arc3", rad=0.2),
                                                        zorder=-10, linewidth=4,
                                                        arrowstyle="->", mutation_scale=10.,
                                                        color=colors[
                                                            np.clip(int(current_value / max_value * 100), 0,
                                                                    100)],
                                                        path_effects=[pe.Stroke(linewidth=4.6, foreground=colors[0]),
                                                                      pe.Normal()])
                        plt.gca().add_patch(arrow)

                    else:
                        draw_circular_arrow(x_origin, y_origin, radius=2 * patch_radius_for_plot, direction=90,
                                            color=colors[np.clip(int(current_value / max_value * 100), 0, 100)],
                                            line_width=4)

            # Then, plot a circle around each of the centers + display patch number in the center
            # Color of the circle depends on probability of revisit of the patch
            for i_patch in range(len(current_patch_positions)):
                x = current_patch_positions[i_patch][0]
                y = current_patch_positions[i_patch][1]
                p_revisit = current_value_matrix[i_patch][i_patch]
                circle = plt.Circle((x, y), radius=patch_radius_for_plot,
                                    color=param.name_to_color[param.nb_to_density[condition]])
                circle_outline = plt.Circle((x, y), radius=patch_radius_for_plot, color=param.name_to_color["0.5"],
                                            linewidth=3, fill=False)
                plt.text(x, y, str(i_patch), horizontalalignment='center', verticalalignment='center',
                         color=param.name_to_color["0.5"], fontsize=16)
                plt.gca().add_artist(circle)
                plt.gca().add_artist(circle_outline)

        plt.xlim(0, 1847)
        plt.ylim(0, 1847)
        plt.gcf().set_size_inches(6, 6)
        plt.show()


def find_nearest_neighbors(results_path, folder_list, condition_list):
    """
    Will return a dictionary with, for each condition in condition_list, a matrix of size NxN with
    N the number of patches in that condition, and containing a 1 in cell [i, j] if patch i and j
    are nearest neighbors, and a 0 otherwise.
    """
    patch_positions = heatmap_script.idealized_patch_centers_mm(results_path, folder_list, 1847)
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
                distance_to_others.append(np.sqrt((curr_x - target_x) ** 2 + (curr_y - target_y) ** 2))
            neighbors_sorted_by_distance = np.argsort(distance_to_others)
            counter = 1  # start at 1 cause 0 is the distance of patch to itself
            i_neighbor = neighbors_sorted_by_distance[counter]
            while counter < nb_of_patches and distance_to_others[i_neighbor] <= distance_to_others[
                neighbors_sorted_by_distance[1]] * 1.3:
                nearest_neighbors[condition][i_patch][i_neighbor] = 1
                counter += 1
                if counter < nb_of_patches:
                    i_neighbor = neighbors_sorted_by_distance[counter]
    return nearest_neighbors


def plot_nearest_neighbor_transition_probability(condition_list, results_path, folder_list, plot_neighbors=False):
    patch_positions = heatmap_script.idealized_patch_centers_mm(results_path, folder_list, 1847)
    nearest_neighbors = find_nearest_neighbors(results_path, folder_list, condition_list)
    if plot_neighbors:
        # Then just plot them 'cause I'm scared
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
                plt.text(x, y, str(np.where(current_nearest_neighbors[i_patch] == 1)[0]), horizontalalignment='center',
                         verticalalignment='center', color="white")
                ax.add_artist(circle)
            plt.title(param.nb_to_distance[condition])
            plt.show()

    values_each_condition = [[] for _ in range(len(condition_list))]
    colors = []
    names = []
    for i_condition, condition in enumerate(condition_list):
        condition_name = param.nb_to_name[condition]
        current_transition_matrix, _ = generate_transition_matrices(results_path, [condition])
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
        values_each_condition[i_condition] += [nearest_prob_each_patch[i] for i in range(len(nearest_prob_each_patch))
                                               if not np.isnan(nearest_prob_each_patch[i])]
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
    if distance == "superfar":
        return [0, 1, 2]
    if distance == "cluster":
        return [1, 3, 6, 11, 13, 18]


def simulate_visit_list(condition, transition_probability, transition_durations, xp_visit_list, nb_of_exp,
                        xp_length=5000, conserve_visit_order=False):
    """
    Will output a list with, for each experiment, a list of visits in the format [start time, end time, patch index].
    - conserve_visit_order: if set to True, then xp_visit_list should have one sublist per rank of visits to the food
                            patch (so a first sublist with first visits to food patches, a second with 2nds, etc.)
                            Then the simulation will take that into account, picking visits with the right rank.
    """
    initial_positions = central_patches(param.nb_to_distance[condition])
    visit_list = [[] for _ in range(nb_of_exp)]
    for i_exp in range(nb_of_exp):
        i_time = 0
        current_patch = random.choice(initial_positions)
        if conserve_visit_order:
            nb_of_visits_each_patch = [0 for _ in range(len(transition_probability))]
            nb_of_visits_each_patch[current_patch] += 1
            # Check that xp_visit_list has the right format (it should have one sublist per visit rank, as returned by
            # the return_value_list() function in analysis.py, with the parameter conserve_visit_order set to True)
            if type(xp_visit_list[0]) is int:
                print("Problem in the simulate_visit_list() function! Mismatch between visit list "
                      "format and value of conserve_visit_order! The output will be only with first visits, probably"
                      "not what you want!!!")
            # Choose among the visits that correspond to the number of visits already made to the patch
            # (so first visit if nb_of_visits is 1, etc.)
            visits_current_rank = xp_visit_list[min(nb_of_visits_each_patch[current_patch], len(xp_visit_list) - 1)]
            visit_end = random.choice(visits_current_rank)
        else:
            visit_end = random.choice(xp_visit_list)
        visit_list[i_exp].append([i_time, visit_end, current_patch])
        i_time = visit_end
        while i_time < xp_length:
            # Memorize where we came from
            previous_patch = current_patch
            # Pick a new place to go to
            current_patch = random.choices(range(len(transition_probability[current_patch])),
                                           weights=transition_probability[current_patch])[0]
            # Add a transit between the two
            visit_start = i_time + random.choice(transition_durations[previous_patch][current_patch])
            if visit_start < xp_length:
                if conserve_visit_order:
                    # If the visit does happen, add a visit to the patch
                    nb_of_visits_each_patch[current_patch] += 1
                    # And choose among the visits that correspond to the number of visits already made to the patch
                    # (so first visit if nb_of_visits is 1, etc.)
                    visits_current_rank = xp_visit_list[min(nb_of_visits_each_patch[current_patch], len(xp_visit_list) - 1)]
                    visit_end = visit_start + random.choice(visits_current_rank)
                else:
                    # If visit order does not matter, randomly choose a visit duration
                    visit_end = visit_start + random.choice(xp_visit_list)
                # Add it to the list and update timer
                visit_list[i_exp].append([visit_start, visit_end, current_patch])
                i_time = visit_end
            else:
                i_time = xp_length
        # If end time is > xp_length, cut the last visit
        if visit_list[i_exp][-1][1] > xp_length:
            visit_list[i_exp][-1][1] = xp_length

        if any([visit[1] > xp_length for visit in visit_list[i_exp]]):
            print("ayayay")

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


def parameter_exchange_matrix(results_path, results_table, condition_list, variable_to_exchange, what_to_plot,
                              nb_of_exp, xp_length, plot_matrix=True):
    # First, load the parameters / distributions for each condition in condition_list
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    xp_visit_duration = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        transition_probability_matrices[i_condition], transition_duration_matrices[
            i_condition] = generate_transition_matrices(results_path, results_table, [condition])
        xp_visit_duration[i_condition] = ana.return_value_list(results_table, "visits", [condition],
                                                               convert_to_duration=True, conserve_visit_order=True)
    # To use as simulation length
    _, xp_total_video_time, _ = ana.results_per_condition(results, condition_list,"total_tracked_time", "")

    value_matrix = np.zeros((len(condition_list), len(condition_list)))
    for i_line in range(len(condition_list)):
        for i_col in range(len(condition_list)):
            where_to_take_each_parameter_from = {"transit_prob": i_line, "transit_times": i_line, "visit_times": i_line,
                                                 "revisit_probability": i_line, variable_to_exchange: i_col}
            transition_probability = copy.deepcopy(
                transition_probability_matrices[where_to_take_each_parameter_from["transit_prob"]])
            revisit_probability_matrix = transition_probability_matrices[
                where_to_take_each_parameter_from["revisit_probability"]]
            # Transform the matrix to match the correct revisit_probability
            # (same-patch transitions are set to be the average revisit probability of the matrix we steal from, and
            # cross-patch transitions are normalized so that the sum of a line is still 1)
            original_revisit_probability = np.mean(
                [transition_probability[i][i] for i in range(len(transition_probability))])
            target_revisit_probability = np.mean(
                [revisit_probability_matrix[i][i] for i in range(len(revisit_probability_matrix))])
            for i_patch in range(len(transition_probability)):
                for j_patch in range(len(transition_probability[i_patch])):
                    if i_patch == j_patch and transition_probability[i_patch][j_patch] != 0:
                        transition_probability[i_patch][j_patch] = target_revisit_probability
                    else:
                        transition_probability[i_patch][j_patch] *= (1 - target_revisit_probability) / (
                                1 - original_revisit_probability)

            transition_durations = transition_duration_matrices[where_to_take_each_parameter_from["transit_times"]]
            visit_durations = xp_visit_duration[where_to_take_each_parameter_from["visit_times"]]
            print("For line of ", str(condition_list[i_line]), ", total sim time is ", xp_total_video_time[i_line])
            list_of_visits = simulate_visit_list(condition_list[i_line], transition_probability, transition_durations,
                                                 visit_durations, nb_of_exp,
                                                 xp_length=xp_total_video_time[i_line],
                                                 conserve_visit_order=True)
            if what_to_plot == "total_visit_time":
                # Total visit time / nb of visited patches
                value_matrix[i_line][i_col] = np.mean(
                    [np.sum(ana.convert_to_durations(visits)) / len(np.unique(np.array(visits)[:, 2])) for visits in
                     list_of_visits])
            if what_to_plot == "avg_visit_duration":
                # Average visit length
                value_matrix[i_line][i_col] = np.mean(
                    [np.mean(ana.convert_to_durations(visits)) for visits in list_of_visits])
            if what_to_plot == "avg_nb_of_visits_per_patch":
                # Average number of visits
                value_matrix[i_line][i_col] = np.mean(
                    [len(visits) / len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits])
            if what_to_plot == "avg_nb_of_visits":
                # Average number of visits
                value_matrix[i_line][i_col] = np.mean([len(visits) for visits in list_of_visits])
            if what_to_plot == "total_xp_time":
                # Average total experimental time
                value_matrix[i_line][i_col] = np.mean([visits[-1][1] for visits in list_of_visits])
            if what_to_plot == "nb_of_explored_patches":
                value_matrix[i_line][i_col] = np.mean(
                    [len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits])

    if plot_matrix:
        exchange_script.plot_matrix(condition_list, value_matrix, variable_to_exchange, what_to_plot)

    else:
        return value_matrix


def behavior_vs_geometry(results_path, results_table, baseline_condition, nb_of_exp, xp_length):
    """
    Function that will plot three total time curves.
    The first one will be the effect of distance in our substitution model, with the experimental parameters that match
    those computed in our actual conditions.
    The second one will be the effect of REVISIT PROBABILITY only (so baseline_condition, but with the revisit probability
    from other distances).
    The third one will be the effect of VISIT TIME only (so baseline_condition, but with the visit times from other
    distances).
    All of those will be computed with a constant food OD, the same as baseline_condition.

    @param results_path:
    @param results_table:
    @param baseline_condition: a string corresponding to a condition (eg "close 0")
    @param nb_of_exp:
    @param xp_length:
    @return:
    """
    baseline_condition_nb = param.name_to_nb[baseline_condition]
    baseline_density = param.nb_to_density[baseline_condition_nb]
    condition_names_this_density = ["close "+baseline_density, "med "+baseline_density, "far "+baseline_density, "superfar "+baseline_density]
    condition_list_this_density = [param.name_to_nb[cond] for cond in condition_names_this_density]

    # Compute the experimental values of total time in patch
    average_per_condition, list_of_avg_each_plate, errorbars = plots.plot_selected_data(results_table, "",
                                                                                        condition_list_this_density,
                                                                                        "total_visit_time",
                                                                                        divided_by="nb_of_visited_patches",
                                                                                        is_plot=False, show_stats=False)
    plt.clf()  # Clear the plot because the previous function plots stuff even though it does not show them

    # Plot the experimental data
    plt.gcf().set_size_inches(7, 9)
    plt.scatter(range(len(average_per_condition)), average_per_condition, marker="x", color="orange", s=100, linewidth=3, label="Experimental values", zorder=10)
    plt.errorbar(range(len(average_per_condition)), average_per_condition, errorbars, color="orange", capsize=5, linewidth=0, elinewidth=3, markeredgewidth=3, zorder=10)

    # Compute the matrices for the simulated data
    visit_exchange_matrix = parameter_exchange_matrix(results_path, results_table, condition_list_this_density, "visit_times", "total_visit_time", nb_of_exp, xp_length, False)
    probability_exchange_matrix = parameter_exchange_matrix(results_path, results_table, condition_list_this_density, "revisit_probability", "total_visit_time", nb_of_exp, xp_length, False)

    # Plot simulated data
    plt.ylabel("Total time per patch (hours)", fontsize=18)
    plt.title("Effect of distance vs. behavior for " + baseline_condition)
    plt.plot([visit_exchange_matrix[i][i]/3600 for i in range(len(visit_exchange_matrix))], color="black", linewidth=4,
             label="Simulation: all parameters change with distance")
    plt.scatter(range(len(visit_exchange_matrix)), [visit_exchange_matrix[i][i]/3600 for i in range(len(visit_exchange_matrix))], color="black", s=67)

    # In order to plot the effect of changing parameters with baseline as an actual baseline, need to find
    # at what index it sits in the matrices
    baseline_index = np.where(np.array(condition_names_this_density) == baseline_condition)[0][0]
    plt.plot([probability_exchange_matrix[baseline_index][i]/3600 for i in range(len(probability_exchange_matrix))], color="cornflowerblue", linewidth=4,
             label="Simulation: only revisit probability changes with distance")
    plt.scatter(range(len(probability_exchange_matrix)), [probability_exchange_matrix[baseline_index][i]/3600 for i in range(len(probability_exchange_matrix))], color="cornflowerblue", s=67)
    plt.plot([visit_exchange_matrix[baseline_index][i]/3600 for i in range(len(visit_exchange_matrix))], color="goldenrod", linewidth=4,
             label="Simulation: only visit time changes with distance")
    plt.scatter(range(len(visit_exchange_matrix)), [visit_exchange_matrix[baseline_index][i]/3600 for i in range(len(visit_exchange_matrix))], color="goldenrod", s=67)
    plt.legend(fontsize=14)
    #plt.ylim(100, 2800)

    # Set the x labels to the distance icons!
    # Stolen from https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings
    for i in range(len(condition_list_this_density)):
        ax = plt.gcf().gca()
        ax.set_xticks([])

        # Image to use
        arr_img = plt.imread(fd.return_icon_path(param.nb_to_distance[condition_list_this_density[i]]))

        # Image box to draw it!
        imagebox = OffsetImage(arr_img, zoom=0.8)
        imagebox.image.axes = ax

        x_annotation_box = AnnotationBbox(imagebox, (i, 0),
                                          xybox=(0, -8),
                                          # that's the shift that the image will have compared to (i, 0)
                                          xycoords=("data", "axes fraction"),
                                          boxcoords="offset points",
                                          box_alignment=(.5, 1),
                                          bboxprops={"edgecolor": "none"})

        ax.add_artist(x_annotation_box)

    plt.show()


if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate()
    results = pd.read_csv(path + "clean_results.csv")
    # trajectories = pd.read_csv(path + "clean_trajectories.csv")
    full_list_of_folders = list(results["folder"])
    #if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
    #    full_list_of_folders.remove(
    #        "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")

    # generate_transition_matrices(path, results, [0, 1, 2, 14], plot_everything=True, plot_transition_matrix=False, is_recompute=True)
    # generate_transition_matrices(path, results, [4, 5, 6, 15], plot_everything=True, plot_transition_matrix=False, is_recompute=True)
    # generate_transition_matrices(path, results, [12, 8, 13, 16], plot_everything=True, plot_transition_matrix=False, is_recompute=True)
    # generate_transition_matrices(path, results, [17, 18, 19, 20], plot_everything=True, plot_transition_matrix=False, is_recompute=True)

    # plot_transition_matrix_graph(path, full_list_of_folders, [14], probability_or_time="probability")
    # plot_transition_matrix_graph(path, full_list_of_folders, [14], probability_or_time="time")

    # behavior_vs_geometry(path, results, "close 0", 2000, 27000)
    behavior_vs_geometry(path, results, "close 0.2", 2000, 30000)
    behavior_vs_geometry(path, results, "close 0.5", 2000, 30000)
    behavior_vs_geometry(path, results, "close 1.25", 2000, 30000)

    # list_of_conditions = list(param.nb_to_name.keys())
    # show_patch_numbers(list_of_conditions)
    # simulate_total_visit_time(results, [0, 1, 2], 30)
    # simulate_nb_of_visits(results, [0, 1, 2], 30)

    # parameter_exchange_matrix(path, results, [0, 1, 2, 14], "visit_times", "total_visit_time", 1000, 30000)
    # parameter_exchange_matrix(path, results, [4, 5, 6, 15], "visit_times", "total_visit_time", 1000, 30000)
    # parameter_exchange_matrix(path, results, [12, 8, 13, 16], "visit_times", "total_visit_time", 1000, 30000)

    # parameter_exchange_matrix(path, results, [0, 1, 2, 14], "revisit_probability", "total_visit_time", 1000, 30000)
    # parameter_exchange_matrix(path, results, [4, 5, 6, 15], "revisit_probability", "total_visit_time", 1000, 30000)
    # parameter_exchange_matrix(path, results, [12, 8, 13, 16], "revisit_probability", "total_visit_time", 1000, 30000)

    # parameter_exchange_matrix(results, [0, 1, 2, 14], "revisit_probability", "avg_visit_duration", 1000, 30000)
    # parameter_exchange_matrix(results, [4, 5, 6, 15], "revisit_probability", "avg_visit_duration", 1000, 30000)
    # parameter_exchange_matrix(results, [12, 8, 13, 16], "revisit_probability", "avg_visit_duration", 1000, 30000)

    # parameter_exchange_matrix(results, [0, 1, 2, 14], "revisit_probability", "total_visit_time", 1000, 30000)
    # parameter_exchange_matrix(results, [4, 5, 6, 15], "revisit_probability", "total_visit_time", 1000, 30000)
    # parameter_exchange_matrix(results, [12, 8, 13, 16], "revisit_probability", "total_visit_time", 1000, 30000)

    # plot_nearest_neighbor_transition_probability(param.list_by_distance, path, full_list_of_folders)
