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
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import time

import plots
from Generating_data_tables import main as gen
from Parameters import parameters as param
from Parameters import custom_legends
import analysis as ana
import find_data as fd
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
from Scripts_models import s202406_exchange_experimental_parameters as exchange_script
from Scripts_analysis import s20240918_total_time_vs_nb_of_visits as TpNv
from plots import plot_variable_distribution


def generate_transition_matrices(results_path, results_table, xp_visits_each_plate, xp_transits_each_plate,
                                 condition_list, plot_everything=False,
                                 plot_transition_matrix=False, is_recompute=False):
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
                current_visits = np.array(xp_visits_each_plate[i_plate])
                current_transits = np.array(xp_transits_each_plate[i_plate])
                if not np.isnan(current_visits).all() and type(current_visits) in [list, np.ndarray] and len(current_visits) > 0:
                    if np.max(current_visits[:, 2]) > nb_of_patches:
                        print("There's a plate with bad patch numbers!!!")
                        nb_of_patches = 0
                    for i_patch in range(nb_of_patches):
                        # print(">>> Patch ", i_patch, " / ", nb_of_patches)
                        # Look for the outgoing transits, and then look at next visits to see which patch they went to
                        visit_end_indices = np.where(current_visits[:, 2] == i_patch) # visits in current patch
                        next_patch_indices = visit_end_indices[0] + 1  # visits next to the ones in current patch
                        next_patch_indices = next_patch_indices[next_patch_indices < len(current_visits)]  # remove index beyond last visit
                        next_patches = current_visits[next_patch_indices, 2]
                        # Fill in the matrices for this patch
                        # NOTE: For each transit, there are three cases:
                        #   1. there is actually a transit from end of visit to start of next visit
                        #   2. there is a transit but separated by a (bad) hole: in this case, do not count it!
                        #   (Note: there cannot be more than one hole, as any hole in the middle of the
                        #       transit would have been filled) (Note 2: I think the best thing would be to count those
                        #       for probabilities but not duration, BUT doing this would mean having non-null transition
                        #       probabilities without the corresponding transit durations...)
                        #   3. there is no transit (between the two visits is just a tracking hole)
                        # NOTE ABOUT NOTE: Now this is useless because we exclude any transit with a hole earlier
                        #                  in the processing. I am still keeping the code as is though, because it
                        #                  might be useful if we want to change the analysis later.
                        for i_transit in range(len(next_patch_indices)):
                            next_patch = next_patches[i_transit]
                            next_patch_index = next_patch_indices[i_transit]
                            previous_visit_end = current_visits[next_patch_index - 1][1]
                            next_visit_start = current_visits[next_patch_index][0]
                            # Check in which case we are!!! (see above)
                            if previous_visit_end in current_transits[:,0] and next_visit_start in current_transits[:, 1]:
                                transition_counts_matrix[int(i_patch), int(next_patch)] += 1
                                travel_time = next_visit_start - previous_visit_end + param.one_frame_in_seconds
                            else:
                                if previous_visit_end in current_transits[:,0]:  # hole between transit end and next visit
                                    i_where = int(np.squeeze(np.where(current_transits[:, 0] == previous_visit_end)))
                                elif next_visit_start in current_transits[:, 1]:  # hole between visit end and transit start
                                    i_where = int(np.squeeze(np.where(current_transits[:, 1] == next_visit_start)))
                                else:  # case where it does visit - hole - transit - hole - visit T-T
                                    transits_starting_after = np.where(current_transits[:, 0] > previous_visit_end)
                                    transits_ending_before = np.where(current_transits[:, 1] < next_visit_start)
                                    i_where = np.intersect1d(transits_starting_after, transits_ending_before)
                                    if len(i_where) != 1:
                                        print("oops")
                                    else:
                                        i_where = int(i_where)
                                travel_time = current_transits[i_where, 1] - current_transits[i_where, 0] + param.one_frame_in_seconds
                            if type(travel_time) is np.ndarray:
                                print("There's a leftover teleportation in plate ", plate, ", excluding the bad transit")
                            elif type(travel_time) not in [int, float, np.float64]:
                                print("You have a problem with times in transition matrices..........")
                            else:
                                transition_times_matrix[i_patch][int(next_patch)].append(travel_time)

            # Then after going through all the plates, one more loop on patches
            for i_patch in range(nb_of_patches):
                # Divide counts by sum of the row to get the probability of reaching patch j from patch i in [i, j]
                if np.sum(transition_counts_matrix[i_patch]) != 0:  # (if it's 0, leave everything as 0)
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
            im = ax0.imshow(distance_matrix, cmap="Greys", vmax=45)
            ax0.set_title("Euclidian distance", fontsize=20)
            clb = fig.colorbar(im, ax=ax0)
            clb.ax.tick_params(labelsize=16)
            ax0.tick_params(labelsize=16)

            # Transition probability
            # Sum the non-diagonal values to normalize edges with that
            cmap = plt.get_cmap('plasma')
            cmap.set_bad('black', 1.)
            # mask = np.zeros(np.array(transition_probability_matrix).shape, dtype=bool)
            # np.fill_diagonal(mask, 1)
            # non_diagonal = np.ma.masked_array(transition_probability_matrix, mask)
            # sum_each_line = np.sum(non_diagonal, axis=1)
            # im = ax1.imshow(non_diagonal/np.transpose(np.atleast_2d(sum_each_line)), cmap=cmap)
            im = ax1.imshow(transition_probability_matrix, cmap=cmap, norm=LogNorm())
            ax1.set_title("Transition probabilities", fontsize=20)
            #ax1.set_xticks([0, 1, 2], labels=["0", "1", "2"])
            #ax1.set_yticks([0, 1, 2], labels=["0", "1", "2"])
            #ax1.tick_params(axis='both', labelsize=18, top=True, labeltop=True, bottom=False, labelbottom=False)
            clb = fig.colorbar(im, ax=ax1, fraction=0.046, extend="min")
            clb.ax.tick_params(labelsize=16)
            ax1.tick_params(labelsize=16)

            # Transition time
            transition_matrix_avg = np.array([list(map(np.nanmean, line)) for line in transition_times_matrix])
            im = ax2.imshow(transition_matrix_avg/60, vmax=np.nanquantile(transition_matrix_avg/60, 0.9))
            ax2.set_title("Transit durations (min)", fontsize=20)
            #ax2.set_xticks([0, 1, 2], labels=["0", "1", "2"])
            #ax2.set_yticks([0, 1, 2], labels=["0", "1", "2"])
            #ax2.tick_params(axis='both', labelsize=18, top=True, labeltop=True, bottom=False, labelbottom=False)
            clb = fig.colorbar(im, ax=ax2, fraction=0.046, extend="max")
            clb.ax.tick_params(labelsize=16)
            ax2.tick_params(labelsize=16)

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


def plot_transition_matrix_graph(results_path, results_table, full_plate_list, condition_list, probability_or_time="probability"):
    # First, load the parameters / distributions for each condition in condition_list
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    first_visits = [[] for _ in range(len(condition_list))]
    xp_visits = [[] for _ in range(len(condition_list))]
    xp_transits = [[] for _ in range(len(condition_list))]
    xp_visit_durations = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        xp_visits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                                   "no_hole_visits", "",
                                                                   remove_censored_events=False, hard_cut=True)
        xp_transits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                                   "aggregated_raw_transits", "",
                                                                   remove_censored_events=False, hard_cut=True)
        for i_plate in range(len(xp_visits[i_condition])):
            v = xp_visits[i_condition][i_plate]
            if type(v) is list and len(v) > 0:
                first_visits[i_condition].append(v[0][0])
                xp_visit_durations[i_condition] += ana.convert_to_durations(v)

        transition_probability_matrices[i_condition], transition_duration_matrices[
            i_condition] = generate_transition_matrices(results_path, results_table, xp_visits, xp_transits,
                                                        [condition])

    # Then, load the patch positions for each distance
    patch_positions = heatmap_script.idealized_patch_centers_mm(results_path, full_plate_list, 1847)

    if probability_or_time == "probability":
        value_matrices = transition_probability_matrices
        colors = plt.cm.plasma(np.logspace(-4, 0, 101))
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

        # if probability_or_time == "probability":
        #     # Sum the non-diagonal values to normalize edges with that
        #     mask = np.zeros(np.array(current_value_matrix).shape, dtype=bool)
        #     np.fill_diagonal(mask, 1)
        #     non_diagonal = np.ma.masked_array(current_value_matrix, mask)
        #     sum_each_line = np.sum(non_diagonal, axis=1)
        #     current_value_matrix = non_diagonal/np.transpose(np.atleast_2d(sum_each_line))
        #     max_value = np.max(current_value_matrix)
        #     min_value = np.min(current_value_matrix)
        #     print(current_value_matrix)

        patch_radius_for_plot = 100
        # Plot lines between the patches
        for i_patch in range(len(current_value_matrix)):
            for j_patch in range(len(current_value_matrix[i_patch])):
                current_value = current_value_matrix[i_patch][j_patch]
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
                                                    color=colors[np.clip(int(current_value * 100), 0, 100)],
                                                    # color=colors[np.clip(int(((current_value-min_value)/(max_value-min_value)) * 100),
                                                    #                      0, 100)],
                                                    path_effects=[pe.Stroke(linewidth=4.6, foreground=colors[0]),
                                                                  pe.Normal()])
                    plt.gca().add_patch(arrow)

                else:
                    draw_circular_arrow(x_origin, y_origin, radius=2 * patch_radius_for_plot, direction=90,
                                        color="white", line_width=4)

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


def show_patch_numbers(results_path, full_plate_list, condition_list=None):
    if condition_list is None:
        condition_list = [0, 1, 2, 14, 4, 5, 6, 15, 12, 8, 13, 16, 17, 18, 19, 20]
    # First, load patch positions
    patch_positions = heatmap_script.idealized_patch_centers_mm(results_path, full_plate_list, 1847)
    for i_condition, condition in enumerate(condition_list):
        current_patch_positions = patch_positions[condition]
        current_initial_patch_indices = central_patches(param.nb_to_distance[condition])
        # Initialize plot
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_size_inches(3.8, 3.3)
        ax.set_xlim(np.min(current_patch_positions[:, 0]) - 100, np.max(current_patch_positions[:, 0]) + 100)
        ax.set_ylim(np.min(current_patch_positions[:, 1]) - 100, np.max(current_patch_positions[:, 1]) + 100)
        # ax.set_facecolor('xkcd:darkish blue')
        plt.axis('scaled')
        # Then, plot a circle around each of the centers + display patch number in the center
        for i_patch in range(len(current_patch_positions)):
            x = current_patch_positions[i_patch][0]
            y = current_patch_positions[i_patch][1]
            # if i_patch not in current_initial_patch_indices:
            circle = plt.Circle((x, y), radius=45, color="xkcd:darkish blue")
            # else:
            #     circle = plt.Circle((x, y), radius=45, color="xkcd:rust")
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
                        first_transit_length_and_patch=None, xp_length=5000, conserve_visit_order=False):
    """
    Will output a list with, for each experiment, a list of visits in the format [start time, end time, patch index].
    - conserve_visit_order: if set to True, then xp_visit_list should have one sublist per rank of visits to the food
                            patch (so a first sublist with first visits to food patches, a second with 2nds, etc.)
                            Then the simulation will take that into account, picking visits with the right rank.
    - first_transit_length_and_patch: [[d, id], [d, id],...] list with one element per experimental worm, d is the
                            duration of the first transit (start of first visit of the video) and id is the id of the
                            patch where this visit happens.
    """
    if first_transit_length_and_patch is not None:
        initial_positions = [int(f[1]) for f in first_transit_length_and_patch]
        first_transit_length = [f[0] for f in first_transit_length_and_patch]
    else:
        initial_positions = central_patches(param.nb_to_distance[condition])
        first_transit_length = [0 for _ in range(len(initial_positions))]

    visit_list = [[] for _ in range(nb_of_exp)]
    for i_exp in range(nb_of_exp):
        i_time = 0
        if conserve_visit_order:
            nb_of_visits_each_patch = [0 for _ in range(len(transition_probability))]
            # Check that xp_visit_list has the right format (it should have one sublist per visit rank, as returned by
            # the return_value_list() function in analysis.py, with the parameter conserve_visit_order set to True)
            if type(xp_visit_list[0]) is int:
                print("Problem in the simulate_visit_list() function! Mismatch between visit list "
                      "format and value of conserve_visit_order! The output will be only with first visits, probably"
                      "not what you want!!!")
        while i_time < xp_length:
            if i_time == 0:
                current_patch = random.choice(initial_positions)
                visit_start = random.choice(first_transit_length)
            else:
                # Memorize where we came from
                previous_patch = current_patch
                # (Case that happens only in OD=1.25 where transits are scarce)
                # To handle dead-end patches, if a patch has no outgoing transits, then set back the simulation to
                # one of the initial positions.
                if np.sum(transition_probability[current_patch]) == 0 :
                    previous_patch = random.choice(initial_positions)
                # Pick a new place to go to
                current_patch = random.choices(range(len(transition_probability[previous_patch])),
                                               weights=transition_probability[previous_patch])[0]
                # Add a transit between the two
                visit_start = i_time + random.choice(transition_durations[previous_patch][current_patch]) - param.one_frame_in_seconds

            if visit_start < xp_length:
                if conserve_visit_order:
                    # If the visit does happen, add a visit to the patch
                    nb_of_visits_each_patch[current_patch] += 1
                    # And choose among the visits that correspond to the number of visits already made to the patch
                    # (so first visit if nb_of_visits is 1, etc.)
                    visits_current_rank = xp_visit_list[min(nb_of_visits_each_patch[current_patch], len(xp_visit_list)) - 1]
                    # Remove one frame to the visit end, because that is how visits are encoded in the xp
                    visit_end = visit_start + random.choice(visits_current_rank) - param.one_frame_in_seconds
                else:
                    # If visit order does not matter, randomly choose a visit duration
                    visit_end = visit_start + random.choice(xp_visit_list[0]) - param.one_frame_in_seconds
                # Add it to the list and update timer
                visit_list[i_exp].append([visit_start, visit_end, current_patch])
                i_time = visit_end

            else:
                i_time = xp_length

            if not type(i_time) in [np.float64, int]:
                print("Problem with timestamps in simulations")

        # If there is no visit, add a fake one so that this xp is considered to have total time of 0 in one patch
        if len(visit_list[i_exp]) == 0:
            visit_list[i_exp].append([0, 0, 0])
        # If there is at least one visit, and the last visit's end time is > xp_length, cut it
        elif visit_list[i_exp][-1][1] > xp_length:
            visit_list[i_exp][-1][1] = xp_length
            # visit_list[i_exp][-1][1] = visit_list[i_exp][-1][1]

        if any([visit[1] > xp_length for visit in visit_list[i_exp]]):
            print("Problem with time cutting in simulations")

    return visit_list


def parameter_exchange_matrix(results_path, results_table, condition_list, variable_to_exchange, what_to_plot,
                              nb_of_exp, xp_length, plot_matrix=True, conserve_visit_order=False,
                              plot_variable_distributions=False, plot_tpnv=False, plot_1stvisit=False):
    if plot_tpnv:
        fig, axs = plt.subplots(1, 4)
        fig.set_size_inches(8, 5)
        for ax in axs:
            ax.set_xscale("log")
    # First, load the parameters / distributions for each condition in condition_list
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    first_visits = [[] for _ in range(len(condition_list))]
    xp_visits = [[] for _ in range(len(condition_list))]
    xp_transits = [[] for _ in range(len(condition_list))]
    xp_visit_durations = [[] for _ in range(len(condition_list))]
    if conserve_visit_order:
        xp_visits = [[[]] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        xp_visits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                                   "no_hole_visits", "",
                                                                   remove_censored_events=False, hard_cut=True)
        xp_transits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                                   "aggregated_raw_transits", "",
                                                                   remove_censored_events=False, hard_cut=True)

        if plot_tpnv:
            nb_of_visit_bins = list(np.logspace(0, 3, 20))  # note: the code bugs if there's a value superior to the last bin
            nb_of_bins = len(nb_of_visit_bins)
            (bin_with_values,
             average_each_bin_this_cond, error_inf_each_bin_this_cond, error_sup_each_bin_this_cond,
             nb_of_points_each_bin_this_cond) = TpNv.Tp_vs_Nv_mix_plates([], [], [], nb_of_bins, nb_of_visit_bins,
                                                              bypass_results=xp_visits[i_condition])
            current_curve_name = param.nb_to_name[condition]
            axs[i_condition].errorbar(bin_with_values, average_each_bin_this_cond,
                         [error_inf_each_bin_this_cond, error_sup_each_bin_this_cond],
                         color="black", capsize=5, capthick=2,
                         marker=param.distance_to_marker[
                             param.nb_to_distance[param.name_to_nb[current_curve_name]]],
                         markersize=10, linewidth=3)
            axs[i_condition].set_title(param.nb_to_name[condition])

        for i_plate in range(len(xp_visits[i_condition])):
            v = xp_visits[i_condition][i_plate]
            if type(v) is list and len(v) > 0:
                first_visits[i_condition].append([v[0][0], v[0][2]])
                if not conserve_visit_order:
                    xp_visit_durations[i_condition] += ana.convert_to_durations(v)
                if conserve_visit_order:
                    v = np.array(v)
                    for i_patch in np.unique(v[:,2]):
                        visits_this_patch = v[v[:,2] == i_patch]
                        for i_visit in range(len(visits_this_patch)):
                            visit = visits_this_patch[i_visit]
                            if len(xp_visit_durations[i_condition]) <= i_visit:
                                xp_visit_durations[i_condition].append(ana.convert_to_durations([visit]))
                            else:
                                xp_visit_durations[i_condition][i_visit].append(ana.convert_to_durations([visit])[0])

        transition_probability_matrices[i_condition], transition_duration_matrices[i_condition] = generate_transition_matrices(results_path, results_table,
                                                                                                    xp_visits[i_condition],
                                                                                                    xp_transits[i_condition],
                                                                                         [condition],
                                                                                                    is_recompute=True)

        if not conserve_visit_order:
            xp_visit_durations[i_condition] = [xp_visit_durations[i_condition]]  # putting it in a list for compatibility with conserve_order=TRUE case


    # To use as simulation length
    if type(xp_length) is str:
        if "length" in xp_length:  # Any expression, including "use_condition_length", will produce this
            # (sum visit and transit times, because other values like "total_tracked_time" are pre-hole filling
            _, xp_total_visit, _ = ana.results_per_condition(results, condition_list,
                                                             "total_visit_time", "", remove_censored_events=False)
            _, xp_total_transit, _ = ana.results_per_condition(results, condition_list,
                                                               "total_transit_time", "", remove_censored_events=False)
            xp_total_video_time = xp_total_visit * 3600 + xp_total_transit
            if xp_length == "use_basal_length":
                xp_total_video_time = [xp_total_video_time[0] for _ in range(len(condition_list))]
        else:
            _, xp_total_video_time, _ = ana.results_per_condition(results, condition_list, "total_tracked_time", "")

    else:
        xp_total_video_time = [xp_length for _ in range(len(condition_list))]

    avg_value_matrix = np.zeros((len(condition_list), len(condition_list)))
    error_inf_matrix = np.zeros((len(condition_list), len(condition_list)))
    error_sup_matrix = np.zeros((len(condition_list), len(condition_list)))
    all_values_lists = [[[] for _ in range(len(condition_list))] for _ in range(len(condition_list))]
    for i_line in range(len(condition_list)):
        if plot_variable_distributions:
            plt.suptitle("Condition " + str(param.nb_to_name[condition_list[i_line]]) + ", changing "+ variable_to_exchange)
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
            ax0 = fig.subplots(1, 1)
            # ax0, ax1 = fig.subplots(1, 2)
            # ax1.set_yscale("log")
            ax0.set_yscale("log")
            ax0.set_title("Visits")
            # ax1.set_title("Transits")
            # Experimental values
            xp_1st_visits = xp_visit_durations[i_line][0]
            # xp_1st_transit_duration = [v[0][0] for v in xp_visits[i_line]]
            ax0.hist(xp_1st_visits, color="k", bins=100, histtype="step", cumulative=True, density=True, label="Xp values",
                     linestyle=('dashed'), linewidth=2)
            # xp_visits = [x[i] for x in xp_visit_durations[i_line] for i in range(len(x))]
            # ax0.hist(xp_visits, color="k", bins=100, histtype="step", cumulative=True, density=True, label="Xp values",
            #          linestyle=('dashed'), linewidth=2)
            # xp_transits = ana.return_value_list(results_table, "transits", [condition_list[i_line]],
            #                                     convert_to_duration=True, remove_censored=False)
            # ax1.hist(xp_transits, color="k", bins=100, histtype="step", cumulative=True, density=True, label="Xp values",
            #          linestyle=('dashed'), linewidth=2)

        for i_col in range(len(condition_list)):
            # i_col=3
            where_to_take_each_parameter_from = {"transit_prob": i_line, "transit_times": i_line, "visit_times": i_line,
                                                 "revisit_probability": i_line, variable_to_exchange: i_col}
            transition_probability = copy.deepcopy(transition_probability_matrices[where_to_take_each_parameter_from["transit_prob"]])
            revisit_probability_matrix = transition_probability_matrices[where_to_take_each_parameter_from["revisit_probability"]]
            # Transform the matrix to match the correct revisit_probability
            # (same-patch transitions are set to be the average revisit probability of the matrix we steal from, and
            # cross-patch transitions are normalized so that the sum of a line is still 1)
            original_revisit_probability = np.mean([transition_probability[i][i] for i in range(len(transition_probability))])
            target_revisit_probability = np.mean([revisit_probability_matrix[i][i] for i in range(len(revisit_probability_matrix))])
            if original_revisit_probability != target_revisit_probability:
                for i_patch in range(len(transition_probability)):
                    for j_patch in range(len(transition_probability[i_patch])):
                        if i_patch == j_patch and transition_probability[i_patch][j_patch] not in [0, 1]:
                            transition_probability[i_patch][j_patch] = target_revisit_probability
                        elif not transition_probability[i_patch][i_patch] == 1:
                            transition_probability[i_patch][j_patch] *= (1 - target_revisit_probability) / (
                                    1 - transition_probability[i_patch][i_patch])
                        # If the REVISIT probability for this line is = 1, leave all line as is (happens only in rare cases in OD=1.25)

            transition_durations = copy.deepcopy(transition_duration_matrices[where_to_take_each_parameter_from["transit_times"]])
            visit_durations = copy.deepcopy(xp_visit_durations[where_to_take_each_parameter_from["visit_times"]])
            print("For line of ", str(condition_list[i_line]), ", total sim time is ", xp_total_video_time[i_line])
            list_of_visits = simulate_visit_list(condition_list[i_line], transition_probability, transition_durations,
                                                 visit_durations, nb_of_exp,
                                                 xp_length=xp_total_video_time[i_line],
                                                 conserve_visit_order=conserve_visit_order,
                                                 first_transit_length_and_patch= first_visits[where_to_take_each_parameter_from["transit_times"]])

            if plot_variable_distributions:
                # Keep first visit to each patch
                list_1st_visits_sim=[]
                for v in list_of_visits:
                    v = np.array(v)
                    for i_patch in np.unique(v[:,2]):
                        visits_this_patch = v[v[:,2] == i_patch]
                        list_1st_visits_sim.append(visits_this_patch[0])
                sim_1st_visit_durations = ana.convert_to_durations(list_1st_visits_sim, add_one=False)
                # sim_1st_transits = [v[0][0] for v in list_of_visits]
                ax0.hist(sim_1st_visit_durations, bins=100, histtype="step", cumulative=True,
                         density=True, color = param.name_to_color[param.nb_to_name[condition_list[i_col]]], label=param.nb_to_name[condition_list[i_col]])
               # sim_visit_durations = [ana.convert_to_durations(v) for v in list_of_visits]
               #  ax0.hist([s[i] for s in sim_visit_durations for i in range(len(s))], bins=100, histtype="step", cumulative=True,
               #           density=True, color = param.name_to_color[param.nb_to_name[condition_list[i_col]]], label=param.nb_to_name[condition_list[i_col]])
                #
                # list_of_transits = []
                # for i_xp in range(len(list_of_visits)):
                #     list_of_transits.append([0, list_of_visits[i_xp][0][0]])
                #     for i_visit in range(len(list_of_visits[i_xp]) - 1):
                #         list_of_transits.append([list_of_visits[i_xp][i_visit][1], list_of_visits[i_xp][i_visit+1][0]])
                # ax1.hist(ana.convert_to_durations(list_of_transits), bins=100, histtype="step", cumulative=True, density=True,
                #          color = param.name_to_color[param.nb_to_name[condition_list[i_col]]], label=param.nb_to_name[condition_list[i_col]])

            list_of_values = []
            if what_to_plot == "total_visit_time":
                # Total visit time / nb of visited patches
                list_of_values = [np.sum(ana.convert_to_durations(visits)) /
                                              len(np.unique(np.array(visits)[:, 2]))
                                              for visits in list_of_visits]
            if what_to_plot == "avg_visit_duration":
                # Average visit length
                list_of_values = [np.mean(ana.convert_to_durations(visits)) for visits in list_of_visits]
            if what_to_plot == "nb_of_visits_per_patch":
                # Average number of visits
                list_of_values = [len(visits) / len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits]
            if what_to_plot == "avg_nb_of_visits":
                # Average total number of visits
                list_of_values = [len(visits) for visits in list_of_visits]
            if what_to_plot == "total_xp_time":
                # Average total experimental time
                list_of_values = [visits[-1][1] for visits in list_of_visits]
            if what_to_plot == "nb_of_visited_patches":
                list_of_values = [len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits]

            # Average for the current condition
            avg_value_matrix[i_line][i_col] = np.nanmean(list_of_values)

            # Bootstrapping on the plate avg duration
            bootstrap_ci = ana.bottestrop_ci(list_of_values, 1000)
            error_inf_matrix[i_line][i_col] = avg_value_matrix[i_line][i_col] - bootstrap_ci[0]
            error_sup_matrix[i_line][i_col] = bootstrap_ci[1] - avg_value_matrix[i_line][i_col]

            # Keep in memory full list of values
            all_values_lists[i_line][i_col] = list_of_values

            if plot_1stvisit and i_line == i_col:
                plt.gcf().clf()
                print("FIRST VISITS FROM THE EXPERIMENTS")
                print(sorted(np.unique(visit_durations[0])))
                print("FIRST VISITS FOUND IN SIM")
                ordered_visits = [[]]
                for i_plate in range(len(list_of_visits)):
                    v = list_of_visits[i_plate]
                    if type(v) is list and len(v) > 0:
                        v = np.array(v)
                        for i_patch in np.unique(v[:, 2]):
                            visits_this_patch = v[v[:, 2] == i_patch]
                            for i_visit in range(len(visits_this_patch)):
                                visit = visits_this_patch[i_visit]
                                if len(ordered_visits) <= i_visit:
                                    ordered_visits.append(ana.convert_to_durations([visit]))
                                else:
                                    ordered_visits[i_visit].append(ana.convert_to_durations([visit])[0])
                print(sorted(np.unique(ordered_visits[0])))
                plt.title("First visits, visits from condition "+param.nb_to_name[condition_list[where_to_take_each_parameter_from["visit_times"]]])
                plt.hist(visit_durations[0], histtype="step", density=True, label="xp",
                         bins=[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 , 10, 20, 30, 40, 50, 60 , 70, 100, 200, 300, 500],
                         cumulative=True)
                plt.hist(ordered_visits[0], histtype="step", density=True, label="sim",
                         bins=[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 , 10, 20, 30, 40, 50, 60 , 70, 100, 200, 300, 500],
                         cumulative=True)

                random_choices = []
                for i in range(6000):
                    random_choices.append(random.choice(visit_durations[0]))
                plt.hist(random_choices, histtype="step", density=True, label="random6000",
                         bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 100, 200, 300, 500],
                         cumulative=True)

                plt.xscale("log")
                plt.yscale("log")
                plt.legend(frameon=False, )
                plt.show()


            if plot_tpnv:
                
                nb_of_visit_bins = list(np.logspace(0, 3.1, 20))  # note: the code bugs if there's a value superior to the last bin
                nb_of_bins = len(nb_of_visit_bins)
                (bin_with_values,
                 average_each_bin_this_cond, error_inf_each_bin_this_cond, error_sup_each_bin_this_cond,
                 nb_of_points_each_bin_this_cond) = TpNv.Tp_vs_Nv_mix_plates([], [], [], nb_of_bins, nb_of_visit_bins,
                                                                  bypass_results=list_of_visits)
                current_curve_name = param.nb_to_name[condition_list[i_line]]
                axs[i_line].errorbar(bin_with_values, average_each_bin_this_cond,
                             [error_inf_each_bin_this_cond, error_sup_each_bin_this_cond],
                             color=param.name_to_color[current_curve_name], capsize=5, capthick=2,
                             marker=param.distance_to_marker[
                                 param.nb_to_distance[param.name_to_nb[current_curve_name]]],
                             markersize=10, linewidth=3)


        if plot_variable_distributions:
            plt.legend(frameon=False, )
            plt.show()
        if plot_tpnv and i_line == len(condition_list)-1:
            plt.legend(frameon=False, )
            plt.title("OD = " + param.nb_to_density[condition_list[0]], fontsize=24)
            plt.xlabel("Number of visits", fontsize=20)
            plt.xscale("log")
            plt.ylabel("Total time in patch (hours)", fontsize=24)
            plt.ylim(0, 2.72)
            plt.show()
            plt.gcf().clf()

    if plot_matrix:
        exchange_script.plot_matrix(condition_list, avg_value_matrix, variable_to_exchange, what_to_plot)

    else:
        return avg_value_matrix, [error_inf_matrix, error_sup_matrix], all_values_lists


def parameter_exchange_matrix_double_sim_shenanigans(results_path, results_table, condition_list, variable_to_exchange, what_to_plot,
                                                      nb_of_exp, xp_length, plot_matrix=True, conserve_visit_order=False,
                                                      plot_variable_distributions=False):
    # First, load the parameters / distributions for each condition in condition_list
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    first_visits = [[] for _ in range(len(condition_list))]
    xp_visits = [[] for _ in range(len(condition_list))]
    xp_transits = [[] for _ in range(len(condition_list))]
    xp_visit_durations = [[] for _ in range(len(condition_list))]
    if conserve_visit_order:
        xp_visits = [[[]] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        xp_visits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                           "no_hole_visits", "",
                                                           remove_censored_events=False, hard_cut=True)
        xp_transits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                             "aggregated_raw_transits", "",
                                                             remove_censored_events=False, hard_cut=True)
        for i_plate in range(len(xp_visits[i_condition])):
            v = xp_visits[i_condition][i_plate]
            if type(v) is list and len(v) > 0:
                first_visits[i_condition].append(v[0][0])
                if not conserve_visit_order:
                    xp_visit_durations[i_condition] += ana.convert_to_durations(v)
                if conserve_visit_order:
                    v = np.array(v)
                    for i_patch in np.unique(v[:, 2]):
                        visits_this_patch = v[v[:, 2] == i_patch]
                        for i_visit in range(len(visits_this_patch)):
                            visit = visits_this_patch[i_visit]
                            if len(xp_visit_durations[i_condition]) <= i_visit:
                                xp_visit_durations[i_condition].append([visit[1] - visit[0] + 1])
                            else:
                                xp_visit_durations[i_condition][i_visit].append(visit[1] - visit[0] + 1)

        transition_probability_matrices[i_condition], transition_duration_matrices[
            i_condition] = generate_transition_matrices(results_path, results_table, xp_visits, xp_transits,
                                                        [condition])

        if not conserve_visit_order:
            xp_visit_durations[i_condition] = [
                xp_visit_durations[i_condition]]  # putting it in a list for compatibility with conserve_order=TRUE case

    #######
    # FIRST ROUND OF SIMULATIONS: JUST TO EXTRACT SIMULATED VISIT + TRANSIT LISTS
    # TO BE USED IN THE SECOND ROUND OF SIMULATION.
    # For this first round, we generate only the diagonal of the matrix, where no experimental variable is exchanged
    # between the conditions.
    sim_visits_each_cond = [[] for _ in range(len(condition_list))]
    sim_transits_each_cond = [[] for _ in range(len(condition_list))]
    for i_line in range(len(condition_list)):
        where_to_take_each_parameter_from = {"transit_prob": i_line, "transit_times": i_line, "visit_times": i_line,
                                             "revisit_probability": i_line}
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
                            1 - transition_probability[i_patch][i_patch])

        transition_durations = transition_duration_matrices[where_to_take_each_parameter_from["transit_times"]]
        visit_durations = xp_visit_durations[where_to_take_each_parameter_from["visit_times"]]
        print("For line of ", str(condition_list[i_line]), ", total sim time is ", xp_length)
        list_of_visits = simulate_visit_list(condition_list[i_line], transition_probability, transition_durations,
                                             visit_durations, nb_of_exp,
                                             xp_length=xp_length,
                                             conserve_visit_order=conserve_visit_order,
                                             first_transit_length_and_patch=first_visits[
                                                 where_to_take_each_parameter_from["transit_times"]])

        list_of_transits = []
        for i_xp in range(len(list_of_visits)):
            list_of_transits.append([0, list_of_visits[i_xp][0][0]])
            for i_visit in range(len(list_of_visits[i_xp]) - 1):
                list_of_transits.append(
                    [list_of_visits[i_xp][i_visit][1], list_of_visits[i_xp][i_visit + 1][0]])

        sim_visits_each_cond[i_line] = list_of_visits
        sim_transits_each_cond[i_line] = list_of_transits

    #######
    # SECOND ROUND OF SIMULATIONS, USING THE SIMULATED DATA AS A BASIS.
    # I DID NOT CHANGE THE VARIABLE NAMES XP_VISITS / XP_TRANSITS but they are actually the sim ones
    transition_probability_matrices = [[] for _ in range(len(condition_list))]
    transition_duration_matrices = [[] for _ in range(len(condition_list))]
    first_visits = [[] for _ in range(len(condition_list))]
    xp_visits = [[] for _ in range(len(condition_list))]
    xp_transits = [[] for _ in range(len(condition_list))]
    xp_visit_durations = [[] for _ in range(len(condition_list))]
    if conserve_visit_order:
        xp_visits = [[[]] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        xp_visits[i_condition] = sim_visits_each_cond[i_condition]
        xp_transits[i_condition] = sim_transits_each_cond[i_condition]
        for i_plate in range(len(xp_visits[i_condition])):
            v = xp_visits[i_condition][i_plate]
            if type(v) is list and len(v) > 0:
                first_visits[i_condition].append(v[0][0])
                if not conserve_visit_order:
                    xp_visit_durations[i_condition] += ana.convert_to_durations(v)
                if conserve_visit_order:
                    v = np.array(v)
                    for i_patch in np.unique(v[:, 2]):
                        visits_this_patch = v[v[:, 2] == i_patch]
                        for i_visit in range(len(visits_this_patch)):
                            visit = visits_this_patch[i_visit]
                            if len(xp_visit_durations[i_condition]) <= i_visit:
                                xp_visit_durations[i_condition].append([visit[1] - visit[0] + 1])
                            else:
                                xp_visit_durations[i_condition][i_visit].append(visit[1] - visit[0] + 1)

        transition_probability_matrices[i_condition], transition_duration_matrices[
            i_condition] = generate_transition_matrices(results_path, results_table, xp_visits, xp_transits,
                                                        [condition])

        if not conserve_visit_order:
            xp_visit_durations[i_condition] = [
                xp_visit_durations[
                    i_condition]]  # putting it in a list for compatibility with conserve_order=TRUE case


    avg_value_matrix = np.zeros((len(condition_list), len(condition_list)))
    error_inf_matrix = np.zeros((len(condition_list), len(condition_list)))
    error_sup_matrix = np.zeros((len(condition_list), len(condition_list)))
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
                                1 - transition_probability[i_patch][i_patch])

            transition_durations = transition_duration_matrices[where_to_take_each_parameter_from["transit_times"]]
            visit_durations = xp_visit_durations[where_to_take_each_parameter_from["visit_times"]]
            print("For line of ", str(condition_list[i_line]), ", total sim time is ", xp_length)
            list_of_visits = simulate_visit_list(condition_list[i_line], transition_probability, transition_durations,
                                                 visit_durations, nb_of_exp,
                                                 xp_length=xp_length,
                                                 conserve_visit_order=conserve_visit_order,
                                                 first_transit_length_and_patch= first_visits[where_to_take_each_parameter_from["transit_times"]])

            list_of_values = []
            if what_to_plot == "total_visit_time":
                # Total visit time / nb of visited patches
                list_of_values = [np.sum(ana.convert_to_durations(visits)) /
                                              len(np.unique(np.array(visits)[:, 2]))
                                              for visits in list_of_visits]
            if what_to_plot == "avg_visit_duration":
                # Average visit length
                list_of_values = [np.mean(ana.convert_to_durations(visits)) for visits in list_of_visits]
            if what_to_plot == "nb_of_visits_per_patch":
                # Average number of visits
                list_of_values = [len(visits) / len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits]
            if what_to_plot == "avg_nb_of_visits":
                # Average total number of visits
                list_of_values = [len(visits) for visits in list_of_visits]
            if what_to_plot == "total_xp_time":
                # Average total experimental time
                list_of_values = [visits[-1][1] for visits in list_of_visits]
            if what_to_plot == "nb_of_visited_patches":
                list_of_values = [len(np.unique(np.array(visits)[:, 2])) for visits in list_of_visits]

            # Average for the current condition
            avg_value_matrix[i_line][i_col] = np.nanmean(list_of_values)

            # Bootstrapping on the plate avg duration
            bootstrap_ci = ana.bottestrop_ci(list_of_values, 1000)
            error_inf_matrix[i_line][i_col] = avg_value_matrix[i_line][i_col] - bootstrap_ci[0]
            error_sup_matrix[i_line][i_col] = bootstrap_ci[1] - avg_value_matrix[i_line][i_col]

    if plot_matrix:
        exchange_script.plot_matrix(condition_list, avg_value_matrix, variable_to_exchange, what_to_plot)

    else:
        # COMPUTE THE FAKE EXPERIMENTAL VALUES (which are actually 1st round simulation results
        # For now just total time per patch because i'm lazy
        avg_total_time=[0 for _ in range(len(condition_list))]
        error_inf_total_time=[[] for _ in range(len(condition_list))]
        error_sup_total_time=[[] for _ in range(len(condition_list))]
        for i_condition in range(len(condition_list)):
            current_visits = xp_visits[i_condition]
            plates=[]
            for i_plate in range(len(current_visits)):
                if type(current_visits[i_plate]) is list and len(current_visits[i_plate]) > 0:
                    total_duration = np.sum(ana.convert_to_durations(current_visits[i_plate]))
                    nb_of_patches = len(np.unique(np.array(current_visits[i_plate])[:, 2]))
                    plates.append(total_duration/nb_of_patches)
            avg_total_time[i_condition] = np.mean(plates) / 3600
            # Bootstrapping on the plate avg duration
            bootstrap_ci = ana.bottestrop_ci(plates, 1000)
            error_inf_total_time[i_condition] = avg_total_time[i_condition] - bootstrap_ci[0]/ 3600
            error_sup_total_time[i_condition] = bootstrap_ci[1]/ 3600 - avg_total_time[i_condition]

        return avg_total_time, [error_inf_total_time, error_sup_total_time], avg_value_matrix, [error_inf_matrix, error_sup_matrix]


def save_visit_distrib_each_rank(results_path, results_table, condition_name_list):
    """
    Function that will plot histogram of duration distribution for 1st, 2nd, etc visits to food patches.
    """
    condition_list = [param.name_to_nb[cond] for cond in condition_name_list]
    xp_visit_durations = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        folder_list = fd.return_folders_condition_list(results_table["folder"], condition)
        for i_folder in range(len(folder_list)):
            current_data = results[results["folder"] == folder_list[i_folder]].reset_index()
            visit_list = fd.load_list(current_data, "visits_to_uncensored_patches")
            if type(visit_list) is list and len(visit_list) > 0:
                v = np.array(visit_list)
                for i_patch in np.unique(v[:,2]):
                    visits_this_patch = v[v[:,2] == i_patch]
                    for i_visit in range(len(visits_this_patch)):
                        visit = visits_this_patch[i_visit]
                        if len(xp_visit_durations[i_condition]) <= i_visit:
                            xp_visit_durations[i_condition].append(ana.convert_to_durations([visit]))
                        else:
                            xp_visit_durations[i_condition][i_visit].append(ana.convert_to_durations([visit])[0])

    table_condition = []
    table_visit_rank = []
    table_visit_list = []
    for i_condition, condition in enumerate(condition_list):
        for i_visits in range(len(xp_visit_durations[i_condition])):
            current_visits = xp_visit_durations[i_condition][i_visits]
            table_condition.append(param.nb_to_name[condition])
            table_visit_rank.append(i_visits)
            table_visit_list.append(current_visits)

    output_datatable = pd.DataFrame()
    output_datatable["condition"] = table_condition
    output_datatable["visit_rank"] = table_visit_rank
    output_datatable["visit_list"] = table_visit_list

    output_datatable.to_csv(results_path + "visit_durations_each_rank.csv")


def plot_visit_distrib_each_rank(results_table, condition_name_list, min_nb_points=600, plot_or_save="plot"):
    """
    Function that will plot histogram of duration distribution for 1st, 2nd, etc visits to food patches.
    """
    condition_list = [param.name_to_nb[cond] for cond in condition_name_list]
    visits_each_cond = [[[]] for _ in range(len(condition_list))]
    durations_by_rank = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        visits_each_cond[i_condition] = ana.results_per_condition(results_table, [condition],
                                                                   "no_hole_visits", "",
                                                                   remove_censored_events=False, hard_cut=True)
        for i_plate in range(len(visits_each_cond[i_condition])):
            v = visits_each_cond[i_condition][i_plate]
            if type(v) is list and len(v) > 0:
                v = np.array(v)
                for i_patch in np.unique(v[:,2]):
                    visits_this_patch = v[v[:,2] == i_patch]
                    for i_visit in range(len(visits_this_patch)):
                        visit = visits_this_patch[i_visit]
                        if len(durations_by_rank[i_condition]) <= i_visit:  # if the list is not long enough create new item
                            durations_by_rank[i_condition].append(ana.convert_to_durations([visit]))
                        else:  # else add to existing list
                            durations_by_rank[i_condition][i_visit].append(ana.convert_to_durations([visit])[0])
    for i_condition, condition in enumerate(condition_list):
        colors = plt.cm.turbo(np.linspace(0, 1, 30))
        dynamic_binning_visits = []
        bin_inf = 0
        bin_sup = 0
        for i_visits in range(len(durations_by_rank[i_condition])):
            dynamic_binning_visits += durations_by_rank[i_condition][i_visits]
            if len(dynamic_binning_visits) > min_nb_points:
                if bin_inf == bin_sup:
                    label = str(bin_inf)
                else:
                    label = str(bin_inf) + "-" + str(bin_sup)
                n, bins, patches = plt.hist(np.array(dynamic_binning_visits)/60, bins=16, histtype="step",
                         linewidth=3.4, edgecolor=colors[bin_inf*2], label=label)
                # patches[0].set_xy(patches[0].get_xy()[:-1]) # just delete the last point
                bin_inf = i_visits + 1
                bin_sup = i_visits + 1
                dynamic_binning_visits = []
            else:
                bin_sup = i_visits + 1
        # If the last bin was never plotted (not enough points)
        if dynamic_binning_visits:
            plt.hist(np.array(dynamic_binning_visits) / 60, bins=16, histtype="step",
                     linewidth=3.4, edgecolor=colors[bin_inf + 4], label="etc")

    plt.semilogy()
    x_min, x_max = plt.gca().get_xlim()
    plt.xlim(0.05, x_max)
    plt.xlabel("Visit duration (min)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(frameon=False, draggable=True, fontsize=16)
    plt.show()


def behavior_vs_geometry(results_path, results_table, baseline_condition, nb_of_exp, xp_length, plot_distrib=False, visit_order=False,
                         variable_to_plot="total_visit_time", double_sim_shenanigans=False, plot_tpnv=False, mix_xp_plates=False,
                         plot_1stvisit=False):
    """
    Function that will plot three total time curves.
    The first one will be the effect of distance in our substitution model, with the experimental parameters that match
    those computed in our actual conditions.
    The third one will be the effect of VISIT TIME only (so baseline_condition, but with the visit times from other
    distances).
    The third one will be the effect of everything EXCEPT visit times (so visit times from baseline_condition, and the
    rest from other distances).
    All of those will be computed with a constant food OD, the same as baseline_condition.

    @param results_path:
    @param results_table:
    @param baseline_condition: a string corresponding to a condition (eg "close 0")
    @param nb_of_exp:
    @param xp_length: can be int (used as is as simulation length), or equal to "use_basal_length", and then all
                      the points are computed using the length from the basal condition
    @return:
    """
    baseline_condition_nb = param.name_to_nb[baseline_condition]
    baseline_density = param.nb_to_density[baseline_condition_nb]
    # condition_names_this_density = ["superfar "+baseline_density, "far "+baseline_density, "med "+baseline_density, "close "+baseline_density]
    condition_names_this_density = ["close "+baseline_density, "med "+baseline_density, "far "+baseline_density, "superfar "+baseline_density]
    condition_list_this_density = [param.name_to_nb[cond] for cond in condition_names_this_density]

    # Experimental values that we will compare to the model later
    if variable_to_plot == "total_visit_time":
        if not mix_xp_plates:
            # Compute the experimental values of total time in patch
            average_per_condition, full_list_values, errorbars = plots.plot_selected_data(results_table, "",
                                                                                                condition_list_this_density,
                                                                                                "total_visit_time",
                                                                                                divided_by="nb_of_visited_patches",
                                                                                                is_plot=False, show_stats=False,
                                                                                                remove_censored_events=False,
                                                                                                hard_cut=True)
        else:
            average_per_condition = [0 for _ in range(len(condition_list_this_density))]
            errors_inf = [0 for _ in range(len(condition_list_this_density))]
            errors_sup = [0 for _ in range(len(condition_list_this_density))]
            full_list_values = [[] for _ in range(len(condition_list_this_density))]
            for i_condition, condition in enumerate(condition_list_this_density):
                visits = ana.results_per_condition(results_table, [condition],
                                                       "no_hole_visits", "",
                                                       remove_censored_events=False, hard_cut=True)
                for i_plate in range(len(visits)):
                    if type(visits[i_plate]) != float and len(visits[i_plate]) > 0:
                        current_visits = np.array(visits[i_plate])
                        visited_patches = np.unique(current_visits[:, 2])
                        for patch in visited_patches:
                            visits_to_patch = current_visits[current_visits[:, 2] == patch]
                            full_list_values[i_condition].append(np.sum(ana.convert_to_durations(list(visits_to_patch)))/60)

                # Average for the current condition
                average_per_condition[i_condition] = np.mean(full_list_values[i_condition])

                # Bootstrapping on the plate avg duration
                bootstrap_ci = ana.bottestrop_ci(full_list_values[i_condition], 1000)
                errors_inf[i_condition] = average_per_condition[i_condition] - bootstrap_ci[0]
                errors_sup[i_condition] = bootstrap_ci[1] - average_per_condition[i_condition]

            errorbars = [errors_inf, errors_sup]
    if variable_to_plot == "avg_visit_duration":
        # Compute the experimental values of average visit duration
        average_per_condition, full_list_values, errorbars = plots.plot_selected_data(results_table, "",
                                                                                            condition_list_this_density,
                                                                                            "total_visit_time",
                                                                                            divided_by="nb_of_visits",
                                                                                            is_plot=False, show_stats=False,
                                                                                            remove_censored_events=False,
                                                                                            hard_cut=True)
    if variable_to_plot == "nb_of_visited_patches":
        average_per_condition, full_list_values, errorbars = plots.plot_selected_data(results_table, "",
                                                                                            condition_list_this_density,
                                                                                            "nb_of_visited_patches",
                                                                                            divided_by="",
                                                                                            is_plot=False, show_stats=False,
                                                                                            remove_censored_events=False,
                                                                                            hard_cut=True)
    if variable_to_plot == "nb_of_visits_per_patch":
        average_per_condition, full_list_values, errorbars = plots.plot_selected_data(results_table, "",
                                                                                            condition_list_this_density,
                                                                                            "nb_of_visits",
                                                                                            divided_by="nb_of_visited_patches",
                                                                                            is_plot=False, show_stats=False,
                                                                                            remove_censored_events=False,
                                                                                            hard_cut=True)
    plt.clf()  # Clear the plot because the previous function plots stuff even when it does not show them

    # Compute the matrices for the simulated data
    if not double_sim_shenanigans:
        visit_exchange_matrix, visit_errors, visits_all_values = parameter_exchange_matrix(results_path, results_table, condition_list_this_density,
                                                                         "visit_times", variable_to_plot,
                                                                                           nb_of_exp, xp_length, False, conserve_visit_order=visit_order,
                                                                                           plot_variable_distributions=plot_distrib, plot_tpnv=plot_tpnv,
                                                                                           plot_1stvisit=plot_1stvisit)

    else:
        average_per_condition, errorbars, visit_exchange_matrix, visit_errors = parameter_exchange_matrix_double_sim_shenanigans(
                                                          results_path, results_table, condition_list_this_density,
                                                          "visit_times", variable_to_plot,
                                                          nb_of_exp, xp_length, False,
                                                          conserve_visit_order=visit_order,
                                                          plot_variable_distributions=plot_distrib)
        average_per_condition, errorbars, probability_exchange_matrix, prob_errors = parameter_exchange_matrix_double_sim_shenanigans(
                                                                results_path, results_table, condition_list_this_density,
                                                                "revisit_probability", variable_to_plot,
                                                                nb_of_exp, xp_length, False,
                                                                conserve_visit_order=visit_order,
                                                                plot_variable_distributions=plot_distrib)

    # Plot the experimental data
    if double_sim_shenanigans:
        label = "First round of simulated data"
    else:
        label = "Experimental values"
    plt.gcf().set_size_inches(7.5, 7.5)
    plt.scatter(range(len(average_per_condition)), average_per_condition, marker="x", color="sienna", s=100, linewidth=3, label=label, zorder=12)
    plt.errorbar(range(len(average_per_condition)), average_per_condition, errorbars, color="sienna", capsize=5, linewidth=0, elinewidth=3, markeredgewidth=3, zorder=12)

    # Plot simulated data
    title = "?"
    factor = 1
    if variable_to_plot == "total_visit_time":
        title="Total time per patch (minutes)"
        factor=60
    if variable_to_plot == "nb_of_visited_patches":
        title = "Nb visited patches"
        factor = 1
    if variable_to_plot == "avg_visit_duration":
        title = "Avg visit duration (minutes)"
        factor=60
    if variable_to_plot == "nb_of_visits_per_patch":
        title = "Nb of visits in each patch"
        factor = 1
    plt.ylabel(title, fontsize=20)
    plt.title("OD = "+baseline_density, fontsize=24)
    plt.errorbar([i for i in range(len(visit_exchange_matrix))],
                 [visit_exchange_matrix[i][i]/factor for i in range(len(visit_exchange_matrix))],
                 [[visit_errors[0][i][i]/factor for i in range(len(visit_errors[0]))],
                  [visit_errors[1][i][i]/factor for i in range(len(visit_errors[1]))]],
                 color=(0.1, 0.1, 0.1), capsize=0, linewidth=4, elinewidth=0, markeredgewidth=3, zorder=10,
                 marker=".", markersize=12,
                 label="All effects\n(in-patch + out-of-patch)")

    # In order to plot the effect of changing parameters with baseline as an actual baseline, need to find
    # at what index it sits in the matrices
    baseline_index = np.where(np.array(condition_names_this_density) == baseline_condition)[0][0]

    # Old blue curve
    # plt.errorbar([i+0.04 for i in range(len(visit_exchange_matrix))],
    #              [probability_exchange_matrix[baseline_index][i]/factor for i in range(len(probability_exchange_matrix))],
    #              np.array(prob_errors)[:,baseline_index]/factor, color="cornflowerblue",
    #              capsize=5, linewidth=4, elinewidth=3, markeredgewidth=3, zorder=10, marker=".", markersize=12,
    #              label="Only revisit probability \n changes with distance")


    # AVERAGE EFFECT PLOT: plots the experimental values, then the model without exchanging variables in black,
    # Then blue area whose height is the effect of changing everything except visit times, minus first condition
    # Then yellow area whose height is the effect of changing only visit time, minus first condition
    all_effects = np.array([visit_exchange_matrix[i][i] / factor for i in range(len(visit_exchange_matrix))])
    basal_all_effects = np.array([visit_exchange_matrix[baseline_index][baseline_index] / factor for _ in range(len(visit_exchange_matrix))])
    out_of_patch_effects = np.array([visit_exchange_matrix[i][baseline_index]/factor for i in range(len(visit_exchange_matrix))])
    in_patch_effects = np.array([visit_exchange_matrix[baseline_index][i]/factor for i in range(len(visit_exchange_matrix))])
    print("Baseline condition: ", baseline_condition)
    print("Black curve: ", all_effects)
    print("Out of patch: ", out_of_patch_effects)
    print("In patch: ", in_patch_effects)
    mpl.rcParams['hatch.linewidth'] = 3  # previous svg hatch linewidth
    plt.fill_between([i for i in range(len(visit_exchange_matrix))],
                     all_effects - (in_patch_effects - basal_all_effects),
                     all_effects,
                     color="goldenrod", alpha=0.3, edgecolor="goldenrod", zorder=10, linewidth=3, hatch="|",
                     label="In-patch effects only")
    plt.fill_between([i for i in range(len(visit_exchange_matrix))],
                 out_of_patch_effects,
                 basal_all_effects,
                 color="lightsteelblue", alpha=0.3, edgecolor="lightsteelblue", zorder=10, linewidth=4, hatch='//',
                 label="Out-of-patch effects only")
    plt.fill_between([i for i in range(len(visit_exchange_matrix))],
                     all_effects - (in_patch_effects - basal_all_effects),
                     out_of_patch_effects,
                     where= all_effects - (in_patch_effects - basal_all_effects) > out_of_patch_effects,
                     color="tomato", alpha=0.3, edgecolor="tomato", zorder=10, linewidth=0, hatch="",
                     interpolate=True,
                     label="Interaction")
    plt.legend(frameon=False, fontsize=18, draggable=True)
    ax = plt.gca()
    ax = custom_legends.distance_x_labels(condition_list_this_density, ax)

    plt.gcf().set_size_inches(6, 7)
    plt.ylim(0, 180)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    color_first_density = param.name_to_color[param.nb_to_density[baseline_condition_nb]]
    ax.spines['bottom'].set(color=color_first_density, linewidth=2.5)
    ax.spines['left'].set(color=color_first_density, linewidth=2.5)
    ax.spines['top'].set(color=color_first_density, linewidth=2.5)
    ax.spines['right'].set(color=color_first_density, linewidth=2.5)

    plt.tight_layout(pad=1)
    plt.show()


    # PROPORTION OF EACH EFFECT PLOT
    # Same basis as previous one but this time we want to plot the proportion of black which is blue or yellow
    # In order to do so, we take the values (all effects, out of patch, in patch) of all simulations,
    # and bootstrap the proportion of each effect
    all_effects_all = [(np.array(visits_all_values[i][i]) -
                        np.array(visits_all_values[baseline_index][baseline_index]))
                       / factor for i in range(len(visit_exchange_matrix))]
    out_of_patch_effects_all = [(np.array(visits_all_values[i][baseline_index]) -
                                np.array(visits_all_values[baseline_index][baseline_index]))
                                / factor for i in range(len(visit_exchange_matrix))]
    in_patch_effects_all = [(np.array(visits_all_values[baseline_index][i]) -
                           np.array(visits_all_values[baseline_index][baseline_index]))
                            / factor for i in range(len(visit_exchange_matrix))]
    avg_out = [0 for _ in range(len(all_effects))]
    errors_inf_out = [0 for _ in range(len(all_effects))]
    errors_sup_out = [0 for _ in range(len(all_effects))]
    avg_in = [0 for _ in range(len(all_effects))]
    errors_inf_in = [0 for _ in range(len(all_effects))]
    errors_sup_in = [0 for _ in range(len(all_effects))]
    avg_inter = [0 for _ in range(len(all_effects))]
    errors_inf_inter = [0 for _ in range(len(all_effects))]
    errors_sup_inter = [0 for _ in range(len(all_effects))]
    for i_cond in range(len(all_effects)):
        # First, average 40 random points picked from data 1000 times
        # This returns a list of random samples from data (nb_resample lines, with each as many elements as data)
        random_all = np.apply_along_axis(ana.random_average, 1, np.array([all_effects_all[i_cond]] * 1000), nb_of_points_to_pick=100)
        random_out = np.apply_along_axis(ana.random_average, 1, np.array([out_of_patch_effects_all[i_cond]] * 1000), nb_of_points_to_pick=100)
        random_in = np.apply_along_axis(ana.random_average, 1, np.array([in_patch_effects_all[i_cond]] * 1000), nb_of_points_to_pick=100)
        bootstrapped_out = np.divide(random_out, random_all)
        bootstrapped_in = np.divide(random_in, random_all)
        bootstrapped_interaction = np.divide(random_all - random_out - random_in, random_all)
        avg_out[i_cond] = np.mean(bootstrapped_out)
        avg_in[i_cond] = np.mean(bootstrapped_in)
        avg_inter[i_cond] = np.mean(bootstrapped_interaction)
        errors_inf_out[i_cond] = avg_out[i_cond] - np.percentile(bootstrapped_out, 2.5)
        errors_sup_out[i_cond] = np.percentile(bootstrapped_out, 97.5) - avg_out[i_cond]
        errors_inf_in[i_cond] = avg_in[i_cond] - np.percentile(bootstrapped_in, 2.5)
        errors_sup_in[i_cond] = np.percentile(bootstrapped_in, 97.5) - avg_in[i_cond]
        errors_inf_inter[i_cond] = avg_inter[i_cond] - np.percentile(bootstrapped_interaction, 2.5)
        errors_sup_inter[i_cond] = np.percentile(bootstrapped_interaction, 97.5) - avg_inter[i_cond]

    plt.errorbar([i for i in range(len(visit_exchange_matrix))],
                 avg_in, yerr=[errors_inf_in, errors_sup_in],
                 color="goldenrod", capsize=3, linewidth=4, elinewidth=3, markeredgewidth=3, zorder=10,
                 marker=".", markersize=12,
                 label="Proportion of in-patch")
    plt.errorbar([i for i in range(len(visit_exchange_matrix))],
                 avg_out, yerr=[errors_inf_out, errors_sup_out],
                 color="lightsteelblue", capsize=3, linewidth=4, elinewidth=3, markeredgewidth=3, zorder=10,
                 marker=".", markersize=12,
                 label="Proportion of out-patch")
    plt.errorbar([i for i in range(len(visit_exchange_matrix))],
                 avg_inter, yerr=[errors_inf_inter, errors_sup_inter],
                 color="tomato", capsize=3, linewidth=4, elinewidth=3, markeredgewidth=3, zorder=10,
                 marker=".", markersize=12,
                 label="Proportion of interaction")


    plt.legend(frameon=False, fontsize=18, draggable=True)
    ax = plt.gca()
    ax = custom_legends.distance_x_labels(condition_list_this_density, ax)

    plt.gcf().set_size_inches(6, 7)
    # plt.ylim(0, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    color_first_density = param.name_to_color[param.nb_to_density[baseline_condition_nb]]
    ax.spines['bottom'].set(color=color_first_density, linewidth=2.5)
    ax.spines['left'].set(color=color_first_density, linewidth=2.5)
    ax.spines['top'].set(color=color_first_density, linewidth=2.5)
    ax.spines['right'].set(color=color_first_density, linewidth=2.5)

    plt.tight_layout(pad=1)
    plt.show()


def plot_simulated_timeline(results_path, results_table, condition_transits,
                            condition_revisit_probability, condition_visit_duration,
                            nb_of_exp, xp_length, visit_order):
    # Generate transition time + probability matrix
    transition_probability, transition_duration = generate_transition_matrices(results_path, results_table,
                                                                      [condition_transits])
    # Generate transition probability matrix (the one we'll take revisit prob from)
    revisit_probability, _ = generate_transition_matrices(results_path, results_table,
                                                                      [condition_revisit_probability])
    # Get visit list from right condition
    if not visit_order:
        current_visits = [ana.return_value_list(results_table, "visits", [condition_visit_duration],
                                            convert_to_duration=True, conserve_visit_order=False)]
    else:
        current_visits = ana.return_value_list(results_table, "visits", [condition_visit_duration],
                                            convert_to_duration=True, conserve_visit_order=True)
    # current_visits = ana.return_value_list(results_table, "visits", [condition_visit_duration],
    #                                     convert_to_duration=True, conserve_visit_order=True)
    first_visits = ana.return_value_list(results_table, "visits", [condition_transits],
                                         convert_to_duration=False,
                                         conserve_visit_order=False,
                                         remove_censored=False, only_first_of_plate=True)
    first_visits = [first_visits[i][0] for i in range(len(first_visits))]

    # Transform the matrix to match the correct revisit_probability
    # (same-patch transitions are set to be the average revisit probability of the matrix we steal from, and
    # cross-patch transitions are normalized so that the sum of a line is still 1)
    original_revisit_probability = np.mean([transition_probability[i][i] for i in range(len(transition_probability))])
    target_revisit_probability = np.mean([revisit_probability[i][i] for i in range(len(revisit_probability))])
    for i_patch in range(len(transition_probability)):
        for j_patch in range(len(transition_probability[i_patch])):
            if i_patch == j_patch and transition_probability[i_patch][j_patch] != 0:
                transition_probability[i_patch][j_patch] = target_revisit_probability
            else:
                transition_probability[i_patch][j_patch] *= (1 - target_revisit_probability) / (
                        1 - original_revisit_probability)

    # SIMULATION BABY
    list_of_visits = simulate_visit_list(condition_transits, transition_probability, transition_duration,
                                         current_visits, nb_of_exp,
                                         xp_length=xp_length,
                                         conserve_visit_order=visit_order, first_transit_length_and_patch= first_visits)

    # TIME TO DRAWWWWWWWWWWW
    current_timeline = np.zeros((200*nb_of_exp, xp_length))
    current_timeline[:] = np.nan
    for i_exp in range(nb_of_exp):
        print(i_exp)
        current_visits = list_of_visits[i_exp]
        # All simulations start with a transit, so always plot 1st transit
        # And then visit + transit pairs
        first_visit_start = current_visits[0][0]
        current_timeline[200 * i_exp:200 * (i_exp + 1), 0:int(np.rint(first_visit_start))] = -1
        i_visit = 0
        i_time = 0
        while i_visit < len(current_visits):
            visit = current_visits[i_visit]
            current_timeline[200 * i_exp:200 * (i_exp + 1), int(np.rint(visit[0])):int(np.rint(visit[1]+1))] = visit[2]
            i_time += visit[1] - visit[0] + 1
            i_visit += 1
            if i_time < xp_length:  # if this visit was not the last ibento
                # if there is a next visit, transit is between current visit end and next visit start
                if i_visit < len(current_visits) - 1:
                    next_visit = current_visits[i_visit + 1]
                    current_timeline[200 * i_exp:200 * (i_exp + 1), int(np.rint(visit[1])):int(np.rint(next_visit[0] + 1))] = -1
                # if there is no next visit, transit is between current visit end and xp end
                else:
                    current_timeline[200 * i_exp:200 * (i_exp + 1), int(np.rint(visit[1])):xp_length] = -1

    # Set the NaN's to white
    masked_array = np.ma.array(current_timeline, mask=np.isnan(current_timeline))
    cmap = plt.get_cmap('rainbow')
    cmap.set_bad('white', 1.)
    # Then plot it
    plt.imshow(masked_array, cmap=cmap)
    plt.colorbar()
    plt.show()


def plot_transition_matrices(results_path, results_table):
    condition_list = [0, 1, 2, 14, 4, 5, 6, 15, 12, 8, 13, 16, 17, 18, 19, 20]
    xp_visits = [[] for _ in range(len(condition_list))]
    xp_transits = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        xp_visits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                           "no_hole_visits", "",
                                                           remove_censored_events=False, hard_cut=True)
        xp_transits[i_condition] = ana.results_per_condition(results_table, [condition],
                                                             "aggregated_raw_transits", "",
                                                             remove_censored_events=False, hard_cut=True)
        generate_transition_matrices(results_path, results_table, xp_visits[i_condition], xp_transits[i_condition], [condition],
                                     plot_everything=True, plot_transition_matrix=False, is_recompute=True)



if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate()
    results = pd.read_csv(path + "clean_results.csv")
    # trajectories = pd.read_csv(path + "clean_trajectories.csv")
    full_list_of_folders = list(results["folder"])
    #if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
    #    full_list_of_folders.remove(
    #        "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")

    # MAIN TEXT
    # Graphs
    # plot_transition_matrix_graph(path, results, full_list_of_folders, [14], probability_or_time="probability")
    # plot_transition_matrix_graph(path, results, full_list_of_folders, [14], probability_or_time="time")

    behavior_vs_geometry(path, results, "close 0.2", 4000, 25000, variable_to_plot="total_visit_time", visit_order=True, plot_tpnv=False, mix_xp_plates=False)
    behavior_vs_geometry(path, results, "close 0.5", 4000, 25000, variable_to_plot="total_visit_time", visit_order=True, plot_tpnv=False, mix_xp_plates=False)
    behavior_vs_geometry(path, results, "close 1.25", 4000, 25000, variable_to_plot="total_visit_time", visit_order=True, plot_tpnv=False, mix_xp_plates=False)

    # SUPPS
    # Transition matrices
    # plot_transition_matrices(path, results)
    # Patch numbers
    # show_patch_numbers(path, full_list_of_folders)



    # save_visit_distrib_each_rank(path, results, ["close 0", "med 0", "far 0", "superfar 0",
    #                                                               "close 0.2", "med 0.2", "far 0.2", "superfar 0.2",
    #                                                               "close 0.5", "med 0.5", "far 0.5", "superfar 0.5",
    #                                                               "close 1.25", "med 1.25", "far 1.25", "superfar 1.25"])


    # behavior_vs_geometry(path, results, "close 0.5", 2000, 25000, visit_order=False, plot_tpnv=True)
    # behavior_vs_geometry(path, results, "close 0", 2000, 25000, visit_order=False)
    # behavior_vs_geometry(path, results, "close 0.2", 2000, 25000, visit_order=False, plot_tpnv=True)
    # behavior_vs_geometry(path, results, "close 1.25", 2000, 25000, visit_order=False, plot_tpnv=True

    # behavior_vs_geometry(path, results, "close 0", 2000, "use_condition_length")
    # behavior_vs_geometry(path, results, "close 0.2", 2000, "use_condition_length")
    # behavior_vs_geometry(path, results, "close 0.5", 2000, "use_condition_length")
    # behavior_vs_geometry(path, results, "close 1.25", 2000, "use_condition_length")


    # behavior_vs_geometry(path, results, "close 0.5", 1000, 25000,
    #                      plot_distrib=False, variable_to_plot="total_visit_time", visit_order=False,
    #                      double_sim_shenanigans=True)
    # behavior_vs_geometry(path, results, "close 0.5", 1000, 25000,
    #                      plot_distrib=False, variable_to_plot="total_visit_time", visit_order=True,
    #                      double_sim_shenanigans=True)


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
