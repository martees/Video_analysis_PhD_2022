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


def generate_transition_matrices(condition_list, plot_everything=False, plot_transition_matrix=False):
    # First, load the visits, number of patches and patch positions for each condition in condition_list
    visit_list = [[] for _ in range(len(condition_list))]
    patch_positions = [[] for _ in range(len(condition_list))]
    nb_of_patches_list = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        visit_list[i_condition] = np.array(ana.return_value_list(results, "visits", [condition], convert_to_duration=False))
        patch_positions[i_condition] = param.distance_to_xy[param.nb_to_distance[condition]]
        nb_of_patches_list[i_condition] = len(patch_positions[i_condition])

    # Then, for each condition, create the matrices and save them
    for i_condition, condition in enumerate(condition_list):
        print("Condition ", i_condition, " / ", len(condition_list))
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
            distance_matrix[i_patch] = [np.sqrt((curr_patches[i_patch][0] - curr_patches[i][0])**2 + (curr_patches[i_patch][1] - curr_patches[i][1])**2) for i in range(nb_of_patches)]

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
            plt.title("Transition probability "+param.nb_to_name[condition]+" , vmax=0.1")
            plt.colorbar()
            plt.show()

        return transition_probability_matrix, transition_times_matrix


def show_patch_numbers(condition_list):
    # First, load patch positions
    patch_positions = heatmap_script.idealized_patch_centers_mm(1847)
    for i_condition, condition in enumerate(condition_list):
        current_patch_positions = patch_positions[i_condition]
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


def simulate_visit_list(condition, nb_of_exp, xp_length=30000):
    print("Generating transition matrix...")
    transition_probability, transition_durations = generate_transition_matrices([condition])

    print("Simulating...")
    visit_list = []
    for i_exp in range(nb_of_exp):
        print("")


# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=False)
results = pd.read_csv(path + "clean_results.csv")
trajectories = pd.read_csv(path + "clean_trajectories.csv")
full_list_of_folders = list(results["folder"])
if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
    full_list_of_folders.remove(
        "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")

list_of_conditions = list(param.nb_to_name.keys())
show_patch_numbers(list_of_conditions)



