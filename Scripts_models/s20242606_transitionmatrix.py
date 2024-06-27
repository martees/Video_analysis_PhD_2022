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
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt
from Scripts_analysis import s20240606_distancetoedgeanalysis as script_distance
from Parameters import parameters as param
from Parameters import patch_coordinates
import find_data as fd
import analysis as ana
import ReferencePoints


# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=False)
results = pd.read_csv(path + "clean_results.csv")
trajectories = pd.read_csv(path + "clean_trajectories.csv")
full_list_of_folders = list(results["folder"])
if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
    full_list_of_folders.remove(
        "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")

condition_list = list(param.nb_to_name.keys())

# First, load the visits, transits and number of patches for each condition in condition_list
#transit_list = [[] for _ in range(len(condition_list))]
visit_list = [[] for _ in range(len(condition_list))]
nb_of_patches_list = [[] for _ in range(len(condition_list))]
for i_condition, condition in enumerate(condition_list):
    #transit_list[i_condition] = ana.return_value_list(results, "transits", [condition])
    visit_list[i_condition] = np.array(ana.return_value_list(results, "visits", [condition], convert_to_duration=False))
    nb_of_patches_list[i_condition] = len(param.distance_to_xy[param.nb_to_distance[condition]])

# Then, for each condition, create the matrices and save them
for i_condition, condition in enumerate(condition_list):
    nb_of_patches = nb_of_patches_list[i_condition]
    current_visits = visit_list[i_condition]
    transition_probability_matrix = np.zeros((nb_of_patches, nb_of_patches))
    transition_times_matrix = [[[] for _ in range(nb_of_patches)] for _ in range(nb_of_patches)]
    transition_times_matrix = np.array(transition_times_matrix, dtype=object)
    for i_patch in range(nb_of_patches):
        # Look for the outgoing transits, and then look at next visits to see which patch they went to
        outgoing_transits_indices = np.where(current_visits[:, 2] == i_patch)
        next_patch_indices = np.clip(outgoing_transits_indices + 1, 0, len(current_visits) - 1)
        next_patches = current_visits[next_patch_indices, :, 2]
        # Count how many outgoing transits go to each of the patches
        counts_each_patch = np.array([len(np.where(next_patches == patch)) for patch in range(len(nb_of_patches))])
        probability_each_patch = counts_each_patch / np.sum(counts_each_patch)
        transition_probability_matrix[i_patch] = probability_each_patch
        for i_next_patch in range(len(next_patch_indices)):
            travel_time = current_visits[i_next_patch]
            previous_visit = current_visits[i_next_patch - 1]
            next_visit = current_visits[i_next_patch]
            transition_times_matrix[i_patch, i_next_patch].append(next_visit[0] - previous_visit[1] + 1)

plt.imshow(transition_probability_matrix)
plt.imshow(transition_times_matrix)
plt.show()






