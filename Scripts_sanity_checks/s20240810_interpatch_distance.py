# This is a script to plot the distribution of inter-patch distance

import matplotlib.pyplot as plt
import numpy as np
import datatable as dt
import pandas as pd
import os

import analysis as ana
from Generating_data_tables import main as gen
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
from Scripts_models import s20240626_transitionmatrix as transition_script
import find_data as fd
from Parameters import parameters as param


def generate_patch_distances():
    path = gen.generate("")
    results = dt.fread(path + "clean_results.csv")
    trajectories = dt.fread(path + "clean_trajectories.csv")
    full_list_of_folders = results[:, "folder"].to_list()[0]
    full_cond_list = list(param.nb_to_name.keys())
    print("Finished loading tables!")

    distances = ["close", "med", "far", "superfar", "cluster"]
    nearest_neighbors_each_cond = transition_script.find_nearest_neighbors(path, full_list_of_folders, full_cond_list)
    # Lists to fill in following loop
    list_nearest_neighbor_distances = [[] for _ in range(len(distances))]
    for i_distance, distance in enumerate(distances):
        print("Computing the inter-patch distance for distance " + distance + "!")
        current_conditions = param.name_to_nb_list[distance]
        # Remove controls 'cause they don't have them patches
        for cond in current_conditions:
            if param.nb_to_density[cond] == "0":
                current_conditions.remove(cond)
        current_folder_list = fd.return_folders_condition_list(full_list_of_folders, current_conditions)
        for i_folder, folder in enumerate(current_folder_list):
            if i_folder % (len(current_folder_list) // 4) == 0:
                print("> Folder "+str(i_folder)+" / "+str(len(current_folder_list)))
            current_metadata = fd.folder_to_metadata(folder)
            condition = current_metadata["condition"][0]
            patch_centers = current_metadata["patch_centers"]
            for i_patch in range(len(patch_centers)):
                current_patch_xy = patch_centers[i_patch]
                for j_neighbor in np.where(nearest_neighbors_each_cond[condition][i_patch] == 1)[0]:
                    neighbor_xy = patch_centers[j_neighbor]
                    list_nearest_neighbor_distances[i_distance].append(ana.distance(current_patch_xy, neighbor_xy))
    print("Finished retrieving neighbor distances!")

    boxplot = plt.boxplot(list_nearest_neighbor_distances, patch_artist=True, labels=distances, bootstrap=1000)
    colors = [param.name_to_color[dist] for dist in distances]
    # Fill with colors
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("Average distance to the closest neighbor")
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.show()

    data_for_saving = pd.DataFrame({"distance": distances, "interpatch_distance": [np.mean(d) for d in list_nearest_neighbor_distances]})
    data_for_saving.to_csv(path + "interpatch_distance.csv")
