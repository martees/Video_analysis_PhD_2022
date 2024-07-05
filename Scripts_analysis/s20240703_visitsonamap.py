import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from Generating_data_tables import main as gen
import analysis as ana
from Parameters import parameters as param
import find_data as fd
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script


def visit_duration_on_patches(results_table, condition_list, variable="total"):
    fig, axs = plt.subplots(1, len(condition_list))
    fig.set_size_inches(5*len(condition_list), 5)

    # First, load the visit list for each condition in condition_list
    visit_list_each_condition = [[] for _ in range(len(condition_list))]
    visit_list_each_condition_each_plate = [[] for _ in range(len(condition_list))]
    for i_condition, condition in enumerate(condition_list):
        results_this_cond = results_table[results_table["condition"] == condition].reset_index(drop=True)
        visit_list_each_condition[i_condition] = ana.return_value_list(results_this_cond, "no_hole_visits", convert_to_duration=False)
        if variable == "discovery" or variable == "rank":
            for i_folder in range(len(results_this_cond)):
                results_this_plate = results_this_cond[results_this_cond["folder"] == results_this_cond["folder"][i_folder]]
                visit_list_each_condition_each_plate[i_condition].append(ana.return_value_list(results_this_plate, "no_hole_visits", convert_to_duration=False))
    # Then, load the patch positions for each distance
    patch_positions = heatmap_script.idealized_patch_centers_mm(1847)

    colors = plt.cm.viridis(np.linspace(0, 1, 101))
    for i_condition, condition in enumerate(condition_list):
        if variable == "avg":
            axs[i_condition].set_title("Average visit time in "+param.nb_to_name[condition])
        if variable == "total":
            axs[i_condition].set_title("Total visit time in "+param.nb_to_name[condition])
        if variable == "rank":
            axs[i_condition].set_title("Median visit rank in "+param.nb_to_name[condition])
        if variable == "start":
            axs[i_condition].set_title("Average visit start // 100 in "+param.nb_to_name[condition])
        if variable == "discovery":
            axs[i_condition].set_title("Minimal discovery time in "+param.nb_to_name[condition])
        visit_list_this_condition = np.array(visit_list_each_condition[i_condition])
        current_patch_positions = patch_positions[condition]

        #Compute the average visit duration / total visit time to each patch
        avg_value_each_patch = np.zeros(len(current_patch_positions))
        for i_patch in range(len(current_patch_positions)):
            visits_this_patch = visit_list_this_condition[visit_list_this_condition[:, 2] == i_patch]
            if variable == "avg":
                avg_value_each_patch[i_patch] = np.nanmean(ana.convert_to_durations(list(visits_this_patch)))
            if variable == "total":
                avg_value_each_patch[i_patch] = np.nansum(ana.convert_to_durations(list(visits_this_patch)))
            if variable == "rank":
                visit_ranks = []
                for i_plate in range(len(visit_list_each_condition_each_plate[i_condition])):
                    visits_this_plate = np.array(visit_list_each_condition_each_plate[i_condition][i_plate])
                    if len(visits_this_plate) > 0:
                        visit_ranks.append(np.nanmean(np.where(visits_this_plate[:, 2] == i_patch)))
                avg_value_each_patch[i_patch] = np.nanmedian(visit_ranks)
            if variable == "start":
                if len(visits_this_patch) > 0:
                    avg_value_each_patch[i_patch] = np.nanmean(visits_this_patch[0, :]//100)
            if variable == "discovery":
                first_discoveries = []
                for i_plate in range(len(visit_list_each_condition_each_plate[i_condition])):
                    visits_this_plate = np.array(visit_list_each_condition_each_plate[i_condition][i_plate])
                    if len(visits_this_plate) > 0:
                        visits_this_patch = visits_this_plate[visits_this_plate[:, 2] == i_patch]
                        if len(visits_this_patch) > 0:
                            first_discoveries.append(visits_this_patch[0, 0])
                if len(first_discoveries) > 0:
                    avg_value_each_patch[i_patch] = np.nanmin(first_discoveries)
                else:
                    avg_value_each_patch[i_patch] = 9999

        # Plot a circle around each of the patch centers
        # Write the average visit length inside
        # Color of the circle depends on average visit length
        for i_patch in range(len(current_patch_positions)):
            x = current_patch_positions[i_patch][0]
            y = current_patch_positions[i_patch][1]
            if not np.isnan(avg_value_each_patch[i_patch]):
                normalized_color_index = np.clip(int(avg_value_each_patch[i_patch]/np.nanmax(avg_value_each_patch) * 100), 0, 100)
                circle = plt.Circle((x, y), radius=60, color=colors[normalized_color_index])
                axs[i_condition].add_artist(circle)
                text_color = int(100 - np.rint(normalized_color_index/100)*100)
                axs[i_condition].text(x, y, str(int(avg_value_each_patch[i_patch])), horizontalalignment='center',
                                      verticalalignment='center',
                                      color=colors[text_color])
            circle_outline = plt.Circle((x, y), radius=60, color=colors[0], fill=False)
            axs[i_condition].add_artist(circle_outline)

        axs[i_condition].axis("scaled")
        if param.nb_to_distance[condition] == "close":
            axs[i_condition].set_ylim(570, 1280)
            axs[i_condition].set_xlim(466, 1380)
        if param.nb_to_distance[condition] == "med":
            axs[i_condition].set_ylim(460, 1375)
            axs[i_condition].set_xlim(290, 1560)
        if param.nb_to_distance[condition] == "far":
            axs[i_condition].set_ylim(440, 1400)
            axs[i_condition].set_xlim(400, 1420)
        if param.nb_to_distance[condition] == "cluster":
            axs[i_condition].set_ylim(460, 1500)
            axs[i_condition].set_xlim(350, 1600)

    plt.show()


if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(test_pipeline=False)
    results = pd.read_csv(path + "clean_results.csv")
    visit_duration_on_patches(results, [12, 13, 14, 15], variable="rank")
    visit_duration_on_patches(results, [0, 1, 2, 3], variable="rank")
    visit_duration_on_patches(results, [4, 5, 6, 7], variable="rank")
    visit_duration_on_patches(results, [1, 5, 8, 9, 10], variable="rank")
