# This is a script to plot the theoretical + recorded reference points,
# scaled to all be on a 1847x1847 plate

import matplotlib.pyplot as plt
import numpy as np
import datatable as dt
import pandas as pd
import os

import analysis as ana
from Generating_data_tables import main as gen
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
import find_data as fd
from Parameters import parameters as param

path = gen.generate("")
results = dt.fread(path + "clean_results.csv")
trajectories = dt.fread(path + "clean_trajectories.csv")
full_list_of_folders = results[:, "folder"].to_list()[0]
print("Finished loading tables!")

all_conditions_list = param.nb_to_name.keys()

if not os.path.isdir(path + "perfect_heatmaps"):
    os.mkdir(path + "perfect_heatmaps")

distances_each_condition_top = []
distances_each_condition_left = []
distances_each_condition_right = []
distances_each_condition_bottom = []

heatmap_points = np.zeros((1847, 1847))

condition_names = []
condition_colors = []
for i_condition, condition in enumerate(all_conditions_list):
    if i_condition % 3 == 0:
        print(">>>>>> Condition ", i_condition, " / ", len(all_conditions_list))
    # Compute average radius from a few plates of this condition
    plates_this_condition = fd.return_folders_condition_list(full_list_of_folders, condition)
    for i_plate, plate in enumerate(plates_this_condition):
        plate_metadata = fd.folder_to_metadata(plate)
        xy_holes = plate_metadata["holes"][0]
        if len(xy_holes) == 4:
            # Reorder points according to y then x, to get lower left corner then lower right then upper left then upper right
            xy_holes = sorted(xy_holes, key=lambda x: x[1])
            xy_holes = sorted(xy_holes[0:2], key=lambda x: x[0]) + sorted(xy_holes[2:4], key=lambda x: x[0])

            point1, point4, point2, point3 = xy_holes
            left_dist = ana.distance(point1, point2)
            top_dist = ana.distance(point2, point3)
            right_dist = ana.distance(point3, point4)
            bottom_dist = ana.distance(point4, point1)

            distances_each_condition_top.append(top_dist)
            distances_each_condition_left.append(left_dist)
            distances_each_condition_right.append(right_dist)
            distances_each_condition_bottom.append(bottom_dist)
        heatmap_points[int(point1[1]), int(point1[0])] += 1
        heatmap_points[int(point2[1]), int(point2[0])] += 1
        heatmap_points[int(point3[1]), int(point3[0])] += 1
        heatmap_points[int(point4[1]), int(point4[0])] += 1

    condition_names.append(param.nb_to_name[condition])
    condition_colors.append(param.name_to_color[param.nb_to_name[condition]])

fig, [ax0, ax1] = plt.subplots(1, 2)

# Heatmap of the points
ax0.imshow(heatmap_points, cmap="hot", vmax=0.1)
ax0.set_title("Heatmap of the reference points, vmax=0.1")

# Boxplot with one box for top edge, left edge, etc.
ax1.boxplot([distances_each_condition_left, distances_each_condition_top, distances_each_condition_right, distances_each_condition_bottom])
ax1.set_xticks([1, 2, 3, 4], ["Left", "Top", "Right", "Bottom"])
ax1.set_title("Reference point distance for each edge")

all_distances = distances_each_condition_top + distances_each_condition_bottom + distances_each_condition_left + distances_each_condition_right
print("You can copy this number in Parameters/parameters.py, for the variable one_pixel_in_mm: ", np.mean(all_distances))

plt.show()

