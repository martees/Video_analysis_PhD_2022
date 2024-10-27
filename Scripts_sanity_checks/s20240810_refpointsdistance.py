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


def rotate_points(points, angle):
    """
    Rotates the points counter-clockwise by the given angle (in radians).
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(points, rotation_matrix.T)


def optimal_rotation(points, reference_square):
    """
    Finds the optimal rotation angle to align points with a reference square.
    """
    def loss_function(angle):
        # Rotate points by the given angle and compute the error with the reference square
        rotated_points = rotate_points(points, angle)
        return np.sum(np.linalg.norm(rotated_points - reference_square, axis=1))

    # Optimize the rotation by minimizing the loss function
    from scipy.optimize import minimize
    result = minimize(lambda x: loss_function(x[0]), [0.0], bounds=[(-np.pi, np.pi)])
    return rotate_points(points, result.x[0])


path = gen.generate("", shorten_traj=True, modeled_data=True)
results = dt.fread(path + "clean_results.csv")
full_list_of_folders = results[:, "folder"].to_list()[0]
print("Finished loading tables!")

all_conditions_list = param.nb_to_name.keys()

if not os.path.isdir(path + "perfect_heatmaps"):
    os.mkdir(path + "perfect_heatmaps")

distances_each_condition_top = []
distances_each_condition_left = []
distances_each_condition_right = []
distances_each_condition_bottom = []
pixel_to_mm_ratios = []

heatmap_points = np.zeros((1847, 1847))
plate_center = [1847/2, 1847/2]

condition_names = []
condition_colors = []
for i_condition, condition in enumerate(all_conditions_list):
    print(">>>>>> Condition ", i_condition, " / ", len(all_conditions_list))
    # Compute average distance from a few plates of this condition
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
            pixel_to_mm_ratios += [32 / top_dist, 32 / left_dist, 32 / right_dist, 32 / bottom_dist]

            # Then rotate them to minimize distance with a perfect square
            points_array = np.array([point1, point2, point3, point4])
            # plt.plot(points_array[:, 0], points_array[:, 1], label="before rotation")
            x_centroid = np.mean(points_array[:, 0])
            y_centroid = np.mean(points_array[:, 1])
            perfect_square = np.array([[plate_center[0] - 500, plate_center[1] - 500], [plate_center[0] - 500, plate_center[1] + 500], [plate_center[0] + 500, plate_center[1] + 500], [plate_center[0] + 500, plate_center[1] - 500]])
            points_array = optimal_rotation(points_array, perfect_square)
            # plt.plot(perfect_square[:, 0], perfect_square[:, 1], label="perfecto")
            # plt.plot(points_array[:, 0], points_array[:, 1], label="after rotation")

            # Then center them to the center of the plate
            x_centroid_shift = plate_center[0] - np.mean(points_array[:, 0])
            y_centroid_shift = plate_center[1] - np.mean(points_array[:, 1])
            points_array[:, 0] += x_centroid_shift
            points_array[:, 1] += y_centroid_shift
            # plt.plot(points_array[:, 0], points_array[:, 1], label="after shift")

            # Add them to the heatmap
            heatmap_points[int(points_array[0, 0]), int(points_array[0, 1])] += 1
            heatmap_points[int(points_array[1, 0]), int(points_array[1, 1])] += 1
            heatmap_points[int(points_array[2, 0]), int(points_array[2, 1])] += 1
            heatmap_points[int(points_array[3, 0]), int(points_array[3, 1])] += 1

            # plt.legend()
            # plt.show()

    condition_names.append(param.nb_to_name[condition])
    condition_colors.append(param.name_to_color[param.nb_to_name[condition]])

fig, [ax0, ax1, ax2] = plt.subplots(1, 3)
fig.set_size_inches(22, 5.6)

# Heatmap of the points
ax0.imshow(heatmap_points, cmap="hot", vmax=0.1)
ax0.set_title("Heatmap of reference points positions", fontsize=20)

# Boxplot with one box for top edge, left edge, etc.
ax1.boxplot([distances_each_condition_left, distances_each_condition_top, distances_each_condition_right, distances_each_condition_bottom])
ax1.set_xticks([1, 2, 3, 4], ["Left", "Top", "Right", "Bottom"])
ax1.set_title("Reference point distance for each edge", fontsize=20)

# Pixel to mm ratio histogram
ax2.hist(pixel_to_mm_ratios / np.mean(pixel_to_mm_ratios), 100)
all_distances = distances_each_condition_top + distances_each_condition_bottom + distances_each_condition_left + distances_each_condition_right
print("You can copy this number in Parameters/parameters.py, for the variable one_pixel_in_mm: ", 32 / np.mean(all_distances))
ax2.set_title("Pixel to mm ratio distribution", fontsize=20)

plt.show()

