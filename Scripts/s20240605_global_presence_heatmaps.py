# A script to plot a heatmap of the duration of visit to each pixel
# But munching all the conditions together mouahahahaha
import copy

from scipy import ndimage
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random

from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt
from Scripts import s20240606_distancetoedgeanalysis as script_distance
from Parameters import parameters as param
from Parameters import patch_coordinates
import find_data as fd
import analysis as ana


def generate_polar_map(plate):
    """
    Function that takes the in_patch_matrix of a folder (indicating which patch each pixel belongs to, -1 = outside)
    and the distance_to_patch_map.npy (indicating distance to the closest patch boundary)
    and, from it, generate the polar coordinates of each pixel with respect to its closest patch BOUNDARY.
    So in the end, it will save a matrix with, for each pixel of the image, [i, r_b, theta], where:
        i = index of the closest food patch
        r_b = radial distance to the boundary of that food patch
        theta = angular position with respect to the closest food patch
    """
    # Load the matrix with patch to which each pixel belongs
    in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
    in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()

    # Load the matrix with distance to the closest food patch boundary for each pixel
    distance_map_path = plate[:-len(plate.split("/")[-1])] + "distance_to_patch_map.csv"
    if not os.path.isdir(distance_map_path):
        script_distance.generate_patch_distance_map(in_patch_matrix, plate)
    distance_map = np.load(plate[:-len(plate.split("/")[-1])] + "distance_to_patch_map.npy")

    # Load patch centers
    patch_centers = fd.folder_to_metadata(plate)["patch_centers"]

    # For every patch, generate a distance transform from the center
    distance_transform_patch_centers = [[] for _ in range(len(patch_centers))]
    for i_patch, patch_center in enumerate(patch_centers):
        zero_in_center = np.ones(in_patch_matrix.shape)
        zero_in_center[int(patch_center[0]), int(patch_center[1])] = 0
        distance_transform_patch_centers[i_patch] = ndimage.distance_transform_edt(zero_in_center)

    # Then, for each pixel, take the index of the patch for which the distance is the smallest
    closest_patch_map = np.argmin(distance_transform_patch_centers, axis=0)

    # In order to compute the angular coordinate with respect to the closest food patch for each pixel, we need to
    # compute theta = arctan2( y_pixel - y_patch, x_pixel - x_patch )
    # In order to compute this with array operations, we need to make the distance to the closest patch center matrix,
    # the x_pixel matrix, and the x_of_closest_patch matrix
    # Just believe me on these xD
    # (it makes a matrix with a column of 0's, a column of 1's, ... etc. for the same size as the image)
    x_pixel_matrix = np.repeat(np.arange(0, len(in_patch_matrix[0])), len(in_patch_matrix)).reshape(
        in_patch_matrix.shape).transpose()
    y_pixel_matrix = np.repeat(np.arange(0, len(in_patch_matrix[0])), len(in_patch_matrix)).reshape(
        in_patch_matrix.shape)
    # (I intentionally invert x and y because when I plot them with imshow they appear inverted)
    x_of_closest_patch_center = [list(map(lambda x: patch_centers[x][1], y)) for y in closest_patch_map]
    y_of_closest_patch_center = [list(map(lambda x: patch_centers[x][0], y)) for y in closest_patch_map]
    radial_coordinates = np.arctan2(
        (y_pixel_matrix - y_of_closest_patch_center), (x_pixel_matrix - x_of_closest_patch_center))

    # Stack all of those so that you get the array with for each pixel [index, distance to border, radial coordinate]
    # relative to the closest food patch
    polar_coordinate_map = np.stack((closest_patch_map, distance_map, radial_coordinates), axis=2)

    np.save(plate[:-len(plate.split("/")[-1])] + "polar_map.npy", polar_coordinate_map)


def generate_average_patch_radius_each_condition(results_path, full_plate_list):
    """
    Will compute the average radius of patches in all conditions, based on 20 radius for 3 random patches for 6 random
    plates from each condition. Will save it in a csv, in path/average_patch_radius_each_condition.csv.
    @param results_path: path to the whole dataset, where the map will be saved
    @param full_plate_list: list of data path ("./*traj.csv" format)
    @return: None, saves a matrix
    """
    print(">>> Computing average radius for all conditions...")
    all_conditions_list = param.nb_to_name.keys()

    average_radius = pd.DataFrame(
        {"condition": all_conditions_list, "avg_patch_radius": np.zeros(len(all_conditions_list))})
    for i_condition in range(len(all_conditions_list)):
        if i_condition % 3 == 0:
            print(">>>>>> Condition ", i_condition, " / ", len(all_conditions_list))
        # Compute average radius from a few plates of this condition
        plates_this_condition = fd.return_folders_condition_list(full_plate_list, i_condition)
        a_few_random_plates = random.sample(plates_this_condition, k=min(len(plates_this_condition), 6))
        radiuses = []
        for i_plate, plate in enumerate(a_few_random_plates):
            in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
            in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()
            # Take a few random patches in each
            plate_metadata = fd.folder_to_metadata(plate)
            patch_centers = plate_metadata["patch_centers"]
            patch_spline_breaks = plate_metadata["spline_breaks"]
            patch_spline_coefs = plate_metadata["spline_coefs"]
            some_patches = random.sample(range(len(patch_centers)), k=min(len(patch_centers), 3))
            for i_patch in some_patches:
                # For a range of 20 angular positions
                angular_pos = np.linspace(-np.pi, np.pi, 20)
                # Compute the local spline value for each of those angles
                for i_angle in range(len(angular_pos)):
                    radiuses.append(gt.spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                                    patch_spline_coefs[i_patch]))
        average_radius.loc[i_condition, "avg_patch_radius"] = np.mean(radiuses)

    average_radius.to_csv(results_path + "average_patch_radius_each_condition.csv")


def experimental_to_perfect_pixel_indices(folder_to_save, polar_map, ideal_patch_centers,
                                          ideal_patch_radius):
    """
    Function that converts pixel coordinates in the experimental plates to the equivalent ones in a "perfect"
    environment (where patches are perfectly round), conserving the closest patch, the distance to the patch boundary,
    and the angular coordinate with respect to the patch center.
    @param folder_to_save: where to save the resulting npy
    @param polar_map: a map containing, for each pixel of the image, a list containing [index of the closest patch,
                      distance to the patch boundary, angular coordinate with respect to the closest patch center].
    @param ideal_patch_centers: coordinates of the ideal patch centers [[x0, y0], [x1, y1], ...].
    @param ideal_patch_radius: average radius of patches for this condition.
    @return: saves a matrix with the same size as polar_map, in folder_to_save, named "xp_to_perfect.npy"
             with, for each cell, the corresponding [x,y] in the "perfect" landscape.
    """
    nb_of_lines = len(polar_map)
    nb_of_col = len(polar_map[0])
    experimental_to_perfect = [[[] for _ in range(nb_of_col)] for _ in range(nb_of_lines)]
    print(">>>>>> Converting to perfect coordinates...")

    # That's the computation:
    # i_patch, distance_boundary, angular_coord = polar_map[i_line, i_col]
    # new_x = ideal_patch_radius * np.cos(angular_coord) + ideal_patch_centers[int(i_patch)][0]
    # new_y = ideal_patch_radius * np.sin(angular_coord) + ideal_patch_centers[int(i_patch)][1]

    # I do it directly on numpy arrays
    closest_patch_index = polar_map[:, :, 0]
    radius_matrix = np.ones((nb_of_lines, nb_of_col)) * ideal_patch_radius
    closest_patch_x_matrix =
    closest_patch_y_matrix =


    for i_line in range(nb_of_lines):
        if i_line % (nb_of_lines // 6) == 0:
            print(">>>>>> Line ", i_line, " / ", nb_of_lines)
        for i_col in range(nb_of_col):
            i_patch, distance_boundary, angular_coord = polar_map[i_line, i_col]
            new_x = ideal_patch_radius * np.cos(angular_coord) + ideal_patch_centers[int(i_patch)][0]
            new_y = ideal_patch_radius * np.sin(angular_coord) + ideal_patch_centers[int(i_patch)][1]
            # Appending in case there are more than one experimental pixel leading to the same "perfect" one
            experimental_to_perfect[i_line][i_col] = [new_x, new_y]
    np.save(folder_to_save + "xp_to_perfect.npy", experimental_to_perfect)


def plot_heatmap_of_all_silhouettes(results_path, traj, full_plate_list, curve_list, curve_names,
                                    regenerate_pixel_visits=False,
                                    regenerate_polar_maps=False, regenerate_perfect_map=False):
    # Plot initialization
    fig, axes = plt.subplots(1, len(curve_list))
    fig.suptitle("Heatmap of worm presence")
    heatmap_each_curve = [np.zeros((1847, 1847)) for _ in range(len(curve_list))]

    # If it's not already done, compute the average patch radiuses for each condition
    if not os.path.isfile(results_path + "average_patch_radius_each_condition.csv"):
        generate_average_patch_radius_each_condition(results_path, full_plate_list)
    average_patch_radius_each_cond = pd.read_csv(results_path + "average_patch_radius_each_condition.csv")

    tic = time.time()
    for i_curve in range(len(curve_list)):
        print(int(time.time() - tic), "s: Curve ", i_curve, " / ", len(curve_list))
        plate_list, condition_each_plate = fd.return_folders_condition_list(full_plate_list, curve_list[i_curve],
                                                                            return_conditions=True)
        for i_plate, plate in enumerate(plate_list):
            if i_plate % 10 == 0:
                print(">>> ", int(time.time() - tic), "s: plate ", i_plate, " / ", len(plate_list))

            # If it's not already done, or has to be redone, compute the experimental to perfect mapping
            if not os.path.isfile(
                    plate[:-len("traj.csv")] + "xp_to_perfect.npy") or regenerate_perfect_map:
                # Then, if it's not already done, or has to be redone, compute the polar coordinates for the plate
                polar_map_path = plate[:-len("traj.csv")] + "polar_map.npy"
                if not os.path.isfile(polar_map_path) or regenerate_polar_maps:
                    generate_polar_map(plate)
                current_polar_map = np.load(polar_map_path)

                # Load patch centers and radiuses
                current_condition = condition_each_plate[i_plate]
                current_patch_centers = param.distance_to_xy[param.nb_to_distance[current_condition]]
                current_average_radius = \
                    average_patch_radius_each_cond[average_patch_radius_each_cond["condition"] == current_condition][
                        "avg_patch_radius"]

                # Then FINALLY convert the current pixel wise visits to their "perfect" equivalent (in an environment with
                # perfectly round patches, while conserving distance to border and angular coordinate w/ respect to center)
                experimental_to_perfect_pixel_indices(plate[:-len("traj.csv")], current_polar_map, current_patch_centers,
                                                      current_average_radius)
            # Matrix with, in each cell, the corresponding "perfect" coordinates
            xp_to_perfect_indices = np.load(plate[:-len("traj.csv")] + "xp_to_perfect.npy")

            # If it's not already done, or has to be redone, compute the pixel visit durations
            pixelwise_visits_path = plate[:-len("traj.csv")] + "pixelwise_visits.npy"
            if not os.path.isfile(pixelwise_visits_path) or regenerate_pixel_visits:
                gr.generate_pixelwise_visits(traj, plate)
            # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
            current_pixel_wise_visits = np.load(pixelwise_visits_path)

            perfect_durations = [[[] for _ in range(len(current_pixel_wise_visits[0]))] for _ in range(len(current_pixel_wise_visits))]
            # Convert visits to durations, and sum them, to get the total time spent in each pixel
            for i in range(len(current_pixel_wise_visits)):
                for j in range(len(current_pixel_wise_visits[i])):
                    perfect_i, perfect_j = xp_to_perfect_indices[i][j]
                    perfect_durations[perfect_i][perfect_j] = int(
                        np.sum(ana.convert_to_durations(current_pixel_wise_visits[i][j])))
            # Add them to the corresponding curve data
            heatmap_each_curve[i_curve] = heatmap_each_curve[i_curve] + current_pixel_wise_visits

        heatmap_each_curve[i_curve] /= np.max(heatmap_each_curve[i_curve])
        axes[i_curve].imshow(heatmap_each_curve[i_curve].astype(float), vmax=0.5)
        axes[i_curve].set_title(curve_names[i_curve])
    plt.show()


if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(test_pipeline=True)
    results = pd.read_csv(path + "clean_results.csv")
    trajectories = pd.read_csv(path + "clean_trajectories.csv")
    full_list_of_folders = results["folder"]

    plot_heatmap_of_all_silhouettes(path, trajectories, full_list_of_folders, [[0], [1], [2]],
                                    ["close 0.2", "med 0.2", "far 0.2"], False, regenerate_polar_maps=True)
