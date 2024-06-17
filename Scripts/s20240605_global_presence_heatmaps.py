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


def generate_polar_map(plate, dont_save_and_return=False):
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
        (y_pixel_matrix - y_of_closest_patch_center) / (x_pixel_matrix - x_of_closest_patch_center))

    # Stack all of those so that you get the array with for each pixel [index, distance to border, radial coordinate]
    # relative to the closest food patch
    polar_coordinate_map = np.stack((closest_patch_map, distance_map, radial_coordinates), axis=2)

    if dont_save_and_return:
        return polar_coordinate_map
    else:
        np.save(plate + "polar_map.npy", polar_coordinate_map)


def average_patch_radius_all_conditions(results_path, full_plate_list):
    """
    Will compute the average radius of patches in all conditions, based on 20 radius for 3 random patches for 6 random
    plates from each condition. Will save it in a csv, in path/perfect_plates.
    @param results_path: path to the whole dataset, where the map will be saved
    @param full_plate_list: list of data path ("./*traj.csv" format)
    @return: None, saves a matrix
    """
    all_conditions_list = param.nb_to_name.keys()
    average_radius = pd.DataFrame({"condition": all_conditions_list, "avg_patch_radius": np.zeros(len(all_conditions_list))})
    for i_condition in range(len(all_conditions_list)):
        # Compute average radius from a few plates of this condition
        plates_this_condition = fd.return_folders_condition_list(full_plate_list, i_condition)
        a_few_random_plates = random.choices(plates_this_condition, k=6)
        radiuses = []
        for i_plate, plate in enumerate(a_few_random_plates):
            in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
            in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()
            # Take a few random patches in each
            plate_metadata = fd.folder_to_metadata(plate)
            patch_centers = plate_metadata["patch_centers"]
            patch_spline_breaks = plate_metadata["spline_breaks"]
            patch_spline_coefs = plate_metadata["spline_coefs"]
            some_patches = random.choices(range(len(patch_centers)), k=3)
            for i_patch in some_patches:
                # For a range of 20 angular positions
                angular_pos = np.linspace(-np.pi, np.pi, 20)
                # Compute the local spline value for each of those angles
                for i_angle in range(len(angular_pos)):
                    radiuses.append(gt.spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                                    patch_spline_coefs[i_patch]))
        average_radius["avg_patch_radius"][i_condition] = np.mean(radiuses)

    average_radius.to_csv(results_path+"perfect_plates/average_patch_radius_each_condition.csv")


def generate_perfect_plates_polar_map(results_path, full_plate_list):
    """
    If it does not exist, will create a path/perfect_plates subfolder in path, and then will save inside a
    "perfect_polar_map_[i_condition]_[frame_width].npy" array for all conditions.
    This npy will contain a map with the size frame_width, and for each pixel, a list of three elements:
        - the index of the closest patch
        - the distance to the boundary of this patch
        - the radial coordinates relative to the center of this patch
    All of this, based on idealized food patches, perfectly round, with a radius equal to the average radius of the
    patches in this condition, and with the theoretical locations.
    @param results_path: path to the whole dataset, where the map will be saved
    @param full_plate_list: list of data path ("./*traj.csv" format)
    @return: None, saves a matrix.
    """
    print("Generating perfect plates...")
    all_conditions_list = param.nb_to_name.keys()

    if not os.path.isdir(results_path + "perfect_plates"):
        os.mkdir(results_path + "perfect_plates")

    # Load the average patch radius for all conditions
    if not os.path.isdir(results_path + "average_patch_radius_each_condition.csv"):
        average_patch_radius_all_conditions(results_path, full_plate_list)
    average_patch_radiuses = pd.read_csv(results_path + "perfect_plates/average_patch_radius_each_condition.csv")

    for i_condition in range(len(all_conditions_list)):
        print(">>> Condition ", i_condition, " / ", len(all_conditions_list))
        # First, generate the perfect plate map, with the patch to which each pixel belongs
        # Load patch information
        distance_this_condition = param.nb_to_distance[i_condition]
        patch_centers = patch_coordinates.distance_to_patches[distance_this_condition]
        avg_patch_radius = average_patch_radiuses["avg_patch_radius"][average_patch_radiuses["condition"] == i_condition]

        # List of discrete positions for the patch boundaries
        boundary_position_list = []

        # For each patch, generate boundary points (using average radius computed previously)
        for i_patch in range(len(patch_centers)):
            # For a range of 480 angular positions
            angular_pos = np.linspace(-np.pi, np.pi, 480)
            # Add to position list discrete (int) cartesian positions
            # (due to discretization, positions will be added multiple times, but we don't care)
            for point in range(len(angular_pos)):
                x = int(patch_centers[i_patch][0] + (avg_patch_radius * np.cos(angular_pos[point])))
                y = int(patch_centers[i_patch][1] + (avg_patch_radius * np.sin(angular_pos[point])))
                boundary_position_list.append((x, y, i_patch))  # 3rd tuple element is patch number

        # Not all images are the same size, but we take the majority size
        # Load the frame size for all the plates of this condition
        plates_this_condition = fd.return_folders_condition_list(full_plate_list, i_condition)
        image_sizes = np.zeros(len(plates_this_condition))
        for i_plate, plate in enumerate(plates_this_condition):
            image_sizes[i_plate] = fd.load_silhouette(plate)[2]
        plate_size = np.argmax(np.bincount(image_sizes))

        # Compute the mapssss teehee
        # Compute plate map, using just one plate because "map_from_boundaries" only asks for
        # a plate path to load a frame size
        one_plate_this_size = np.where(image_sizes == plate_size)[0]
        in_patch_map, _ = gt.map_from_boundaries(one_plate_this_size, boundary_position_list, patch_centers)
        # I add bbb in the end because the function removes the last subfolder (to take paths ending in ./traj.csv)
        distance_map = script_distance.generate_patch_distance_map(in_patch_map, results_path + "perfect_plates/bbb")
        # Generate and save a polar map from these perfect plates!
        perfect_polar_map = generate_polar_map(in_patch_map, distance_map, patch_centers, "", dont_save_and_return=True)
        np.save(results_path + "perfect_plates/polar_map_"+param.nb_to_name[i_condition]+".npy", perfect_polar_map)


def experimental_to_perfect_pixel_visits(pixelwise_visits, polar_map, ideal_patch_centers, ideal_patch_radius):
    """
    Function that converts pixel-level visits in the experimental plates to the equivalent ones in a "perfect"
    environment (where patches are perfectly round), conserving the closest patch, the distance to the patch boundary,
    and the angular coordinate with respect to the patch center.
    @param pixelwise_visits: a numpy array (dtype = object), containing one list per pixel in the image, and in each
                             list, one sublist per visit to the pixel, containing [visit start, visit end].
    @param polar_map: a map containing, for each pixel of the image, a list containing [index of the closest patch,
                      distance to the patch boundary, angular coordinate with respect to the closest patch center].
    @param ideal_patch_centers: coordinates of the ideal patch centers [[x0, y0], [x1, y1], ...].
    @param ideal_patch_radius: average radius of patches for this condition.
    @return: same format as pixelwise_visits, but transformed spatially to match the perfect landscape.
    """
    perfect_pixel_wise_visits = [[[[]] for _ in range(len(pixelwise_visits[0]))] for _ in range(len(pixelwise_visits))]
    for i_line in range(len(pixelwise_visits)):
        for i_col in range(len(pixelwise_visits[i_line])):
            i_patch, distance_boundary, angular_coord = polar_map[i_line, i_col]
            new_x = ideal_patch_radius * np.cos(angular_coord) + ideal_patch_centers[i_patch][0]
            new_y = ideal_patch_radius * np.sin(angular_coord) + ideal_patch_centers[i_patch][1]
            perfect_pixel_wise_visits[int(new_y), int(new_x)] = pixelwise_visits[i_line, i_col]
    return perfect_pixel_wise_visits


def plot_heatmap_of_all_silhouettes(traj, full_plate_list, curve_list, curve_names, regenerate_pixel_visits=False, regenerate_polar_map=False, regenerate_perfect_maps=False):
    # Plot initialization
    fig, axes = plt.subplots(1, len(curve_list))
    fig.suptitle("Heatmap of worm presence")
    heatmap_each_curve = [np.zeros((1847, 1847)) for _ in range(len(curve_list))]

    tic = time.time()
    for i_curve in range(len(curve_list)):
        print(int(time.time() - tic), "s: Curve ", i_curve, " / ", len(curve_list))
        plate_list = fd.return_folders_condition_list(full_plate_list, curve_list[i_curve])
        for i_plate, plate in enumerate(plate_list):
            if i_plate % 10 == 0:
                print(">>> ", int(time.time() - tic), "s: plate ", i_plate, " / ", len(plate_list))

            # If it's not already done, or has to be redone, compute the pixel visit durations
            pixelwise_visits_path = plate[:-len("traj.csv")] + "pixelwise_visits.npy"
            if not os.path.isfile(pixelwise_visits_path) or regenerate_pixel_visits:
                gr.generate_pixelwise_visits(traj, plate)
            # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
            current_pixel_wise_visits = np.load(pixelwise_visits_path, allow_pickle=True)

            # Then, if it's not already done, or has to be redone, compute the polar coordinates for the plate
            polar_map_path = plate[:-len("traj.csv")] + "polar_map.npy"
            if not os.path.isfile(pixelwise_visits_path) or regenerate_polar_map:

                # If it's not already done, compute the "perfect patch" matrix
                if not os.path.isdir(plate[:-len(plate.split("/")[-1])] + "perfect_map.npy"):
                generate_polar_map()
            # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
            current_pixel_wise_visits = np.load(pixelwise_visits_path, allow_pickle=True)



            # Load path and clean_results.csv, because that's where the list of folders we work on is stored
            path = gen.generate(test_pipeline=False)
            results = pd.read_csv(path + "clean_results.csv")
            trajectories = pd.read_csv(path + "clean_trajectories.csv")
            full_folder_list = results["folder"]

            if len(current_pixel_wise_visits) == 1847:
                # Load patch info for this folder
                in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
                if not os.path.isfile(in_patch_matrix_path):
                    gt.in_patch_all_pixels(plate)
                in_patch_matrix = pd.read_csv(in_patch_matrix_path)
                # Convert visits to durations, and sum them, to get the total time spent in each pixel
                for i in range(len(current_pixel_wise_visits)):
                    for j in range(len(current_pixel_wise_visits[i])):
                        if in_patch_matrix[str(j)][i] == -1:
                            current_pixel_wise_visits[i][j] = int(
                                np.sum(ana.convert_to_durations(current_pixel_wise_visits[i][j])))
                        else:
                            current_pixel_wise_visits[i][j] = 0
                # Add them to the corresponding curve data
                heatmap_each_curve[i_curve] = heatmap_each_curve[i_curve] + current_pixel_wise_visits
            # If the plate is not 1944 x 1944, print something out
            else:
                print("Plate ", plate, " is not the standard size, it's ", len(current_pixel_wise_visits))

        heatmap_each_curve[i_curve] /= np.max(heatmap_each_curve[i_curve])
        axes[i_curve].imshow(heatmap_each_curve[i_curve].astype(float), vmax=0.5)
        #if i_curve == len(curve_list) - 1:
        #    axes[i_curve].colorbar()
    plt.show()



#plot_heatmap_of_all_silhouettes(trajectories, full_plate_list, [[0], [1], [2]], ["close 0.2", "med 0.2", "far 0.2"], False)
generate_polar_map(full_folder_list[0])
