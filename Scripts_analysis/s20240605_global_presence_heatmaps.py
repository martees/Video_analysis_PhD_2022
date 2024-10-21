# A script to plot a heatmap of the duration of visit to each pixel
# But munching all the conditions together mouahahahaha

from scipy import ndimage
import pandas as pd
import datatable as dt
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import random
mpl.use("TkAgg")

import plots
from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt
from Scripts_analysis import s20240606_distancetoedgeanalysis as script_distance
from Parameters import parameters as param
import find_data as fd
import analysis as ana
import ReferencePoints


def keep_only_short_visits(traj, threshold, longer_or_shorter):
    """
    Returns traj but keeping only time points that correspond to visits longer/shorter than threshold
    """
    return 0


#TODO then, in generate pixelwise visits, keep only visits that correspond to something in traj??? :thonk:


def generate_pixelwise_speeds(traj, folder):
    """
    Function that takes a folder containing a time series of silhouettes, and returns a list of lists with the dimension
    of the plate in :folder:, and in each cell, a list with the speed of the centroid for every time step where this
    pixel is visited.
    When this function is called, it also saves this output under the name "pixelwise_speeds.npy" in folder.
    Takes trajectory in argument to access speeds.
    """
    # Get the pixelwise_visits
    pixelwise_visit_timestamps = load_pixel_visits(traj[:, dt.f.time], folder, regenerate=True, return_durations=False)
    # Get the frame size
    _, _, frame_size = fd.load_silhouette(folder)
    # Initialize the table: one list per pixel
    speeds_each_pixel = [[[[]] for _ in range(frame_size[0])] for _ in range(frame_size[1])]
    for i_line in range(len(speeds_each_pixel)):
        for i_col in range(len(speeds_each_pixel[i_line])):
            current_pixel_visits = pixelwise_visit_timestamps[i_line][i_col]
            #if len(current_pixel_visits) > 0:
            #    print("hhhh")
            #if i_line == 307 and i_col == 1144:
            #    print("ccc")
            # Convert them to indices in the trajectory (for now visits are in seconds, but second 26 might be traj[1])
            if len(current_pixel_visits) > 0:
                current_pixel_visits = [[np.where(traj[:, dt.f.time].to_numpy() == visit[i])[0][0]
                                         for i in range(len(visit))]
                                        for visit in current_pixel_visits]
                # Go from [[0, 6, x], [12, 13, x]] to [[0, 1, 2, 3, 4, 5], [12]]
                list_of_frames = [list(range(int(np.rint(visit[0])), int(np.rint(visit[1]) + 1))) for visit in
                                  current_pixel_visits]
                # Go from [[0, 1, 2, 3, 4, 5], [12]] to [0, 1, 2, 3, 4, 5, 12] (list of frames for which this pixel was visited)
                list_of_frames = [list_of_frames[i][j] for i in range(len(list_of_frames)) for j in
                                  range(len(list_of_frames[i]))]
                # Put the speeds in the tableee
                speeds_each_pixel[i_line][i_col] = traj[list_of_frames, dt.f.speeds].to_list()[0]
            if len(speeds_each_pixel[i_line][i_col]) > 0 and speeds_each_pixel[i_line][i_col][0] is None:
                speeds_each_pixel[i_line][i_col] = [np.nan]
            if speeds_each_pixel[i_line][i_col] is None or len(speeds_each_pixel) == 0:
                speeds_each_pixel[i_line][i_col] = [np.nan]

    np.save(folder[:-len("traj.csv")] + "pixelwise_speeds.npy", np.array(speeds_each_pixel, dtype=object))
    return speeds_each_pixel


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
    distance_map_path = plate[:-len(plate.split("/")[-1])] + "distance_to_patch_map.npy"
    if not os.path.isdir(distance_map_path):
        script_distance.generate_patch_distance_map(in_patch_matrix, plate)
    distance_map = np.load(distance_map_path)

    # Load patch centers
    patch_centers = fd.folder_to_metadata(plate)["patch_centers"]

    # For every patch, generate a distance transform from the center
    distance_transform_patch_centers = [[] for _ in range(len(patch_centers))]
    for i_patch, patch_center in enumerate(patch_centers):
        zero_in_center = np.ones(in_patch_matrix.shape)
        zero_in_center[int(patch_center[1]), int(patch_center[0])] = 0
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
    x_of_closest_patch_center = [list(map(lambda x: patch_centers[x][0], y)) for y in closest_patch_map]
    y_of_closest_patch_center = [list(map(lambda x: patch_centers[x][1], y)) for y in closest_patch_map]
    radial_coordinates = np.arctan2(
        (y_pixel_matrix - y_of_closest_patch_center), (x_pixel_matrix - x_of_closest_patch_center))

    # Stack all of those so that you get the array with for each pixel [index, distance to border, radial coordinate]
    # relative to the closest food patch
    polar_coordinate_map = np.stack((closest_patch_map, distance_map, radial_coordinates), axis=2)

    np.save(plate[:-len(plate.split("/")[-1])] + "polar_map.npy", polar_coordinate_map)


def generate_average_patch_radius_each_condition(results_path, full_plate_list):
    """
    Will compute the average radius of patches in all conditions, based on 100 radii from each patch.
    Will save it in a csv, in path/average_patch_radius_each_condition.csv.
    Column "condition" contains the condition. Column "avg_patch_radius" contains the corresponding radius.
    Will also plot the distribution of all the radii for each condition in a boxplot (sanity check).
    @param results_path: path to the whole dataset, where the map will be saved
    @param full_plate_list: list of data path ("./*traj.csv" format)
    @return: None, saves a csv
    """
    print(">>> Computing average radius for all conditions...")
    all_conditions_list = param.nb_to_name.keys()

    if not os.path.isdir(results_path + "perfect_heatmaps"):
        os.mkdir(results_path + "perfect_heatmaps")

    average_radius = {"condition": [], "avg_patch_radius": []}
    radiuses_each_condition = [[] for _ in range(len(all_conditions_list))]
    condition_names = []
    condition_colors = []
    for i_condition, condition in enumerate(all_conditions_list):
        if i_condition % 3 == 0:
            print(">>>>>> Condition ", i_condition, " / ", len(all_conditions_list))
        # Compute average radius from a few plates of this condition
        plates_this_condition = fd.return_folders_condition_list(full_plate_list, condition)
        for i_plate, plate in enumerate(plates_this_condition):
            # Load patch info
            plate_metadata = fd.folder_to_metadata(plate)
            patch_centers = plate_metadata["patch_centers"]
            patch_spline_breaks = plate_metadata["spline_breaks"]
            patch_spline_coefs = plate_metadata["spline_coefs"]
            for i_patch in range(len(patch_centers)):
                # For a range of 100 angular positions
                angular_pos = np.linspace(-np.pi, np.pi, 100)
                # Compute the local spline value for each of those angles
                for i_angle in range(len(angular_pos)):
                    radiuses_each_condition[i_condition].append(
                        gt.spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                        patch_spline_coefs[i_patch]))
        average_radius["condition"].append(condition)
        average_radius["avg_patch_radius"].append(np.mean(radiuses_each_condition[i_condition]))
        condition_names.append(param.nb_to_name[condition])
        condition_colors.append(param.name_to_color[param.nb_to_name[condition]])

    plt.hist([rad[i] for rad in radiuses_each_condition for i in range(len(rad))], bins=20)
    plt.title("Radius distribution computed over 100 radiuses for each patch")
    plt.show()

    pd.DataFrame(average_radius).to_csv(results_path + "perfect_heatmaps/average_patch_radius_each_condition.csv")


def compute_average_ref_points_distance(results_path, full_plate_list):
    """
    Will compute the average distance between reference points in all conditions.
    Will save it in a csv, in path/average_ref_points_distance_each_condition.csv.
    Will also plot the distribution of all the distances for each condition in a boxplot (sanity check),
    and a boxplot showing the average for each of the distances (top, left, right, bottom).
    @param results_path: path to the whole dataset, where the map will be saved
    @param full_plate_list: list of data path ("./*traj.csv" format)
    @return: None, saves a csv
    """
    print(">>> Computing average reference point distance for all conditions...")
    all_conditions_list = param.nb_to_name.keys()

    if not os.path.isdir(results_path + "perfect_heatmaps"):
        os.mkdir(results_path + "perfect_heatmaps")

    average_distance = {"condition": [], "avg_ref_points_distance": []}
    distances_each_condition = [[] for _ in range(len(all_conditions_list))]
    distances_each_condition_top = []
    distances_each_condition_left = []
    distances_each_condition_right = []
    distances_each_condition_bottom = []

    condition_names = []
    condition_colors = []
    for i_condition, condition in enumerate(all_conditions_list):
        if i_condition % 3 == 0:
            print(">>>>>> Condition ", i_condition, " / ", len(all_conditions_list))
        # Compute average radius from a few plates of this condition
        plates_this_condition = fd.return_folders_condition_list(full_plate_list, condition)
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
                distances_each_condition[i_condition].append(np.mean([left_dist, top_dist, right_dist, bottom_dist]))
                distances_each_condition_top.append(top_dist)
                distances_each_condition_left.append(left_dist)
                distances_each_condition_right.append(right_dist)
                distances_each_condition_bottom.append(bottom_dist)
        average_distance["condition"].append(condition)
        average_distance["avg_ref_points_distance"].append(np.nanmean(distances_each_condition[i_condition]))
        condition_names.append(param.nb_to_name[condition])
        condition_colors.append(param.name_to_color[param.nb_to_name[condition]])

    fig, [ax0, ax1] = plt.subplots(1, 2)
    # Boxplot with values for each condition
    ax0.boxplot(distances_each_condition)
    ax0.set_xticks(range(1, len(all_conditions_list) + 1), condition_names, rotation=45)
    ax0.set_title("Reference point distance for each condition")

    # Boxplot with one box for top edge, left edge, etc.
    ax1.boxplot([distances_each_condition_left, distances_each_condition_top, distances_each_condition_right,
                 distances_each_condition_bottom])
    ax1.set_xticks([1, 2, 3, 4], ["Left", "Top", "Right", "Bottom"])
    ax1.set_title("Reference point distance for each edge")
    plt.show()
    pd.DataFrame(average_distance).to_csv(
        results_path + "perfect_heatmaps/average_reference_points_distance_each_condition.csv")


def idealized_patch_centers_mm(results_path, full_plate_list, output_frame_size):
    print(">>> Computing patch positions...")
    all_conditions_list = param.nb_to_name.keys()

    # Load the average distance between reference points
    if not os.path.isfile(results_path + "perfect_heatmaps/average_reference_points_distance_each_condition.csv"):
        if len(full_plate_list) == 1:
            print("Run compute_average_ref_points_distance() on all the plates first!!!")
            return 0
        compute_average_ref_points_distance(results_path, full_plate_list)
    ref_points = pd.read_csv(results_path + "perfect_heatmaps/average_reference_points_distance_each_condition.csv")
    average_ref_points_distance = np.mean(ref_points["avg_ref_points_distance"])

    # Deduce reference points from that
    margin = (output_frame_size - average_ref_points_distance) / 2
    bottom_left = [margin, margin]
    bottom_right = [output_frame_size - margin, margin]
    top_left = [margin, output_frame_size - margin]
    top_right = [output_frame_size - margin, output_frame_size - margin]

    patch_centers_each_cond = {}
    robot_xy_each_cond = param.distance_to_xy
    for i_condition, condition in enumerate(all_conditions_list):
        small_ref_points = ReferencePoints.ReferencePoints([[-20, 20], [20, 20], [20, -20], [-20, -20]])
        big_ref_points = ReferencePoints.ReferencePoints([bottom_left, bottom_right, top_left, top_right])
        robot_xy = np.array(robot_xy_each_cond[param.nb_to_distance[condition]])
        robot_xy[:, 0] = - robot_xy[:, 0]
        patch_centers_each_cond[condition] = big_ref_points.mm_to_pixel(small_ref_points.pixel_to_mm(robot_xy))
        # For superfar distance patches are mirrored on the x-axis??? idk why but fix that
        if len(patch_centers_each_cond[condition]) == 3:
            patch_centers_to_fix = patch_centers_each_cond[condition]
            y_max = np.max([patch_centers_to_fix[0][1], patch_centers_to_fix[1][1], patch_centers_to_fix[2][1]])
            y_min = np.min([patch_centers_to_fix[0][1], patch_centers_to_fix[1][1], patch_centers_to_fix[2][1]])
            y_extent = y_max - y_min
            for i_patch in range(len(patch_centers_to_fix)):
                patch_centers_to_fix[i_patch][1] = y_min + (y_extent - (patch_centers_to_fix[i_patch][1] - y_min))
            patch_centers_each_cond[condition] = patch_centers_to_fix
    return patch_centers_each_cond


def experimental_to_perfect_pixel_indices(folder_to_save, polar_map, ideal_patch_centers,
                                          ideal_patch_radius, frame_size, collapse_all_patches=False):
    """
    Function that converts pixel coordinates in the experimental plates to the equivalent ones in a "perfect"
    environment (where patches are perfectly round), conserving the closest patch, the distance to the patch boundary,
    and the angular coordinate with respect to the patch center.
    @param folder_to_save: where to save the resulting npy
    @param polar_map: a map containing, for each pixel of the image, a list containing [index of the closest patch,
                      distance to the patch boundary, angular coordinate (rad) with respect to the closest patch center].
    @param ideal_patch_centers: coordinates of the ideal patch centers [[x0, y0], [x1, y1], ...].
    @param ideal_patch_radius: average radius of patches.
    @param collapse_all_patches: if set to TRUE, will give indices to collapse everything on a single patch, in the center of the plate!!!
    @return: saves a matrix with the same size as polar_map, in folder_to_save, named "xp_to_perfect.npy"
             with, for each cell, the corresponding [x,y] in the "perfect" landscape.
    """
    nb_of_lines = len(polar_map)
    nb_of_col = len(polar_map[0])

    # That's the computation for each cell in the polar map:
    # i_patch, distance_boundary, angular_coord = polar_map[i_line, i_col]
    # new_x = (ideal_patch_radius + distance_boundary) * np.cos(angular_coord) + ideal_patch_centers[int(i_patch)][0]
    # new_y = (ideal_patch_radius + distance_boundary) * np.sin(angular_coord) + ideal_patch_centers[int(i_patch)][1]

    # I do it directly on numpy arrays
    # First, initialize the arrays
    closest_patch_index = polar_map[:, :, 0]
    distance_to_boundary_matrix = polar_map[:, :, 1]
    angular_coordinates_matrix = polar_map[:, :, 2]
    if collapse_all_patches:
        column_shift_matrix = frame_size // 2
        row_shift_matrix = frame_size // 2
    else:
        column_shift_matrix = np.array(
            [list(map(lambda x: ideal_patch_centers[int(x)][0], y)) for y in closest_patch_index])
        row_shift_matrix = np.array(
            [list(map(lambda x: ideal_patch_centers[int(x)][1], y)) for y in closest_patch_index])

    # Then, array operations
    perfect_x = (ideal_patch_radius + distance_to_boundary_matrix) * np.cos(
        angular_coordinates_matrix) + column_shift_matrix
    perfect_y = (ideal_patch_radius + distance_to_boundary_matrix) * np.sin(
        angular_coordinates_matrix) + row_shift_matrix

    perfect_x = np.rint(perfect_x)
    perfect_y = np.rint(perfect_y)

    # # Debug for one of those bloody patches
    # x_indices = np.where(column_shift_matrix == np.unique(column_shift_matrix)[6])
    # y_indices = np.where(row_shift_matrix == np.unique(row_shift_matrix)[4])
    # # Fancy way to select coordinates that are in both x_indices and y_indices
    # array1 = np.stack((x_indices[0], x_indices[1]), axis=1)
    # array2 = np.stack((y_indices[0], y_indices[1]), axis=1)
    # n_rows, n_cols = array1.shape
    # dtype = {'names': ['f{}'.format(i) for i in range(n_cols)],
    #          'formats': n_cols * [array1.dtype]}
    # unique_couples = np.intersect1d(array1.view(dtype), array2.view(dtype))
    # unique_couples = unique_couples.view(array1.dtype).reshape(-1, n_cols)
    #
    # colors = plt.cm.jet(np.linspace(0, 1, 10))
    # for i_distance, distance in enumerate([-4, -3, -2, -1, 0, 1, 2, 3, 4]):
    #     print("distance ", distance)
    #     for i_coord, coord in enumerate(unique_couples):
    #         i_line = coord[0]
    #         i_col = coord[1]
    #         if int(distance_to_boundary_matrix[i_line, i_col]) == distance:
    #             plt.plot([perfect_x[i_line, i_col], i_col], [perfect_y[i_line, i_col], i_line], color=colors[i_distance], alpha=0.2)
    #             plt.scatter([perfect_x[i_line, i_col]], [perfect_y[i_line, i_col]], s=200, color=colors[i_distance], marker="*")
    #             plt.scatter([i_col], [i_line], s=200, color=colors[i_distance])
    #     plt.scatter([], [], color=colors[i_distance], label=str(distance))
    # plt.scatter(np.where(distance_to_boundary_matrix == 0)[1], np.where(distance_to_boundary_matrix == 0)[0], color="grey")
    # plt.xlim(849, 856)
    # plt.ylim(990, 993)
    # plt.legend()
    # plt.show()

    # Stack all of those so that you get the array with for each pixel [x, y]
    experimental_to_perfect = np.stack((perfect_y, perfect_x), axis=2)
    # Clip them so that their values are not too high
    experimental_to_perfect = np.clip(experimental_to_perfect, 0, frame_size - 1)

    if not collapse_all_patches:
        np.save(folder_to_save + "xp_to_perfect.npy", experimental_to_perfect.astype(int))
    else:
        np.save(folder_to_save + "xp_to_perfect_collapsed.npy", experimental_to_perfect.astype(int))


def load_pixel_visits(list_of_time_stamps, plate, regenerate=False, return_durations=True):
    # If it's not already done, or has to be redone, compute the pixel visit durations
    pixelwise_visits_path = plate[:-len("traj.csv")] + "pixelwise_visits.npy"
    if not os.path.isfile(pixelwise_visits_path) or regenerate:
        gr.generate_pixelwise_visits(list_of_time_stamps, plate)
    # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
    pixel_wise_visits = np.load(pixelwise_visits_path, allow_pickle=True)
    if return_durations:
        # Convert all pixel visits to durations
        visit_durations = [list(map(ana.convert_to_durations, y)) for y in pixel_wise_visits]
        visit_durations = [list(map(np.sum, y)) for y in visit_durations]
        return visit_durations
    else:
        return pixel_wise_visits


def load_avg_pixel_speed(traj, plate, regenerate):
    # If it's not already done, or has to be redone, compute the pixel visit durations
    pixelwise_speed_path = plate[:-len("traj.csv")] + "pixelwise_speeds.npy"
    if not os.path.isfile(pixelwise_speed_path) or regenerate:
        generate_pixelwise_speeds(traj, plate)
    # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
    pixelwise_speeds = np.load(pixelwise_speed_path, allow_pickle=True)
    cells_with_values = np.where(pixelwise_speeds)
    pixelwise_avg_speeds = np.empty((len(pixelwise_speeds), len(pixelwise_speeds[0])))
    pixelwise_avg_speeds.fill(np.nan)
    for i_cell in range(len(cells_with_values[0])):
        line, column = cells_with_values[0][i_cell], cells_with_values[1][i_cell]
        pixelwise_avg_speeds[line][column] = np.nanmean(pixelwise_speeds[line][column])
    return pixelwise_avg_speeds


def load_short_pixel_visits(traj, plate, regenerate=False):
    return 0


def plot_heatmap(results_path, traj, full_plate_list, curve_list, variable="pixel_visits",
                 regenerate_pixel_values=False,
                 regenerate_polar_maps=False, regenerate_perfect_map=False,
                 frame_size=1847, collapse_patches=False, show_plot=True):
    """
    Plots a heatmap with color that varies according to a variable that can be inputted as a string parameter.
    In order to maintain patch structure across all plates, will plot this heatmap in an idealized landscape,
    where patches are perfectly round and equidistant.
    Will compute the average of that variable for each pixel, dividing the value by the number of pixels that got mapped
    in this idealized pixel (to avoid biases linked to the idealization).
    Will save the resulting heatmap to save computing cost.
    @param show_plot:
    @param results_path:
    @param traj:
    @param full_plate_list:
    @param curve_list:
    @param curve_names:
    @param variable:
    @param regenerate_pixel_values:
    @param regenerate_polar_maps:
    @param regenerate_perfect_map:
    @param frame_size:
    @param collapse_patches:
    @return:
    """
    # Plot initialization
    fig, axes = plt.subplots(1, len(curve_list))
    curve_names = [param.nb_list_to_name[str(curve)] for curve in curve_list]
    if variable == "pixel_visits":
        color_map = "viridis"
        fig.suptitle("Heatmap of worm presence")
    if variable == "speed":
        color_map = "plasma"
        fig.suptitle("Heatmap of worm speed")
    heatmap_each_curve = [np.zeros((frame_size, frame_size)) for _ in range(len(curve_list))]
    counts_each_curve = [np.zeros((frame_size, frame_size)) for _ in range(len(curve_list))]

    # If it's not already done, compute the average patch radius
    if not os.path.isfile(results_path + "perfect_heatmaps/average_patch_radius_each_condition.csv"):
        generate_average_patch_radius_each_condition(results_path, full_plate_list)
    average_patch_radius_each_cond = pd.read_csv(
        results_path + "perfect_heatmaps/average_patch_radius_each_condition.csv")
    average_radius = np.mean(average_patch_radius_each_cond["avg_patch_radius"])

    # Compute the idealized patch positions by converting the robot xy data to mm in a "perfect" reference frame
    ideal_patch_centers_each_cond = idealized_patch_centers_mm(results_path, full_plate_list, frame_size)

    tic = time.time()
    for i_curve in range(len(curve_list)):
        print(int(time.time() - tic), "s: Curve ", i_curve, " / ", len(curve_list))
        plate_list, condition_each_plate = fd.return_folders_condition_list(full_plate_list, curve_list[i_curve],
                                                                            return_conditions=True)
        for i_plate, plate in enumerate(plate_list):
            print(">>> ", int(time.time() - tic), "s: plate ", i_plate, " / ", len(plate_list))

            # Correct the times for plates that have only NaNs or jumps in the time
            current_traj = traj[dt.f.folder == plate, :]
            corrected_times = fd.correct_time_stamps(current_traj.to_pandas(), True)["time"]
            current_traj[:, dt.f.time] = corrected_times
            list_of_time_stamps = current_traj["time"].to_list()[0]

            # Perfect index matrix path
            if not collapse_patches:
                xp_to_perfect_path = plate[:-len("traj.csv")] + "xp_to_perfect.npy"
            else:
                xp_to_perfect_path = plate[:-len("traj.csv")] + "xp_to_perfect_collapsed.npy"

            # If it's not already done, or has to be redone, compute the experimental to perfect mapping
            if not os.path.isfile(xp_to_perfect_path) or regenerate_perfect_map:
                print(">>>>>> Generating perfect coordinates...")
                # Then, if it's not already done, or has to be redone, compute the polar coordinates for the plate
                polar_map_path = plate[:-len("traj.csv")] + "polar_map.npy"
                if not os.path.isfile(polar_map_path) or regenerate_polar_maps:
                    generate_polar_map(plate)
                current_polar_map = np.load(polar_map_path)

                # Load patch centers and radiuses
                current_condition = condition_each_plate[i_plate]
                current_patch_centers = ideal_patch_centers_each_cond[current_condition]

                # Then FINALLY convert the current pixel wise visits to their "perfect" equivalent (in an environment with
                # perfectly round patches, while conserving distance to border and angular coordinate w/ respect to center)
                experimental_to_perfect_pixel_indices(plate[:-len("traj.csv")], current_polar_map,
                                                      current_patch_centers,
                                                      average_radius, frame_size=frame_size,
                                                      collapse_all_patches=collapse_patches)
            # Matrix with, in each cell, the corresponding "perfect" coordinates
            xp_to_perfect_indices = np.load(xp_to_perfect_path)

            print(">>>>>> Loading pixelwise values...")
            if variable == "pixel_visits":
                values_each_pixel = load_pixel_visits(list_of_time_stamps, plate, regenerate=regenerate_pixel_values)
            if variable == "short_pixel_visits":
                values_each_pixel = load_short_pixel_visits(current_traj, plate, regenerate=regenerate_pixel_values)
            if variable == "speed":
                values_each_pixel = load_avg_pixel_speed(current_traj, plate, regenerate_pixel_values)

            print(">>>>>> Putting the values in the perfect plate...")
            # For each pixel of the perfect plate, load the visits that correspond to them in the experimental plates
            for i in range(len(values_each_pixel)):
                for j in range(len(values_each_pixel[i])):
                    current_values = values_each_pixel[i][j]
                    if not np.isnan(current_values):
                        perfect_i, perfect_j = xp_to_perfect_indices[i][j]
                        heatmap_each_curve[i_curve][perfect_i][perfect_j] += values_each_pixel[i][j]
                        counts_each_curve[i_curve][perfect_i][perfect_j] += 1

        # Divide the values by the number of pixels that went there
        heatmap_each_curve[i_curve] = ana.array_division_ignoring_zeros(heatmap_each_curve[i_curve],
                                                                        counts_each_curve[i_curve])

        if len(curve_list) > 1:
            # For mixed densities, plot the 0.5 patch centers in orange
            if "+" in curve_names[i_curve]:
                metadata = fd.folder_to_metadata(plate)
                for i_patch in range(len(ideal_patch_centers_each_cond[curve_list[i_curve][0]])):
                    axes[i_curve].scatter(ideal_patch_centers_each_cond[curve_list[i_curve][0]][i_patch][1],
                                          ideal_patch_centers_each_cond[curve_list[i_curve][0]][i_patch][0],
                                          color="white")
                    if metadata["patch_densities"][i_patch][0] == 0.5:
                        axes[i_curve].scatter(ideal_patch_centers_each_cond[curve_list[i_curve][0]][i_patch][1],
                                              ideal_patch_centers_each_cond[curve_list[i_curve][0]][i_patch][0],
                                              color="orange")

        heatmap_each_curve[i_curve] /= np.sum(heatmap_each_curve[i_curve])

        # Save iiitttt
        heatmap_path = results_path + "perfect_heatmaps/" + variable + "_heatmap_cond_" + str(
            curve_list[i_curve]) + collapse_patches * "_collapsed" + ".npy"
        np.save(heatmap_path, heatmap_each_curve[i_curve])
        counts_path = results_path + "perfect_heatmaps/" + variable + "_heatmap_counts_cond_" + str(
            curve_list[i_curve]) + collapse_patches * "_collapsed" + ".npy"
        np.save(counts_path, counts_each_curve[i_curve])

        if show_plot:
            if len(curve_list) == 1:
                plt.imshow(heatmap_each_curve[i_curve].astype(float), cmap=color_map, vmax=0.00001)
                plt.title(curve_names[i_curve])
                plt.colorbar()
            else:
                axes[i_curve].imshow(heatmap_each_curve[i_curve].astype(float), cmap=color_map, vmax=0.000001)
                axes[i_curve].set_title(curve_names[i_curve])

    if show_plot:
        plt.show()
        print("")


def plot_existing_heatmap(condition_list, variable, v_min=0, v_max=1):
    heatmap_path = "/media/admin/T7 Shield/Results_minipatches_retracked_shortened/perfect_heatmaps/" + variable + "_heatmap_cond_" + str(
        condition_list) + ".npy"
    if not os.path.isfile(heatmap_path):
        print("This heatmap does not exist! è_é")
    else:
        heatmap = np.load(heatmap_path)
        plt.imshow(heatmap, vmin=v_min, vmax=v_max, cmap="plasma")
        plt.title(heatmap_path[-13:-4])
        plt.xlim(250, 1600)
        plt.ylim(250, 1600)
        plt.title(str([param.nb_to_name[c] for c in condition_list]) + ", v_max=" + str(v_max))
        plt.colorbar()
        plt.show()


def plot_distance_map_and_patches(results_path, plate):
    # Load the matrix with patch to which each pixel belongs
    in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
    in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()

    # Load the matrix with distance to the closest food patch boundary for each pixel
    distance_map_path = plate[:-len(plate.split("/")[-1])] + "distance_to_patch_map.csv"
    if not os.path.isdir(distance_map_path):
        script_distance.generate_patch_distance_map(in_patch_matrix, plate)
    distance_map = np.load(plate[:-len(plate.split("/")[-1])] + "distance_to_patch_map.npy")

    # Load patch centers and boundaries
    patch_centers = fd.folder_to_metadata(plate)["patch_centers"]
    patch_boundaries = plots.patches(plate, show_composite=False, is_plot=False)
    patch_x = [patch_centers[i][0] for i in range(len(patch_centers))]
    patch_y = [patch_centers[i][1] for i in range(len(patch_centers))]

    # Load theoretical patch centers
    ideal_patch_centers_each_cond = idealized_patch_centers_mm(results_path, [plate], len(distance_map))
    ideal_patch_centers = ideal_patch_centers_each_cond[fd.load_condition(plate)]

    # Load the average patch radius
    average_patch_radius_each_cond = pd.read_csv(
        results_path + "perfect_heatmaps/average_patch_radius_each_condition.csv")
    average_radius = np.mean(average_patch_radius_each_cond["avg_patch_radius"])

    plt.title(str(plate[-48:-9]))
    plt.imshow(distance_map)
    plt.scatter(patch_boundaries[0], patch_boundaries[1], color="yellow")
    plt.scatter(patch_x, patch_y, color="yellow", label="experimental patches")
    plt.scatter(ideal_patch_centers[:, 0], ideal_patch_centers[:, 1], color="white", label="ideal patches")
    for i_patch in range(len(ideal_patch_centers)):
        circle = plt.Circle(ideal_patch_centers[i_patch], average_radius, color="white", fill=False)
        plt.gca().add_patch(circle)
    plt.show()


if __name__ == "__main__":
    # 0
    #plot_existing_heatmap([17], "speed", v_max=0.000002)
    #plot_existing_heatmap([18], "speed", v_max=0.000002)
    #plot_existing_heatmap([19], "speed", v_max=0.000002)
    #plot_existing_heatmap([20], "speed", v_max=0.000002)
    # 0.2
    #plot_existing_heatmap([0], "speed", v_max=0.000002)
    #plot_existing_heatmap([1], "speed", v_max=0.000002)
    #plot_existing_heatmap([2], "speed", v_max=0.000002)
    #plot_existing_heatmap([14], "speed", v_max=0.000002)
    # 0.5
    #plot_existing_heatmap([4], "speed", v_max=0.000002)
    #plot_existing_heatmap([5], "speed", v_max=0.000002)
    #plot_existing_heatmap([6], "speed", v_max=0.000002)
    #plot_existing_heatmap([15], "speed", v_max=0.000002)
    # 1.25
    #plot_existing_heatmap([12], "speed", v_max=0.000002)
    #plot_existing_heatmap([8], "speed", v_max=0.000002)
    #plot_existing_heatmap([13], "speed", v_max=0.000002)
    #plot_existing_heatmap([16], "speed", v_max=0.000002)

    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(starting_from="", shorten_traj=True)
    #path2 = gen.generate(shorten_traj=False)

    #plate = path + '20221011T111254_SmallPatches_C3-CAM3/traj.csv'
    #plate2 = path2 + '20221011T111254_SmallPatches_C3-CAM3/traj.csv'
    traj = dt.fread(path + "clean_trajectories.csv")
    #traj2 = dt.fread(path2 + "clean_trajectories.csv")
    #current_traj = traj[dt.f.folder == plate, :]
    #current_traj2 = traj2[dt.f.folder == plate2, :]
    #speeds = generate_pixelwise_speeds(current_traj, plate)
    #speeds2 = generate_pixelwise_speeds(current_traj2, plate2)

    #print("")

    #plot_existing_heatmap(path + "perfect_heatmaps/pixel_visits_heatmap_conditions_[0].npy", path + "perfect_heatmaps/pixel_visits_heatmap_conditions_[12].npy", v_max=10)

    results = pd.read_csv(path + "clean_results.csv")
    trajectories = dt.fread(path + "clean_trajectories.csv")
    full_list_of_folders = list(results["folder"])
    if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
        full_list_of_folders.remove(
            "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")

    #compute_average_ref_points_distance(path, full_list_of_folders)

    #compute_average_ref_points_distance(path, full_list_of_folders)
    #generate_average_patch_radius_each_condition(path, full_list_of_folders)
    #plot_distance_map_and_patches(path, full_list_of_folders[11])

    #import cProfile
    #import pstats

    #profiler = cProfile.Profile()
    #profiler.enable()

    list_by_distance = [17, 0, 4, 12, 18, 1, 9, 5, 10, 8, 19, 2, 6, 13, 14, 15, 20, 16, 21, 3, 7]
    list_by_density = [17, 18, 19, 20, 21, 0, 1, 2, 14, 3, 4, 5, 6, 15, 7, 12, 8, 13, 16, 9, 10]

    # plot_heatmap(path, trajectories, full_list_of_folders, [[17], [18], [19], [20], [21]], variable="speed",
    #              regenerate_pixel_values=True, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #              collapse_patches=False, show_plot=False)
    plot_heatmap(path, trajectories, full_list_of_folders, [[14]], variable="speed",
                 regenerate_pixel_values=True, regenerate_polar_maps=True, regenerate_perfect_map=True,
                 collapse_patches=False, show_plot=False)
    # plot_heatmap(path, trajectories, full_list_of_folders, [[0], [1], [2], [3]], variable="speed",
    #              regenerate_pixel_values=True, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #              collapse_patches=False, show_plot=False)
    # plot_heatmap(path, trajectories, full_list_of_folders, [[4], [5], [6], [15], [7]], variable="speed",
    #              regenerate_pixel_values=True, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #              collapse_patches=False, show_plot=False)
    # plot_heatmap(path, trajectories, full_list_of_folders, [[12], [8], [13], [16]], variable="speed",
    #              regenerate_pixel_values=True, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #              collapse_patches=False, show_plot=False)
    # plot_heatmap(path, trajectories, full_list_of_folders, [[9], [10]], variable="speed",
    #              regenerate_pixel_values=True, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #              collapse_patches=False, show_plot=False)

    # Pixel visits
    #plot_heatmap(path, trajectories, full_list_of_folders, [[17], [18], [19], [20], [21]], variable="pixel_visits",
    #             regenerate_pixel_values=False, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #             collapse_patches=False, show_plot=False)
    #plot_heatmap(path, trajectories, full_list_of_folders, [[0], [1], [2], [14], [3]], variable="pixel_visits",
    #             regenerate_pixel_values=False, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #             collapse_patches=False, show_plot=False)
    #plot_heatmap(path, trajectories, full_list_of_folders, [[1], [2], [14], [3]], variable="pixel_visits",
    #             regenerate_pixel_values=False, regenerate_polar_maps=False, regenerate_perfect_map=False,
    #             collapse_patches=False, show_plot=False)
    plot_heatmap(path, trajectories, full_list_of_folders, [[4], [5], [6], [15], [7]], variable="pixel_visits",
                 regenerate_pixel_values=False, regenerate_polar_maps=False, regenerate_perfect_map=False,
                 collapse_patches=False, show_plot=False)
    plot_heatmap(path, trajectories, full_list_of_folders, [[12], [8], [13], [16]], variable="pixel_visits",
                 regenerate_pixel_values=False, regenerate_polar_maps=False, regenerate_perfect_map=False,
                 collapse_patches=False, show_plot=False)
    plot_heatmap(path, trajectories, full_list_of_folders, [[9], [10]], variable="pixel_visits",
                 regenerate_pixel_values=False, regenerate_polar_maps=False, regenerate_perfect_map=False,
                 collapse_patches=False, show_plot=False)

#profiler.disable()
#stats = pstats.Stats(profiler).sort_stats('cumtime')
#stats.print_stats()
