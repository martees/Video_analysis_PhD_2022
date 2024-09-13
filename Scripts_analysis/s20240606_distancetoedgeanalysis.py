# In this script, I will look at speed and visit time to pixels as a function of the distance to the edge of a patch

from scipy import ndimage
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datatable as dt

from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Parameters import parameters as param
import find_data as fd
import analysis as ana
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script


def generate_patch_distance_map(in_patch_matrix, folder_to_save):
    """
    Function that saves a map with for each pixel the distance to the closest food patch boundary, with positive values
    outside food patches, and negative values inside (and boundary = 0).
    @param in_patch_matrix: a numpy array containing for each pixel the patch to which it belongs (-1 for outside)
    @param folder_to_save: the folder where the output map should be saved
    @return: None
    """
    zeros_inside = np.zeros(in_patch_matrix.shape)
    zeros_outside = np.zeros(in_patch_matrix.shape)
    for i in range(len(in_patch_matrix)):
        for j in range(len(in_patch_matrix[i])):
            if in_patch_matrix[i, j] == -1:
                zeros_inside[i, j] = 1
                zeros_outside[i, j] = 0
            else:
                zeros_inside[i, j] = 0
                zeros_outside[i, j] = 1

    # Create a distance matrix with 0 inside food patches and distance to boundary outside
    distance_transform_outside = ndimage.distance_transform_edt(zeros_inside)
    # Create a distance matrix with 0 outside food patches and distance to boundary inside
    distance_transform_inside = ndimage.distance_transform_edt(zeros_outside)
    # Subtract them from one another so that distance to patch boundary is positive outside, negative inside
    # Add 1 to make inside (negative) distances go to zero at boundary
    # And then remove 1 outside otherwise there'd be no 1 in the distances
    distance_transform = distance_transform_outside - distance_transform_inside + 1 - zeros_inside

    np.save(folder_to_save[:-len(folder_to_save.split("/")[-1])] + "distance_to_patch_map.npy", distance_transform)


def pixel_visits_vs_distance_to_boundary(folder_list, traj, bin_list, variable="Total"):
    visit_values_each_bin_each_plate = np.zeros((len(bin_list), len(folder_list)))
    visit_values_each_bin_each_plate[:] = np.nan
    for i_folder, folder in enumerate(folder_list):
        print(">>> Folder ", i_folder, " / ", len(folder_list))
        # Correct the times for plates that have only NaNs or jumps in the time
        current_traj = traj[dt.f.folder == folder, :]
        corrected_times = fd.correct_time_stamps(current_traj.to_pandas(), True)["time"]
        current_traj[:, dt.f.time] = corrected_times

        # Load the distance map
        distance_map_path = folder[:-len(folder.split("/")[-1])] + "distance_to_patch_map.csv"
        if not os.path.isdir(distance_map_path):
            in_patch_matrix_path = folder[:-len("traj.csv")] + "in_patch_matrix.csv"
            in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()
            generate_patch_distance_map(in_patch_matrix, folder)
        distance_map = np.load(folder[:-len(folder.split("/")[-1])] + "distance_to_patch_map.npy")
        print(">>>>>> Loaded distance map!")

        if variable == "Total":
            pixel_wise_visits = heatmap_script.load_pixel_visits(current_traj[:, dt.f.time].to_list(),
                                                                 folder)
            print(">>>>>> Loaded pixel_wise visit durations!")
            # Make it a list
            current_folder_values = np.ravel(pixel_wise_visits)

        if variable == "Speed":
            pixel_wise_avg_speed = heatmap_script.load_avg_pixel_speed(current_traj, folder, regenerate=True)
            print(">>>>>> Loaded pixel_wise average speeds!")
            # Note: the pixels that were never visited are returned as NaN values by the load_avg_pixel_speed() function,
            #       so they are not taken into account in the averaging.
            current_folder_values = np.ravel(pixel_wise_avg_speed)

        # In all cases, linearize the distance map to match pixel_wise values (one value per pixel)
        current_folder_distances = np.ravel(distance_map)

        # Put all the gathered values in bins corresponding to the bin_List argument
        current_folder_distance_bins, current_folder_bin_values, [_, _], _ = ana.xy_to_bins(
            list(current_folder_distances),
            list(current_folder_values),
            bin_size=None,
            print_progress=False,
            custom_bins=bin_list,
            do_not_edit_xy=False,
            compute_bootstrap=False)
        for i_bin, distance_bin in enumerate(current_folder_distance_bins):
            if distance_bin in bin_list:  # check this because ana.xy_to_bins() adds max value to the output bin list
                bin_index = np.where(np.asarray(bin_list) == distance_bin)[0]
                # Sometimes bin_index is a list with one element for some reason, if that's the case convert to int
                if type(bin_index.astype(int)) != int and len(bin_index) > 0:
                    bin_index = bin_index[0]
                visit_values_each_bin_each_plate[bin_index][i_folder] = current_folder_bin_values[i_bin]
    return visit_values_each_bin_each_plate


def plot_visit_duration_vs_distance(full_folder_list, traj, curve_names, bin_list, variable):
    """
    Function that will make a plot with the duration of visits as a function of distance to the closest patch boundary
    (negative distance => worm is inside the patch). Average is made over all pixels pooled together for each curve.
    @param full_folder_list:
    @param traj:
    @param curve_names:
    @param bin_list:
    @param variable:
    @return:
    """
    tic = time.time()
    curve_list = [param.name_to_nb_list[curve] for curve in curve_names]
    for i_curve, curve in enumerate(curve_list):
        print(int(time.time() - tic), "s: Curve ", i_curve + 1, " / ", len(curve_list))
        folder_list = fd.return_folders_condition_list(full_folder_list, curve)
        visit_values_each_bin_each_plate = pixel_visits_vs_distance_to_boundary(folder_list,
                                                                                traj, bin_list,
                                                                                variable=variable)

        # At this point, visit_values has one value per bin/folder in each cell => average and bootstrap all that
        nb_of_bins = len(bin_list)
        avg_each_bin = np.empty(nb_of_bins)
        avg_each_bin[:] = np.nan  # just a convenient way of having a table full of NaNs
        # Errors
        errors_inf = np.empty(nb_of_bins)
        errors_sup = np.empty(nb_of_bins)
        errors_inf[:] = np.nan
        errors_sup[:] = np.nan
        for i_bin in range(nb_of_bins):
            # Rename
            values_this_time_bin = visit_values_each_bin_each_plate[i_bin]
            # and remove nan values for bootstrapping
            values_this_time_bin = [values_this_time_bin[i] for i in range(len(values_this_time_bin)) if
                                    not np.isnan(values_this_time_bin[i])]
            if values_this_time_bin:
                current_avg = np.nanmean(values_this_time_bin)
                avg_each_bin[i_bin] = current_avg
                bootstrap_ci = ana.bottestrop_ci(values_this_time_bin, 1000)
                errors_inf[i_bin] = current_avg - bootstrap_ci[0]
                errors_sup[i_bin] = bootstrap_ci[1] - current_avg

        # Plot and add error bars
        condition_name = "OD = "+param.nb_to_density[curve[0]]
        if condition_name == "OD = 0":
            condition_name = "control"
        condition_color = param.name_to_color[param.nb_to_density[curve[0]]]
        plt.plot(bin_list, avg_each_bin, color=condition_color, label=condition_name, linewidth=3)
        plt.errorbar(bin_list, avg_each_bin, [errors_inf, errors_sup], fmt='.k', capsize=5)

    print("Total time: ", int((time.time() - tic) // 60), "min")

    if variable == "Total":
        plt.title("Total time in pixel as a function of distance to the edge of the patch in " + condition_name)
        plt.ylabel("Total visit time to pixel")
    if variable == "Speed":
        plt.title("Average centroid speed while in pixel as a function of distance to the edge of the patch in " + condition_name)
        plt.ylabel("Average centroid speed while in pixel")
    plt.xlabel("Distance to closest patch boundary ( <0 = inside)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(shorten_traj=True)
    results = pd.read_csv(path + "clean_results.csv")
    trajectories = dt.fread(path + "clean_trajectories.csv")
    full_list_of_folders = list(results["folder"])
    list_of_distance_bins = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    plot_visit_duration_vs_distance(full_list_of_folders, trajectories,
                                    ['med 0', 'med 0.2', 'med 0.5', 'med 1.25'], list_of_distance_bins,
                                    variable="Speed")
