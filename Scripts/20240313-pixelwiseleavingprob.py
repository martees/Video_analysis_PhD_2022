# This is a script that I start with the intention of connecting more precisely pixel-level depletion (using overlap with
# worm silhouette as a proxy) and instantaneous behavior (looking at whether the worm moves away from a pixel).
import copy

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mplcolors

import find_data as fd
import plots
from Generating_data_tables import main as gen
from Generating_data_tables import generate_trajectories as gt
import main
from Parameters import parameters
import analysis as ana


# PUT IT IN ANOTHER SCRIPT
def distance_to_closest_patch_map(folder):
    """
    Function that takes a folder leading to a traj.csv file, containing a trajectory, and in a folder containing a matrix
    with the patch location of every pixel in the image (so -1 if outside patches, and otherwise the patch number).
    """


def pixel_wise_visit_durations(folder):
    """
    Function that takes a folder containing a time series of silhouettes, and returns a list of lists with the dimension
    of the plate in :folder:, and in each cell, a list with the duration of the successive visits to this pixel.
    (a visit starts when a pixel of the worm overlaps with the pixel, and ends when this overlap stops)
    When this function is called, it also saves this output under the name "pixelwise_visit_durations.npy" in folder.
    """
    # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    pixels, intensities, frame_size = fd.load_silhouette(folder)
    pixels = fd.reindex_silhouette(pixels, frame_size)

    # Create a table with a list containing, for each pixel in the image, a sublist with the duration of visits
    # to this pixel. In the following algorithm, when the last element of a sublist is -1, it means that the pixel
    # was not being visited at the previous time point.
    # We start by creating an array with one sublist per pixel, each sublist only containing -1 in the beginning
    visit_times_each_pixel = [[[-1] for _ in range(frame_size[0])] for _ in range(frame_size[1])]
    # For each time point, create visits in pixels that just started being visited, continue those that have already
    # started, and end those that are finished
    for j_time in range(len(pixels)):
        current_visited_pixels = pixels[j_time]
        for i_pixel in range(len(current_visited_pixels[0])):
            current_pixel = [current_visited_pixels[0][i_pixel], current_visited_pixels[1][i_pixel]]
            # If visit just started, start it
            if visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] == -1:
                visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] = 1
            # If visit is continuing, increment time spent
            else:
                visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] += 1
        # Then, close the visits of the previous time step that are not being continued
        if j_time > 0:
            previous_visited_pixels = pixels[j_time - 1]
            for i_pixel in range(len(previous_visited_pixels[0])):
                current_pixel = [previous_visited_pixels[0][i_pixel], previous_visited_pixels[1][i_pixel]]
                # If one of this current pixel's coordinates is not in the current visited pixels, then close the visit
                if current_pixel[0] not in current_visited_pixels[0] or current_pixel[1] not in current_visited_pixels[
                    1]:
                    visit_times_each_pixel[current_pixel[1]][current_pixel[0]].append(-1)
                # Else, both are here, so check that they are for the same point
                else:
                    if True not in np.logical_and(np.array(current_visited_pixels[0]) == current_pixel[0],
                                                  np.array(current_visited_pixels[0]) == current_pixel[0]):
                        visit_times_each_pixel[current_pixel[1]][current_pixel[0]].append(-1)

    # Remove the -1 because they were only useful for the previous algorithm
    for j_line in range(len(visit_times_each_pixel)):
        for i_column in range(len(visit_times_each_pixel[j_line])):
            if visit_times_each_pixel[j_line][i_column][-1] == -1:
                visit_times_each_pixel[j_line][i_column] = visit_times_each_pixel[j_line][i_column][:-1]

    np.save(folder[:-len("traj.csv")] + "pixelwise_visit_durations.npy", np.array(visit_times_each_pixel, dtype=object))

    return visit_times_each_pixel


def pixel_wise_delay_analysis(condition_list, is_recompute_pixelwise_visits=False, is_recompute_full_data_table=False,
                              exclude_first_visits=False):
    """
    = Function that takes a list of condition numbers (as defined in ./Parameters/parameters.py), and will return a list
    of matrices, one for each condition. This matrix will contain 4001 lines and 4001 columns, corresponding to a time
    already spent in the pixel and to the delay before next exit, respectively.
    Each cell of a matrix contains the number of events for the corresponding time and delay.
    This function only looks at pixels inside food patches!

    = Example: for condition_list = [4, 5, 6], the output will be a list of 3 matrices. If in line 10, column 3, of the
            1st matrix contains 100, it means that in plates of condition 4, it happened 100 times that there was a
            pixel that had been visited for 10 time steps already, and would be left 3 time steps later.
    = This function also saves each matrix of each condition in the folder pixelwise_analysis, in the path defined in
    ./Generate_data_tables/main.py, under the name pixelwise_delay_analysis_X.npy, where X is the condition number.

    @param condition_list: list of condition numbers (as defined in ./Parameters/parameters.py).
    @param is_recompute_pixelwise_visits: if TRUE, call pixel_wise_visit_durations for each folder. If FALSE, load the
    existing pixel_wise_visit_durations.npy file in each folder.
    @param is_recompute_full_data_table: if FALSE, load the pixelwise_delay_analysis_X.npy file saved in
    ./pixelwise_analysis for each condition. If True, recompute that from the pixelwise visit durations of each folder.

    """
    global tic  # Load timer info

    # Look at delay vs. time spent in pixels, for pixels inside food patches
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(test_pipeline=False)
    results = pd.read_csv(path + "clean_results.csv")

    if not os.path.isdir(path + "pixelwise_analysis"):
        os.mkdir(path + "pixelwise_analysis")

    full_folder_list = pd.unique(results["folder"])
    list_of_matrices = []
    for condition in condition_list:
        print(int(time.time() - tic), ": Computing pixelwise analysis for condition ", condition, " / ",
              len(condition_list), "...")
        current_folder_list = fd.return_folders_condition_list(full_folder_list, condition)
        if is_recompute_full_data_table:

            # We create a matrix to store the data, with each line corresponding to a time in pixel, each column to a
            # delay until next exit, and the value in the cell, to the number of points in this delay / time couple
            # We treat visits
            data_matrix = [[0 for _ in range(4001)] for _ in range(4001)]

            tic2 = time.time()
            for i_folder in range(len(current_folder_list)):
                # Print progress info
                if i_folder % 1 == 0:
                    print(int(time.time() - tic), ": > Folder ", i_folder, " / ", len(current_folder_list), "...")
                if i_folder % 10 == 4:
                    curr_t = (time.time() - tic2) / 60
                    print(int(time.time() - tic), ": > Current run time: ", curr_t,
                          " minutes, estimated time left for this condition: ",
                          (curr_t / i_folder) * (len(current_folder_list) - i_folder), " min.")
                # Compute or load the data
                if is_recompute_pixelwise_visits == True:
                    visits_to_pixels_matrix = pixel_wise_visit_durations(current_folder_list[i_folder])
                else:
                    visits_to_pixels_matrix = np.load(
                        current_folder_list[i_folder][:-len("traj.csv")] + "pixelwise_visit_durations.npy",
                        allow_pickle=True)
                # Load patch info for this folder
                in_patch_matrix_path = current_folder_list[i_folder][:-len("traj.csv")] + "in_patch_matrix.csv"
                if os.path.isfile(in_patch_matrix_path):
                    in_patch_matrix = pd.read_csv(in_patch_matrix_path)
                else:
                    in_patch_matrix = gt.in_patch_all_pixels(current_folder_list[i_folder])
                # Fill the data matrix with pixels only from inside the patches
                # We want to go from a matrix containing list of visits to each pixel, to a matrix containing, in each cell
                # t, d, the number of events with this time and delay. In order to do so, we go through the visits to each
                # pixel, and add +1 to all relevant cells of the matrix.
                # Note: all values with time or delay > 4000 are stored in the last line / column of the data matrix.
                for i_line in range(len(visits_to_pixels_matrix)):
                    for i_col in range(len(visits_to_pixels_matrix[0])):
                        # Only take into account pixels that are inside food patches
                        if in_patch_matrix[str(i_col)][i_line] != -1:  # str(i_col) because the matrix is a pandas
                            current_visit_durations = visits_to_pixels_matrix[i_line][i_col]
                            time_already_spent_in_patch = 0
                            start_index = int(exclude_first_visits)  # start at 1 if exclude_1st is TRUE, 0 if FALSE
                            for i_visit in range(start_index, len(current_visit_durations)):
                                if i_visit > 0:
                                    time_already_spent_in_patch += current_visit_durations[i_visit - 1]
                                current_visit_duration = current_visit_durations[i_visit]
                                for i_time in range(current_visit_duration):
                                    data_matrix[min(time_already_spent_in_patch + i_time, 4000)][
                                        min(current_visit_duration - i_time, 4000)] += 1
            print(int(time.time() - tic), ": > Saving data matrix...")
            np.save(path + "/pixelwise_analysis/pixelwise_delay_analysis_" + str(condition) + exclude_first_visits * "_exclude_1st_visit" + ".npy", data_matrix)

        else:
            data_matrix = np.load(
                path + "/pixelwise_analysis/pixelwise_delay_analysis_" + str(condition) + exclude_first_visits * "_exclude_1st_visit" + ".npy")
        list_of_matrices.append(data_matrix)
    return list_of_matrices


def plot_pixel_wise_leaving_probability(condition_list):
    """
    Function that takes a list of condition numbers (as defined in ./Parameters/parameters.py), and plots a leaving
    probability as a function of time already spent in pixel, for all the conditions of condition_list mashed together.
    If exclude_1st_visit is TRUE, then the algorithm will exclude first visits from the analysis. I am implementing this
    because we think that first visits are more numerous and only reproduce the visit duration distribution rather than
    the depletion signal that we are looking for.
    """

    global tic
    tic = time.time()

    # Load a list of matrices containing 4001 lines and 4001 columns, corresponding to a time already spent in the pixel
    # and to the delay before next exit, respectively. The list has one matrix by condition.
    list_of_delay_matrices = pixel_wise_delay_analysis(condition_list, False, False, exclude_first_visits=True)
    print(int(time.time() - tic), ": Starting to compute leaving probability...")
    # For every condition (aka every time vs. delay matrix)
    for i_condition in range(len(list_of_delay_matrices)):
        # Load data and exclude the rows which do not have enough data
        matrix_this_condition = list_of_delay_matrices[i_condition]
        nb_of_points_each_time = np.sum(matrix_this_condition, axis=1)  # make the sum of each row
        possible_times_spent_in_pixel = np.where(nb_of_points_each_time > 300)[
            0]  # only take times with > 300 datapoints
        # Initialize leaving probability list
        leaving_prob_each_time = [0 for _ in range(len(possible_times_spent_in_pixel))]
        # For every line of the matrix
        for i_time_spent, time_spent in enumerate(possible_times_spent_in_pixel):
            if i_time_spent % 400 == 0:
                print(int(time.time() - tic), "> Time ", i_time_spent, " / ", len(possible_times_spent_in_pixel))
            current_data = matrix_this_condition[time_spent]
            leaving_prob_each_time[i_time_spent] = current_data[1] / np.sum(current_data)

        # Some code to visualize the number of data points in each bin
        # colors = plt.cm.jet(np.linspace(0, 1, len(possible_times_spent_in_pixel)))
        # plt.scatter(possible_times_spent_in_pixel, leaving_prob_each_time, alpha=0.3,
        #            label=parameters.nb_to_name[condition_list[i_condition]], c=np.array(nb_of_points_each_time)[np.where(nb_of_points_each_time > 200)[0]],
        #            cmap="hot")
        #                color=parameters.name_to_color[parameters.nb_to_name[i_condition]])
        # plt.imshow(matrix_this_condition)

        # Bin the values
        bin_list, avg_list, [errors_inf, errors_sup], _ = ana.xy_to_bins(list(possible_times_spent_in_pixel),
                                                                         leaving_prob_each_time, 50)
        condition_name = parameters.nb_to_name[condition_list[i_condition]]
        condition_color = parameters.name_to_color[condition_name]
        # Plot all the values with some transparency
        plt.scatter(possible_times_spent_in_pixel, leaving_prob_each_time, alpha=0.1, color=condition_color)
        # Plot error bars
        plt.plot(bin_list, avg_list, color=parameters.name_to_color[parameters.nb_to_name[i_condition]], linewidth=4,
                 label=condition_name)
        plt.errorbar(bin_list, avg_list, [errors_inf, errors_sup], fmt='.k', capsize=5)

    plt.title("Leaving probability graph for conditions " + str(condition_list))
    plt.legend()
    # plt.colorbar()
    plt.xlabel("Time spent in pixel")
    plt.ylabel("Leaving probability")
    plt.yscale("log")
    plt.show()


def save_pixel_visit_duration_in_results_table():
    """
    Function that loads the clean_results.csv table, and adds a column in it, containing the average duration of visits
    to a pixel for this plate. Requires that pixel_wise_visit_durations function has already been executed.
    """

    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(test_pipeline=False)
    results = pd.read_csv(path + "clean_results.csv")
    plate_list = results["folder"]

    avg_duration_inside_list = np.zeros(len(plate_list))
    avg_duration_outside_list = np.zeros(len(plate_list))
    for i_plate, plate in enumerate(plate_list):
        # if i_plate % 10 == 0:
        print("Computing average visit duration in pixels for plate ", i_plate, " / ", len(plate_list))

        # If it's not already done, compute the pixel visit durations
        pixelwise_durations_path = plate[:-len("traj.csv")] + "pixelwise_visit_durations.npy"
        if not os.path.isfile(pixelwise_durations_path):
            pixel_wise_visit_durations(plate)
        # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
        current_pixel_wise_visit_durations = np.load(pixelwise_durations_path, allow_pickle=True)

        # Load patch info for this folder
        in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
        if os.path.isfile(in_patch_matrix_path):
            in_patch_matrix = pd.read_csv(in_patch_matrix_path)
        else:
            in_patch_matrix = gt.in_patch_all_pixels(plate)

        # Separate inside / outside food patch visit durations
        current_pixel_wise_visit_durations_inside = current_pixel_wise_visit_durations[in_patch_matrix != -1]
        current_pixel_wise_visit_durations_outside = current_pixel_wise_visit_durations[in_patch_matrix == -1]
        # Remove the pixelwise structure (from sublists in lists of lists, just a list with all the visit durations)
        current_pixel_wise_visit_durations_inside = [current_pixel_wise_visit_durations_inside[i][j] for i in
                                                     range(len(current_pixel_wise_visit_durations_inside)) for j in
                                                     range(len(current_pixel_wise_visit_durations_inside[i]))]
        current_pixel_wise_visit_durations_outside = [current_pixel_wise_visit_durations_outside[i][j] for i in
                                                      range(len(current_pixel_wise_visit_durations_outside)) for j in
                                                      range(len(current_pixel_wise_visit_durations_outside[i]))]
        # Then, compute the average visit duration inside and outside
        avg_duration_inside_list[i_plate] = np.mean(current_pixel_wise_visit_durations_inside)
        avg_duration_outside_list[i_plate] = np.mean(current_pixel_wise_visit_durations_outside)

    results["avg_visit_duration_to_pixels_inside_patches"] = avg_duration_inside_list
    results["avg_visit_duration_to_pixels_outside_patches"] = avg_duration_outside_list

    results.to_csv(path + "clean_results.csv")

    return results


def visit_duration_previous_visit_pixel(condition_list):
    """
        Function that plots the visit duration to a pixel as a function of the sum of the previous visits made to that
        same pixel. Does that for a few plates (for now)
    """

    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(test_pipeline=False)
    results = pd.read_csv(path + "clean_results.csv")
    plate_list = results["folder"]

    current_visit_durations_by_condition_and_plate = [[] for _ in range(len(condition_list))]
    previous_visit_durations_by_condition_and_plate = [[] for _ in range(len(condition_list))]
    for i_plate, plate in enumerate(plate_list):
        if i_plate % 30 == 0:
            print("Computing average visit duration in pixels for plate ", i_plate, " / ", len(plate_list))
        current_condition = fd.load_condition(plate)
        # Only if the current plate is in the list of conditions that we need
        if current_condition in condition_list:
            # If it's not already done, compute the pixel visit durations
            pixelwise_durations_path = plate[:-len("traj.csv")] + "pixelwise_visit_durations.npy"
            if not os.path.isfile(pixelwise_durations_path):
                pixel_wise_visit_durations(plate)
            # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
            current_pixel_wise_visit_durations = np.load(pixelwise_durations_path, allow_pickle=True)

            # Load patch info for this folder
            in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
            if os.path.isfile(in_patch_matrix_path):
                in_patch_matrix = pd.read_csv(in_patch_matrix_path)
            else:
                in_patch_matrix = gt.in_patch_all_pixels(plate)

            # Separate inside / outside food patch visit durations
            current_pixel_wise_visit_durations_inside = current_pixel_wise_visit_durations[in_patch_matrix != -1]
            # Add values to the relevant lists (current visit duration and corresponding "time already spent")
            curr_condition_index = np.where(condition_list == current_condition)[0][0]
            # In the _by_condition_by_plate lists, add a new sublist for the current plate
            current_visit_durations_by_condition_and_plate[curr_condition_index].append([])
            previous_visit_durations_by_condition_and_plate[curr_condition_index].append([])
            for i_pixel in range(len(current_pixel_wise_visit_durations_inside)):  # for every pixel
                current_visit_list = current_pixel_wise_visit_durations_inside[i_pixel]
                time_already_spent = 0
                for i_visit in range(len(current_visit_list)):  # for every visit
                    current_visit_duration = current_visit_list[i_visit]
                    current_visit_durations_by_condition_and_plate[curr_condition_index][-1].append(current_visit_duration)
                    previous_visit_durations_by_condition_and_plate[curr_condition_index][-1].append(time_already_spent)
                    time_already_spent += current_visit_duration

    plot_title = "Current vs previous visits in conditions: "
    for i_condition, condition in enumerate(condition_list):
        condition_name = parameters.nb_to_name[condition]
        condition_color = parameters.name_to_color[condition_name]
        plot_title += condition_name + " "

        bin_size = 10

        # Build a list with one sublist per bin of time_already_spent_in_pixel, and in each sublist, the average current
        # visit lengths corresponding to that time for plates which have more than some critical nb of data points
        # So we go from current_visit_durations_by_condition_and_plate[i_condition] = [ [0, 1, 1], [0, 20] ]
        # to binned values for each plate => plate 0 : [[0, 1, 1]], plate 1 : [[0], [], [20]]
        # to an average for each of those bins => list_of_plate_avg_each_bin = [[0.6666, 0], [1], [20]]
        list_of_plate_avg_each_bin = []
        # For each plate of this condition, bin the values and get an average current_visit length for every time spent
        for i_plate in range(len(current_visit_durations_by_condition_and_plate[i_condition])):
            if previous_visit_durations_by_condition_and_plate[i_condition][i_plate] and current_visit_durations_by_condition_and_plate[i_condition][i_plate]:
                previous_visit_bins, curr_visit_values, [inf, sup], binned_current_visits = ana.xy_to_bins(
                    previous_visit_durations_by_condition_and_plate[i_condition][i_plate],
                    current_visit_durations_by_condition_and_plate[i_condition][i_plate], bin_size, print_progress=False)
                for i_bin in range(len(previous_visit_bins)):
                    # If this bin has never been encountered yet, add an element to the list for it
                    # We do this even if this plate does not have enough values for this bin in order to keep the i_bin
                    # ordered and corresponding to the right time_already_spent
                    if i_bin > len(list_of_plate_avg_each_bin) - 1:
                        list_of_plate_avg_each_bin.append([])
                    # If the bin contains enough values, put its average
                    if len(binned_current_visits[i_bin]) > 10:
                        list_of_plate_avg_each_bin[i_bin].append(curr_visit_values[i_bin])
                # plt.plot(previous_visit_bins, curr_visit_values, color=condition_color, linewidth=6, alpha=0.1)

        # Then, for every time_already_spent bin, bootstrappppppp
        previous_visit_bins = [bin_size*i for i in range(len(list_of_plate_avg_each_bin))]
        current_visits_avg = [-1 for i in range(len(list_of_plate_avg_each_bin))]
        errors_inf = [-1 for i in range(len(list_of_plate_avg_each_bin))]
        errors_sup = [-1 for i in range(len(list_of_plate_avg_each_bin))]
        for i_bin in range(len(list_of_plate_avg_each_bin)):
            current_visits_avg[i_bin] = np.mean(list_of_plate_avg_each_bin[i_bin])
            bootstrap_ci = ana.bottestrop_ci(list_of_plate_avg_each_bin[i_bin], 100)
            # bootstrap_ci returns error bar absolute positions => convert them to error bar lengths
            [errors_inf[i_bin], errors_sup[i_bin]] = [current_visits_avg[i_bin] - bootstrap_ci[0], bootstrap_ci[1] - current_visits_avg[i_bin]]

        # Keep only bins with > 10 plates
        valid_bins = [previous_visit_bins[i] for i in range(len(previous_visit_bins)) if
                      len(list_of_plate_avg_each_bin[i]) > 10]
        valid_visits = [current_visits_avg[i] for i in range(len(current_visits_avg)) if
                        len(list_of_plate_avg_each_bin[i]) > 10]
        errors_inf = [errors_inf[i] for i in range(len(errors_inf)) if len(list_of_plate_avg_each_bin[i]) > 10]
        errors_sup = [errors_sup[i] for i in range(len(errors_sup)) if len(list_of_plate_avg_each_bin[i]) > 10]

        plt.plot(valid_bins, valid_visits, label=condition_name, color=condition_color, linewidth=4)
        plt.errorbar(valid_bins, valid_visits, [errors_inf, errors_sup], fmt='.k', capsize=5)

        # plt.scatter(previous_visit_durations_by_condition_and_plate[i_condition], current_visit_durations_by_condition_and_plate[i_condition], label=condition_name, color=condition_color, alpha=0.2)

    plt.title(plot_title)
    # plt.yscale("log")
    plt.xlabel("Time spent there before this visit (to a pixel)")
    plt.ylabel("Current visit duration (to a pixel)")
    plt.legend()
    plt.show()

    return 0

# plot_pixel_wise_leaving_probability([0, 4])

# results = save_pixel_visit_duration_in_results_table()
# path = gen.generate(test_pipeline=False)
# results = pd.read_csv(path + "clean_results.csv")
# main.plot_graphs(results, "pixels_avg_visit_duration", [["close 0", "med 0", "far 0", "cluster 0"]])
# main.plot_graphs(results, "pixels_avg_visit_duration", [["close 0.2", "med 0.2", "far 0.2", "cluster 0.2"]])
# main.plot_graphs(results, "pixels_avg_visit_duration", [["close 0.5", "med 0.5", "far 0.5", "cluster 0.5"]])

visit_duration_previous_visit_pixel([0, 1, 2])
visit_duration_previous_visit_pixel([4, 5, 6])
visit_duration_previous_visit_pixel([12, 13, 14])
