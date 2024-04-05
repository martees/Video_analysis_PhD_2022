# This is a script that I start with the intention of connecting more precisely pixel-level depletion (using overlap with
# worm silhouette as a proxy) and instantaneous behavior (looking at whether the worm moves away from a pixel).
import copy

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

import find_data as fd
from Generating_data_tables import main as gen
from Parameters import parameters
import analysis as ana


def distance_to_closest_patch_map(folder):
    """
    Function that takes a folder leading to a traj.csv file, containing a trajectory, and in a folder containing a matrix
    with the patch location of every pixel in the image (so -1 if outside patches, and otherwise the patch number).
    """


def pixel_wise_leaving_delay(folder):
    """
    Function that takes a folder containing a time series of silhouettes, and returns:
        - a list of delay before leaving for every visited pixel
        - a list of corresponding times already spent in each visited pixel
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
    for i_time in range(len(pixels)):
        current_visited_pixels = pixels[i_time]
        for i_pixel in range(len(current_visited_pixels[0])):
            current_pixel = [current_visited_pixels[0][i_pixel], current_visited_pixels[1][i_pixel]]
            # If visit just started, start it
            if visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] == -1:
                visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] = 1
            # If visit is continuing, increment time spent
            else:
                visit_times_each_pixel[current_pixel[1]][current_pixel[0]][-1] += 1
        # Then, close the visits of the previous time step that are not being continued
        if i_time > 0:
            previous_visited_pixels = pixels[i_time - 1]
            for i_pixel in range(len(previous_visited_pixels[0])):
                current_pixel = [previous_visited_pixels[0][i_pixel], previous_visited_pixels[1][i_pixel]]
                # if current_pixel == [479, 1455]:
                #    print("hehe")
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
    for i_line in range(len(visit_times_each_pixel)):
        for i_column in range(len(visit_times_each_pixel[i_line])):
            if visit_times_each_pixel[i_line][i_column][-1] == -1:
                visit_times_each_pixel[i_line][i_column] = visit_times_each_pixel[i_line][i_column][:-1]

    # Then, go through the list of visits to pixel, and create a list of "delay before leaving" and the corresponding
    # list of "time already spent in pixel", while recording x and y of the pixels
    pixel_x_list = []
    pixel_y_list = []
    time_already_spent_in_pixel = []
    delay_before_leaving = []
    for i_line in range(len(visit_times_each_pixel)):
        for i_column in range(len(visit_times_each_pixel[i_line])):
            current_list_of_visits = visit_times_each_pixel[i_line][i_column]
            # if current_list_of_visits:
            #    print("hihi")
            # Register the current x and y of the pixel
            pixel_x_list += [i_column] * int(np.sum(current_list_of_visits))  # the int is just because len([]) = 0.0
            pixel_y_list += [i_line] * int(
                np.sum(current_list_of_visits))  # (in all other cases, visit duration is already integer)
            # In time already spent in pixel we have a range equal to total time spent inside
            # (so if worm spent a visit of length 4, then one of length 2, we get [0, 1, 2, 3, 4, 5])
            time_already_spent_in_pixel += list(range(int(np.sum(current_list_of_visits))))
            # In delays before leaving, we put a range going from visit length to zero, for each visit
            # (so if worm spent a visit of length 4, then one of length 2, then we get [4, 3, 2, 1, 2, 1])
            delays = [range(visit, 0, -1) for visit in current_list_of_visits]
            delays = [delays[i][j] for i in range(len(delays)) for j in range(len(delays[i]))]
            delay_before_leaving += delays

    # Dataframe in which to save everything
    pixel_analysis_dataframe = pd.DataFrame()
    pixel_analysis_dataframe["folder"] = [folder for _ in range(len(pixel_x_list))]
    pixel_analysis_dataframe["x"] = pixel_x_list
    pixel_analysis_dataframe["y"] = pixel_y_list
    pixel_analysis_dataframe["time"] = time_already_spent_in_pixel
    pixel_analysis_dataframe["delay_before_next_exit"] = delay_before_leaving
    pixel_analysis_dataframe.to_csv(folder[:-len("traj.csv")] + "pixel_analysis_dataframe.csv", index=False)

    return pixel_analysis_dataframe


## Look at delay vs. time spent in patch
path = gen.generate(test_pipeline=False)
is_recompute_pixelwise_delays = False
is_recompute_full_data_table = True

if not os.path.isdir(path + "pixelwise_analysis"):
    os.mkdir(path + "pixelwise_analysis")

condition_list = list(parameters.nb_to_name.keys())
full_folder_list = fd.path_finding_traj(path)
list_of_matrices = []

for condition in condition_list[0:3]:
    print("Computing pixelwise analysis for condition ", condition, " / ", len(condition_list), "...")
    current_folder_list = fd.return_folders_condition_list(full_folder_list, condition)
    if is_recompute_full_data_table:

        # Create pandas where we'll store the number of points for each delay / time in pixel couple
        # (we do this otherwise it's > 100,000 points per worm so storing all the info would be a mess
        dataframe_this_condition = pd.DataFrame({'time': [], 'delay': [], 'nb_of_points': []})
        # We will fill this dataframe based on the values in a matrix, with each line corresponding to time in patch, each
        # column to delay until next exit, and the value in the cell, to the number of points in this delay / time couple
        data_matrix = np.array([np.array([0 for _ in range(4001)]) for _ in range(4001)])
        # Since some data will be over 4000, store those points in a separate list that will require more work
        #extra_data_list = []

        tic = time.time()
        for i_folder in range(len(current_folder_list)):
            # for i_folder in range(len(current_folder_list)):
            # Print progress info
            if i_folder % 1 == 0:
                print("> Folder ", i_folder, " / ", len(current_folder_list), "...")
            if i_folder % 10 == 4:
                curr_t = (time.time() - tic) / 60
                print("> Current run time: ", curr_t, " minutes, estimated time left: ",
                      (curr_t / i_folder) * (len(current_folder_list) - i_folder))
            # Compute or load the data
            if is_recompute_pixelwise_delays == True:
                current_dataframe = pixel_wise_leaving_delay(current_folder_list[i_folder])
            else:
                current_dataframe = pd.read_csv(
                    current_folder_list[i_folder][:-len("traj.csv")] + "pixel_analysis_dataframe.csv")
            # Fill the data matrix
            tic = time.time()
            nb_of_datapoints = len(current_dataframe)
            for i_datapoint in range(nb_of_datapoints):
                if i_datapoint % (nb_of_datapoints // 2) == 0:
                    print(">> Registering analysis for data point ", i_datapoint, " / ", nb_of_datapoints)
                current_point = current_dataframe.iloc[i_datapoint]
                t = int(current_point["time"])
                d = int(current_point["delay_before_next_exit"])
                if t < len(data_matrix) and d < len(data_matrix[t]):
                    data_matrix[t][d] += 1
                # All the points with t or d > 4000 are put together because f*ck it
                else:
                    if t < len(data_matrix) < d:
                        data_matrix[t][len(data_matrix) - 1] += 1
                    elif d < len(data_matrix) < t:
                        data_matrix[len(data_matrix) - 1][d] += 1
                    else:
                        data_matrix[len(data_matrix) - 1][len(data_matrix) - 1] += 1
            print(time.time() - tic)
        print("> Saving data matrix...")
        np.save(path + "/pixelwise_analysis/pixelwise_delay_analysis_" + str(condition) + ".npy", data_matrix)
        #print(">> Data with delay and time < 4000...")
        # Filling it with values from the data_matrix (time and delay < 4000), which should be most points
        #for t in range(len(data_matrix)):
        #    for d in range(len(data_matrix[0])):
        #        if data_matrix[t][d] != 0:
        #            dataframe_this_condition = pd.concat((dataframe_this_condition, pd.DataFrame(
        #                {'time': [t], 'delay': [d], 'nb_of_points': [data_matrix[t][d]]}))).reset_index(drop=True)
        # Filling it with values from extra_data_list (time or delay >= 4000)
        #print(">> Data with delay and time > 4000...")
        # for i_extra in range(len(extra_data_list)):
        #     t = extra_data_list[i_extra][0]
        #     d = extra_data_list[i_extra][1]
        #     index_current_point = \
        #     np.where(np.logical_and(dataframe_this_condition["time"] == t, dataframe_this_condition["delay"] == d))[0]
        #     # If there is already a line with the correct time and delay, just add one to its nb of point column
        #     if len(index_current_point) > 0:
        #         dataframe_this_condition["nb_of_points"][index_current_point] += 1
        #     else:  # Otherwise, create a new line with the current time and delay
        #         dataframe_this_condition = pd.concat((dataframe_this_condition, pd.DataFrame(
        #             {'time': [t], 'delay': [d], 'nb_of_points': [1]}))).reset_index(drop=True)

        #dataframe_this_condition.to_csv(
        #    path + "/pixelwise_analysis/pixelwise_delay_analysis_" + str(condition) + ".csv")
    else:
        dataframe_this_condition = np.load(
            path + "/pixelwise_analysis/pixelwise_delay_analysis_" + str(condition) + ".npy")
    list_of_matrices.append(dataframe_this_condition)

print("Starting to compute leaving probability...")
for i_condition in range(len(list_of_matrices)):
    dataframe_this_condition = list_of_matrices[i_condition]
    dataframe_this_condition = dataframe_this_condition.sort_values(by="time")
    possible_times_spent_in_pixel = pd.unique(dataframe_this_condition["time"])
    leaving_prob_each_time = [0 for _ in range(len(possible_times_spent_in_pixel))]

    for i_time_spent, time_spent in enumerate(possible_times_spent_in_pixel):
        if i_time_spent % 400 == 0:
            print("> Time ", i_time_spent, " / ", len(possible_times_spent_in_pixel))
        current_data = dataframe_this_condition[dataframe_this_condition["time"] == time_spent].reset_index(drop=True)
        leaving_next_timestep = current_data[current_data["delay"] == 1].reset_index(drop=True)
        if len(leaving_next_timestep) > 0:
            leaving_prob_each_time[i_time_spent] = leaving_next_timestep["nb_of_points"][0] / np.sum(
                current_data["nb_of_points"])
        else:
            leaving_prob_each_time[i_time_spent] = 0

    plt.scatter(possible_times_spent_in_pixel, leaving_prob_each_time, alpha=0.2)
    plt.title("Leaving probability graph for condition " + parameters.nb_to_name[condition_list[i_condition]])
    plt.xlabel("Time spent in pixel")
    plt.ylabel("Leaving probability")
    plt.yscale("log")
    plt.show()
