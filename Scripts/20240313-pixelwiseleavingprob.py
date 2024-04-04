# This is a script that I start with the intention of connecting more precisely pixel-level depletion (using overlap with
# worm silhouette as a proxy) and instantaneous behavior (looking at whether the worm moves away from a pixel).
import copy

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import find_data as fd
from Generating_data_tables import main as gen


def distance_to_closest_patch_map(folder):
    return 0


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
                #if current_pixel == [479, 1455]:
                #    print("hehe")
                # If one of this current pixel's coordinates is not in the current visited pixels, then close the visit
                if current_pixel[0] not in current_visited_pixels[0] or current_pixel[1] not in current_visited_pixels[1]:
                    visit_times_each_pixel[current_pixel[1]][current_pixel[0]].append(-1)
                # Else, both are here, so check that they are for the same point
                else:
                    if True not in np.logical_and(np.array(current_visited_pixels[0]) == current_pixel[0], np.array(current_visited_pixels[0]) == current_pixel[0]):
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
            #if current_list_of_visits:
            #    print("hihi")
            # Register the current x and y of the pixel
            pixel_x_list += [i_column] * int(np.sum(current_list_of_visits))  # the int is just because len([]) = 0.0
            pixel_y_list += [i_line] * int(np.sum(current_list_of_visits))  # (in all other cases, visit duration is already integer)
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
folder_list = fd.path_finding_traj(path)


if is_recompute_full_data_table:
    # Create pandas where we'll store the number of points for each delay / time in pixel couple
    # (we do this otherwise it's > 100,000 points per worm so storing all the info would be a mess
    full_dataframe = pd.DataFrame({'time': [], 'delay': [], 'nb_of_points': []})
    # We will fill this dataframe based on the values in a matrix, with each line corresponding to time in patch, each
    # column to delay until next exit, and the value in the cell, to the number of points in this delay / time couple
    data_matrix = [[[] for _ in range(20000)] for _ in range(20000)]
    tic = time.time()
    for i_folder in range(150):
        # Print progress info
        if i_folder % 1 == 0:
            print("Computing pixelwise analysis for folder ", i_folder, " / ", len(folder_list), "...")
        if i_folder % 100 == 6:
            curr_t = (time.time() - tic)/60
            print("Current run time: ", curr_t, " minutes, estimated time left: ", (curr_t / i_folder) * (len(folder_list) - i_folder))
        # Compute or load the data
        if is_recompute_pixelwise_delays == True and i_folder > 151:
            current_dataframe = pixel_wise_leaving_delay(folder_list[i_folder])
        else:
            current_dataframe = pd.read_csv(folder_list[i_folder][:-len("traj.csv")] + "pixel_analysis_dataframe.csv")
        # Fill the full dataframe
        nb_of_datapoints = len(current_dataframe)
        for i_datapoint in range(nb_of_datapoints):
            if i_datapoint % (nb_of_datapoints/4) == 0:
                print(">>> Registering analysis for data point ", i_datapoint, " / ", nb_of_datapoints)
            current_point = current_dataframe.iloc[i_datapoint]
            t = int(current_point["time"])
            d = int(current_point["delay_before_next_exit"])
            index_current_point = np.where(np.logical_and(full_dataframe["time"] == t, full_dataframe["delay"] == d))[0]
            # If there is already a line with the correct time and delay, just add one to its nb of point column
            if len(index_current_point) > 0:
                full_dataframe["nb_of_points"][index_current_point] += 1
            else:
                full_dataframe = pd.concat((full_dataframe, pd.DataFrame({'time': [t], 'delay': [d], 'nb_of_points': [1]}))).reset_index(drop=True)
    full_dataframe.to_csv(path+"pixelwise_delay_analysis.csv")

else:
    full_dataframe = pd.read_csv(path+"pixelwise_delay_analysis.csv")

print("Starting to compute leaving probability...")
full_dataframe = full_dataframe.sort_values(by="time")
possible_times_spent_in_pixel = pd.unique(full_dataframe["time"])
leaving_prob_each_time = [0 for _ in range(len(possible_times_spent_in_pixel))]
for i_time_spent, time_spent in enumerate(possible_times_spent_in_pixel):
    if i_time_spent % 100 == 0:
        print("Time ", i_time_spent, " / ", len(possible_times_spent_in_pixel))
    current_data = full_dataframe[full_dataframe["time"] == time_spent].reset_index(drop=True)
    leaving_next_timestep = current_data[current_data["delay"] == 1]
    leaving_prob_each_time[i_time_spent] = leaving_next_timestep["nb_of_points"] / np.sum(current_data["nb_of_points"])

plt.scatter(possible_times_spent_in_pixel, leaving_prob_each_time, alpha=0.2)
plt.xlabel("Time spent in pixel")
plt.ylabel("Leaving probability")
plt.yscale("log")
plt.show()








