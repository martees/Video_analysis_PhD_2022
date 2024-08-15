# In the full datasets, there seems to be quite a lot of frames with very big worm silhouettes due to integration of
# condensation droplets into worm blob.
# I want to exclude some silhouettes based on their number of pixels.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datatable as dt

from Generating_data_tables import main as gen
import find_data as fd


def distribution_silhouette_size(trajectories, folder_list, end_time_list):
    """
    Will plot survival curves with the distribution of silhouette sizes found in folder_list.
    Will plot one curve per end_time in end_time_list, plotting the distribution of silhouettes until frame number
    reaches end time.
    This way, if abnormal silhouettes appear after X hours of recording in each video, it will be visible on the plot as
    a difference between the curves.
    @param trajectories:
    @param folder_list:
    @param end_time_list:
    @return:
    """
    end_time_list = sorted(end_time_list)
    previous_list_of_silhouette_sizes = []
    previous_end_time = 0
    colors = plt.cm.plasma(np.linspace(0, 1, len(end_time_list)))
    for i_end_time, end_time in enumerate(end_time_list):
        print("Computing distribution for end_time ", i_end_time, "/", len(end_time_list))
        list_of_silhouette_sizes = previous_list_of_silhouette_sizes
        for i_folder in range(len(folder_list)):
            current_folder = folder_list[i_folder, 0]
            frame_list = trajectories[dt.f.folder == current_folder, "frame"]
            # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
            pixels, _, frame_size = fd.load_silhouette(current_folder)
            pixels = fd.reindex_silhouette(pixels, frame_size)
            # Setting a global counter: when looking at end_time 1000, then 2000, you just need to add new silhouettes
            # starting on time 1000,
            i_silhouette = previous_end_time
            while frame_list[i_silhouette, 0] < end_time:
                list_of_silhouette_sizes.append(len(pixels[i_silhouette][0]))
                i_silhouette += 1
        if i_end_time == 0:
            initial_silhouette_size = np.mean(list_of_silhouette_sizes)
        plt.hist(list_of_silhouette_sizes/initial_silhouette_size, bins=100, density=True, cumulative=-1, histtype="step", label=str(end_time), color=colors[i_end_time], linewidth=3)
        previous_end_time = end_time
    plt.title("Distribution of silhouette sizes, divided by avg until t="+str(end_time_list[0]))
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_bad_silhouette_events(trajectories, folder_list, time_bin_size, min_ratio, max_ratio):
    """
    Plots one line per folder, and will plot for every time frame of size time_bin in the video, the number of
    silhouettes that have fewer pixels than min_size, or more than max_size.
    """
    events_matrix = np.zeros((folder_list.shape[0], 40000 // time_bin_size))
    for i_folder in range(folder_list.shape[0]):
        print(i_folder)
        folder = folder_list[i_folder, 0]
        frame_list = trajectories[dt.f.folder == folder, "frame"]
        # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
        pixels, _, frame_size = fd.load_silhouette(folder)
        pixels = fd.reindex_silhouette(pixels, frame_size)
        # First use the first 1000 frames to look at the average size of the worm
        i_frame = 0
        beginning_size_list = []
        while frame_list[i_frame, 0] < 1000:
            beginning_size_list.append(len(pixels[i_frame][0]))
            i_frame += 1
        beginning_size = np.mean(beginning_size_list)
        current_bin_max = 0
        while current_bin_max < 40000:
            counter = 0
            while counter < frame_list.shape[0] and frame_list[counter, 0] < current_bin_max:
                if len(pixels[counter][0])/beginning_size > max_ratio or len(pixels[counter][0])/beginning_size < min_ratio:
                    events_matrix[i_folder, current_bin_max // time_bin_size] += 1
                counter += 1
            current_bin_max += time_bin_size

    # Plot the events matrix
    cmap = plt.cm.plasma
    plt.imshow(events_matrix, interpolation='nearest', cmap=cmap, vmax=20)
    plt.xticks(range(len(events_matrix[0])), [i * time_bin_size for i in range(len(events_matrix[0]))], rotation=45)
    plt.xlabel("Time in video")
    plt.ylabel("Folder")
    plt.title("Nb of bad silhouettes, min size " + str(min_ratio) + ", max size " + str(max_ratio))
    plt.colorbar()
    plt.show()


path = gen.generate("")
results = dt.fread(path + "clean_results.csv")
traj = dt.fread(path + "clean_trajectories.csv")
print("Finished loading data tables")
#distribution_silhouette_size(traj, results["folder"], [1000, 10000, 15000, 20000, 25000, 30000])
plot_bad_silhouette_events(traj, results[200:300, "folder"], 1000, 0.75, 1.4)



