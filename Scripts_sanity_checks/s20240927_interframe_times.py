# This is a script to plot the distribution of inter-frame times in our videos.
# Expected results: a gaussian centered on 0.82 seconds
# If that's not the case, and there are other peaks, then we're in the panade

import matplotlib.pyplot as plt
import numpy as np
import datatable as dt
import pandas as pd
import os

import analysis as ana
from Generating_data_tables import main as gen
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
from Scripts_models import s20240626_transitionmatrix as transition_script
import find_data as fd
from Parameters import parameters as param


def plot_inter_frame_histogram():
    path = gen.generate("")
    results = dt.fread(path + "clean_results.csv")
    trajectories = dt.fread(path + "clean_trajectories.csv")
    full_list_of_folders = results[:, "folder"].to_list()[0]
    print("Finished loading tables!")
    list_of_interframe_times = []

    for i_folder, folder in enumerate(full_list_of_folders):
        if i_folder % 10 == 0:
            print("Folder ", i_folder, " / ", len(full_list_of_folders))
        current_traj = trajectories[dt.f.folder == folder, :]
        list_of_frames = current_traj[:, "frame"].to_numpy()
        list_of_times = current_traj[:, "time"].to_numpy()
        _, time_bug = fd.correct_time_stamps(current_traj.to_pandas(), False, True)
        if time_bug == "nothing_wrong" or "time_reset":
            frame_delta_each_line = list_of_frames[1:] - list_of_frames[:-1]
            time_delta_each_line = list_of_times[1:] - list_of_times[:-1]
            time_per_frame = ana.array_division_ignoring_zeros(time_delta_each_line, frame_delta_each_line)
            list_of_interframe_times += list(np.ravel(time_per_frame))

    plt.hist(list_of_interframe_times, bins=[-7000, -5000, -3000, -1000, -500, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 2], log=True)
    print("You can copy this number in Parameters/parameters.py, for the variable one_frame_in_seconds: ", np.mean(list_of_interframe_times))
    plt.show()


plot_inter_frame_histogram()

