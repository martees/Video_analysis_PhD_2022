# In this script, I will plot one line per folder and in each line show a timeline of in which patch the worm was

from scipy import ndimage
import pandas as pd
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datatable as dt

from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Parameters import parameters as param
import find_data as fd


def plot_timeline(path, full_folder_list, traj, curve_names, is_plot):
    tic = time.time()
    curve_list = [param.name_to_nb_list[curve] for curve in curve_names]
    nb_of_curves = len(curve_list)
    # If there's an even nb of "curves", make two lines, otherwise just all in one line
    if nb_of_curves > 1:
        if nb_of_curves % 2 == 0:
            fig, axs = plt.subplots(2, nb_of_curves // 2)
        else:
            fig, axs = plt.subplots(1, nb_of_curves)
    for i_curve, curve in enumerate(curve_list):
        print(int(time.time() - tic), "s: Curve ", i_curve + 1, " / ", len(curve_list))
        folder_list = fd.return_folders_condition_list(full_folder_list, curve)
        current_timeline = np.zeros((300*len(folder_list), param.time_to_cut_videos))
        current_timeline[:] = np.nan
        for i_folder, folder in enumerate(folder_list):
            current_trajectory = traj[dt.f.folder == folder, :]
            patch_list_this_folder = current_trajectory[:, "patch_silhouette"].to_numpy().astype(int)
            timestamp_list_this_folder = current_trajectory[:, "time"].to_numpy().astype(int)
            current_timeline[300*i_folder:300*(i_folder + 1), timestamp_list_this_folder] = patch_list_this_folder
        # Set the NaN's to white
        masked_array = np.ma.array(current_timeline, mask=np.isnan(current_timeline))
        cmap = plt.get_cmap('rainbow')
        cmap.set_bad('white', 1.)
        # Then plot it
        if nb_of_curves == 1:
            plt.title("Timeline for condition "+curve_names[i_curve])
            plt.imshow(masked_array, cmap=cmap)
        elif nb_of_curves % 2 == 0:
            axs[i_curve // 2, i_curve % 2].set_title("Timeline for condition "+curve_names[i_curve])
            axs[i_curve // 2, i_curve % 2].imshow(masked_array, cmap=cmap)

    plt.colorbar()
    if is_plot:
        plt.show()
    else:
        if not os.path.isdir(path + "timeline_plots/"):
            os.mkdir(path + "timeline_plots/")
        plt.savefig(path + "timeline_plots/timeline_condition_"+curve_names[i_curve]+".png")
        plt.clf()


if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(shorten_traj=True)
    results = pd.read_csv(path + "clean_results.csv")
    trajectories = dt.fread(path + "clean_trajectories.csv")
    full_list_of_folders = list(results["folder"])

    plot_timeline(path, full_list_of_folders, trajectories, ["close 0"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["close 0.2"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["close 0.5"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["close 1.25"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["med 0"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["med 0.2"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["med 0.5"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["med 1.25"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["far 0"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["far 0.2"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["far 0.5"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["far 1.25"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["superfar 0"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["superfar 0.2"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["superfar 0.5"], is_plot=False)
    plot_timeline(path, full_list_of_folders, trajectories, ["superfar 1.25"], is_plot=False)

