# In this script, I will plot one line per folder and in each line show a timeline of in which patch the worm was

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datatable as dt
import cv2

from Generating_data_tables import main as gen
from Parameters import parameters as param
import find_data as fd


def plot_timeline_with_holes(results_path, full_folder_list, traj, curve_names, is_plot):
    """
    A function that will plot a matrix with one line per folder in full_folder_list, and one column per frame
    in the videos, until the time point "time_to_cut_videos" defined in parameters.py.
    In each line, the color will correspond to a food patch.
    Purple = -1 = outside the food patches.
    White = any hole in the tracking.
    In this function, the way we fill holes is not shown!
    @param results_path: a path to the folder containing all the result subfolders.
    @param full_folder_list: a list of the subfolders for each plate (paths including "traj.csv")
    @param traj: the clean_trajectories.csv file, loaded with the datatable module.
    @param curve_names: a list of condition names, eg ["close", "med 0.2"]
    @param is_plot: if True, will show the matrix, if False, will save it.
    @return: nuthin
    """
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
        current_timeline = np.zeros((200*len(folder_list), param.times_to_cut_videos))
        current_timeline[:] = np.nan
        for i_folder, folder in enumerate(folder_list):
            current_trajectory = traj[dt.f.folder == folder, :]
            patch_list_this_folder = current_trajectory[:, "patch_silhouette"].to_numpy().astype(int)
            timestamp_list_this_folder = current_trajectory[:, "time"].to_numpy().astype(int)
            current_timeline[200*i_folder:200*(i_folder + 1), timestamp_list_this_folder] = patch_list_this_folder
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
        if not os.path.isdir(results_path + "timeline_plots/"):
            os.mkdir(results_path + "timeline_plots/")
        plt.savefig(results_path + "timeline_plots/timeline_condition_" + curve_names[i_curve] + "_with_holes.png", dpi=2000)
        plt.clf()


def plot_timeline_with_bad_holes(results_path, result_datatable, full_folder_list,
                                 curve_names, is_plot, line_height=200, plot_only_uncensored=False):
    """
    A function that will plot a matrix with one line per folder in full_folder_list, and one column per frame
    in the videos, until the time point "time_to_cut_videos" defined in parameters.py.
    In each line, the color will correspond to a food patch.
    Purple = -1 = outside the food patches.
    White = tracking holes that were considered "bad" (when the tracking ends and starts in different situations, like
            if the hole starts inside a food patch and ends outside). Other tracking holes (which end and start in the
            same place) are filled with the relevant color (so if worm was outside when we lost it and when we got it
            back, the hole will be purple).
    @param results_path: path to the folder containing the clean_results.csv + all the other results folders
    @param result_datatable: a csv loaded from clean_results.csv, using datatable module
    @param full_folder_list: a list of the subfolders for each plate (paths including "traj.csv")
    @param curve_names: a list of condition names, eg ["close", "med 0.2"]
    @param is_plot: if True, will show the matrix, if False, will save it.
    @return: nuthin
    """
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
        # current_timeline = np.zeros((200*len(folder_list), param.time_to_cut_videos))
        current_timeline = np.zeros((line_height*len(folder_list), 30000))
        current_timeline[:] = np.nan
        for i_folder, folder in enumerate(folder_list):
            current_results = result_datatable[dt.f.folder == folder, :]
            if plot_only_uncensored:
                visit_list = fd.load_list(current_results.to_pandas(), "uncensored_visits")
                transit_list = fd.load_list(current_results.to_pandas(), "uncensored_transits")
            else:
                visit_list = fd.load_list(current_results.to_pandas(), "no_hole_visits")
                transit_list = fd.load_list(current_results.to_pandas(), "aggregated_raw_transits")
            i = 0
            while i < len(visit_list) or i < len(transit_list):
                if i < len(visit_list):
                    current_timeline[line_height * i_folder:line_height * (i_folder + 1), int(np.rint(visit_list[i][0])):int(np.rint(visit_list[i][1]+1))] = visit_list[i][2]
                if i < len(transit_list):
                    current_timeline[line_height * i_folder:line_height * (i_folder + 1), int(np.rint(transit_list[i][0])):int(np.rint(transit_list[i][1]+1))] = -1
                i += 1
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
        if not os.path.isdir(results_path + "timeline_plots/"):
            os.mkdir(results_path + "timeline_plots/")
        plt.savefig(results_path + "timeline_plots/timeline_condition_" + curve_names[i_curve] + "_holes_filled.png", dpi=2000)
        # Calculate mean and STD
        #mean = np.nanmean(current_timeline)
        #std = np.nanstd(current_timeline)
        # Clip frame to lower and upper STD
        #offset = 0.2
        #clipped = np.clip(current_timeline, mean - offset * std, mean + offset * std).astype(np.uint8)
        # Normalize to range
        #image = cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
        # Save it with nice colormap
        #colormap = plt.get_cmap('rainbow')
        #heatmap = (colormap(image) * 2 ** 16).astype(np.uint16)[:, :, :3]
        #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(results_path + "timeline_plots/timeline_condition_" + curve_names[i_curve] + ".png", heatmap)
        plt.clf()


def plot_timeline_with_censorship(results_path, result_datatable, full_folder_list, curve_names, is_plot):
    """
    A function that will plot a matrix with one line per folder in full_folder_list, and one column per frame
    in the videos, until the time point "time_to_cut_videos" defined in parameters.py.
    In each line, the color will correspond to a food patch.
    Purple = -1 = outside the food patches.
    White = tracking holes that were considered "bad" (when the tracking ends and starts in different situations, like
            if the hole starts inside a food patch and ends outside). Other tracking holes (which end and start in the
            same place) are filled with the relevant color (so if worm was outside when we lost it and when we got it
            back, the hole will be purple).
    Black = -10 = events considered as censored in our analysis. This is all events touching the start/end of video,
                 or touching a tracking hole.

    @param results_path: path to the folder containing the clean_results.csv + all the other results folders
    @param result_datatable: a csv loaded from clean_results.csv, using datatable module
    @param full_folder_list: a list of the subfolders for each plate (paths including "traj.csv")
    @param curve_names: a list of condition names, eg ["close", "med 0.2"]
    @param is_plot: if True, will show the matrix, if False, will save it.
    @return: nuthin
    """
    tic = time.time()
    curve_list = [param.name_to_nb_list[curve] for curve in curve_names]
    nb_of_curves = len(curve_list)
    # If there's an even nb of "curves", make two lines, otherwise just all in one line
    if nb_of_curves > 1:
        if nb_of_curves % 2 == 0:
            fig, axs = plt.subplots(nb_of_curves // 2, nb_of_curves // 2 + nb_of_curves % 2)
        else:
            fig, axs = plt.subplots(1, nb_of_curves)
    for i_curve, curve in enumerate(curve_list):
        print(int(time.time() - tic), "s: Curve ", i_curve + 1, " / ", len(curve_list))
        folder_list = fd.return_folders_condition_list(full_folder_list, curve)
        # current_timeline = np.zeros((200*len(folder_list), param.time_to_cut_videos))
        current_timeline = np.zeros((200*len(folder_list), 30000))
        current_timeline[:] = np.nan
        for i_folder, folder in enumerate(folder_list):
            current_results = result_datatable[dt.f.folder == folder, :]
            # Load all the visits and transits
            visit_list = fd.load_list(current_results.to_pandas(), "no_hole_visits")
            transit_list = fd.load_list(current_results.to_pandas(), "aggregated_raw_transits")
            # Load the uncensored visits and transits
            uncensored_visit_list = fd.load_list(current_results.to_pandas(), "uncensored_visits")
            uncensored_transit_list = fd.load_list(current_results.to_pandas(), "uncensored_transits")

            # First pass: set all visits and transits (so everything except holes) as -2 (black)
            i = 0
            while i < len(visit_list) or i < len(transit_list):
                if i < len(visit_list):
                    current_timeline[200 * i_folder:200 * (i_folder + 1), int(np.rint(visit_list[i][0])):int(np.rint(visit_list[i][1]+1))] = -10
                if i < len(transit_list):
                    current_timeline[200 * i_folder:200 * (i_folder + 1), int(np.rint(transit_list[i][0])):int(np.rint(transit_list[i][1]+1))] = -10
                i += 1

            # Second pass: set all *uncensored* visits and transits to their actual value (-1 for transits, patch id for visits)
            i = 0
            while i < len(uncensored_visit_list) or i < len(uncensored_transit_list):
                if i < len(uncensored_visit_list):
                    current_timeline[200 * i_folder:200 * (i_folder + 1), int(np.rint(uncensored_visit_list[i][0])):int(np.rint(uncensored_visit_list[i][1]+1))] = uncensored_visit_list[i][2]
                if i < len(uncensored_transit_list):
                    current_timeline[200 * i_folder:200 * (i_folder + 1), int(np.rint(uncensored_transit_list[i][0])):int(np.rint(uncensored_transit_list[i][1]+1))] = -1
                i += 1

        # Create masked array with the mask on holes
        masked_array = np.ma.array(current_timeline, mask=np.isnan(current_timeline))
        # Set the NaN's (holes) to white and lower values (censored) to black
        cmap = plt.get_cmap('rainbow')
        cmap.set_bad('white', 1.)
        cmap.set_under('black')
        # Then plot it
        if nb_of_curves == 1:
            plt.title("Timeline for condition "+curve_names[i_curve])
            plt.imshow(current_timeline, cmap=cmap, vmin=-2)
            plt.colorbar(extend="min")
        elif nb_of_curves % 2 == 0:
            axs[i_curve // 2 - 1, i_curve // 2 + i_curve % 2 - 1].set_title("Timeline for condition "+curve_names[i_curve])
            img = axs[i_curve // 2 - 1, i_curve // 2 + i_curve % 2 - 1].imshow(masked_array, cmap=cmap)
            plt.gcf().colorbar(img, extend="min")

    if is_plot:
        plt.show()
    else:
        if not os.path.isdir(results_path + "timeline_plots/"):
            os.mkdir(results_path + "timeline_plots/")
        plt.savefig(results_path + "timeline_plots/timeline_condition_" + curve_names[i_curve] + "_censorship_in_black.png", dpi=2000)
        # Calculate mean and STD
        #mean = np.nanmean(current_timeline)
        #std = np.nanstd(current_timeline)
        # Clip frame to lower and upper STD
        #offset = 0.2
        #clipped = np.clip(current_timeline, mean - offset * std, mean + offset * std).astype(np.uint8)
        # Normalize to range
        #image = cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
        # Save it with nice colormap
        #colormap = plt.get_cmap('rainbow')
        #heatmap = (colormap(image) * 2 ** 16).astype(np.uint16)[:, :, :3]
        #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(results_path + "timeline_plots/timeline_condition_" + curve_names[i_curve] + ".png", heatmap)
        plt.clf()


def plot_overlap_timelines(results_path, result_datatable, full_folder_list, curve_names, plot_only_uncensored=False, is_plot=True):
    """
    A function that will plot a matrix with one line per folder in full_folder_list, and one column per frame
    in the videos, until the time point "time_to_cut_videos" defined in parameters.py.
    In each line, the color will correspond to the number of time points assigned to the same time point.
    This is to detect bugs in the visit / transit functions of generate_results(), that could create multiple
    visits in the same time point etc.
    """
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
        # current_timeline = np.zeros((200*len(folder_list), param.times_to_cut_videos))
        current_timeline = np.zeros((200*len(folder_list), 30000))
        for i_folder, folder in enumerate(folder_list):
            print(folder)
            current_results = result_datatable[dt.f.folder == folder, :]
            if plot_only_uncensored:
                visit_list = fd.load_list(current_results.to_pandas(), "uncensored_visits")
                transit_list = fd.load_list(current_results.to_pandas(), "uncensored_transits")
            else:
                visit_list = fd.load_list(current_results.to_pandas(), "no_hole_visits")
                transit_list = fd.load_list(current_results.to_pandas(), "aggregated_raw_transits")
            i = 0
            while i < len(visit_list) or i < len(transit_list):
                if i < len(visit_list):
                    if visit_list[i][1] < visit_list[i][0]:
                        print("Negative visit: ", int(np.rint(visit_list[i][0])), ", ", int(np.rint(visit_list[i][1])), ", folder: ", folder[-40:])
                    current_timeline[200 * i_folder:200 * (i_folder + 1), int(np.rint(visit_list[i][0])):int(np.rint(visit_list[i][1]))] += 1
                if i < len(transit_list):
                    if transit_list[i][1] < transit_list[i][0]:
                        print("Negative transit: ", int(np.rint(transit_list[i][0])), ", ", int(np.rint(transit_list[i][1])), ", folder: ", folder[-40:])
                    current_timeline[200 * i_folder:200 * (i_folder + 1), int(np.rint(transit_list[i][0])):int(np.rint(transit_list[i][1]))] += 1
                #plt.imshow(current_timeline)
                #plt.colorbar()
                #plt.show()
                i += 1
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
            axs[i_curve // 2, i_curve % 2].colorbar()

    if is_plot:
        plt.show()
    else:
        if not os.path.isdir(results_path + "timeline_plots/"):
            os.mkdir(results_path + "timeline_plots/")
        plt.savefig(results_path + "timeline_plots/timeline_condition_" + curve_names[i_curve] + "_overlap_check.png", dpi=2000)
        plt.clf()

if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(starting_from="", test_pipeline=False)

    results = dt.fread(path + "clean_results.csv")
    trajectories = dt.fread(path + "clean_trajectories.csv")
    full_list_of_folders = results[:, "folder"].to_list()[0]
    print("Finished loading tables!")

    # list_of_bad_folders = ["E:/Results_minipatches_retracked/20221014T103022_SmallPatches_C4-CAM3/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221015T112001_SmallPatches_C2-CAM7/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221015T113417_SmallPatches_C5-CAM2/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221015T113423_SmallPatches_C6-CAM1/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221015T113423_SmallPatches_C6-CAM2/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221017T203313_SmallPatches_C5-CAM2/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221017T203320_SmallPatches_C6-CAM6/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221017T203320_SmallPatches_C6-CAM8/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221110T193022_SmallPatches_C1-CAM2/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221115T113217_SmallPatches_C5-CAM4/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221116T194409_SmallPatches_C5-CAM7/traj.csv",
    #                       "E:/Results_minipatches_retracked/20221117T104032_SmallPatches_C1-CAM7/traj.csv"]
    # full_list_of_folders = list_of_bad_folders

    # plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 0"], is_plot=True, plot_only_uncensored=False)
    # plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 0.2"], is_plot=True, plot_only_uncensored=False)
    # plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 0.5"], is_plot=True, plot_only_uncensored=False)
    # plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 1.25"], is_plot=True, plot_only_uncensored=False)

    #plot_timeline_with_holes(path, full_list_of_folders, trajectories, ["superfar 0.2"], is_plot=False)
    #plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["far 0.5"],
    #                             plot_only_uncensored=False, is_plot=True, line_height=100)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["far 0.5"], is_plot=True)

    print("OVERRRRRRRRRRRR")

    # Sanity check: check that events do not overlap
    plot_overlap_timelines(path, results, full_list_of_folders, ["close 0"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["med 0"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["far 0"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["superfar 0"], plot_only_uncensored=False, is_plot=False)

    plot_overlap_timelines(path, results, full_list_of_folders, ["close 0.2"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["med 0.2"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["far 0.2"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["superfar 0.2"], plot_only_uncensored=False, is_plot=False)

    plot_overlap_timelines(path, results, full_list_of_folders, ["close 0.5"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["med 0.5"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["far 0.5"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["superfar 0.5"], plot_only_uncensored=False, is_plot=False)

    plot_overlap_timelines(path, results, full_list_of_folders, ["close 1.25"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["med 1.25"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["far 1.25"], plot_only_uncensored=False, is_plot=False)
    plot_overlap_timelines(path, results, full_list_of_folders, ["superfar 1.25"], plot_only_uncensored=False, is_plot=False)

    # Plotting what happens at each time step, with long bad tracking holes in white
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 0"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 0.2"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 0.5"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["close 1.25"], is_plot=False)

    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["med 0"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["med 0.2"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["med 0.5"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["med 1.25"], is_plot=False)

    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["far 0"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["far 0.2"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["far 0.5"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["far 1.25"], is_plot=False)

    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["superfar 0"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["superfar 0.2"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["superfar 0.5"], is_plot=False)
    plot_timeline_with_bad_holes(path, results, full_list_of_folders, ["superfar 1.25"], is_plot=False)

    # Plotting what happens at each time step, censored events in black
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["close 0"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["close 0.2"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["close 0.5"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["close 1.25"], is_plot=False)

    plot_timeline_with_censorship(path, results, full_list_of_folders, ["med 0"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["med 0.2"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["med 0.5"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["med 1.25"], is_plot=False)

    plot_timeline_with_censorship(path, results, full_list_of_folders, ["far 0"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["far 0.2"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["far 0.5"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["far 1.25"], is_plot=False)

    plot_timeline_with_censorship(path, results, full_list_of_folders, ["superfar 0"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["superfar 0.2"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["superfar 0.5"], is_plot=False)
    plot_timeline_with_censorship(path, results, full_list_of_folders, ["superfar 1.25"], is_plot=False)


