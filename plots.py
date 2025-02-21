import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patheffects as pe
import random

import scipy.stats
import os
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import analysis as ana
from Generating_data_tables import main as gen
from Generating_data_tables import generate_trajectories as gt
import find_data as fd
from Parameters import parameters as param
from Parameters import custom_legends
from Parameters import colored_line_plot
from Scripts_analysis import s20240828_barplot_first_visit as first_visit_script


def data_coverage(traj):
    """
    Takes a dataframe with the trajectories implemented as in our trajectories.csv folder.
    Returns a plot with plates in y, time in x, and a color depending on whether:
    - there is or not a data point for this frame
    - the worm in this frame is in a patch or not
    """
    list_of_plates = np.unique(traj["folder"])
    nb_of_plates = len(list_of_plates)
    list_of_frames = [list(i) for i in np.zeros((nb_of_plates, 1),
                                                dtype='int')]  # list of list of frames for each plate [[0],[0],...,[0]]
    list_of_coverages = np.zeros(len(list_of_plates))  # proportion of coverage for each plate
    # to plot data coverage
    list_x = []
    list_y = []
    for i_plate in range(nb_of_plates):
        current_plate = list_of_plates[i_plate]
        current_plate_data = traj[traj["folder"] == current_plate]  # select one plate
        current_list_of_frames = list(current_plate_data["frame"])  # extract its frames
        current_coverage = len(current_list_of_frames) / current_list_of_frames[-1]  # coverage
        list_of_coverages[i_plate] = current_coverage
        if current_coverage > 0.85:
            for frame in current_list_of_frames:
                list_x.append(frame)
                list_y.append(current_plate)
    plt.scatter(list_x, list_y, s=.8)
    plt.show()


def patches(folder_list, show_composite=True, is_plot=True):
    """
    Function that takes a folder list, and for each folder, will either:
    - plot the patch positions on the composite patch image, to check if our metadata matches our actual data (is_plot = True)
    - return a list of border positions for each patch (is_plot = False)
    """
    if type(folder_list) is str:
        folder_list = [folder_list]

    for folder in folder_list:
        if is_plot:
            fig, ax = plt.subplots()
            if show_composite:
                composite = plt.imread(fd.load_file_path(folder, "composite_patches.tif"))
                ax.imshow(composite)
            else:
                background = plt.imread(fd.load_file_path(folder, "background.tif"))
                ax.imshow(background, cmap='gray')

        # Load metadata
        metadata = fd.folder_to_metadata(folder)
        patch_centers = metadata["patch_centers"]
        patch_densities = metadata["patch_densities"]
        patch_spline_breaks = metadata["spline_breaks"]
        patch_spline_coefs = metadata["spline_coefs"]
        patch_guides = metadata["spline_guides"]

        colors = plt.cm.jet(np.linspace(0, 1, len(patch_centers)))
        x_list = []
        y_list = []
        # For each patch
        for i_patch in range(len(patch_centers)):
            # For a range of 100 angular positions
            angular_pos = np.linspace(-np.pi, np.pi, 100)
            radiuses = np.zeros(len(angular_pos))
            # Compute the local spline value for each of those radiuses
            for i_angle in range(len(angular_pos)):
                radiuses[i_angle] = gt.spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                                    patch_spline_coefs[i_patch])

            # Create lists of cartesian positions out of this
            x_pos = []
            y_pos = []
            for point in range(len(angular_pos)):
                x_pos.append(patch_centers[i_patch][0] + (radiuses[point] * np.cos(angular_pos[point])))
                y_pos.append(patch_centers[i_patch][1] + (radiuses[point] * np.sin(angular_pos[point])))

            # Either plot them
            if is_plot:
                plt.plot(x_pos, y_pos, color=colors[i_patch])
                plt.scatter([patch_guides[i_patch][i][0] for i in range(len(patch_guides[i_patch]))],
                            [patch_guides[i_patch][i][1] for i in range(len(patch_guides[i_patch]))], color="white",
                            s=15)
            # Or add them to a list for later
            else:
                x_list.append(x_pos)
                y_list.append(y_pos)

        if is_plot:
            plt.title(folder)
            plt.show()
        else:
            return x_list, y_list


def trajectories_1condition(path, traj, condition_list, n_max=4, is_plot_patches=False, show_composite=True,
                            plot_in_patch=False,
                            plot_continuity=False, plot_speed=False, plot_time=False, plate_list=None, is_plot=True,
                            save_fig=False, plot_lines=False, zoom_in=False):
    """
    Function that takes in our dataframe format, using columns: "x", "y", "id_conservative", "folder"
    and extracting "condition" info in metadata
    Extracts list of series of positions from indicated condition and draws them, with one color per id
    traj: dataframe containing the series of (x,y) positions ([[x0,x1,x2...] [y0,y1,y2...])
    i_condition: the experimental condition that you want displayed
    n_max: max number of subplots in each graphic displayed
    plate_list: list of plates to display. Overrides condition selection
    :return: trajectory plot
    """
    # If there is a plate list
    if plate_list is not None:
        if type(plate_list) == str:
            plate_list = [plate_list]
        #folder_list = []
        condition_list = []
        #for i_plate in range(len(plate_list)):
        #    folder_list.append(traj[traj["folder"] == plate_list[i_plate]]["folder"])
        folder_list = np.unique(plate_list)
        # Change condition accordingly
        for plate in folder_list:
            condition_list.append(fd.load_condition(plate))
        condition_list = np.unique(condition_list)
    # Otherwise take all the plates of the condition
    else:
        folder_list = fd.return_folders_condition_list(np.unique(traj["folder"]), condition_list)

    # If save_fig is True, check that there is a trajectory_plots folder in path, otherwise create it
    if save_fig:
        if not os.path.isdir(path + "trajectory_plots"):
            os.mkdir(path + "trajectory_plots")

    nb_of_folders = len(folder_list)
    #colors = plt.cm.jet(np.linspace(0, 1, nb_of_folders))
    previous_folder = 0
    n_plate = 1
    for i_folder in range(nb_of_folders):
        current_folder = folder_list[i_folder]
        current_traj = traj[traj["folder"] == current_folder]
        current_list_x = current_traj.reset_index()["x"]
        current_list_y = current_traj.reset_index()["y"]
        metadata = fd.folder_to_metadata(current_folder)
        plt.suptitle("Trajectories for condition " + str(condition_list))

        # If we just changed plate or if it's the 1st, plot the background elements
        if previous_folder != current_folder or previous_folder == 0:
            if n_plate > n_max:  # If we exceeded the max nb of plates per graphic
                if save_fig:
                    plt.savefig(path + "trajectory_plots/condition_" + str(condition_list) + "_" +
                                current_folder.split("/")[-2] + ".png")
                if is_plot:
                    plt.show()
                else:
                    plt.clf()
                n_plate = 1
            if n_plate <= n_max or len(plate_list) > 1:
                plt.subplot(n_max // 2, n_max // 2, n_plate)
                n_plate += 1
            # Show background and patches
            fig = plt.gcf()
            fig.set_size_inches(20, 20)
            ax = fig.gca()
            fig.set_tight_layout(True)  # make the margins tighter

            if show_composite:  # show composite with real patches
                composite = plt.imread(fd.load_file_path(current_folder, "composite_patches.tif"))
                ax.imshow(composite)
                # White scale bar
                plt.plot([100, 100 + 5 * (1 / param.one_pixel_in_mm)], [1800, 1800], color="white", linewidth=4)
                plt.text(110, 1750, "5 mm", color="white")
            else:  # show cleaner background without the patches
                background = plt.imread(fd.load_file_path(current_folder, "background.tif"))
                ax.imshow(background, cmap='gray')
                # Black scale bar
                plt.plot([100, 100 + 5 * (1 / param.one_pixel_in_mm)], [1800, 1800], color="black", linewidth=4)
                plt.text(110, 1750, "5 mm")

            if zoom_in:
                plt.xlim(670, 1080)
                plt.ylim(670, 1080)
                # White scale bar
                plt.plot([1000, 1000 + (1 / param.one_pixel_in_mm)], [1050, 1050], color="white", linewidth=4)
                plt.text(1000, 1030, "1 mm", color="white")

            ax.set_title(str(current_folder[-48:-9]))
            # Plot them patches
            if is_plot_patches:
                # If the speeds are plotted, set the color of the patches to one that depends on the condition
                if plot_speed:
                    color = param.name_to_color[param.nb_to_distance[condition_list[0]]]
                else:
                    color = "black"
                patch_densities = metadata["patch_densities"]
                patch_centers = metadata["patch_centers"]
                x_list, y_list = patches([current_folder], is_plot=False)
                for i_patch in range(len(patch_centers)):
                    ax.plot(x_list[i_patch], y_list[i_patch], color=color, zorder=0, linewidth=4)
                    #ax.plot(x_list[i_patch], y_list[i_patch], color="black", alpha=0.5, zorder=10, linewidth=4)
                    # to show density, add this to previous call: , alpha=patch_densities[i_patch][0]
                    # ax.annotate(str(i_patch), xy=(patch_centers[i_patch][0] + 80, patch_centers[i_patch][1] + 80), color='white')

        # In any case, plot worm trajectory
        # Plot the trajectory with a colormap based on the speed of the worm
        if plot_speed:
            speed_list = current_traj.reset_index()["speeds"]
            lines = colored_line_plot.colored_line(current_list_x, current_list_y, c=speed_list*param.one_pixel_in_mm, ax=ax, cmap="hot", zorder=1.3, linewidths=3)
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                clb = plt.gcf().colorbar(lines)
                clb.ax.tick_params(labelsize=12)
                clb.ax.set_title('Speed (mm / s)', fontsize=16)

        # Plot the trajectory with a colormap based on time
        if plot_time:
            nb_of_timepoints = len(current_list_x)
            bin_size = 100
            for current_bin in range(nb_of_timepoints // bin_size):
                lower_bound = current_bin * bin_size
                upper_bound = min((current_bin + 1) * bin_size, len(current_list_x))
                plt.scatter(current_list_x[lower_bound:upper_bound], current_list_y[lower_bound:upper_bound],
                            c=range(lower_bound, upper_bound), cmap="hot", s=0.5)
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                plt.colorbar()

        # Plot black dots when the worm is inside
        if plot_in_patch:
            plt.scatter(current_list_x, current_list_y,
                        color=param.name_to_color[param.nb_to_distance[condition_list[0]]], s=.5)
            if "patch_silhouette" in current_traj.columns:
                indexes_in_patch = np.where(current_traj["patch_silhouette"] != -1)
            else:
                indexes_in_patch = np.where(current_traj["patch_centroid"] != -1)
            plt.scatter(current_list_x.iloc[indexes_in_patch], current_list_y.iloc[indexes_in_patch], color='black',
                        s=.5, zorder=10)
            plt.text(100, 100, "total_visit_time=" + str(len(indexes_in_patch[0])), color='white')

        # Plot markers where the tracks start, interrupt and restart
        # Good hole: green markers
        # Bad hole: blue markers
        # Start of a hole: cross
        # End of a hole: star
        # Black star: start of the tracking
        # THIS IS BROKEN: to fix it, look for tracking holes by looking for frames where worm id switches
        if plot_continuity:
            list_of_track_id = np.array(current_traj["id_conservative"])
            track_id_changes = list_of_track_id[1:] - list_of_track_id[:-1]
            track_id_changes_index = np.where(track_id_changes != 0)[0]
            for i_hole in range(len(track_id_changes_index)):
                current_hole_start_index = track_id_changes_index[i_hole]
                current_hole_end_index = track_id_changes_index[i_hole] + 1
                current_hole_start = current_traj.reset_index(drop=True).loc[current_hole_start_index]
                current_hole_end = current_traj.reset_index(drop=True).loc[current_hole_end_index]
                # If good hole, plot start and end as a GREEN cross / star
                if current_hole_start["patch_silhouette"] == current_hole_end["patch_silhouette"]:
                    plt.scatter(current_hole_start["x"], current_hole_start["y"], marker='X', color="green", s=100)
                    plt.scatter(current_hole_end["x"], current_hole_end["y"], marker='*', color="green", s=200)
                # If bad hole, plot start and end as a BLUE cross / star
                if current_hole_start["patch_silhouette"] != current_hole_end["patch_silhouette"]:
                    plt.scatter(current_hole_start["x"], current_hole_start["y"], marker='X', color="blue", s=100)
                    plt.scatter(current_hole_end["x"], current_hole_end["y"], marker='*', color="blue", s=200)
            # First tracked point
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                plt.scatter(current_list_x[0], current_list_y[0], marker='*', color="black", s=200)

        # Plot the trajectory, one color per worm
        else:
            plt.scatter(current_list_x, current_list_y,
                        color=param.name_to_color[param.nb_to_distance[condition_list[0]]], s=1.6)
            if plot_lines:
                plt.plot(np.array(current_list_x), np.array(current_list_y),
                         color=param.name_to_color[param.nb_to_distance[condition_list[0]]], linewidth=3)

        previous_folder = current_folder

    if save_fig:
        plt.savefig(
            path + "trajectory_plots/condition_" + str(condition_list) + "_" + folder_list[-1].split("/")[-2] + ".png")
    if is_plot:
        plt.show()
    else:
        plt.clf()


# Analysis functions
def plot_speed_time_window_list(traj, list_of_time_windows, nb_resamples, in_patch=False, out_patch=False,
                                is_plot=True):
    # TODO take care of holes in traj.csv
    """
    Will take the trajectory dataframe and exit the following plot:
    x-axis: time-window size
    y-axis: average proportion of time spent on food over that time-window
    color: current speed
    in_patch / out_patch: only take time points such as the worm is currently inside / outside a patch
                          if both are True or both are False, will take any time point
    This function will show nb_resamples per time window per plate.
    """
    # Will take all the different folders (plate names) in traj
    plate_list = np.unique(traj["folder"])
    nb_of_plates = len(plate_list)
    normalize = mplcolors.Normalize(vmin=0, vmax=3.5)  # for the plots

    # For each time window
    for i_window in range(len(list_of_time_windows)):
        # Initialize
        window_size = list_of_time_windows[i_window]  # size of current time window
        average_food_list = np.zeros(nb_of_plates * nb_resamples)  # list to fill with average past food
        current_speed_list = np.zeros(nb_of_plates * nb_resamples)  # list to fill with current speed of worm
        # For each plate
        for i_plate in range(nb_of_plates):
            plate = plate_list[i_plate]
            current_traj = traj[traj["folder"] == plate].reset_index()
            # Pick a random time to look at, cannot be before window_size otherwise not enough past for avg food
            for i_resample in range(nb_resamples):
                if len(current_traj) > 10000:
                    random_time = random.randint(window_size, len(current_traj) - 1)

                    # Only take current times such as the worm is inside a patch
                    if in_patch and not out_patch:
                        n_trials = 0
                        while current_traj["patch"][random_time] == -1 and n_trials < 100:
                            random_time = random.randint(window_size, len(current_traj) - 1)
                            n_trials += 1
                        if n_trials == 100:
                            print("No in_patch position was  for window, plate:", str(window_size), ", ", plate)

                    # Only take current times such as the worm is outside a patch
                    if out_patch and not in_patch:
                        n_trials = 0
                        while current_traj["patch"][random_time] != -1 and n_trials < 100:
                            random_time = random.randint(window_size, len(current_traj) - 1)
                            n_trials += 1
                        if n_trials == 100:
                            print("No out_patch position was  for window, plate:", str(window_size), ", ", plate)

                    # Look for first index of the trajectory where the frame is at least window_size behind
                    # This is because if we just take random time - window-size there might be a tracking hole in the traj
                    first_index = current_traj[
                        current_traj["frame"] <= current_traj["frame"][random_time] - window_size].index.values
                    if len(first_index) > 0:  # if there is such a set of indices
                        first_index = first_index[-1]  # take the latest one
                        traj_window = traj[first_index: random_time]
                        # Compute average feeding rate over that window and current speed
                        average_food_list[i_plate * nb_resamples + i_resample] = len(
                            traj_window[traj_window["patch"] != -1]) / window_size
                        current_speed_list[i_plate * nb_resamples + i_resample] = current_traj["distances"][random_time]
                    else:  # otherwise it means the video is not long enough
                        average_food_list[i_plate + i_resample] = -1
                        current_speed_list[i_plate + i_resample] = -1

                else:
                    average_food_list[i_plate + i_resample] = -1
                    current_speed_list[i_plate + i_resample] = -1
        # Sort all lists according to speed (in one line sorry oopsie)
        current_speed_list, average_food_list = zip(*sorted(zip(current_speed_list, average_food_list)))
        # Plot for this window size
        plt.scatter([window_size + current_speed_list[i] for i in range(len(current_speed_list))], average_food_list,
                    c=current_speed_list, cmap="viridis", norm=normalize)
    if is_plot:
        plt.colorbar()
        plt.show()


def plot_speed_time_window_continuous(traj, time_window_min, time_window_max, step_size, nb_resamples, current_speed,
                                      speed_history, past_speed, is_plot=True):
    # TODO take care of holes in traj.csv
    """
    === Will take the trajectory dataframe and:
    start and end for the time windows
    step size by which to increase time window
    nb of times to do a random resample in the video
    3 bool values to describe which speed is plotted as color
    === Exits the following plot:
    x-axis: time-window size
    y-axis: average proportion of time spent on food over that time-window
    color: current speed
    Note: it will show one point per time window per plate because otherwise wtf
    """
    plate_list = np.unique(traj["folder"])
    random_plate = plate_list[random.randint(0, len(plate_list))]
    random_traj = traj[traj["folder"] == random_plate].reset_index()
    condition = fd.folder_to_metadata(random_plate)["condition"].reset_index()
    for n in range(nb_resamples):
        present_time = random.randint(0, len(random_traj))
        normalize = mplcolors.Normalize(vmin=0, vmax=3.5)
        window_size = time_window_min - step_size
        while window_size < min(len(random_traj), time_window_max):
            window_size += step_size
            traj_window = traj[present_time - window_size:present_time]
            average_food = len(traj_window[traj_window["patch"] != -1]) / window_size
            speed = 0
            if current_speed:
                speed = random_traj["distances"][present_time]
            if speed_history:
                speed = np.mean(traj_window["distances"])
            if past_speed:
                speed = np.mean(traj_window["distances"][0:window_size])
            # Plot for this window size
            plt.scatter(window_size, average_food, c=speed, cmap="viridis", norm=normalize)

    if is_plot:
        plt.colorbar()
        plt.ylabel("Average feeding rate")
        plt.xlabel("Time window to compute past average feeding rate")
        plt.title(str(condition["condition"][0]) + ", " + str(random_plate)[-48:-9])
        plt.show()


def binned_speed_as_a_function_of_time_window(traj, condition_list, list_of_time_windows, list_of_food_bins,
                                              nb_resamples, in_patch=False, out_patch=False, is_plot=True):
    """
    Function that takes a table of trajectories, a list of time windows and food bins,
    and will plot the CURRENT SPEED for each time window and for each average food during that time window
    FOR NOW, WILL TAKE nb_resamples RANDOM TIMES IN EACH PLATE
    """
    # Prepare plate list
    plate_list = fd.return_folders_condition_list(np.unique(traj["folder"]), condition_list)
    nb_of_plates = len(plate_list)

    # This is for x ticks for the final plot
    list_of_x_positions = []
    colors = plt.cm.plasma(np.linspace(0, len(list_of_time_windows)))

    # Fill lists with info for each plate
    for i_window in range(len(list_of_time_windows)):
        window_size = list_of_time_windows[i_window]
        average_food_list = np.zeros(nb_of_plates * nb_resamples)
        current_speed_list = np.zeros(nb_of_plates * nb_resamples)
        for i_plate in range(nb_of_plates):
            if i_plate % 20 == 0:
                print("Computing for plate ", i_plate, "/", nb_of_plates)
            plate = plate_list[i_plate]
            current_traj = traj[traj["folder"] == plate].reset_index()

            # Pick a random time to look at, cannot be before window_size otherwise not enough past for avg food
            for i_resample in range(nb_resamples):

                # TODO correct this interval, it should be between last frame and first frame that is at least window size
                random_time = random.randint(window_size, len(current_traj) - 1)

                # Only take current times such as the worm is inside a patch
                if in_patch and not out_patch:
                    n_trials = 0
                    while current_traj["patch"][random_time] == -1 and n_trials < 100:
                        random_time = random.randint(window_size, len(current_traj) - 1)
                        n_trials += 1
                    if n_trials == 100:
                        print("No in_patch position was found for window, plate:", str(window_size), ", ", plate)

                # Only take current times such as the worm is outside a patch
                if out_patch and not in_patch:
                    n_trials = 0
                    while current_traj["patch"][random_time] != -1 and n_trials < 100:
                        random_time = random.randint(window_size, len(current_traj) - 1)
                        n_trials += 1
                    if n_trials == 100:
                        print("No out_patch position was found for window, plate:", str(window_size), ", ", plate)

                # Look for first index of the trajectory where the frame is at least window_size behind
                # This is because if we just take random time - window-size there might be a tracking hole in the traj
                first_index = current_traj[
                    current_traj["frame"] <= current_traj["frame"][random_time] - window_size].index.values
                if len(first_index) > 0:  # if there is such a set of indices
                    first_index = first_index[-1]  # take the latest one
                    traj_window = current_traj[first_index: random_time]
                    # Compute average feeding rate over that window and current speed
                    average_food_list[i_plate * nb_resamples + i_resample] = len(
                        traj_window[traj_window["patch"] != -1]) / window_size
                    current_speed_list[i_plate * nb_resamples + i_resample] = current_traj["distances"][random_time]
                else:  # otherwise, it means the video is not long enough
                    average_food_list[i_plate + i_resample] = -1
                    current_speed_list[i_plate + i_resample] = -1

        # Sort all lists according to average_food (in one line sorry oopsie)
        average_food_list, current_speed_list = zip(*sorted(zip(average_food_list, current_speed_list)))
        print("Finished computing for window = ", window_size)

        # Fill the binsss
        binned_avg_speeds = np.zeros(len(list_of_food_bins))
        errorbars_sup = []
        errorbars_inf = []
        i_food = 0
        for i_bin in range(len(list_of_food_bins)):
            list_curr_speed_this_bin = []
            # While avg food is not above bin, continue filling it
            while i_food < len(average_food_list) and average_food_list[i_food] <= list_of_food_bins[i_bin]:
                list_curr_speed_this_bin.append(current_speed_list[i_food])
                i_food += 1
            # Once the bin is over, fill stat info for global plot
            binned_avg_speeds[i_bin] = np.mean(list_curr_speed_this_bin)
            errors = ana.bottestrop_ci(list_curr_speed_this_bin, 1000)
            errorbars_inf.append(binned_avg_speeds[i_bin] - errors[0])
            errorbars_sup.append(errors[1] - binned_avg_speeds[i_bin])
            # and plot individual points
            plt.scatter([2 * i_window + list_of_food_bins[i_bin] for _ in range(len(list_curr_speed_this_bin))],
                        list_curr_speed_this_bin, zorder=2, color="gray")
            # Plot violins in the bgd
            try:
                violins = plt.violinplot(list_curr_speed_this_bin, positions=[2 * i_window + list_of_food_bins[i_bin]],
                                         showextrema=False)
                # Setting the color of the violins using dark magic
                for pc in violins['bodies']:
                    pc.set_facecolor('gray')

            except ValueError:
                pass
            # Indicate on graph the nb of points in each bin
            if list_curr_speed_this_bin:
                ax = plt.gca()
                ax.annotate(str(len(list_curr_speed_this_bin)),
                            xy=(2 * i_window + list_of_food_bins[i_bin], max(list_curr_speed_this_bin) + 0.5))
            print("Finished binning for bin ", i_bin, "/", len(list_of_food_bins))

        # Plot for this window size
        x_positions = [2 * i_window + list_of_food_bins[i] for i in range(len(list_of_food_bins))]  # for the bars
        list_of_x_positions += x_positions  # for the final plot
        plt.bar(x_positions, binned_avg_speeds, width=min(0.1, 1 / len(list_of_food_bins)), label=str(window_size),
                color=colors[i_window])
        plt.errorbar(x_positions, binned_avg_speeds, [errorbars_inf, errorbars_sup], fmt='.k', capsize=5)

    ax = plt.gca()
    ax.set_xticks(list_of_x_positions)
    ax.set_xticklabels(
        [str(np.round(list_of_food_bins[i], 2)) for i in range(len(list_of_food_bins))] * len(list_of_time_windows))

    if is_plot:
        plt.xlabel("Average amount of food during time window")
        plt.ylabel("Average speed")
        plt.legend(title="Time window size")
        plt.show()


def plot_selected_data(results, plot_title, condition_list, column_name, divided_by="",
                       plot_model=False, is_plot=True, normalize_by_video_length=False,
                       remove_censored_events=False, soft_cut=False, hard_cut=False,
                       only_first_visited_patch=False, visits_longer_than=0):
    """
    This function will make a bar plot from the selected part of the data. Selection is described as follows:
    - condition_list: list of conditions you want to plot (each condition = one bar)
    - column_name: column to plot from the results (y-axis)
    - divided_by: name of a column whose values will divide column name
    eg : if column_name="total_visit_time" and divided_by="nb_of_visits", it will output the average visit time
    """
    # Getting results
    if column_name == "first_visit_duration":
        list_of_avg_each_plate, average_per_condition, errorbars = first_visit_script.bar_plot_first_visit_each_patch(
            results, condition_list, is_plot=False, remove_censored_events=remove_censored_events, only_first_visited_patch=only_first_visited_patch, visits_longer_than=visits_longer_than)
    else:
        list_of_avg_each_plate, average_per_condition, errorbars = ana.results_per_condition(results, condition_list,
                                                                                             column_name, divided_by,
                                                                                             remove_censored_events=remove_censored_events,
                                                                                             normalize_by_video_length=normalize_by_video_length,
                                                                                             only_first_visited_patch=only_first_visited_patch,
                                                                                             soft_cut=soft_cut,
                                                                                             hard_cut=hard_cut,
                                                                                             visits_longer_than=visits_longer_than)

    # Plotttt
    plt.title(plot_title, fontsize=24)
    fig = plt.gcf()
    ax = fig.gca()
    fig.set_size_inches(7, 8.6)
    # plt.style.use('dark_background')

    if column_name == "total_visit_time" and divided_by == "nb_of_visited_patches":
        ax.set_ylabel("Total time per patch (hours)", fontsize=20)
        #ax.set_ylim(0, 4)
    if column_name == "total_visit_time" and divided_by == "nb_of_visits":
        ax.set_ylabel("Average time per visit (hours)", fontsize=20)
        #ax.set_ylim(0, 0.6)
    if column_name == "first_visit_duration":
        if only_first_visited_patch:
            ax.set_ylabel("Duration of 1st visit of the video (hours)", fontsize=20)
        else:
            ax.set_ylabel("Duration of 1st visit to each patch (hours)", fontsize=20)
        #ax.set_ylim(0, 2.6)
    if column_name == "average_speed_inside":
        ax.set_ylabel("Average speed inside food patches (pixel/second)", fontsize=20)
    if column_name == "average_speed_outside":
        ax.set_ylabel("Average speed outside food patches (pixel/second)", fontsize=20)

    # Plot condition averages as a bar plot
    condition_names = [param.nb_to_name[cond] for cond in condition_list]
    condition_colors = [param.name_to_color[name] for name in condition_names]
    ax.bar(range(len(condition_list)), average_per_condition, color=condition_colors, label=condition_names)
    ax.set_xticks(range(len(condition_list)))
    ax.set_xticklabels(condition_names, rotation=45)
    # ax.set_xlabel("Inter-patch distance", fontsize=16)

    # Plot plate averages as scatter on top
    for i_cond in range(len(condition_list)):
        ax.scatter([range(len(condition_list))[i_cond] + 0.001 * random.randint(-100, 100) for p in
                    range(len(list_of_avg_each_plate[i_cond]))],
                   list_of_avg_each_plate[i_cond], color="orange", zorder=2, alpha=0.6)

    # Plot error bars
    ax.errorbar(range(len(condition_list)), average_per_condition, errorbars, fmt='.k', capsize=5)

    # Plot model as a dashed line on top
    if plot_model:
        model_per_condition = ana.model_per_condition(results, condition_list, column_name, divided_by)
        ax.plot(range(len(condition_list)), model_per_condition, linestyle="dashed", color="blue")

    if is_plot:
        # Set the x labels to the distance icons!
        # Stolen from https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings
        for i in range(len(condition_list)):
            ax = plt.gcf().gca()
            ax.set_xticks([])

            # Image to use
            arr_img = plt.imread(
                # "/home/admin/Desktop/Camera_setup_analysis/Video_analysis/Parameters/icon_" + param.nb_to_distance[
                "C://Users//Asmar//Desktop//These//2022_summer_videos//analysis//Parameters//icon_" + param.nb_to_distance[
                    condition_list[i]] + '.png')

            # Image box to draw it!
            imagebox = OffsetImage(arr_img, zoom=0.8)
            imagebox.image.axes = ax

            x_annotation_box = AnnotationBbox(imagebox, (i, 0),
                                              xybox=(0, -8),
                                              # that's the shift that the image will have compared to (i, 0)
                                              xycoords=("data", "axes fraction"),
                                              boxcoords="offset points",
                                              box_alignment=(.5, 1),
                                              bboxprops={"edgecolor": "none"})

            ax.add_artist(x_annotation_box)

        # Print statistical test results
        print("Variable = ", column_name, ", divided_by = ", divided_by)
        for i in range(len(condition_list)):
            normality_test = scipy.stats.normaltest(list_of_avg_each_plate[i], nan_policy="omit")
            print("Condition: ", param.nb_to_name[condition_list[i]], ", pvalue = ", normality_test.pvalue)
            if normality_test.pvalue < 0.05:
                print("(The data is not normal.)")
            else:
                print("(The data is normal.)")
        stat_test = scipy.stats.alexandergovern(list_of_avg_each_plate[0], list_of_avg_each_plate[1], list_of_avg_each_plate[2], list_of_avg_each_plate[3], nan_policy="omit")
        y_axis_limits = plt.gca().get_ylim()
        x_axis_limits = plt.gca().get_xlim()
        if not np.isnan(stat_test.pvalue):
            # If the p_value is less than 0.01, write in scientific notation
            if int(stat_test.pvalue*1000) == 0:
                plt.text(0.4*x_axis_limits[1], 0.9*y_axis_limits[1], "p-value = "+str(np.format_float_scientific(stat_test.pvalue, 3)), fontsize=16)
            # Else write it
            else:
                plt.text(0.46*x_axis_limits[1], 0.9*y_axis_limits[1], "p-value = "+str(np.round(stat_test.pvalue, 3)), fontsize=16)
        else:
            plt.text(0.46 * x_axis_limits[1], 0.9 * y_axis_limits[1], "p-value is np.nan")
        plt.show()
    else:
        return average_per_condition, list_of_avg_each_plate, errorbars


def plot_visit_time(results, trajectory, plot_title, condition_list, variable, condition_names, split_conditions=True,
                    is_plot=True, patch_or_pixel="patch", only_first=False, min_nb_data_points=10, custom_bins=None):
    # Call function to obtain list of visit lengths and corresponding list of variable values (one sublist per condition)
    full_visit_list, full_variable_list = ana.visit_time_as_a_function_of(results, trajectory, condition_list, variable,
                                                                          patch_or_pixel, only_first)

    # Plot the thing
    nb_cond = len(condition_list)
    data = pd.DataFrame()
    data["visit_duration"] = []
    data[variable] = []

    # Set bin size for the plots
    if variable == "speed_when_entering":
        bin_size = 0.2
    if variable == "visit_start":
        bin_size = 10000
    if variable == "last_travel_time":
        bin_size = 100

    if split_conditions:
        for i_cond in range(nb_cond):
            variable_values_bins, average_visit_duration, [errors_inf,
                                                           errors_sup], binned_current_visits = ana.xy_to_bins(
                full_variable_list[i_cond], full_visit_list[i_cond], bin_size=bin_size, print_progress=False)

            condition_name = condition_names[i_cond]
            condition_color = param.name_to_color[condition_name]

            if is_plot:
                plt.scatter(full_variable_list[i_cond], full_visit_list[i_cond], color=condition_color, alpha=0.2)
                plt.plot(variable_values_bins, average_visit_duration, color=condition_color, linewidth=4,
                         label=condition_name)
                plt.errorbar(variable_values_bins, average_visit_duration, [errors_inf, errors_sup], fmt='.k',
                             capsize=5)
                plt.title(plot_title)
                label_y = "visit duration to " + patch_or_pixel
                if only_first != False:
                    label_y = "first " + str(only_first) + " " + label_y
                plt.ylabel(label_y)
                plt.xlabel(variable)

    if not split_conditions:
        # Merge the conditions sublists (go from [[values for condition x], [values for condition y]] to [all values])
        full_visit_list = [full_visit_list[i_cond][i_visit] for i_cond in range(len(full_visit_list)) for i_visit in
                           range(len(full_visit_list[i_cond]))]
        full_variable_list = [full_variable_list[i_cond][i_visit] for i_cond in range(len(full_variable_list)) for
                              i_visit in range(len(full_variable_list[i_cond]))]

        variable_values_bins, average_visit_duration, [errors_inf, errors_sup], binned_current_visits = ana.xy_to_bins(
            full_variable_list, full_visit_list, bin_size=bin_size, print_progress=False,
            custom_bins=custom_bins)
        # Exclude the ones that have 100 visits or fewer
        variable_values_bins = [variable_values_bins[i] for i in range(len(variable_values_bins)) if
                                len(binned_current_visits[i]) > min_nb_data_points]
        average_visit_duration = [average_visit_duration[i] for i in range(len(average_visit_duration)) if
                                  len(binned_current_visits[i]) > min_nb_data_points]
        errors_inf = [errors_inf[i] for i in range(len(errors_inf)) if
                      len(binned_current_visits[i]) > min_nb_data_points]
        errors_sup = [errors_sup[i] for i in range(len(errors_sup)) if
                      len(binned_current_visits[i]) > min_nb_data_points]

        condition_name = param.nb_list_to_name[str(sorted(condition_list))]
        condition_color = param.name_to_color[condition_name]

        if is_plot:
            plt.plot(variable_values_bins, average_visit_duration, color=condition_color, linewidth=4,
                     label=condition_name)
            plt.errorbar(variable_values_bins, average_visit_duration, [errors_inf, errors_sup], fmt='.k', capsize=5)
            plt.title(plot_title)
            label_y = "visit duration to " + patch_or_pixel
            if only_first:
                label_y = "first " + label_y

            plt.ylabel(label_y)
            plt.xlabel(variable)
            plt.xscale("log")

    if is_plot:
        plt.legend()
        plt.show()
    else:
        return variable_values_bins, average_visit_duration, [errors_inf, errors_sup]


def plot_variable_distribution(results, curve_list, variable_list=None, scale_list=None,
                               plot_cumulative=True, threshold_list=None, is_plot=True, only_first=False,
                               end_time=False):
    """
    Will plot a distribution of each variable from variable_list in results, for conditions in each curve of curve_list.
        variable_list: can contain any argument that the return_value_list function can take
        scale_list: say if you want to plot the y-axis linear scale, log scale or both
        plot_cumulative: plot cumulative histograms or not
        to_same_patch: if False, will only plot transits that go from one patch to another
        to_different_patch: if False, will only plot transits that leave and come back to the same patch
        threshold_list: list of thresholds for aggregation of visits
    """
    if scale_list is None:
        scale_list = ["linear", "log"]  # could also be "linear"
    if variable_list is None:
        variable_list = ["visits", "same transits", "cross transits"]

    # If an aggregation related column is called for, generate it in results if it's not done yet
    if "aggregated_visits" in variable_list or "aggregated_leaving_events" in variable_list:
        # If aggregation is in variable list, we want to plot all the effects ("distance", "density", etc.) but with
        # one subplot per possible aggregation threshold
        for i_thresh in range(len(threshold_list)):
            column_name = "aggregated_visits_thresh_" + str(threshold_list[i_thresh])
            if column_name not in results.columns:
                # If the aggregated visits have not been generated yet for this threshold values
                results = gen.generate_aggregated_visits(gen.generate(), [
                    threshold_list[i_thresh]])  # add them to the clean_results.csv table
        if "aggregated_visits" in variable_list:
            for thresh in threshold_list:
                variable_list.append("aggregated_visits_thresh_" + str(thresh) + "_visit_durations")
            variable_list.remove("aggregated_visits")
        if "aggregated_leaving_events" in variable_list:
            for thresh in threshold_list:
                variable_list.append("aggregated_visits_thresh_" + str(thresh) + "_leaving_events_time_stamps")
            variable_list.remove("aggregated_leaving_events")

    # Initialize plot
    curve_names = [param.nb_list_to_name[str(curve)] for curve in curve_list]
    fig, axs = plt.subplots(len(scale_list), len(variable_list))
    fig.set_size_inches(7 * len(variable_list), 6 * len(scale_list))
    fig.suptitle("Conditions " + str(curve_names))
    fig.set_tight_layout(True)  # make the margins tighter

    for i_variable in range(len(variable_list)):
        variable = variable_list[i_variable]
        avg_values = []
        median_values = []
        color_list = []
        ax_lim_list = []
        if variable == "cross transits":
            bins = np.linspace(0, 25000, 60)
        else:
            bins = np.linspace(0, 8500, 30)
        for i_scale in range(len(scale_list)):
            if len(scale_list) > 1 and len(variable_list) > 1:
                ax = axs[i_scale, i_variable]
            elif len(scale_list) == 1:
                ax = axs[i_variable]
            elif len(variable_list) == 1:
                ax = axs[i_scale]
            else:
                ax = axs
            if i_variable == 0:
                ax.set(ylabel="normalized " + scale_list[i_scale] + " occurrences")
            if i_scale == 0:
                ax.set_title(
                    "first " * only_first + str(variable) + " values" + (end_time != False) * "until " + str(end_time))
            ax.set_yscale(scale_list[i_scale])
            # For every condition pool in condition_list
            for i_curve in range(len(curve_list)):
                conditions = curve_list[i_curve]
                name = curve_names[i_curve]
                if "aggregated" in variable and "transits" not in variable:
                    # This is because our aggregated visits not including transits are in sub-lists already containing durations
                    values = ana.return_value_list(results, variable, conditions, convert_to_duration=False)
                    values = [sublist[i] for sublist in values for i in range(len(sublist))]
                else:
                    values = ana.return_value_list(results, variable, conditions, convert_to_duration=True,
                                                   only_first=only_first, end_time=end_time)
                avg_values.append(np.mean(values))
                print("Average value for curve ", str(name), ": ", np.mean(values))
                median_values.append(np.median(values))
                print("Median value for curve ", str(name), ": ", np.median(values))
                color_list.append(param.name_to_color[name])
                ax.hist(values, bins=bins, density=True, cumulative=-plot_cumulative, label=name, histtype="step",
                        color=param.name_to_color[name], linewidth=3)
                ax_lim_list.append(ax.get_ylim())

        ax.vlines(avg_values, color=color_list, ymin=0, ymax=1, linestyles="dashed", transform=ax.get_xaxis_transform(),
                  label="average")
        ax.vlines(median_values, color=color_list, ymin=0, ymax=1, linestyles="dotted",
                  transform=ax.get_xaxis_transform(), label="median")

    if is_plot:
        plt.legend()
        plt.show()


def plot_leaving_delays(results, plot_title, condition_list, bin_size, color, is_plot=True):
    leaving_delays, corresponding_time_in_patch = ana.delays_before_leaving(results, condition_list)
    leaving_delays_in_one_list = [leaving_delays[i] for i in range(len(leaving_delays))]
    binned_times_in_patch, avg_leaving_delays, y_err_list, full_value_list = ana.xy_to_bins(corresponding_time_in_patch,
                                                                                            leaving_delays_in_one_list,
                                                                                            bin_size)
    plt.title(plot_title)
    plt.ylabel("Average delay before next exit")
    plt.xlabel("Time already spent in patch")

    # Make a violin plot
    nans = [float('nan'), float('nan')]
    parts = plt.violinplot([val or nans for val in full_value_list],
                           [binned_times_in_patch[i] + bin_size / 2 for i in range(len(binned_times_in_patch))],
                           showmedians=True, showextrema=True, widths=bin_size)

    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(1)

    for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[part_name]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)

    # Plot number of values in each bin
    for i_bin in range(len(binned_times_in_patch)):
        plt.annotate(str(len(full_value_list[i_bin])),
                     [binned_times_in_patch[i_bin] + 100, np.max(full_value_list[i_bin]) + 100])

    if is_plot:
        plt.show()


def plot_leaving_probability(results, plot_title, condition_list, custom_bins, worm_limit, color, label,
                             split_conditions=False, is_plot=True, is_nb_of_worms=False):
    plt.title(plot_title)
    plt.ylabel("Probability of exiting in the next " + str(param.time_threshold) + " time steps")
    plt.xlabel("Time already spent in patch")
    plt.yscale("log")

    if not split_conditions:
        binned_times_in_patch, binned_leaving_probability, error_bars, nb_of_worms = ana.leaving_probability(results,
                                                                                                             condition_list,
                                                                                                             custom_bins,
                                                                                                             worm_limit,
                                                                                                             errorbars=True)

        plt.plot(binned_times_in_patch, binned_leaving_probability, color=color, label=label, linewidth=2)
        if error_bars:
            plt.errorbar(binned_times_in_patch, binned_leaving_probability, error_bars, fmt='.k', capsize=5)

        # Plot number of worms that are in each bin
        if is_nb_of_worms:
            for i_bin in range(len(binned_times_in_patch)):
                plt.annotate(str(int(nb_of_worms[i_bin])),
                             [binned_times_in_patch[i_bin], binned_leaving_probability[i_bin] * 2 + 0.02])

    if split_conditions:
        for i_condition in range(len(condition_list)):
            current_condition = condition_list[i_condition]
            binned_times_in_patch, binned_leaving_probability, error_bars, nb_of_worms = ana.leaving_probability(
                results,
                [current_condition],
                custom_bins,
                worm_limit,
                min_visit_length=0,
                errorbars=True)

            # If there are lines to plot, plot them !
            if len(binned_times_in_patch) > 1:
                plt.plot(binned_times_in_patch, binned_leaving_probability,
                         color=param.name_to_color[param.nb_to_density[current_condition]], linewidth=3)
                plt.errorbar(binned_times_in_patch, binned_leaving_probability, error_bars,
                             label="OD=" + param.nb_to_density[current_condition],
                             color=param.name_to_color[param.nb_to_density[current_condition]],
                             capsize=5)
            # If there's just one point, plot it but put an outline to make it visible (it's usually the control so very light)
            else:
                plt.errorbar(binned_times_in_patch, binned_leaving_probability, error_bars,
                             label="OD=" + param.nb_to_density[current_condition],
                             color=param.name_to_color[param.nb_to_density[current_condition]],
                             capsize=5, path_effects=[pe.Stroke(linewidth=3,
                                                                foreground=param.name_to_color["0.2"]),
                                                      pe.Normal()])

    if is_plot:
        # plt.xscale("log")
        plt.legend()
        plt.show()


def plot_test(results):
    full_visit_list, full_variable_list = ana.visit_time_as_a_function_of(results, [0], "visit_start")
    plt.scatter(full_visit_list, full_variable_list)
    plt.show()
