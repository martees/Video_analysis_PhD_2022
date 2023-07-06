import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import random
import seaborn as sns

import analysis as ana
import generate_results as gr
import find_data as fd
import parameters

# Sanity check functions


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
    for folder in folder_list:
        metadata = fd.folder_to_metadata(folder)
        patch_centers = metadata["patch_centers"]

        lentoremove = len('traj.csv')  # removes traj from the current path, to get to the parent folder
        folder = folder[:-lentoremove]

        if is_plot:
            fig, ax = plt.subplots()
            if show_composite:
                composite = plt.imread(folder + "composite_patches.tif")
                ax.imshow(composite)
            else:
                background = plt.imread(folder + "background.tif")
                ax.imshow(background, cmap='gray')
            ax.use_sticky_edges = False

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
                radiuses[i_angle] = gr.spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                                    patch_spline_coefs[i_patch])

            fig = plt.gcf()

            # Create lists of cartesian positions out of this
            x_pos = []
            y_pos = []
            for point in range(len(angular_pos)):
                x_pos.append(patch_centers[i_patch][0] + (radiuses[point] * np.sin(angular_pos[point])))
                y_pos.append(patch_centers[i_patch][1] + (radiuses[point] * np.cos(angular_pos[point])))

            # Either plot them
            if is_plot:
                plt.plot(x_pos, y_pos, color=colors[i_patch])
                plt.scatter([patch_guides[i_patch][i][0] for i in range(len(patch_guides[i_patch]))], [patch_guides[i_patch][i][1] for i in range(len(patch_guides[i_patch]))], color="white", s=15)
            # Or add them to a list for later
            else:
                x_list.append(x_pos)
                y_list.append(y_pos)

        if is_plot:
            plt.title(folder)
            plt.show()
        else:
            return x_list, y_list


def trajectories_1condition(traj, i_condition, n_max=4, is_plot_patches=False, show_composite=True, plot_in_patch=False,
                            plot_continuity=False, plot_speed=False, plot_time=False, plate_list=[]):
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
    if plate_list:
        folder_list = []
        for i_plate in range(len(plate_list)):
            folder_list.append(traj[traj["folder"] == plate_list[i_plate]]["folder"])
        folder_list = np.unique(folder_list)
    # Otherwise take all the plates of the condition
    else:
        folder_list = fd.return_folders_condition_list(np.unique(traj["folder"]), [i_condition])
    nb_of_folders = len(folder_list)
    colors = plt.cm.jet(np.linspace(0, 1, nb_of_folders))
    previous_folder = 0
    n_plate = 1
    for i_folder in range(nb_of_folders):
        current_folder = folder_list[i_folder]
        current_traj = traj[traj["folder"] == current_folder]
        current_list_x = current_traj.reset_index()["x"]
        current_list_y = current_traj.reset_index()["y"]
        metadata = fd.folder_to_metadata(current_folder)
        plt.suptitle("Trajectories for condition " + str(i_condition))

        # If we just changed plate or if it's the 1st, plot the background elements
        if previous_folder != current_folder or previous_folder == 0:
            if n_plate > n_max:  # If we exceeded the max nb of plates per graphic
                plt.show()
                n_plate = 1
            if n_plate <= n_max or len(plate_list) > 1:
                plt.subplot(n_max // 2, n_max // 2, n_plate)
                n_plate += 1
            # Show background and patches
            fig = plt.gcf()
            ax = fig.gca()
            fig.set_tight_layout(True)  # make the margins tighter
            if show_composite:  # show composite with real patches
                composite = plt.imread(current_folder[:-len("traj.csv")] + "composite_patches.tif")
                ax.imshow(composite)
            else:  # show cleaner background without the patches
                background = plt.imread(current_folder[:-len("traj.csv")] + "background.tif")
                ax.imshow(background, cmap='gray')
            ax.set_title(str(current_folder[-48:-9]))
            # Plot them patches
            if is_plot_patches:
                patch_densities = metadata["patch_densities"]
                patch_centers = metadata["patch_centers"]
                x_list, y_list = patches([current_folder], is_plot=False)
                for i_patch in range(len(patch_centers)):
                    ax.plot(x_list[i_patch], y_list[i_patch], color='yellow')
                    # to show density, add this to previous call: , alpha=patch_densities[i_patch][0]
                    # ax.annotate(str(i_patch), xy=(patch_centers[i_patch][0] + 80, patch_centers[i_patch][1] + 80), color='white')

        # In any case, plot worm trajectory
        # Plot the trajectory with a colormap based on the speed of the worm
        if plot_speed:
            distance_list = current_traj.reset_index()["distances"]
            normalize = mplcolors.Normalize(vmin=0, vmax=3.5)
            plt.scatter(current_list_x, current_list_y, c=distance_list, cmap="hot", norm=normalize, s=1, zorder=1.3)
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                plt.colorbar()

        # Plot the trajectory with a colormap based on time
        if plot_time:
            nb_of_timepoints = len(current_list_x)
            bin_size = 100
            # colors = plt.cm.jet(np.linspace(0, 1, nb_of_timepoints//bin_size))
            for current_bin in range(nb_of_timepoints // bin_size):
                lower_bound = current_bin * bin_size
                upper_bound = min((current_bin + 1) * bin_size, len(current_list_x))
                plt.scatter(current_list_x[lower_bound:upper_bound], current_list_y[lower_bound:upper_bound],
                            c=range(lower_bound, upper_bound), cmap="hot", s=0.5)
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                plt.colorbar()

        # Plot black dots when the worm is inside
        if plot_in_patch:
            plt.scatter(current_list_x, current_list_y, color=colors[i_folder], s=.5)
            indexes_in_patch = np.where(current_traj["patch"] != -1)
            plt.scatter(current_list_x.iloc[indexes_in_patch], current_list_y.iloc[indexes_in_patch], color='black',
                        s=.5, zorder=1.5)

        # Plot markers where the tracks start, interrupt and restart
        if plot_continuity:
            # Tracking stops
            plt.scatter(current_list_x.iloc[-1], current_list_y.iloc[-1], marker='X', color="red")
            # Tracking restarts
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                plt.scatter(current_list_x[0], current_list_y[0], marker='*', color="black", s=100)
            # First tracked point
            else:
                plt.scatter(current_list_x[0], current_list_y[0], marker='*', color="green")

        # Plot the trajectory, one color per worm
        else:
            plt.scatter(current_list_x, current_list_y, color=colors[i_folder], s=.5)

        previous_folder = current_folder

    plt.show()


def interactive_worm_silhouette(folder, frame):
    video.show_frames(folder, frame)


# Analysis functions

def plot_speed_time_window_list(traj, list_of_time_windows, nb_resamples, in_patch=False, out_patch=False):
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
    plt.colorbar()
    plt.show()
    return 0


def plot_speed_time_window_continuous(traj, time_window_min, time_window_max, step_size, nb_resamples, current_speed,
                                      speed_history, past_speed):
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
    plt.colorbar()
    plt.ylabel("Average feeding rate")
    plt.xlabel("Time window to compute past average feeding rate")
    plt.title(str(condition["condition"][0]) + ", " + str(random_plate)[-48:-9])
    plt.show()
    return 0


def binned_speed_as_a_function_of_time_window(traj, condition_list, list_of_time_windows, list_of_food_bins,
                                              nb_resamples, in_patch=False, out_patch=False):
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
    plt.xlabel("Average amount of food during time window")
    plt.ylabel("Average speed")
    plt.legend(title="Time window size")

    plt.show()
    return 0


def plot_selected_data(results, plot_title, condition_list, condition_names, column_name, divided_by="",
                       mycolor="blue", plot_model=False):
    """
    This function will make a bar plot from the selected part of the data. Selection is described as follows:
    - condition_list: list of conditions you want to plot (each condition = one bar)
    - column_name: column to plot from the results (y-axis)
    - divided_by: name of a column whose values will divide column name
    eg : if column_name="total_visit_time" and divided_by="nb_of_visits", it will output the average visit time
    """
    # Getting results
    list_of_avg_each_plate, average_per_condition, errorbars = ana.results_per_condition(results, condition_list,
                                                                                         column_name, divided_by)

    # Model results
    if plot_model:
        model_per_condition = ana.model_per_condition(results, condition_list, column_name, divided_by)

    # Plotttt
    plt.title(plot_title)
    fig = plt.gcf()
    ax = fig.gca()
    fig.set_size_inches(5, 6)

    # Plot condition averages as a bar plot
    ax.bar(range(len(condition_list)), average_per_condition, color=mycolor)
    ax.set_xticks(range(len(condition_list)))
    ax.set_xticklabels(condition_names, rotation=45)
    ax.set(xlabel="Condition number")

    # Plot plate averages as scatter on top
    for i in range(len(condition_list)):
        ax.scatter([range(len(condition_list))[i] for _ in range(len(list_of_avg_each_plate[i]))],
                   list_of_avg_each_plate[i], color="red", zorder=2)

    # Plot error bars
    ax.errorbar(range(len(condition_list)), average_per_condition, errorbars, fmt='.k', capsize=5)

    # Plot model as a dashed line on top
    if plot_model:
        ax.plot(range(len(condition_list)), model_per_condition, linestyle="dashed", color="blue")

    plt.show()


def plot_visit_time(results, trajectory, plot_title, condition_list, variable, condition_names, split_conditions=True):
    # Call function to obtain list of visit lengths and corresponding list of variable values (one sublist per condition)
    full_visit_list, full_variable_list = ana.visit_time_as_a_function_of(results, trajectory, condition_list, variable)

    # Plot the thing
    nb_cond = len(condition_list)
    data = pd.DataFrame()
    data["Visit duration"] = []
    data[variable] = []

    if split_conditions:
        for i_cond in range(nb_cond):
            # Plotting a linear regression on the thing
            data = pd.DataFrame()
            data["Visit duration"] = full_visit_list[i_cond]
            data[variable] = full_variable_list[i_cond]

            sns.jointplot(data=data, x=variable, y="Visit duration", kind="reg", marginal_kws=dict(bins=100),
                          marginal_ticks=True)
            fig = plt.gcf()
            ax = fig.gca()
            max_y = ax.get_ylim()[1]
            max_x = ax.get_xlim()[1]

            fig.suptitle(plot_title + ": " + condition_names[i_cond])
            ax.annotate("R2 = " + str(np.round(ana.r2(data[variable], data["Visit duration"]), 5)),
                        [max_x // 2, max_y * 3 / 4])

    if not split_conditions:
        data["Visit duration"] = pd.DataFrame(
            [full_visit_list[i_cond][i] for i_cond in range(len(full_visit_list)) for i in
             range(len(full_visit_list[i_cond]))])
        data[variable] = pd.DataFrame(
            [full_variable_list[i_cond][i] for i_cond in range(len(full_variable_list)) for i in
             range(len(full_variable_list[i_cond]))])
        sns.jointplot(data=data, x=variable, y="Visit duration", kind="reg", marginal_kws=dict(bins=100),
                      marginal_ticks=True)
        fig = plt.gcf()
        ax = fig.gca()
        max_y = ax.get_ylim()[1]
        max_x = ax.get_xlim()[1]
        fig.suptitle(plot_title + ": " + str(condition_list))
        ax.annotate("R2 = " + str(np.round(ana.r2(data[variable], data["Visit duration"]), 5)),
                    [max_x // 2, max_y * 3 / 4])

    plt.show()


def plot_variable_distribution(results, condition_list, effect_of="nothing", variable_list=None, scale_list=None,
                               plot_cumulative=True, threshold_list=None):
    """
    Will plot a distribution of each variable from variable_list in results, for conditions in condition_list.
        effect_of: if set to "nothing", will plot one curve for each condition in condition_list.
                   if set to "density" will make one curve for each bacterial density, pooling conditions together
                   if set to "distance" same but for distance
                   if set to "food", will pool all food conditions together, apart from control w/o food
                   if set to "aggregation"
        variable_list: can contain any argument that the return_value_list function can take
        scale_list: say if you want to plot the y-axis linear scale, log scale or both
        plot_cumulative: plot cumulative histograms or not
    to_same_patch: if False, will only plot transits that go from one patch to another
    to_different_patch: if False, will only plot transits that leave and come back to the same patch
    threshold_list: list of thresholds for aggregation of visits
    """
    if scale_list is None:
        scale_list = ["log"]  # could also be "linear
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
                results = gr.generate_aggregated_visits(gr.generate(), [
                    threshold_list[i_thresh]])  # add them to the clean_results.csv table
        if "aggregated_visits" in variable_list:
            for thresh in threshold_list:
                variable_list.append("aggregated_visits_thresh_" + str(thresh) + "_visit_durations")
            variable_list.remove("aggregated_visits")
        if "aggregated_leaving_events" in variable_list:
            for thresh in threshold_list:
                variable_list.append("aggregated_visits_thresh_" + str(thresh) + "_leaving_events_time_stamps")
            variable_list.remove("aggregated_leaving_events")

    # Pool conditions depending on the "effect_of" argument
    condition_pools, condition_names = ana.pool_conditions_by(condition_list, effect_of)
    fig, axs = plt.subplots(len(scale_list), len(variable_list))
    fig.set_size_inches(7 * len(variable_list), 6 * len(scale_list))
    colors = plt.cm.jet(np.linspace(0, 1, len(condition_pools)))
    fig.suptitle("Conditions " + str([param.nb_to_name[i] for i in condition_list]))
    fig.set_tight_layout(True)  # make the margins tighter

    for i_variable in range(len(variable_list)):
        variable = variable_list[i_variable]
        if variable == "cross transits":
            bins = np.linspace(0, 25000, 60)
        else:
            bins = np.linspace(0, 8500, 60)
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
                ax.set_title(str(variable) + " values")
            ax.set_yscale(scale_list[i_scale])
            # For every condition pool in condition_list
            for i_cond in range(len(condition_pools)):
                cond = condition_pools[i_cond]
                name = condition_names[i_cond]
                if "aggregated" in variable and "transits" not in variable:
                    # This is because our aggregated visits not including transits are in sub-lists already containing durations
                    values = ana.return_value_list(results, variable, cond, convert_to_duration=False)
                    values = [sublist[i] for sublist in values for i in range(len(sublist))]
                else:
                    values = ana.return_value_list(results, variable, cond, convert_to_duration=True)
                ax.hist(values, bins=bins, density=True, cumulative=-plot_cumulative, label=name, histtype="step",
                        color=colors[i_cond])

    plt.legend()
    plt.show()


def plot_leaving_delays(results, plot_title, condition_list, bin_size, color):
    leaving_delays, corresponding_time_in_patch = ana.delays_before_leaving(results, condition_list)
    leaving_delays_in_one_list = [leaving_delays[i] for i in range(len(leaving_delays))]
    binned_times_in_patch, avg_leaving_delays, y_err_list, full_value_list = ana.xy_to_bins(corresponding_time_in_patch,
                                                                                            leaving_delays_in_one_list, bin_size, bootstrap=False)
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
        plt.annotate(str(len(full_value_list[i_bin])), [binned_times_in_patch[i_bin] + 100, np.max(full_value_list[i_bin]) + 100])

    plt.show()


def plot_leaving_probability(results, plot_title, condition_list, bin_size, color, split_conditions=False):
    plt.title(plot_title)
    plt.ylabel("Probability of exiting in the next " + str(param.time_threshold) + " time steps")
    plt.xlabel("Time already spent in patch")

    colors = plt.cm.jet(np.linspace(0, 1, len(condition_list)))

    if not split_conditions:
        leaving_delays, corresponding_time_in_patch = ana.delays_before_leaving(results, condition_list)
        binned_times_in_patch, avg_leaving_delays, y_err_list, full_delay_list = ana.xy_to_bins(
            corresponding_time_in_patch,
            leaving_delays, bin_size,
            bootstrap=False)
        binned_leaving_probability, errorbars = ana.leaving_probability(full_delay_list)
        plt.plot(binned_times_in_patch, binned_leaving_probability, color=color)
        plt.errorbar(binned_times_in_patch, binned_leaving_probability, errorbars, fmt='.k', capsize=5)

        # Plot number of values in each bin
        for i_bin in range(len(binned_times_in_patch)):
            plt.annotate(str(len(full_delay_list[i_bin])), [binned_times_in_patch[i_bin], binned_leaving_probability[i_bin]+0.005])

    if split_conditions:

        for i_condition in range(len(condition_list)):
            current_condition = condition_list[i_condition]
            leaving_delays, corresponding_time_in_patch = ana.delays_before_leaving(results, [current_condition])
            binned_times_in_patch, avg_leaving_delays, y_err_list, full_delay_list = ana.xy_to_bins(
                corresponding_time_in_patch,
                leaving_delays, bin_size,
                bootstrap=False)
            binned_leaving_probability, errorbars = ana.leaving_probability(full_delay_list, errorbars=False)
            plt.plot(binned_times_in_patch, binned_leaving_probability, color=colors[i_condition], label=parameters.nb_to_name[current_condition])

        plt.legend()

    plt.show()


def plot_test(results):
    full_visit_list, full_variable_list = ana.visit_time_as_a_function_of(results, [0], "visit_start")
    plt.scatter(full_visit_list, full_variable_list)
    plt.show()
