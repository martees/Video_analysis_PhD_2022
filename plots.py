import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import random

import analysis as ana


# Sanity check functions

def plot_data_coverage(traj):
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


def plot_patches(folder_list, show_composite=True, is_plot=True):
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
                composite = ax.imshow(composite)
            else:
                background = plt.imread(folder + "background.tif")
                background = ax.imshow(background, cmap='gray')

        patch_centers = metadata["patch_centers"]
        patch_densities = metadata["patch_densities"]
        patch_spline_breaks = metadata["spline_breaks"]
        patch_spline_coefs = metadata["spline_coefs"]

        colors = plt.cm.jet(np.linspace(0, 1, len(patch_centers)))
        x_list = []
        y_list = []
        # For each patch
        for i_patch in range(len(patch_centers)):
            # For a range of 100 angular positions
            angular_pos = np.linspace(0, 2 * np.pi, 100)
            radiuses = np.zeros(len(angular_pos))
            # Compute the local spline value for each of those radiuses
            for i_angle in range(len(angular_pos)):
                radiuses[i_angle] = gr.spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                                    patch_spline_coefs[i_patch])

            fig = plt.gcf()
            ax = fig.gca()

            # Create lists of cartesian positions out of this
            x_pos = []
            y_pos = []
            for point in range(len(angular_pos)):
                x_pos.append(patch_centers[i_patch][0] + (radiuses[point] * np.sin(angular_pos[point])))
                y_pos.append(patch_centers[i_patch][1] + (radiuses[point] * np.cos(angular_pos[point])))

            # Either plot them
            if is_plot:
                plt.plot(x_pos, y_pos, color=colors[i_patch])
            # Or add them to a list for later
            else:
                x_list.append(x_pos)
                y_list.append(y_pos)

        if is_plot:
            plt.title(folder)
            plt.show()
        else:
            return x_list, y_list


def plot_traj(traj, i_condition, n_max=4, is_plot_patches=False, show_composite=True, plot_in_patch=False,
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
        folder_list = fd.return_folder_list_one_condition(np.unique(traj["folder"]), i_condition)
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
                x_list, y_list = plot_patches([current_folder], is_plot=False)
                for i_patch in range(len(patch_centers)):
                    ax.plot(x_list[i_patch], y_list[i_patch], color='yellow')
                    # to show density, add this to previous call: , alpha=patch_densities[i_patch][0]
                    # ax.annotate(str(i_patch), xy=(patch_centers[i_patch][0] + 80, patch_centers[i_patch][1] + 80), color='white')

        # In any case, plot worm trajectory
        # Plot the trajectory with a colormap based on the speed of the worm
        if plot_speed:
            distance_list = current_traj.reset_index()["distances"]
            normalize = mplcolors.Normalize(vmin=0, vmax=3.5)
            plt.scatter(current_list_x, current_list_y, c=distance_list, cmap="hot", norm=normalize, s=1)
            if previous_folder != current_folder or previous_folder == 0:  # if we just changed plate or if it's the 1st
                plt.colorbar()

        # Plot the trajectory with a colormap based on time
        if plot_time:
            nb_of_timepoints = len(current_list_x)
            bin_size = 100
            # colors = plt.cm.jet(np.linspace(0, 1, nb_of_timepoints//bin_size))
            for bin in range(nb_of_timepoints // bin_size):
                lower_bound = bin * bin_size
                upper_bound = min((bin + 1) * bin_size, len(current_list_x))
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
    full_plate_list = np.unique(traj["folder"])
    plate_list = []
    for condition in condition_list:
        plate_list += fd.return_folder_list_one_condition(full_plate_list, condition)
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
            errors = bottestrop_ci(list_curr_speed_this_bin, 1000)
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


def plot_selected_data(results, plot_title, condition_list, column_name, condition_names, divided_by="", mycolor="blue"):
    """
    This function will plot a selected part of the data. Selection is described as follows:
    - condition_low, condition_high: bounds on the conditions (0,3 => function will plot conditions 0, 1, 2, 3)
    - column_name:
    """
    # Getting results
    list_of_conditions, list_of_avg_each_plate, average_per_condition, errorbars = ana.results_per_condition(results,
                                                                                                         column_name,
                                                                                                         divided_by)

    # Slicing to get condition we're interested in (only take indexes from condition_list)
    list_of_conditions = [list_of_conditions[i] for i in condition_list]
    list_of_avg_each_plate = [list_of_avg_each_plate[i] for i in condition_list]
    average_per_condition = [average_per_condition[i] for i in condition_list]
    errorbars[0] = [errorbars[0][i] for i in condition_list]
    errorbars[1] = [errorbars[1][i] for i in condition_list]

    # Plotttt
    plt.title(plot_title)
    fig = plt.gcf()
    ax = fig.gca()
    fig.set_size_inches(5, 6)

    # Plot condition averages as a bar plot
    ax.bar(range(len(list_of_conditions)), average_per_condition, color=mycolor)
    ax.set_xticks(range(len(list_of_conditions)))
    ax.set_xticklabels(condition_names, rotation=45)
    ax.set(xlabel="Condition number")

    # Plot plate averages as scatter on top
    for i in range(len(list_of_conditions)):
        ax.scatter([range(len(list_of_conditions))[i] for _ in range(len(list_of_avg_each_plate[i]))],
                   list_of_avg_each_plate[i], color="red", zorder=2)
    ax.errorbar(range(len(list_of_conditions)), average_per_condition, errorbars, fmt='.k', capsize=5)
    plt.show()


def plot_visit_time(results, trajectory, plot_title, condition_list, variable, condition_names):

    # Call function to obtain list of visit lengths and corresponding list of variable values (one sublist per condition)
    full_visit_list, full_variable_list = ana.visit_time_as_a_function_of(results, trajectory, condition_list, variable)

    # Plot the thing
    nb_cond = len(condition_list)
    for i_cond in range(nb_cond):
        plt.subplot(1, nb_cond, i_cond + 1)
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_tight_layout(True)
        ax.set_title(str(condition_names[i_cond]))
        ax.set_xlabel(variable)
        ax.set_ylabel("Visit duration")
        plt.hist2d(full_variable_list[i_cond], full_visit_list[i_cond], range=[[0,2000],[0,2000]],
                   bins=[100, 100], norm=mplcolors.LogNorm(), cmap="viridis")
        # for axis limits control, add range= [[x0,xmax],[y0,ymax]] in arguments

        # Plotting a linear regression on the thing
        coef = np.polyfit(full_variable_list[i_cond], full_visit_list[i_cond], 1)
        line_function = np.poly1d(coef)  # function which takes in x and returns an estimate for y
        plt.plot(full_variable_list[i_cond], line_function(full_variable_list[i_cond]))

    # Displaying everything with a nice size
    fig = plt.gcf()
    fig.set_size_inches(5 * nb_cond, 6)
    plt.suptitle(plot_title)

    # Plot legend with every label just once
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())

    plt.show()


def plot_test(results):
    full_visit_list, full_variable_list = ana.visit_time_as_a_function_of(results, [0], "visit_start")
    plt.scatter(full_visit_list, full_variable_list)
    plt.show()

