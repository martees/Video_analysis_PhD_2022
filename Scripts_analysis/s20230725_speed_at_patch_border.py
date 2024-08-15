from main import *
import numpy as np
import matplotlib.pyplot as plt
import find_data as fd
from Generating_data_tables import main as gen
# Analysis of the worms' speeds near the patch border
# The goal is to plot speed right before and after worm enters / exits a food patch
# Note: the script will pool all given conditions into two curves, one for exit, and one for entry

path = gen.generate("")
clean_results = pd.read_csv(path + "/clean_results.csv", index_col=0)
clean_trajectories = pd.read_csv(path + "/clean_trajectories.csv", index_col=0)

# Find conditions and folders
condition_pool_name = "far 0"
condition_list = [param.name_to_nb[condition_pool_name]]
condition_names = [param.nb_to_name[nb] for nb in condition_list]
folder_list = fd.return_folders_condition_list(np.unique(clean_results["folder"]), condition_list)

# Number of time steps before and after exit/entry that we look at
time_window = 10

# Init: lists to fill with average for each time step pre/post entry/exit for each plate
avg_speed_around_entry_all_plates = [np.zeros(len(folder_list)) for _ in range(2*time_window)]
avg_speed_around_exit_all_plates = [np.zeros(len(folder_list)) for _ in range(2*time_window)]

for i_folder in range(len(folder_list)):
    folder = folder_list[i_folder]
    # Init
    speed_around_entry = [[] for _ in range(2*time_window)]
    speed_around_exit = [[] for _ in range(2*time_window)]
    # Load visits
    current_results = clean_results[clean_results["folder"] == folder]
    current_traj = clean_trajectories[clean_trajectories["folder"] == folder].reset_index(drop=True)
    list_of_visits = fd.load_list(current_results, "no_hole_visits")
    list_of_transits = fd.load_list(current_results, "aggregated_raw_transits")
    # Lists of frames where worm enters / exits patches (visit and transit starts) and it lasts more than time window
    # (any visit or transit shorter than time window would lead to an "impure" behavior => excluded from analysis)
    long_enough_visits = [list_of_visits[i] for i in range(len(list_of_visits)) if list_of_visits[i][1]-list_of_visits[i][0] >= time_window]
    long_enough_transits = [list_of_transits[i] for i in range(len(list_of_transits)) if list_of_transits[i][1]-list_of_transits[i][0] >= time_window]
    # Since we filtered the two lists, they now contain visits and transits that are not necessarily consecutive => fix that
    entry_frames = [long_enough_visits[i][0] for i in range(len(long_enough_visits)) if any(long_enough_visits[i][0] in t for t in long_enough_transits)]
    exit_frames = [long_enough_transits[i][0] for i in range(len(long_enough_transits)) if any(long_enough_transits[i][0] in t for t in long_enough_visits)]

    # Fill entry list
    for i_entry in range(len(entry_frames)):
        current_entry_frame = entry_frames[i_entry]
        entry_index = fd.find_closest(current_traj["frame"], current_entry_frame)
        # Check if frames are continuous around entry: otherwise, exclude it completely (for now because I'm tired)
        # (check if the difference between the indexes in the "frame" column is the same as the difference between the frames)
        pre_entry_index = fd.find_closest(current_traj["frame"], current_entry_frame - time_window)
        post_entry_index = fd.find_closest(current_traj["frame"], current_entry_frame + time_window)
        if pre_entry_index == entry_index - time_window and post_entry_index == entry_index + time_window:
            entry_speeds = current_traj["speeds"].iloc[pre_entry_index:post_entry_index].reset_index(drop=True)
            for time in range(2*time_window):
                speed_around_entry[time].append(entry_speeds[time])

    # Fill exit list
    for i_exit in range(len(exit_frames)):
        current_exit_frame = exit_frames[i_exit]
        exit_index = fd.find_closest(current_traj["frame"], current_exit_frame)
        # Check if frames are continuous around exit: otherwise, exclude it completely (for now because I'm tired)
        pre_exit_index = fd.find_closest(current_traj["frame"], current_exit_frame - time_window)
        post_exit_index = fd.find_closest(current_traj["frame"], current_exit_frame + time_window)
        if pre_exit_index == exit_index - time_window and post_exit_index == exit_index + time_window:
            exit_speeds = current_traj["speeds"].iloc[pre_exit_index:post_exit_index].reset_index(drop=True)
            for time in range(2*time_window):
                speed_around_exit[time].append(exit_speeds[time])

    # At this point, speed_before_entry, speed_after_... etc. are filled with one sublist per time step before/after
    # entry/exit, and each sublist contains the worms' speeds during those time steps. Now we average for each time step
    for time in range(2*time_window):
        avg_speed_around_entry_all_plates[time][i_folder] = np.nanmean(speed_around_entry[time])
        avg_speed_around_exit_all_plates[time][i_folder] = np.nanmean(speed_around_exit[time])

# Now that we have the full list of averages for each time step before/after entry/exit, average and bootstrap all that
# Averages
avg_speed_around_entry = np.zeros(2*time_window)
avg_speed_around_exit = np.zeros(2*time_window)
# Errors
errors_inf_around_entry = np.zeros(2*time_window)
errors_sup_around_entry = np.zeros(2*time_window)
errors_inf_around_exit = np.zeros(2*time_window)
errors_sup_around_exit = np.zeros(2*time_window)

for time in range(2*time_window):
    # Rename
    around_entry_values = avg_speed_around_entry_all_plates[time]
    around_exit_values = avg_speed_around_exit_all_plates[time]
    # and remove nan values for bootstrapping
    around_entry_values = [around_entry_values[i] for i in range(len(around_entry_values)) if not np.isnan(around_entry_values[i])]
    around_exit_values = [around_exit_values[i] for i in range(len(around_exit_values)) if not np.isnan(around_exit_values[i])]

    # Around entry
    current_values = around_entry_values
    current_avg = np.nanmean(current_values)
    avg_speed_around_entry[time] = current_avg
    bootstrap_ci = ana.bottestrop_ci(current_values, 1000)
    errors_inf_around_entry[time] = current_avg - bootstrap_ci[0]
    errors_sup_around_entry[time] = bootstrap_ci[1] - current_avg

    # Around exit
    current_values = around_exit_values
    current_avg = np.nanmean(current_values)
    avg_speed_around_exit[time] = current_avg
    bootstrap_ci = ana.bottestrop_ci(current_values, 1000)
    errors_inf_around_exit[time] = current_avg - bootstrap_ci[0]
    errors_sup_around_exit[time] = bootstrap_ci[1] - current_avg

# Make the plotty plots
fig, axs = plt.subplots(1, 2)  # Create one subplot for entry and one for exit

fig.suptitle("Speed as a function of time around patch entry / exit for conditions "+str(condition_list))

axs[0].set_title("Speed when entering")
axs[0].plot(range(-time_window, time_window), avg_speed_around_entry, color=param.name_to_color[condition_pool_name], label=condition_pool_name, linewidth=3)
axs[0].errorbar(range(-time_window, time_window), avg_speed_around_entry, [errors_inf_around_entry, errors_sup_around_entry], fmt='.k', capsize=5)
axs[0].set_ylabel("Average speed")
axs[0].set_xlabel("Time pre/post entry")

axs[1].set_title("Speed when exiting")
axs[1].plot(range(-time_window, time_window), avg_speed_around_exit, color=param.name_to_color[condition_pool_name], label=condition_pool_name, linewidth=3)
axs[1].errorbar(range(-time_window, time_window), avg_speed_around_exit, [errors_inf_around_exit, errors_sup_around_exit], fmt='.k', capsize=5)
axs[1].set_ylabel("Average speed")
axs[1].set_xlabel("Time pre/post exit")

plt.legend()
plt.show()













