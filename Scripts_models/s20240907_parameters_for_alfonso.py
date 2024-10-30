import numpy as np
import pandas as pd
import copy

from Generating_data_tables import main as gen
import find_data as fd
from Parameters import parameters as param
import analysis as ana

# Script to return the stuff that Alfonso wants for his model
# For now, average total time spent outside per food patch, and average nb of visits per food patch, for each condition
# Save it in a nice table

results_path = gen.generate("", shorten_traj=False)
results = pd.read_csv(results_path + "clean_results.csv")
full_folder_list = results["folder"]
condition_names = param.name_to_nb_list.keys()
# condition_names = ["far 0.2", "superfar 0.2"]

# Tables to put average total time out per patch + bootstrapped errorbars
time_out_per_patch_avg_each_cond = np.zeros(len(condition_names))
time_out_per_patch_error_inf_each_cond = np.zeros(len(condition_names))
time_out_per_patch_error_sup_each_cond = np.zeros(len(condition_names))
# Tables to put average number of visits to each patch + bootstrapped errorbars
nb_of_visits_per_patch_avg_each_cond = np.zeros(len(condition_names))
nb_of_visits_per_patch_error_inf_each_cond = np.zeros(len(condition_names))
nb_of_visits_per_patch_error_sup_each_cond = np.zeros(len(condition_names))

# Fill 'em up
for i_condition, condition_name in enumerate(condition_names):
    print("Condition ", condition_name)
    times_out_this_cond = []
    nb_of_visits_this_cond = []
    folder_list = fd.return_folders_condition_list(full_folder_list, param.name_to_nb_list[condition_name])
    for i_folder, folder in enumerate(folder_list):
        # print(">>> Folder ", folder)
        current_results = results[results["folder"] == folder]
        current_visits = fd.load_list(current_results, "no_hole_visits")
        current_transits = fd.load_list(current_results, "aggregated_raw_transits")

        # Only keep the first param.time_to_cut tracked time steps of the video
        # Mix visits and transits and sort them by beginning
        current_events = copy.deepcopy(current_visits) + copy.deepcopy(current_transits)
        current_events = sorted(current_events, key=lambda x: x[0])
        # Loop through them and stop when it's reached time to cut!!!
        cumulated_duration_of_events = 0
        i_event = 0
        new_list_of_events = []
        first_event_found = False
        while i_event < len(current_events) and cumulated_duration_of_events < (
                param.times_to_cut_videos[1] - param.times_to_cut_videos[0]):
            current_event = current_events[i_event]
            if current_event[0] >= param.times_to_cut_videos[0]:
                if not first_event_found:
                    first_event_found = True
                    # If this is the first event that starts after time_to_cut[0] and not the first of the video, add the previous one
                    if i_event > 0:
                        previous_event = current_events[i_event - 1]
                        previous_event[0] = param.times_to_cut_videos[0]  # but set it to start at time_to_cut[0]
                        cumulated_duration_of_events += previous_event[1] - previous_event[0]
                        new_list_of_events.append(previous_event)
                        # If this previous event does not exceed time_to_cut[1], then you can add the current one
                        if cumulated_duration_of_events < (param.times_to_cut_videos[1] - param.times_to_cut_videos[0]):
                            cumulated_duration_of_events += current_event[1] - current_event[0]
                            new_list_of_events.append(current_event)
                    # If this is the first event, just add it! and start it at time_to_cut[0]
                    else:
                        current_event[0] = param.times_to_cut_videos[0]  # but set it to start at time_to_cut[0]
                        cumulated_duration_of_events += current_event[1] - current_event[0]
                        new_list_of_events.append(current_event)

                else:
                    cumulated_duration_of_events += current_event[1] - current_event[0]
                    new_list_of_events.append(current_event)
            i_event += 1
        # In the end of the loop, if we have reached the cut parameter, cut the last event
        if cumulated_duration_of_events > (param.times_to_cut_videos[1] - param.times_to_cut_videos[0]):
            new_list_of_events[-1][1] -= cumulated_duration_of_events - (
                        param.times_to_cut_videos[1] - param.times_to_cut_videos[0])
        # Then, sort the events back to visits!
        current_visits = [event for event in new_list_of_events if event[2] != -1]

        total_time_out = np.sum(ana.convert_to_durations(current_transits))
        nb_of_visits = len(current_visits)
        if nb_of_visits > 0:  # if there's any visit
            nb_of_visited_patches = len(np.unique(np.array(current_visits)[:, 2]))
            # Fill the tablezzz
            times_out_this_cond.append(total_time_out/nb_of_visited_patches)
            nb_of_visits_this_cond.append(nb_of_visits/nb_of_visited_patches)
            # print("Total time out: ", total_time_out, ", Nb of visited patches: ", nb_of_visited_patches, ", Effective time out: ", total_time_out / nb_of_visited_patches)

    # Average and bootstrap times out per visited patch
    times_out_this_cond = np.array(times_out_this_cond)/3600  # convert to hours
    current_avg = np.mean(times_out_this_cond)
    time_out_per_patch_avg_each_cond[i_condition] = current_avg
    bootstrap_ci = ana.bottestrop_ci(times_out_this_cond, 1000)
    time_out_per_patch_error_inf_each_cond[i_condition] = current_avg - bootstrap_ci[0]
    time_out_per_patch_error_sup_each_cond[i_condition] = bootstrap_ci[1] - current_avg
    # Average and bootstrap number of visits per visited patch
    current_avg = np.mean(nb_of_visits_this_cond)
    nb_of_visits_per_patch_avg_each_cond[i_condition] = current_avg
    bootstrap_ci = ana.bottestrop_ci(nb_of_visits_this_cond, 1000)
    nb_of_visits_per_patch_error_inf_each_cond[i_condition] = current_avg - bootstrap_ci[0]
    nb_of_visits_per_patch_error_sup_each_cond[i_condition] = bootstrap_ci[1] - current_avg

# Fill a datatable
output_datatable = pd.DataFrame()
output_datatable["condition"] = condition_names
output_datatable["nb_of_visits_avg"] = nb_of_visits_per_patch_avg_each_cond
output_datatable["nb_of_visits_error_inf"] = nb_of_visits_per_patch_error_inf_each_cond
output_datatable["nb_of_visits_error_sup"] = nb_of_visits_per_patch_error_sup_each_cond
output_datatable["effective_travel_time_avg"] = time_out_per_patch_avg_each_cond
output_datatable["effective_travel_time_error_inf"] = time_out_per_patch_error_inf_each_cond
output_datatable["effective_travel_time_error_sup"] = time_out_per_patch_error_sup_each_cond

output_datatable.to_csv(results_path + "model_parameters_from_alid.csv")





