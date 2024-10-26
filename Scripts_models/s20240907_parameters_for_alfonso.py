import numpy as np
import pandas as pd

from Generating_data_tables import main as gen
import find_data as fd
from Parameters import parameters as param
import analysis as ana

# Script to return the stuff that Alfonso wants for his model
# For now, average total time spent outside per food patch, and average nb of visits per food patch, for each condition
# Save it in a nice table

results_path = gen.generate("", shorten_traj=True)
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
        all_visits = fd.load_list(current_results, "no_hole_visits")
        all_transits = fd.load_list(current_results, "aggregated_raw_transits")
        total_time_out = np.sum(ana.convert_to_durations(all_transits))
        nb_of_visits = len(all_visits)
        if nb_of_visits > 0:  # if there's any visit
            nb_of_visited_patches = len(np.unique(np.array(all_visits)[:, 2]))
            # Fill the tablezzz
            times_out_this_cond.append(total_time_out/nb_of_visited_patches)
            nb_of_visits_this_cond.append(nb_of_visits/nb_of_visited_patches)
            # print("Total time out: ", total_time_out, ", Nb of visited patches: ", nb_of_visited_patches, ", Effective time out: ", total_time_out / nb_of_visited_patches)
        else:
            times_out_this_cond.append(total_time_out)  # don't divide by 0
            nb_of_visits_this_cond.append(nb_of_visits)  # should be 0
    # Average and bootstrap times out per visited patch
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





