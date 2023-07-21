# Analysis of visits in mixed environments
from main import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

results = pd.read_csv(path + "/backup_of_results_using_centroid_for_presence_in_patch/clean_results.csv")

# Set to True to plot information about whether the worms have encountered both patch types or not
plot_encountered_both = True

# Parameters to look at only a subset of visits
min_visit_start = 10000
max_visit_start = 40000

# Systematically find mixed conditions and the corresponding pure densities using the fact that they have a "+" x)
mixed_conditions_names = [list(param.name_to_nb.keys())[i] for i in range(len(param.name_to_nb)) if
                          "+" in list(param.name_to_nb.keys())[i]]
mixed_conditions_nb = [param.name_to_nb[mixed_conditions_names[i]] for i in range(len(mixed_conditions_names))]
corresponding_pure_densities = np.unique(
    [param.nb_to_density[mixed_conditions_nb[i]].split("+") for i in range(len(mixed_conditions_nb))])
# Here I put "med" because they are all medium, but we could look for the distance systematically too
pure_conditions_names = ["med " + corresponding_pure_densities[i] for i in range(len(corresponding_pure_densities))]
pure_conditions_nb = [param.name_to_nb[pure_conditions_names[i]] for i in range(len(pure_conditions_names))]

# Then go through all the folders and do the thing
full_condition_list = pure_conditions_nb + mixed_conditions_nb
full_folder_list = np.unique(results["folder"])

# Create results lists
all_condition_names = mixed_conditions_names + pure_conditions_names
# Dictionary with condition name as key, and list of results as values. For pure conditions, second sublist should be empty.
avg_visit_duration_all_plates = {all_condition_names[i]: [[], []] for i in range(len(all_condition_names))}
# Dictionary with condition name as key, one sublist per density in each condition. Contains True if worm has seen both types of patches already.
encountered_both_patches = {all_condition_names[i]: [[], []] for i in range(len(all_condition_names))}

# Go through all conditions and fill dictionaries with average visit duration for each plate
for i_cond in range(len(full_condition_list)):
    # Load folders
    current_condition = full_condition_list[i_cond]
    current_condition_name = param.nb_to_name[current_condition]
    current_condition_folders = fd.return_folders_condition_list(full_folder_list,
                                                                 current_condition)  # folder list for that condition

    # Create lists to be filled, that will have one sublist for each patch density
    # Should produce two sublists for mixed conditions and one sublist for pure ones, without need for an if
    current_densities = np.sort(np.unique(param.nb_to_density[current_condition].split("+")))  # lower density first
    avg_visit_duration_each_plate = [[] for _ in range(len(current_densities))]
    encountered_both_patches_current_condition = np.zeros(len(current_condition_folders))
    for i_folder in range(len(current_condition_folders)):
        # List of values for this folder
        current_folder_duration_list = [[] for _ in range(len(current_densities))]
        # Load tables and patch info
        current_folder = current_condition_folders[i_folder]
        current_results = results[results["folder"] == current_folder]
        current_metadata = fd.folder_to_metadata(current_folder)
        # Load lists from the tables
        list_of_visits = fd.load_list(current_results, "no_hole_visits")
        list_of_visit_durations = ana.convert_to_durations(list_of_visits)
        patch_densities = current_metadata["patch_densities"]  # import
        patch_densities = [list(patch_densities[i])[0] for i in range(len(patch_densities))]  # reformat
        for i_visit in range(len(list_of_visits)):
            current_patch = list_of_visits[i_visit][2]
            current_patch_density = patch_densities[current_patch]
            current_visit_duration = list_of_visit_durations[i_visit]
            for i_density in range(len(current_densities)):
                if str(current_patch_density) == current_densities[i_density] and min_visit_start < list_of_visits[i_visit][0] < max_visit_start:
                    current_folder_duration_list[i_density].append(current_visit_duration)
                    if encountered_both_patches_current_condition[i_folder] == 0:  # first visit
                        encountered_both_patches_current_condition[i_folder] = current_patch_density  # register current patch density
                    else:
                        if encountered_both_patches_current_condition[i_folder] != current_patch_density:  # if a different density is ever encountered
                            encountered_both_patches_current_condition[i_folder] = True
        # If a different density was never encountered
        if encountered_both_patches_current_condition[i_folder] != True:
            encountered_both_patches_current_condition[i_folder] = False
        # At this point full_list should have one sublist of visit durations per density
        for i_density in range(len(current_densities)):
            avg_visit_duration_each_plate[i_density].append(np.mean(current_folder_duration_list[i_density]))
    # Fill list of plate averages
    for i_density in range(len(current_densities)):
        avg_visit_duration_all_plates[current_condition_name][i_density] += avg_visit_duration_each_plate[i_density]
        encountered_both_patches[current_condition_name][i_density] += list(encountered_both_patches_current_condition)

# Compute the average and errorbars for each condition
avg_visit_duration_list = {all_condition_names[i]: [[], []] for i in range(len(all_condition_names))}
errorbars_list = {all_condition_names[i]: [[], []] for i in range(len(all_condition_names))}
for condition_name in all_condition_names:
    list_of_values_condition = avg_visit_duration_all_plates[condition_name]
    for i_density in range(len(list_of_values_condition)):
        list_of_values = list_of_values_condition[i_density]
        list_of_values = [list_of_values[i] for i in range(len(list_of_values)) if not np.isnan(list_of_values[i])]
        avg_visit_duration_list[condition_name][i_density] = np.mean(list_of_values)
        bootstrap_ci = ana.bottestrop_ci(list_of_values, 1000)
        errors_inf = avg_visit_duration_list[condition_name][i_density] - bootstrap_ci[0]
        errors_sup = bootstrap_ci[1] - avg_visit_duration_list[condition_name][i_density]
        errorbars_list[condition_name][i_density] = [errors_inf, errors_sup]

# Transform the dictionaries into something more useful
# Go from {"med 0.2": [[...], []], "med 0.2+0.5": [[...],[...]], ...}
# To {"0.2 in med 0.2": [...], "0.2 in med 0.2+0.5": [...]} with one list per element
for condition_name in all_condition_names:
    # Pop values so that the new keys replace the old ones
    full_list = avg_visit_duration_all_plates.pop(condition_name)
    avg_list = avg_visit_duration_list.pop(condition_name)
    errorbars = errorbars_list.pop(condition_name)
    encountered_both = encountered_both_patches.pop(condition_name)
    density_list = param.nb_to_density[param.name_to_nb[condition_name]].split("+")  # ["0.2"] or ["0.2", "0.5"]
    for i_density in range(len(density_list)):
        density = density_list[i_density]
        avg_visit_duration_all_plates[density + " in " + condition_name] = full_list[i_density]
        avg_visit_duration_list[density + " in " + condition_name] = avg_list[i_density]
        errorbars_list[density + " in " + condition_name] = errorbars[i_density]
        encountered_both_patches[density + " in " + condition_name] = encountered_both[i_density]  # same for both densities

plt.title("Average visit duration in mixed densities, visits starting between "+str(min_visit_start)+" & "+str(max_visit_start))
fig = plt.gcf()
ax = fig.gca()
fig.set_size_inches(8, 14)

# Plot condition averages as a bar plot
condition_names = list(avg_visit_duration_list.keys())
avg_list = [avg_visit_duration_list[cond] for cond in condition_names]
ax.bar(range(len(condition_names)), avg_list,
       color=[param.name_to_color[cond.split(" ")[0]] for cond in condition_names])
ax.set_xticks(range(len(condition_names)))
ax.set_xticklabels(condition_names, rotation=60)
ax.set(xlabel="Condition name")
ax.set_ylim(0, 6000)

# Plot plate averages as scatter on top
for i_cond in range(len(condition_names)):
    current_condition_name = condition_names[i_cond]
    x_list = [range(len(condition_names))[i_cond] for _ in range(len(avg_visit_duration_all_plates[current_condition_name]))]
    color_list = ["red" if encountered_both_patches[current_condition_name][i] else "blue" for i in range(len(encountered_both_patches[current_condition_name]))]
    ax.scatter(x_list, avg_visit_duration_all_plates[current_condition_name], color=color_list, zorder=2, alpha=0.4)

    if plot_encountered_both:  # to plot information about whether worms have encountered both patches yet or not
        # Plot average of subgroups of plates depending on whether they encountered both patch types or not
        list_encountered_both = []
        list_encountered_only_one = []
        for i_plate in range(len(avg_visit_duration_all_plates[current_condition_name])):
            if encountered_both_patches[current_condition_name][i_plate]:
                list_encountered_both.append(avg_visit_duration_all_plates[current_condition_name][i_plate])
            else:
                list_encountered_only_one.append(avg_visit_duration_all_plates[current_condition_name][i_plate])
        avg_both = np.nanmean(list_encountered_both)
        avg_only_one = np.nanmean(list_encountered_only_one)

        ax.scatter(range(len(condition_names))[i_cond], avg_both, color="salmon", marker="*", s=200, zorder=3)
        ax.scatter(range(len(condition_names))[i_cond], avg_only_one, color="midnightblue", marker="*", s=200, zorder=3)


# Plot error bars
error_list = [[errorbars_list[cond][0] for cond in errorbars_list.keys()],
              [errorbars_list[cond][1] for cond in errorbars_list.keys()]]
ax.errorbar(range(len(condition_names)), avg_list, error_list, fmt='.k', capsize=5)

# Create legend
legend_list = []
for density in corresponding_pure_densities:
    legend_list.append(Patch(facecolor=param.name_to_color[density], label=density))

if plot_encountered_both:  # to plot information about whether worms have encountered both patches yet or not
    legend_list.append(Patch(facecolor="blue", label="Encountered 1 patch type"))
    legend_list.append(Patch(facecolor="red", label="Encountered 2 patch types"))

plt.legend(handles=legend_list)
plt.show()
