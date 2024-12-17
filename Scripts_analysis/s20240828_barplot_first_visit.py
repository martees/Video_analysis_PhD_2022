import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import find_data as fd
from Parameters import parameters as param
from Generating_data_tables import main as gen
import analysis as ana


def bar_plot_first_visit_each_patch(results, condition_list, is_plot=True, remove_censored_events=False,
                                    only_first_visited_patch=False, soft_cut=False, hard_cut=False,
                                    visits_longer_than=0):
    """
    Function that plots a histogram with the average length of first visit to each food patch in each condition in condition_list.
    """
    avg_value_each_condition = [0 for _ in range(len(condition_list))]
    errors_inf_each_condition = [0 for _ in range(len(condition_list))]
    errors_sup_each_condition = [0 for _ in range(len(condition_list))]
    avg_value_each_condition_each_plate = [[] for _ in range(len(condition_list))]
    plate_list = results["folder"]
    for i_plate, plate in enumerate(plate_list):
        current_plate = results[results["folder"] == plate].reset_index()
        condition = current_plate["condition"][0]
        # condition = fd.load_condition(plate)
        if remove_censored_events:
            current_visits = fd.load_list(current_plate, "uncensored_visits")
        else:
            current_visits = fd.load_list(current_plate, "no_hole_visits")

        # Remove visits shorter than threshold
        current_visits = [visit for visit in current_visits if (visit[1] - visit[0]) >= visits_longer_than]

        if len(current_visits) > 0:
            if only_first_visited_patch:
                first_visited_patch = current_visits[0][2]
                for visit in current_visits:
                    if visit[2] != first_visited_patch:
                        current_visits.remove(visit)
            if soft_cut:
                for visit in current_visits:
                    if param.times_to_cut_videos[0] > visit[0] > param.times_to_cut_videos[1]:
                        current_visits.remove(visit)
            if hard_cut:
                print("WARNING, YOU HAVENT IMPLEMENTED THE FULL HARD CUT FOR FIRST VISITS è_é")
                for visit in current_visits:
                    if visit[1] > param.time_to_cut_videos:
                        current_visits.remove(visit)
        # Only select the first visits
        list_of_found_patches = []
        first_value_each_patch = []
        for value in current_visits:
            if value[2] not in list_of_found_patches:
                list_of_found_patches.append(value[2])
                first_value_each_patch.append((value[1] - value[0])/3600)  # convert to hours
        if condition in condition_list and len(first_value_each_patch) > 0:
            condition_index = np.where(condition_list == condition)[0][0]
            avg_value_each_condition_each_plate[condition_index].append(np.nanmean(first_value_each_patch))
    for i_condition in range(len(condition_list)):
        avg_value_each_condition[i_condition] = np.mean(avg_value_each_condition_each_plate[i_condition])
        errors = ana.bottestrop_ci(avg_value_each_condition_each_plate[i_condition], 1000)
        errors_inf_each_condition[i_condition], errors_sup_each_condition[i_condition] = [avg_value_each_condition[i_condition] - errors[0],
                                                                                          errors[1] - avg_value_each_condition[i_condition]]
    if is_plot:
        # Plotty plot
        fig = plt.gcf()
        fig.set_size_inches(6, 10)
        condition_names = [param.nb_to_name[cond] for cond in condition_list]
        condition_colors = [param.name_to_color[name] for name in condition_names]
        plt.title("First visit to each patch")
        plt.ylabel("First visit to each patch (hours)")
        # Bar plot
        plt.bar(range(len(condition_list)), avg_value_each_condition, color=condition_colors)
        plt.xticks(range(len(condition_list)), condition_names, rotation=45)
        plt.xlabel("Condition number")
        # Plate averages as scatter on top
        for i in range(len(condition_list)):
            plt.scatter([range(len(condition_list))[i] for _ in range(len(avg_value_each_condition_each_plate[i]))],
                        avg_value_each_condition_each_plate[i], color="orange", zorder=2, s=6)
        # Error bars
        plt.errorbar(range(len(condition_list)), avg_value_each_condition, [errors_inf_each_condition, errors_sup_each_condition], fmt='.k', capsize=5)

        plt.show()
    else:
        return avg_value_each_condition_each_plate, avg_value_each_condition, [errors_inf_each_condition, errors_sup_each_condition]


if __name__ == "__main__":
    path = gen.generate("")
    clean_results = pd.read_csv(path + "clean_results.csv")
    bar_plot_first_visit_each_patch(clean_results, param.list_by_distance)
    bar_plot_first_visit_each_patch(clean_results, param.list_by_density)

