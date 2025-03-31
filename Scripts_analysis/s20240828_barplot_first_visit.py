import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import find_data as fd
from Parameters import parameters as param
from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
import analysis as ana


def bar_plot_first_visit_each_patch(results, condition_list, is_plot=True,
                                    exclude_censored_events=False,
                                    ignore_subsequent_uncensored_events=True,
                                    only_first_visited_patch=False, soft_cut=False, hard_cut=False,
                                    visits_longer_than=0):
    """
    Function that plots a histogram with the average length of first visit to each food patch in each condition in condition_list.
    If "exclude_censored_events" is True, then censored events will be removed from the visit lists before analysis.
    The "ignore_subsequent_uncensored_events" is only relevant when exclude_censored_events is also True.
        If "ignore_subsequent_uncensored_events" is False, then the analysis is made on the first uncensored event of
                    the patch (so in most cases, it means analyzing the second visit when the first is censored)
        If "ignore_subsequent_uncensored_events" is True, then the analysis will ignore patches when the first visit
                    was censored. In our dataset we also have a list named "visits_to_uncensored_patches", but in this
                    case this is not suitable because it excludes patches that have *any* censored events, and we are
                    just interested on whether their first visit is censored.
    """
    avg_value_each_condition = [0 for _ in range(len(condition_list))]
    errors_inf_each_condition = [0 for _ in range(len(condition_list))]
    errors_sup_each_condition = [0 for _ in range(len(condition_list))]
    avg_value_each_condition_each_plate = [[] for _ in range(len(condition_list))]
    plate_list = fd.return_folders_condition_list(results["folder"], condition_list)
    for i_plate, plate in enumerate(plate_list):
        current_plate = results[results["folder"] == plate].reset_index()
        condition = current_plate["condition"][0]
        # condition = fd.load_condition(plate)

        if exclude_censored_events and not ignore_subsequent_uncensored_events:
            # We remove censored visits BEFORE taking the first
            # visit. This means that later analysis will take the first uncensored visit.
            current_visits = fd.load_list(current_plate, "uncensored_visits")

        if exclude_censored_events and ignore_subsequent_uncensored_events:
            # What we do here is a bit dirty because extracting first visit is also done later,
            # but at least we get it over with. We remove all visits to a patch if the first visit to this patch
            # was censored.
            all_visits = fd.load_list(current_plate, "no_hole_visits")
            uncensored_visits = fd.load_list(current_plate, "uncensored_visits")
            # Use a more convenient visit format, where they are sorted by patch instead of chronologically
            all_visits = gr.sort_visits_by_patch(all_visits, param.nb_to_nb_of_patches[condition])
            for i_patch in range(param.nb_to_nb_of_patches[condition]):
                if len(all_visits[i_patch]) > 0:
                    if all_visits[i_patch][0] + [i_patch] not in uncensored_visits:
                        all_visits[i_patch] = []
            current_visits = gr.sort_visits_chronologically(all_visits)

        else:
            current_visits = fd.load_list(current_plate, "no_hole_visits")

        # Remove visits shorter than threshold
        current_visits = [visit for visit in current_visits if (visit[1] - visit[0]) >= visits_longer_than]

        if len(current_visits) > 0:
            if only_first_visited_patch:
                first_visited_patch = current_visits[0][2]
                original_visits = copy.deepcopy(current_visits)  # Making a copy for proper removal behavior!
                for visit in original_visits:
                    if visit[2] != first_visited_patch:
                        current_visits.remove(visit)
            if soft_cut:
                original_visits = copy.deepcopy(current_visits)  # Making a copy for proper removal behavior!
                for visit in original_visits:
                    if (param.times_to_cut_videos[0] > visit[0]) or (visit[0] > param.times_to_cut_videos[1]):
                        current_visits.remove(visit)
            if hard_cut:
                print("WARNING, YOU HAVENT IMPLEMENTED THE FULL HARD CUT FOR FIRST VISITS è_é")
                original_visits = copy.deepcopy(current_visits)  # Making a copy for proper removal behavior!
                for visit in original_visits:
                    if visit[1] > param.time_to_cut_videos[1]:
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

