# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

#My code
import generate_results as gr


def avg_duration_per_condition(result_table):
    """
    Function that takes our result table with a ["condition"] and a ["avg_duration"] column
    Will return the average of avg_duration for each condition, with a bootstrap confidence interval
    """
    list_of_conditions = np.unique(result_table["condition"])
    list_of_avg = np.zeros(len(list_of_conditions))
    bootstrap_errors_inf = np.zeros(len(list_of_conditions))
    bootstrap_errors_sup = np.zeros(len(list_of_conditions))
    for i_condition in range(len(list_of_conditions)):
        condition = list_of_conditions[i_condition]
        current = result_table[result_table["condition"] == condition]["avg_visit_duration"]
        sum_of_durations = np.sum(current)
        nb = len(current)
        list_of_avg[i_condition] = sum_of_durations / nb
        bootstrap_ci = bootstrap((current,), np.mean, confidence_level=0.95,
                             random_state=1, method='percentile').confidence_interval
        bootstrap_errors_inf[i_condition] = bootstrap_ci[0]
        bootstrap_errors_sup[i_condition] = bootstrap_ci[1]
    return list_of_conditions, list_of_avg, [bootstrap_errors_inf,bootstrap_errors_sup]


#I have two lines, one for Windows and the other for Linux:
# path = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos"
path = "/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_Nov2022_clean/"

# Only run this once in the beginning of your analysis!
# Extracting data, the function looks for all "traj.csv" files in the indicated path (will look into subfolders)
# It will then generate a "results" table, with one line per worm, and these info:
#         results_table["condition"] = condition written on the plate of the worm
#         results_table["worm_id"] = number of the worm (100 times the file number + id attributed by tracking algorithm)
#         results_table["raw_visits"] = list outputed by patch_visits_single_traj (see its description)
#         results_table["avg_visit_duration"] = average visit duration of that worm
#         results_table["furthest_patch_distance"] = furthest patch visited
# Saves these results in a "results.csv" file in path, so no need to run this line every time!
regenerate_data = False
if regenerate_data:
    gr.generate_and_save(path) #run this once, will save results under path+"results.csv"

results = pd.read_csv(path+"results.csv") #run this to retrieve results
condition_nb, average_per_condition, errorbars = avg_duration_per_condition(results)

plt.bar(condition_nb, average_per_condition)
plt.errorbar(condition_nb, average_per_condition, errorbars, fmt='.k')
plt.show()