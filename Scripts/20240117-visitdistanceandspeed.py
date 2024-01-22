# Analysis of the average distance crawled by the worms for each visit, and also their average speed, and 1/that.
# We look at the inverse of the speeds, because in our visit duration graphs, we are plotting time = distance/speed,
# so we wonder if the variability that we see can be found in distance, speed or 1/speed.

import find_data as fd
from main import *
import numpy as np

path = gr.generate(starting_from="")
trajectories = pd.read_csv(path + "clean_trajectories.csv")
results = pd.read_csv(path + "clean_results.csv")

# Parameter you can set to False if you want to use already existing data
recompute = False

if recompute == True:
    # Folder list
    folder_list = pd.unique(results["folder"])
    nb_of_folders = len(folder_list)

    # Output lists, will have one average value for each plate/folder
    list_of_avg_distances = []
    list_of_avg_speeds = []
    list_of_avg_speeds_inv = []

    for i_folder in range(nb_of_folders):
        if i_folder % 10 == 0:
            print("Computing for folder ", i_folder, " / ", nb_of_folders)

        # Define current visits
        current_folder = folder_list[i_folder]
        current_results = results[results["folder"] == current_folder]
        current_visit_list = fd.load_list(current_results, "no_hole_visits")
        nb_of_visits = len(current_visit_list)

        # Variable to store sums (in order to compute average per visit after the for loop)
        distance_all_visits = 0
        speed_all_visits = 0
        speed_all_visits_inv = 0

        for i_visit in range(nb_of_visits):
            current_visit = current_visit_list[i_visit]
            visit_start_index = fd.load_index(current_folder, current_visit[0])
            visit_end_index = fd.load_index(current_folder, current_visit[1])
            distance_all_visits += np.sum(trajectories["distances"][visit_start_index: visit_end_index + 1])

            # Working a bit more for the speeds
            speed_list = trajectories["speeds"][visit_start_index: visit_end_index + 1]
            speed_all_visits += np.mean(speed_list)
            # For the inverse, exclude null speeds
            speed_all_visits_inv += 1/np.mean(speed_list)

        if nb_of_visits != 0:
            list_of_avg_distances.append(distance_all_visits / nb_of_visits)
            list_of_avg_speeds.append(speed_all_visits / nb_of_visits)
            list_of_avg_speeds_inv.append(speed_all_visits_inv / nb_of_visits)
        else:
            list_of_avg_distances.append(0)
            list_of_avg_speeds.append(0)
            list_of_avg_speeds_inv.append(0)

    results["average_distance_each_visit"] = list_of_avg_distances
    results["average_speed_each_visit"] = list_of_avg_speeds
    results["average_speed_each_visit_inv"] = list_of_avg_speeds_inv

    results.to_csv(path + "clean_results.csv")


plot_graphs(results, "visit_smoothed_speeds", [["close 0", "med 0", "far 0"]])
plot_graphs(results, "visit_smoothed_speeds", [["close 0.2", "med 0.2", "far 0.2"]])
plot_graphs(results, "visit_smoothed_speeds", [["close 0.5", "med 0.5", "far 0.5"]])

# Garbage, code for recomputing the inverse of speeds fast
# # Folder list
# folder_list = pd.unique(results["folder"])
# nb_of_folders = len(folder_list)
#
# # Output lists, will have one average value for each plate/folder
# list_of_avg_speeds_inv = []
#
# for i_folder in range(nb_of_folders):
#     current_folder = folder_list[i_folder]
#     current_results = results[results["folder"] == current_folder].reset_index()
#     avg_visit_speed = current_results["average_speed_each_visit"][0]
#     if avg_visit_speed != 0:
#         list_of_avg_speeds_inv.append(1/avg_visit_speed)
#     else:
#         list_of_avg_speeds_inv.append(0)
#
# results["average_speed_each_visit_inv"] = list_of_avg_speeds_inv
# results.to_csv(path + "clean_results.csv")

