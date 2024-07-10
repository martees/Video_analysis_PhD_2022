# In this script I want to compare pixelwise visits and centroid speed.
# Pixelwise visit duration: length for which a pixel is covered.
# Centroid speed: distance covered by the centroid between previous and current time step.
# Goal: explain why there are more short pixel visits in the close condition. If those very short visits are the head
# of the worm wiggling, for low centroid speeds, we should see more dispersion in the pixel visits (with body pixels
# having very long visit times, and head pixels having shorter ones)
import analysis as ana
from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt
from Parameters import parameters as param
import find_data as fd

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time


def pixel_visit_duration_vs_centroid_speed(results_path, trajectories, full_folder_list, condition_list, regenerate=False):

    if not os.path.isdir(results_path + "pixel_visits_vs_centroid_speed"):
        os.mkdir(results_path + "pixel_visits_vs_centroid_speed")

    tic = time.time()
    for i_condition, condition in enumerate(condition_list):
        existing_data_path = results_path + "pixel_visits_vs_centroid_speed/visits_vs_speeds_list_"+str(condition)+".npy"
        if not os.path.isfile(existing_data_path) or regenerate:
            pixel_visit_values = []
            corresponding_centroid_speeds = []
            list_of_folders = fd.return_folders_condition_list(full_folder_list, [condition])
            print("Condition ", i_condition, " / ", len(condition_list))
            for i_folder, folder in enumerate(list_of_folders):
                print(">>> Folder ", i_folder, " / ", len(list_of_folders))
                # If it's not already done, compute the pixel visit durations
                pixelwise_visits_path = folder[:-len("traj.csv")] + "pixelwise_visits.npy"
                if not os.path.isfile(pixelwise_visits_path):
                    gr.generate_pixelwise_visits(trajectories, folder)
                pixelwise_visits = np.load(pixelwise_visits_path, allow_pickle=True)
                # Load the patch to which each pixel belongs
                in_patch_matrix_path = folder[:-len("traj.csv")] + "in_patch_matrix.csv"
                if not os.path.isfile(in_patch_matrix_path):
                    gt.in_patch_all_pixels(folder)
                in_patch_matrix = pd.read_csv(in_patch_matrix_path)
                # Separate inside / outside food patch visit durations
                pixelwise_visits = pixelwise_visits[in_patch_matrix != -1]
                # Lose the pixelwise structure
                pixelwise_visits = [pixelwise_visits[i][j] for i in range(len(pixelwise_visits)) for j in
                                    range(len(pixelwise_visits[i]))]
                # Load the speeds of the centroid
                current_traj = trajectories[trajectories["folder"] == folder].reset_index(drop=True)

                # Visit durations
                pixel_visit_values += ana.convert_to_durations(pixelwise_visits)
                # Corresponding centroid speeds
                for i_visit in range(len(pixelwise_visits)):
                    if i_visit % (len(pixelwise_visits) // 6) == 0:
                        print(">>>>>> Visit ", i_visit, " / ", len(pixelwise_visits))
                    current_visit_traj = current_traj.loc[(pixelwise_visits[i_visit][0] <= current_traj["frame"]) & (current_traj["frame"] <= pixelwise_visits[i_visit][1])]
                    corresponding_centroid_speeds.append(np.mean(current_visit_traj["speeds"]))
            data_to_save = np.array([pixel_visit_values, corresponding_centroid_speeds])
            np.save(existing_data_path, data_to_save)
        corresponding_centroid_speeds, pixel_visit_values = np.load(existing_data_path)
        corresponding_centroid_speeds = list(corresponding_centroid_speeds)
        pixel_visit_values = list(pixel_visit_values)

        bin_list, avg_list, _, binned_y_values = ana.xy_to_bins(corresponding_centroid_speeds,
                                                                pixel_visit_values, 1,
                                                                print_progress=False,
                                                                custom_bins=[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6,
                                                                             1.8, 2, 3, 4, 6, 12, 18, 24, 40, 80, 120, 160, 200, 240, 300, 340, 400, 1200], compute_bootstrap=False)

        # Keep only bins with > 100 points
        valid_bins = [bin_list[i] for i in range(len(bin_list)) if len(binned_y_values[i]) >= 100]
        valid_avg = [avg_list[i] for i in range(len(avg_list)) if len(binned_y_values[i]) >= 100]
        #valid_indices = np.where(corresponding_centroid_speeds <= np.max(valid_bins))[0]

        #plt.hist2d(np.array(corresponding_centroid_speeds)[i_condition, valid_indices],
        #np.array(pixel_visit_values)[i_condition, valid_indices], cmap="YlOrBr", norm="log", bins=100)

        # plt.scatter(np.array(corresponding_centroid_speeds)[i_condition, valid_indices],
        #            np.array(pixel_visit_values)[i_condition, valid_indices], color="black", alpha=0.1)

        plt.plot(valid_bins, valid_avg, color="white", linewidth=6)
        plt.plot(valid_bins, valid_avg, label=param.nb_to_name[condition],
                 color=param.name_to_color[param.nb_to_name[condition]], linewidth=4)

    plt.ylabel("Average duration of a visit to this pixel")
    plt.xlabel("Average centroid speed when this pixel is visited")
    plt.title(str([param.nb_to_name[cond] for cond in condition_list]))

    print("Total time: ", int((time.time() - tic) // 60), "min")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(test_pipeline=False)
    results = pd.read_csv(path + "clean_results.csv")
    traj = pd.read_csv(path + "clean_trajectories.csv")
    full_list_of_folders = list(results["folder"])
    if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
        full_list_of_folders.remove(
            "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")

    #pixel_visit_duration_vs_centroid_speed(path, traj, full_list_of_folders, [0, 1, 2], regenerate=True)
    #pixel_visit_duration_vs_centroid_speed(path, traj, full_list_of_folders, [4, 5, 6], regenerate=True)
    #pixel_visit_duration_vs_centroid_speed(path, traj, full_list_of_folders, [12, 13, 14], regenerate=True)
    #pixel_visit_duration_vs_centroid_speed(path, traj, full_list_of_folders, [12, 0, 4])
    #pixel_visit_duration_vs_centroid_speed(path, traj, full_list_of_folders, [13, 1, 5])
    #pixel_visit_duration_vs_centroid_speed(path, traj, full_list_of_folders, [14, 2, 6])
