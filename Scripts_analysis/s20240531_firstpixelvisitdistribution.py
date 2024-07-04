# A script to look at the distribution of first visits to pixels in our different conditions

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt
from Parameters import parameters as param
import find_data as fd
import analysis as ana


def histogram_visit_distribution(results, traj, curve_list, curve_names, regenerate_pixel_visits, only_first_visits, y_scale="linear", all_pixels=False):
    full_plate_list = results["folder"]
    px_visit_durations_each_curve = [[] for _ in range(len(curve_list))]

    tic = time.time()
    for i_curve in range(len(curve_list)):
        print(int(time.time() - tic), "s: Curve ", i_curve, " / ", len(curve_list))
        plate_list = fd.return_folders_condition_list(full_plate_list, curve_list[i_curve])
        for i_plate, plate in enumerate(plate_list):
            if i_plate % 10 == 0:
                print(">>> ", int(time.time() - tic), "s: plate ", i_plate, " / ", len(plate_list))
            # If it's not already done, compute the pixel visit durations
            pixelwise_visits_path = plate[:-len("traj.csv")] + "pixelwise_visits.npy"
            if not os.path.isfile(pixelwise_visits_path) or regenerate_pixel_visits:
                gr.generate_pixelwise_visits(traj, plate)
            # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
            current_pixel_wise_visits = np.load(pixelwise_visits_path, allow_pickle=True)

            # Load patch info for this folder
            in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
            if not os.path.isfile(in_patch_matrix_path):
                gt.in_patch_all_pixels(plate)
            in_patch_matrix = pd.read_csv(in_patch_matrix_path)

            # Separate inside / outside food patch visit durations
            if not all_pixels:
                current_pixel_wise_visits_inside = current_pixel_wise_visits[in_patch_matrix != -1]
            else:
                current_pixel_wise_visits_inside = current_pixel_wise_visits[in_patch_matrix > -1000]

            if only_first_visits:
                # Remove empty lists, and make it a list of visits (instead of a list of lists)
                current_pixel_wise_visits_inside = [current_pixel_wise_visits_inside[i][j]
                                                    for i in range(len(current_pixel_wise_visits_inside))
                                                    for j in range(min(1, len(current_pixel_wise_visits_inside[i])))
                                                    if len(current_pixel_wise_visits_inside[i]) > 0]
            else:
                # Remove empty lists, and make it a list of visits (instead of a list of lists)
                current_pixel_wise_visits_inside = [current_pixel_wise_visits_inside[i][j]
                                                    for i in range(len(current_pixel_wise_visits_inside))
                                                    for j in range(len(current_pixel_wise_visits_inside[i]))
                                                    if len(current_pixel_wise_visits_inside[i]) > 0]
            # Convert them to durations
            current_visit_durations = ana.convert_to_durations(current_pixel_wise_visits_inside)
            # Add them to the corresponding curve data
            px_visit_durations_each_curve[i_curve] += current_visit_durations

        curve_name = curve_names[i_curve]
        curve_color = param.name_to_color[curve_name]
        plt.hist(px_visit_durations_each_curve[i_curve], histtype="step", density=True, label=curve_name, color=curve_color,
                 bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30], linewidth=2)

    fig = plt.gcf()
    ax = fig.gca()
    ymin, ymax = ax.get_ylim()
    plt.vlines([np.mean(px_visit_durations_each_curve[i_curve]) for i_curve in range(len(curve_list))], ymin=ymin, ymax=ymax, colors=[param.name_to_color[c] for c in curve_names])

    plt.legend()
    plt.title("Distribution of " + only_first_visits * "first " + "pixel visit durations in the different conditions " + (1 - all_pixels) * "(inside patch)" + all_pixels * "(all pixels)")
    plt.yscale(y_scale)
    plt.show()


# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=False)
resultos = pd.read_csv(path + "clean_results.csv")
traj = pd.read_csv(path + "clean_trajectories.csv")

#histogram_visit_distribution(resultos, traj, [[0], [1], [2]], ["close 0.2", "med 0.2", "far 0.2"], False, True, y_scale="log")
#histogram_visit_distribution(resultos, traj, [[4], [5], [6]], ["close 0.5", "med 0.5", "far 0.5"], False, True, y_scale="log")
histogram_visit_distribution(resultos, traj, [[0], [1], [2]], ["close 0", "med 0", "far 0"], False, True, y_scale="log")
histogram_visit_distribution(resultos, traj, [[12], [13], [14]], ["close 0", "med 0", "far 0"], False, True, y_scale="log")
histogram_visit_distribution(resultos, traj, [[12], [13], [14]], ["close 0", "med 0", "far 0"], False, True, y_scale="log")
#histogram_visit_distribution(resultos, traj, [[0, 4], [1, 5], [2, 6]], ["close", "med", "far"], False, False)
#histogram_visit_distribution(resultos, traj, [[0, 1, 2, 3], [4, 5, 6, 7], [8], [12, 13, 14, 15]], ["0.2", "0.5", "1.25", "control"], False, True, y_scale="log")
#histogram_visit_distribution(resultos, traj, [[0, 4], [1, 5], [2, 6]], ["close", "med", "far"], False, False, "linear", True)

