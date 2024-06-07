# A script to plot a heatmap of the duration of visit to each pixel
# But munching all the conditions together mouahahahaha


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

# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=False)
results = pd.read_csv(path + "clean_results.csv")
traj = pd.read_csv(path + "clean_trajectories.csv")
full_plate_list = results["folder"]

regenerate_pixel_visits = False
curve_list = [[0], [1], [2]]
curve_names = ["close 0.2", "med 0.2", "far 0.2"]
heatmap_each_curve = [np.zeros((1847, 1847)) for _ in range(len(curve_list))]

# Plot initialization
fig, axes = plt.subplots(1, len(curve_list))
fig.suptitle("Heatmap of worm presence")

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

        if len(current_pixel_wise_visits) == 1847:
            # Load patch info for this folder
            in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
            if not os.path.isfile(in_patch_matrix_path):
                gt.in_patch_all_pixels(plate)
            in_patch_matrix = pd.read_csv(in_patch_matrix_path)
            # Convert visits to durations, and sum them, to get the total time spent in each pixel
            for i in range(len(current_pixel_wise_visits)):
                for j in range(len(current_pixel_wise_visits[i])):
                    if in_patch_matrix[str(j)][i] == -1:
                        current_pixel_wise_visits[i][j] = int(np.sum(ana.convert_to_durations(current_pixel_wise_visits[i][j])))
                    else:
                        current_pixel_wise_visits[i][j] = 0
            # Add them to the corresponding curve data
            heatmap_each_curve[i_curve] = heatmap_each_curve[i_curve] + current_pixel_wise_visits
        # If the plate is not 1944 x 1944, print something out
        else:
            print("Plate ", plate, " is not the standard size, it's ", len(current_pixel_wise_visits))

    heatmap_each_curve[i_curve] /= np.max(heatmap_each_curve[i_curve])
    axes[i_curve].imshow(heatmap_each_curve[i_curve].astype(float), vmax=0.5)
    #if i_curve == len(curve_list) - 1:
    #    axes[i_curve].colorbar()
plt.show()
