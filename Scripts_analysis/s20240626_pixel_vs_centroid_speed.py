# In this script I want to compare pixelwise visits and centroid speed.
# Pixelwise visit duration: length for which a pixel is covered.
# Centroid speed: distance covered by the centroid between previous and current time step.
# Goal: explain why there are more short pixel visits in the close condition. If those very short visits are the head
# of the worm wiggling, for low centroid speeds, we should see more dispersion in the pixel visits (with body pixels
# having very long visit times, and head pixels having shorter ones)

from Generating_data_tables import main as gen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load path and clean_results.csv, because that's where the list of folders we work on is stored
    path = gen.generate(test_pipeline=False)
    results = pd.read_csv(path + "clean_results.csv")
    trajectories = pd.read_csv(path + "clean_trajectories.csv")
    full_list_of_folders = list(results["folder"])
    if "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv" in full_list_of_folders:
        full_list_of_folders.remove(
            "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/20221011T191711_SmallPatches_C2-CAM7/traj.csv")
