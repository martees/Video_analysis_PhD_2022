import pandas as pd
import numpy as np

folder_path = "/home/admin/Desktop/Camera_setup_analysis/Egg_counting/"
egg_counts_each_patch = pd.read_csv(folder_path + "Minipatches_egg_counts.csv")
plate_info = pd.read_csv(folder_path + "Minipatches_egg_counts_metadata.csv")
patch_locations = pd.read_csv(folder_path + "Minipatches_patch_positions.csv")
inventory_plates = pd.read_csv(folder_path + "inventory_pretracking.csv")

dates_with_egg_counts = []
for plate in plate_info["plate_id"]:
    # Go from "1711-..." to "20221117T"
    dates_with_egg_counts.append("2022"+plate[2:4]+plate[0:2]+"T")
dates_with_egg_counts = np.unique(dates_with_egg_counts)

plates_with_those_dates = []
hard_drives_with_those_dates = []
for i_folder, folder in enumerate(inventory_plates["folder"]):
    if folder[14:23] in dates_with_egg_counts:
        plates_with_those_dates.append(folder)
        hard_drives_with_those_dates.append(inventory_plates["name_disk"][i_folder])

print(dates_with_egg_counts)
print(plates_with_those_dates)
print(np.unique(hard_drives_with_those_dates))



