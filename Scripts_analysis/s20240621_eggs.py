import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import find_data as fd
from Parameters import parameters as param
from Generating_data_tables import main as gen
import analysis as ana

folder_path = "/home/admin/Desktop/Camera_setup_analysis/Egg_counting/"
egg_counts_each_patch = pd.read_csv(folder_path + "Minipatches_egg_counts.csv")
egg_counts_each_plate = pd.read_csv(folder_path + "Minipatches_egg_counts_metadata.csv")
patch_locations = pd.read_csv(folder_path + "Minipatches_patch_positions.csv")
inventory_plates = pd.read_csv(folder_path + "inventory_pretracking.csv")


def print_dates_and_hard_drives():
    dates_with_egg_counts = []
    for plate in egg_counts_each_plate["plate_id"]:
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


def hist_nb_of_eggs_each_plate(condition_list):
    """
    Function that plots a histogram with the average nb of eggs laid in each condition in condition_list.
    """
    avg_nb_eggs_each_condition = [0 for _ in range(len(condition_list))]
    errors_inf_each_condition = [0 for _ in range(len(condition_list))]
    errors_sup_each_condition = [0 for _ in range(len(condition_list))]
    avg_nb_eggs_each_condition_each_plate = [[] for _ in range(len(condition_list))]
    plate_list = pd.unique(egg_counts_each_plate["vid_path"])
    plate_list = [plate_list[i] for i in range(len(plate_list)) if str(plate_list[i]) != "nan"]
    missing_plates = ["20221116T102308_SmallPatches_C3-CAM5", "20221119T115009_SmallPatches_C5-CAM4",
                      "20221121T121358_SmallPatches_C5-CAM3", "20221121T121358_SmallPatches_C5-CAM6",
                      "20221121T121358_SmallPatches_C5-CAM7", "20221123T120808_SmallPatches_C4-CAM3",
                      "20221121T121358_SmallPatches_C5-CAM5"]
    plate_list = [plate_list[i] for i in range(len(plate_list)) if plate_list[i] not in missing_plates]
    results_path = gen.generate("")
    for i_plate, plate in enumerate(plate_list):
        #print(plate)
        current_plate = egg_counts_each_plate[egg_counts_each_plate["vid_path"] == plate].reset_index()
        #if len(current_plate) > 0:
        condition = fd.load_condition(results_path + plate + "/traj.csv")
        if condition in condition_list:
            condition_index = np.where(condition_list == condition)[0][0]
            avg_nb_eggs_each_condition_each_plate[condition_index].append(current_plate["egg_total"][0])
    for i_condition in range(len(condition_list)):
        avg_nb_eggs_each_condition[i_condition] = np.mean(avg_nb_eggs_each_condition_each_plate[i_condition])
        errors = ana.bottestrop_ci(avg_nb_eggs_each_condition_each_plate[i_condition], 1000)
        errors_inf_each_condition[i_condition], errors_sup_each_condition[i_condition] = [avg_nb_eggs_each_condition[i_condition] - errors[0],
                                                                                          errors[1] - avg_nb_eggs_each_condition[i_condition]]
    # Plotty plot
    condition_names = [param.nb_to_name[cond] for cond in condition_list]
    condition_colors = [param.name_to_color[name] for name in condition_names]
    plt.title("Number of eggs")
    plt.ylabel("Total time per patch")
    # Bar plot
    plt.bar(range(len(condition_list)), avg_nb_eggs_each_condition, color=condition_colors)
    plt.xticks(range(len(condition_list)), condition_names, rotation=45)
    plt.xlabel("Condition number")
    # Plate averages as scatter on top
    for i in range(len(condition_list)):
        plt.scatter([range(len(condition_list))[i] for _ in range(len(avg_nb_eggs_each_condition_each_plate[i]))],
                    avg_nb_eggs_each_condition_each_plate[i], color="orange", zorder=2)
    # Error bars
    plt.errorbar(range(len(condition_list)), avg_nb_eggs_each_condition, [errors_inf_each_condition, errors_sup_each_condition], fmt='.k', capsize=5)

    plt.show()


list_by_density = param.name_to_nb_list["0"] + param.name_to_nb_list["0.2"] + param.name_to_nb_list["0.5"] + param.name_to_nb_list["1.25"]
list_by_distance = (param.name_to_nb_list["close"] + param.name_to_nb_list["med"] + param.name_to_nb_list["far"]
                    + param.name_to_nb_list["superfar"] + param.name_to_nb_list["cluster"])
hist_nb_of_eggs_each_plate(list_by_distance)












