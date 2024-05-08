import pandas as pd
import numpy as np

import find_data as fd
from Parameters import parameters as param
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_controls as gc
from Generating_data_tables import generate_trajectories as gt


def exclude_invalid_videos(trajectories, results_per_plate, bad_patches_folders):
    # Remove the plates which have overlapping patches
    folders_with_bad_patches = bad_patches_folders[bad_patches_folders["overlapping_patches"] == True].reset_index(drop=True)
    for i_folder in range(len(results_per_plate)):
        if results_per_plate.loc[i_folder, "folder"] in list(folders_with_bad_patches["folder"]):
            results_per_plate.drop([i_folder], inplace=True)

    # Remove plates which don't have enough video time, or have more than ~10-30 double frames (=> could be two worms)
    cleaned_results = results_per_plate[results_per_plate["total_video_time"] >= 10000]
    cleaned_results = cleaned_results[cleaned_results["avg_proportion_double_frames"] <= 0.01]

    # Once the bad folders have been excluded from clean_results, remove them from the trajectory file
    valid_folders = np.unique(cleaned_results["folder"])
    cleaned_traj = pd.DataFrame(columns=trajectories.columns)
    cleaned_traj["is_smoothed"] = cleaned_traj["is_smoothed"].astype(bool)
    for plate in valid_folders:
        cleaned_traj = pd.concat([cleaned_traj, trajectories[trajectories["folder"] == plate]])
    return cleaned_traj, cleaned_results


def generate_smooth_trajectories(path):
    # Retrieve trajectories from the folder path and save them in one dataframe
    trajectories = fd.trajcsv_to_dataframe(fd.path_finding_traj(path))
    print("Finished retrieving trajectories")
    # Smooth the trajectory
    trajectories = gt.smooth_trajectory(trajectories, 2)
    trajectories.to_csv(path + "trajectories.csv")


def generate_trajectories(path):
    trajectories = pd.read_csv(path + "trajectories.csv")
    # Add a column with the patch where the worm is (-1 is outside)
    print("Computing where the worm is...")
    #trajectories["patch_centroid"] = in_patch_list(trajectories, using="centroid")
    trajectories["patch_silhouette"], overlapping_patches = gt.in_patch_list(trajectories, using="silhouette")
    overlapping_patches.to_csv(path + "overlapping_patches.csv")
    print("Finished computing in which patch the worm is at each time step")
    print("Computing distances...")
    # Add a column with the distance the worm crawled since last time step, for each time step
    # It should put 0 for the first point of every folder, but record distance even when there's a tracking hole
    # Distances are computed now because they are used for average speed analysis in results_per_id.
    # Doing speed analysis later is tiring because of the tracking holes that are not in the results_per_plate
    # table anymore.
    trajectories["distances"] = gt.trajectory_distances(trajectories)
    trajectories.to_csv(path + "trajectories.csv")
    print("Finished computing distance covered by the worm at each time step")


def generate_results_per_id(path):
    print("Building results_per_id...")
    trajectories = pd.read_csv(path + "trajectories.csv")
    print("Starting to build results_per_id from trajectories...")
    results_per_id = gr.make_results_per_id_table(trajectories)
    print("Finished!")
    results_per_id.to_csv(path + "results_per_id.csv")
    return 0


def generate_results_per_plate(path):
    print("Aggregating and preprocessing results per plate...")
    print("Retrieving results...")
    trajectories = pd.read_csv(path + "trajectories.csv",
                               index_col=0)  # index_col=0 is to prevent the addition of new index columns at each import
    results_per_id = pd.read_csv(path + "results_per_id.csv", index_col=0)
    print("Starting to build results_per_plate from results_per_id...")
    results_per_plate = gr.make_results_per_plate(results_per_id, trajectories)
    results_per_plate.to_csv(path + "results_per_plate.csv")
    return 0


def generate_clean_tables_and_speed(path):
    print("Retrieving results and trajectories...")
    results_per_plate = pd.read_csv(path + "results_per_plate.csv", index_col=0)
    trajectories = pd.read_csv(path + "trajectories.csv", index_col=0)
    overlapping_patches = pd.read_csv(path + "overlapping_patches.csv")
    print("Cleaning results...")
    clean_trajectories, clean_results = exclude_invalid_videos(trajectories, results_per_plate, overlapping_patches)
    clean_results.to_csv(path + "clean_results.csv")
    print("Computing speeds...")
    # For faster execution, we only compute speeds here (and not at the same time as distances), when invalid
    # trajectories have been excluded
    clean_trajectories["speeds"] = gt.trajectory_speeds(clean_trajectories)
    clean_trajectories.to_csv(path + "clean_trajectories.csv")
    return 0


def generate_aggregated_visits(path, threshold_list):
    print("Retrieving results and trajectories...")
    clean_results = pd.read_csv(path + "clean_results.csv", index_col=0).reset_index()
    print("Adding aggregated visits info for thresholds " + str(threshold_list) + "...")
    new_results = gr.add_aggregate_visit_info_to_results(clean_results, threshold_list)
    new_results.to_csv(path + "clean_results.csv", index=False)
    print("Done!")
    return new_results  # return this because this function is also used dynamically


def generate(starting_from="", test_pipeline=False, modeled_data=False):
    """
    Will generate the data tables starting more or less from scratch.
    Argument = from which level to regenerate stuff.
    Returns path.
    test_pipeline: if set to True, will run in a subset of the path that has to be hardcoded in this function.
    """

    # Data path
    if fd.is_linux():  # Linux path
        path = "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp/"
        if test_pipeline:
            path = "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp_less/"
        if modeled_data:
            path = "/media/admin/Expansion/Only_Copy_Probably/Results_minipatches_20221108_clean_fp_model_rw/"
    else:  # Windows path
        path = "C:/Users/Asmar/Desktop/Thèse/2022_summer_videos/Results_minipatches_20221108_clean_fp_less/"

    if starting_from == "controls":
        gc.generate_controls(path)
        generate_smooth_trajectories(path)
        generate_trajectories(path)
        generate_results_per_id(path)
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, param.threshold_list)

    if starting_from == "smoothing":
        generate_smooth_trajectories(path)
        generate_trajectories(path)
        generate_results_per_id(path)
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, param.threshold_list)

    if starting_from == "trajectories":
        generate_trajectories(path)
        generate_results_per_id(path)
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, param.threshold_list)

    elif starting_from == "results_per_id":
        generate_results_per_id(path)
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, param.threshold_list)

    elif starting_from == "results_per_plate":
        generate_results_per_plate(path)
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, param.threshold_list)

    elif starting_from == "clean":
        generate_clean_tables_and_speed(path)
        generate_aggregated_visits(path, param.threshold_list)

    return path
