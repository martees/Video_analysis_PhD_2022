import pandas as pd
from Generating_data_tables import main as gen

# Just a short script to take an existing clean_results.csv file and replace the folders in "folder" column by
# ones starting with the right path for the results (which changes when switching from linux to windows)
# Use the switch_path to change to a custom path.


# CODE TO RUN WHEN ON WINDOWS WITH RESULTS PRODUCED IN LINUX
def linux_to_windows():
    linux_path = gen.generate("", shorten_traj=False, modeled_data=False, force_linux=True)
    windows_path = gen.generate("", shorten_traj=False, modeled_data=False)
    results = pd.read_csv(windows_path + "clean_results.csv")  # the results are saved in the windows path
    for i_folder in range(len(results["folder"])):
        current_folder = results["folder"][i_folder]
        if str(current_folder) != "nan" and linux_path in current_folder:
            results.loc[i_folder, "folder"] = windows_path + current_folder[len(linux_path):]
    results.to_csv(windows_path + "clean_results.csv")


# CODE TO RUN WHEN ON LINUX WITH RESULTS PRODUCED IN WINDOWS
def windows_to_linux():
    linux_path = gen.generate("", shorten_traj=False)
    windows_path = gen.generate("", shorten_traj=False, force_windows=True)
    results = pd.read_csv(linux_path + "clean_results.csv")  # the results are saved in the linux path
    for i_folder in range(len(results["folder"])):
        current_folder = results["folder"][i_folder]
        if str(current_folder) != "nan" and windows_path in current_folder:
            results.loc[i_folder, "folder"] = linux_path + current_folder[len(windows_path):]
    results.to_csv(linux_path + "clean_results.csv")


# CODE TO RUN WHEN RUNNING THE CODE ON A MACHINE WHICH IS NOT ALID'S
def switch_path(custom_path):
    results = pd.read_csv(custom_path + "clean_results.csv")
    for i_folder in range(len(results["folder"])):
        current_folder = results["folder"][i_folder]
        if str(current_folder) != "nan":
            # For non-control folders, format is results_path/plate/traj.csv
            if "control" not in str(current_folder):
                path_elements = str(current_folder).split("/")
                results.loc[i_folder, "folder"] = custom_path + path_elements[-2] + "/traj.csv"
            # For control folders, format is results_path/parent_plate/plate/traj.csv
            else:
                path_elements = str(current_folder).split("/")
                results.loc[i_folder, "folder"] = custom_path + path_elements[-3] + "/" + path_elements[-2] + "/traj.csv"
    results.to_csv(custom_path + "clean_results.csv")


linux_to_windows()
