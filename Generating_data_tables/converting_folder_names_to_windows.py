import pandas as pd
import numpy as np
from Generating_data_tables import main as gen

# Just a short script to take an existing clean_results.csv file and replace the folders in "folder" column by
# ones starting with the right path for the results (which changes when switching from linux to windows)


# CODE TO RUN WHEN ON WINDOWS WITH RESULTS PRODUCED IN LINUX
def linux_to_windows():
    linux_path = gen.generate("", shorten_traj=True, modeled_data=True, force_linux=True)
    windows_path = gen.generate("", shorten_traj=True, modeled_data=True)
    results = pd.read_csv(windows_path + "clean_results.csv")
    for i_folder in range(len(results["folder"])):
        current_folder = results["folder"][i_folder]
        if str(current_folder) != "nan":
            results.loc[i_folder, "folder"] = windows_path + current_folder[len(linux_path):]
    results.to_csv(windows_path + "clean_results.csv")


# CODE TO RUN WHEN ON LINUX WITH RESULTS PRODUCED IN WINDOWS
def windows_to_linux():
    linux_path = gen.generate("", shorten_traj=True)
    windows_path = gen.generate("", shorten_traj=True, force_windows=True)
    results = pd.read_csv(linux_path + "clean_results.csv")
    for i_folder in range(len(results["folder"])):
        current_folder = results["folder"][i_folder]
        if str(current_folder) != "nan":
            results.loc[i_folder, "folder"] = linux_path + current_folder[len(windows_path):]
    results.to_csv(linux_path + "clean_results.csv")


linux_to_windows()
