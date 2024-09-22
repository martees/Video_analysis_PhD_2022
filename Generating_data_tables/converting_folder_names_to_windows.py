import pandas as pd
import numpy as np
from Generating_data_tables import main as gen

# Just a short script to take an existing clean_results.csv file and replace the folders in "folder" column by
# ones starting with the right path for the results (which changes when switching from linux to windows)


# CODE TO RUN WHEN ON WINDOWS WITH RESULTS PRODUCED IN LINUX
def linux_to_windows():
    linux_path = gen.generate("", shorten_traj=False, force_linux=True)
    windows_path = gen.generate("", shorten_traj=False)
    results = pd.read_csv(windows_path + "clean_results.csv")
    for i_folder in range(len(results["folder"])):
        current_folder = results["folder"][i_folder]
        if str(current_folder) != "nan":
            results.loc[i_folder, "folder"] = windows_path + current_folder[len(linux_path):]
    results.to_csv(windows_path + "clean_results.csv")


linux_to_windows()
