# Just a small script to print the N for each condition
import datatable as dt
from Generating_data_tables import main as gen
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
import find_data as fd
from Parameters import parameters as param

path = gen.generate("")
results = dt.fread(path + "clean_results.csv")
trajectories = dt.fread(path + "clean_trajectories.csv")
full_list_of_folders = results[:, "folder"].to_list()[0]
print("Finished loading tables!")

for nb, name in param.nb_to_name.items():
    nb_of_folders = len(fd.return_folders_condition_list(full_list_of_folders, nb))
    print(name, ", n=", nb_of_folders)
