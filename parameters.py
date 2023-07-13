import copy

import matplotlib.pyplot as plt
import patch_coordinates

# General parameters

# NOT IMPLEMENTED YET
# Tolerance for the worm exiting a food patch, so when it's radius+tolerance away we still count it as being inside
radial_tolerance = 0
# Just a parameter to toggle extensive printing in the functions (for debugging purposes)
verbose = False

# Time threshold list for visit aggregation
threshold_list = [0, 10, 100, 100000]

# Time threshold for leaving probability (to compute P_leave of a worm, look at probability that it leaves in the next
#   N time steps, with N being this threshold)
time_threshold = 20

# Condition names
nb_to_name = {0: "close 0.2", 1: "med 0.2", 2: "far 0.2", 3: "cluster 0.2", 4: "close 0.5", 5: "med 0.5", 6: "far 0.5",
              7: "cluster 0.5", 8: "med 1.25", 9: "med 0.2+0.5", 10: "med 0.5+1.25", 12: "close 0", 13: "med 0",
              14: "far 0", 15: "cluster 0"}
name_to_nb = {v: k for k, v in nb_to_name.items()}

# Distance to number of patch dictionary (lower we build a condition number to number of patches dictionary from that)
distance_to_nb_of_patches = {"close": 52, "med": 24, "far": 7, "superfar": 3, "cluster": 25}
# nb_to_nb_of_patches = {0: 52, 1: 24, 2: 7, 3: 25, 4: 52, 5: 24, 6: 7, 7: 25, 8: 24, 9: 24, 10: 24, 11: 24}

# Loops to make nice dictionaries from that:
# nb_to_distance
# nb_to_density
# nb_to_nb_of_patches
# name_to_nb_list

nb_to_distance = {}
for condition in nb_to_name.keys():
    if "close" in nb_to_name[condition]:
        nb_to_distance[condition] = "close"
    elif "med" in nb_to_name[condition]:
        nb_to_distance[condition] = "med"
    elif "far" in nb_to_name[condition]:
        nb_to_distance[condition] = "far"
    elif "superfar" in nb_to_name[condition]:
        nb_to_distance[condition] = "superfar"
    elif "cluster" in nb_to_name[condition]:
        nb_to_distance[condition] = "cluster"

nb_to_density = {}
for condition in nb_to_name.keys():
    if "0.2+0.5" in nb_to_name[condition]:
        nb_to_density[condition] = "0.2+0.5"
    elif "0.5+1.25" in nb_to_name[condition]:
        nb_to_density[condition] = "0.5+1.25"
    # Put those after otherwise 0.5+1.25 would be classified as 0.5
    elif "0.2" in nb_to_name[condition]:
        nb_to_density[condition] = "0.2"
    elif "0.5" in nb_to_name[condition]:
        nb_to_density[condition] = "0.5"
    elif "1.25" in nb_to_name[condition]:
        nb_to_density[condition] = "1.25"
    # Put this after otherwise all conditions go to 0
    elif "0" in nb_to_name[condition]:
        nb_to_density[condition] = "0"

nb_to_nb_of_patches = {}
for condition in nb_to_name.keys():
    if "close" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["close"]
    elif "med" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["med"]
    elif "far" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["far"]
    elif "superfar" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["superfar"]
    elif "cluster" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["cluster"]

# Convert condition names into lists of condition numbers
# eg {"close": [0, 4], "med": [1, 5, 8, 9, 10, 11], "far": [2, 6]}
name_to_nb_list = {"all": [], "close": [], "med": [], "far": [], "superfar": [], "cluster": [], "0.5+1.25": [],
                   "0.2+0.5": [], "0.5": [], "1.25": [], "0.2": [], "0": []}
for condition in nb_to_name.keys():
    name_to_nb_list["all"].append(condition)  # the all should have everyone
    name_to_nb_list[nb_to_name[condition]] = [
        condition]  # "condition" to [condition_nb] conversion for single conditions
    # For every pool defined in name_to_nb_list initialization, add all conditions that have this name in their name
    # Note: first it finds distance, then will stop at first density found (which should be the right one)
    # (this is to avoid all conditions being put in 0)
    distance_found = False
    for condition_pool in name_to_nb_list.keys():
        if condition_pool in nb_to_name[condition]:
            name_to_nb_list[condition_pool].append(condition)
            if not distance_found:
                distance_found = True
            else:
                break  # order in name_to_nb list matters

# Same but to list of names (eg "close" => ["close 0", "close 0.2", "close 0.5"]
name_to_name_list = {"all": [], "close": [], "med": [], "far": [], "superfar": [], "cluster": [], "0.5+1.25": [],
                     "0.2+0.5": [], "0.5": [], "1.25": [], "0.2": [], "0": []}
for name in name_to_name_list.keys():
    condition_list = copy.deepcopy(name_to_nb_list[name])
    for i_cond in range(len(condition_list)):
        condition_list[i_cond] = nb_to_name[condition_list[i_cond]]  # convert each nb of nb_list to a name
    name_to_name_list[name] = condition_list

# close: purple
# med: blue
# far: green
# superfar: yellowish green
# clusters: teal
# 0.2, 0.5, 1.25: shades of brown

name_to_color = {"close": "purple", "med": "blue", "far": "cornflowerblue", "superfar": "teal",
                 "cluster": "yellowgreen",
                 "0.2": "burlywood", "0.5": "darkgoldenrod", "1.25": "brown", "0.2+0.5": "chocolate",
                 "0.5+1.25": "orange",
                 "control": "gray", "all": "pink"}
# Add colors for single conditions, distance override
for condition in nb_to_name.keys():
    name_to_color[nb_to_name[condition]] = name_to_color[nb_to_distance[condition]]


def test_colors():
    x = 0
    y = 1
    for pool in name_to_color.keys():
        plt.scatter(x, y, color=name_to_color[pool], label=pool)
        x += 1
    plt.legend()
    plt.show()


# Centers of patches for each condition
nb_to_xy = {}
for condition in nb_to_distance.keys():
    if nb_to_distance[condition] == "close":
        nb_to_xy[condition] = patch_coordinates.xy_patches_close
    if nb_to_distance[condition] == "med":
        nb_to_xy[condition] = patch_coordinates.xy_patches_med
    if nb_to_distance[condition] == "far":
        nb_to_xy[condition] = patch_coordinates.xy_patches_far
    if nb_to_distance[condition] == "cluster":
        nb_to_xy[condition] = patch_coordinates.xy_patches_cluster

# Centers of patches for each condition
distance_to_xy = {"close": patch_coordinates.xy_patches_close, "med": patch_coordinates.xy_patches_med,
                  "far": patch_coordinates.xy_patches_far, "cluster": patch_coordinates.xy_patches_cluster}
