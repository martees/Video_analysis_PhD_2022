import matplotlib.pyplot as plt
import numpy as np
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
              7: "cluster 0.5", 8: "med 1.25", 9: "med 0.2+0.5", 10: "med 0.5+1.25", 11: "med 0"}

# Distance to number of patch dictionary (lower we build a condition number to number of patches dictionary from that)
distance_to_nb_of_patches = {"close":52, "med":24, "far":7, "extrafar":3, "cluster":25}
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
    elif "extrafar" in nb_to_name[condition]:
        nb_to_distance[condition] = "extrafar"
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
    elif "extrafar" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["extrafar"]
    elif "cluster" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["cluster"]

# Convert condition names into lists of condition numbers
# eg {"close": [0, 4], "med": [1, 5, 8, 9, 10, 11], "far": [2, 6]}
name_to_nb_list = {"all": [], "close": [], "med": [], "far": [], "extrafar": [], "cluster": [], "0.5+1.25": [],
                   "0.2+0.5": [], "0.5": [], "1.25": [], "0.2": [], "0": []}
for condition in nb_to_name.keys():
    name_to_nb_list["all"].append(condition)  # the all should have everyone
    name_to_nb_list[nb_to_name[condition]] = [condition]  # "condition" to [condition_nb] conversion for single conditions
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

# close: purple
# med: blue
# far: green
# extrafar: yellowish green
# clusters: teal
# 0.2, 0.5, 1.25: shades of brown

name_to_color = {"close": "purple", "med": "blue", "far": "cornflowerblue", "extrafar": "teal", "cluster": "yellowgreen",
                 "0.2": "burlywood", "0.5": "darkgoldenrod", "1.25": "brown", "0.2+0.5": "chocolate", "0.5+1.25": "orange",
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


# x-y coordinates of the patches in the reference points system
xy_patches_far = [
    [-9.0, -15.59],
    [9.0, -15.59],
    [-18.0, 0.0],
    [0.0, 0.0],
    [18.0, 0.0],
    [-9.0, 15.59],
    [9.0, 15.59]
]

xy_patches_med = [
    [-13.5, -15.59],
    [-4.5, -15.59],
    [4.5, -15.59],
    [13.5, -15.59],
    [-18.0, -7.79],
    [-9.0, -7.79],
    [0.0, -7.79],
    [9.0, -7.79],
    [18.0, -7.79],
    [-22.5, 0.0],
    [-13.5, 0.0],
    [-4.5, 0.0],
    [4.5, 0.0],
    [13.5, 0.0],
    [22.5, 0.0],
    [-18.0, 7.79],
    [-9.0, 7.79],
    [0.0, 7.79],
    [9.0, 7.79],
    [18.0, 7.79],
    [-13.5, 15.59],
    [-4.5, 15.59],
    [4.5, 15.59],
    [13.5, 15.59]
]

alpha = -15 / 180 * np.pi
mediumSpaceListOrig = xy_patches_med.copy()
for iPatch in range(len(xy_patches_med)):
    xy = xy_patches_med[iPatch]
    xy_patches_med[iPatch] = [xy[0] * np.cos(alpha) - xy[1] * np.sin(alpha),
                              xy[0] * np.sin(alpha) + xy[1] * np.cos(alpha)]

xy_patches_close = [
    [-15.75, -11.69],
    [-11.25, -11.69],
    [-6.75, -11.69],
    [-2.25, -11.69],
    [2.25, -11.69],
    [6.75, -11.69],
    [11.25, -11.69],
    [15.75, -11.69],
    [-13.5, -7.79],
    [-9.0, -7.79],
    [-4.5, -7.79],
    [0.0, -7.79],
    [4.5, -7.79],
    [9.0, -7.79],
    [13.5, -7.79],
    [-15.75, -3.90],
    [-11.25, -3.90],
    [-6.75, -3.90],
    [-2.25, -3.90],
    [2.25, -3.90],
    [6.75, -3.90],
    [11.25, -3.90],
    [15.75, -3.90],
    [-13.5, 0.0],
    [-9.0, 0.0],
    [-4.5, 0.0],
    [4.5, 0.0],
    [9.0, 0.0],
    [13.5, 0.0],
    [-15.75, 3.90],
    [-11.25, 3.90],
    [-6.75, 3.90],
    [-2.25, 3.90],
    [2.25, 3.90],
    [6.75, 3.90],
    [11.25, 3.90],
    [15.75, 3.90],
    [-13.5, 7.79],
    [-9.0, 7.79],
    [-4.5, 7.79],
    [0.0, 7.79],
    [4.5, 7.79],
    [9.0, 7.79],
    [13.5, 7.79],
    [-15.75, 11.69],
    [-11.25, 11.69],
    [-6.75, 11.69],
    [-2.25, 11.69],
    [2.25, 11.69],
    [6.75, 11.69],
    [11.25, 11.69],
    [15.75, 11.69]
]

xy_patches_cluster = [
    [-16.14, -9.6],
    [-12.1, -9.96],
    [-18.94, -6.54],
    [-14.48, -5.68],
    [-20.09, -11.42],
    [-2.71, 12.51],
    [-4.6, 7.81],
    [0.47, 15.37],
    [0.65, 10.3],
    [-6.82, 11.76],
    [5.97, -16.54],
    [7.33, -11.5],
    [9.23, -18.94],
    [2.66, -14.34],
    [5.56, -20.51],
    [17.69, 4.04],
    [15.78, 8.05],
    [20.36, 6.92],
    [13.3, 3.47],
    [16.1, 0.44],
    [-19.21, 8.89],
    [-22.63, 6.53],
    [-15.15, 8.95],
    [-18.35, 5.02]
]

mediumSpaceHighDensityMask = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]

# Centers of patches for each condition
nb_to_xy = {}
for condition in nb_to_distance.keys():
    if nb_to_distance[condition] == "close":
        nb_to_xy[condition] = xy_patches_close
    if nb_to_distance[condition] == "med":
        nb_to_xy[condition] = xy_patches_med
    if nb_to_distance[condition] == "far":
        nb_to_xy[condition] = xy_patches_far
    if nb_to_distance[condition] == "cluster":
        nb_to_xy[condition] = xy_patches_cluster


