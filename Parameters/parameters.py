import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

from Parameters import patch_coordinates

# General parameters

# Just a parameter to toggle extensive printing in the functions (for debugging purposes)
verbose = False

# Time threshold list for visit aggregation
threshold_list = [100000]

# Time threshold for leaving probability (to compute P_leave of a worm, look at probability that it leaves in the next
#   N time steps, with N being this threshold)
time_threshold = 1

# Number of pixel threshold to consider that a silhouette is invalid (if it has more than the threshold, invalid)
invalid_silhouette_threshold = 150

# Ratio for conversion between seconds and frames
# (for now measured on just one plate)
one_frame_in_seconds = 0.82817

# Condition names
nb_to_name = {0: "close 0.2", 1: "med 0.2", 2: "far 0.2", 14: "superfar 0.2", 3: "cluster 0.2",
              4: "close 0.5", 5: "med 0.5", 6: "far 0.5", 15: "superfar 0.5", 7: "cluster 0.5",
              12: "close 1.25", 8: "med 1.25", 13: "far 1.25", 16: "superfar 1.25",
              9: "med 0.2+0.5", 10: "med 0.5+1.25",
              17: "close 0", 18: "med 0", 19: "far 0", 20: "superfar 0", 21: "cluster 0"}
name_to_nb = {v: k for k, v in nb_to_name.items()}

list_by_distance = [17, 0, 4, 12, 18, 1, 5, 8, 19, 2, 6, 13, 20, 14, 15, 20, 16, 21, 3, 7]
list_by_density = [17, 18, 19, 20, 0, 1, 2, 14, 4, 5, 6, 15, 12, 8, 13, 16]
list_by_density_with_clusters = [17, 18, 19, 20, 21, 0, 1, 2, 14, 3, 4, 5, 6, 15, 7, 12, 8, 13, 16, 9, 10]

# Distance to number of patch dictionary (lower we build a condition number to number of patches dictionary from that)
distance_to_nb_of_patches = {"close": 52, "med": 24, "far": 7, "superfar": 3, "cluster": 25}
# nb_to_nb_of_patches = {0: 52, 1: 24, 2: 7, 3: 25, 4: 52, 5: 24, 6: 7, 7: 25, 8: 24, 9: 24, 10: 24, 11: 24}

# Loops to make nice dictionaries from that:
# nb_to_distance
# nb_to_density
# nb_to_nb_of_patches
# name_to_nb_list

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
    else:
        nb_to_density[condition] = "all"

nb_to_distance = {}
for condition in nb_to_name.keys():
    if "0" in nb_to_density[condition]:
        nb_to_distance[condition] = "control"
    if "close" in nb_to_name[condition]:
        nb_to_distance[condition] = "close"
    elif "med" in nb_to_name[condition]:
        nb_to_distance[condition] = "med"
    elif "superfar" in nb_to_name[condition]:  # superfar before because "far" is in "superfar"
        nb_to_distance[condition] = "superfar"
    elif "far" in nb_to_name[condition]:
        nb_to_distance[condition] = "far"
    elif "cluster" in nb_to_name[condition]:
        nb_to_distance[condition] = "cluster"

nb_to_nb_of_patches = {}
for condition in nb_to_name.keys():
    if "close" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["close"]
    elif "med" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["med"]
    elif "superfar" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["superfar"]
    elif "far" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["far"]
    elif "cluster" in nb_to_name[condition]:
        nb_to_nb_of_patches[condition] = distance_to_nb_of_patches["cluster"]

# Convert condition names into lists of condition numbers
# eg {"close": [0, 4], "med": [1, 5, 8, 9, 10, 11], "far": [2, 6]}
name_to_nb_list = {"all": [], "close": [], "med": [], "far": [], "superfar": [], "cluster": [],
                   "0.5+1.25": [], "0.2+0.5": [],
                   "0.2": [], "0.5": [], "1.25": [],
                   "0": []}
for condition in nb_to_name.keys():
    name_to_nb_list["all"].append(condition)  # the all should have everyone
    name_to_nb_list[nb_to_name[condition]] = [
        condition]  # "condition" to [condition_nb] conversion for single conditions
    # For every pool defined in name_to_nb_list initialization, add all conditions that have this name in their name
    # Note: first it finds distance, then will stop at first density found (which should be the right one)
    # (this is to avoid all conditions being put in 0)
    distance_found = False
    for condition_pool in name_to_nb_list.keys():
        if condition_pool in nb_to_name[condition] and (
                distance_found or nb_to_name[condition][0] == condition_pool[0]):
            name_to_nb_list[condition_pool].append(condition)
            if not distance_found:
                distance_found = True
            else:
                break  # order in name_to_nb list matters

# Takes "[12, 13, 14, 15]" and returns "0"
# Only works for lists that include all of the conditions
nb_list_to_name = {str(v): k for k, v in name_to_nb_list.items()}
nb_list_to_name = {str(sorted(v)): k for k, v in name_to_nb_list.items()}
nb_list_to_name["[0, 4]"] = "close"
nb_list_to_name["[1, 5]"] = "med"
nb_list_to_name["[2, 6]"] = "far"
nb_list_to_name["[3, 7]"] = "cluster"
nb_list_to_name["[3, 7]"] = "cluster"
nb_list_to_name["[0, 4, 12]"] = "close"
nb_list_to_name["[1, 5, 13]"] = "med"
nb_list_to_name["[2, 6, 14]"] = "far"
nb_list_to_name["[3, 7, 15]"] = "cluster"
nb_list_to_name["[9, 10]"] = "mixed"
nb_list_to_name["[0, 1, 2, 14]"] = "0.2"
nb_list_to_name["[0, 1, 2, 14, 3]"] = "0.2"
nb_list_to_name["[4, 5, 6, 15]"] = "0.5"
nb_list_to_name["[4, 5, 6, 15, 7]"] = "0.5"
nb_list_to_name["[8, 12, 13, 16]"] = "1.25"
nb_list_to_name["[12, 8, 13, 16]"] = "1.25"

# Same but to list of names (eg "close" => ["close 0", "close 0.2", "close 0.5"]
name_to_name_list = {"all": [], "close": [], "med": [], "far": [], "superfar": [], "cluster": [], "0.5+1.25": [],
                     "0.2+0.5": [], "0.5": [], "1.25": [], "0.2": [], "0": []}
for name in name_to_name_list.keys():
    condition_list = copy.deepcopy(name_to_nb_list[name])
    for i_cond in range(len(condition_list)):
        condition_list[i_cond] = nb_to_name[condition_list[i_cond]]  # convert each nb of nb_list to a name
    name_to_name_list[name] = condition_list

# Centers of patches for each condition
nb_to_xy = {}
for condition in nb_to_distance.keys():
    if nb_to_distance[condition] == "close":
        nb_to_xy[condition] = patch_coordinates.xy_patches_close
    if nb_to_distance[condition] == "med":
        nb_to_xy[condition] = patch_coordinates.xy_patches_med
    if nb_to_distance[condition] == "far":
        nb_to_xy[condition] = patch_coordinates.xy_patches_far
    if nb_to_distance[condition] == "superfar":
        nb_to_xy[condition] = patch_coordinates.xy_patches_super_far
    if nb_to_distance[condition] == "cluster":
        nb_to_xy[condition] = patch_coordinates.xy_patches_cluster

# Centers of patches for each condition
distance_to_xy = {"close": patch_coordinates.xy_patches_close, "med": patch_coordinates.xy_patches_med,
                  "far": patch_coordinates.xy_patches_far, "superfar": patch_coordinates.xy_patches_super_far,
                  "cluster": patch_coordinates.xy_patches_cluster}


## Colors

# close: purple
# med: blue
# far: green
# superfar: yellowish green
# clusters: teal
# 0.2, 0.5, 1.25: shades of brown

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    r, g, b = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))


name_to_color = {"close": "mediumslateblue", "med": "royalblue", "far": "deepskyblue", "superfar": "turquoise",
                 "cluster": "forestgreen", "0": "bisque",
                 "0.2": "burlywood", "0.5": "darkgoldenrod", "1.25": "brown", "0.2+0.5": "chocolate",
                 "0.5+1.25": "orange",
                 "control": "gray", "all": "teal",
                 "food": "brown",
                 "mixed": "goldenrod"}

# Add colors for single conditions, distance override but color is lighter when density is 0.2, darker when 1.25
for condition in nb_to_name.keys():
    distance_color = name_to_color[nb_to_distance[condition]]
    density = nb_to_density[condition]
    if "+" not in density:
        density = float(density)
    else:
        density1, density2 = density.split("+")
        # for mixed densities, density will be intermediate
        density = (float(density1) + float(density2)) / 2
    name_to_color[nb_to_name[condition]] = lighten_color(mc.cnames[distance_color],
                                                         amount=min(1.35, max(0.1, density) * 2))


#name_to_color["med 1.25"] = "black"
#name_to_color["med 0.5+1.25"] = "grey"


def test_colors():
    y = 1
    for pool_name, pool in name_to_nb_list.items():
        x = 1
        for cond in pool:
            if len(pool) > 1:
                plt.text(0, y, pool_name)
                plt.scatter(x, y, color=name_to_color[nb_to_name[cond]], s=300)
                x += 1
        y += 1
    plt.show()


if __name__ == "__main__":
    test_colors()
