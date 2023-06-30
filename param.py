# General parameters

# NOT IMPLEMENTED YET
# Tolerance for the worm exiting a food patch, so when it's radius+tolerance away we still count it as being inside
radial_tolerance = 0
# Just a parameter to toggle extensive printing in the functions (for debugging purposes)
verbose = False
condition_names = ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                   "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "control"]

# Time threshold list for visit aggregation
threshold_list = [0, 10, 100, 100000]

# Time threshold for leaving probability (to compute P_leave of a worm, look at probability that it leaves in the next
#   N time steps, with N being this threshold)
time_threshold = 20

# Conditions
nb_to_name = {0: "close 0.2", 1: "med 0.2", 2: "far 0.2", 3: "cluster 0.2", 4: "close 0.5", 5: "med 0.5", 6: "far 0.5",
              7: "cluster 0.5", 8: "med 1.25", 9: "med 0.2+0.5", 10: "med 0.5+1.25", 11: "control"}
nb_to_distance = {0: "close", 1: "med", 2: "far", 3: "cluster", 4: "close", 5: "med", 6: "far", 7: "cluster", 8: "med",
                  9: "med", 10: "med"}
nb_to_density = {0: "0.2", 1: "0.2", 2: "0.2", 3: "0.2", 4: "0.5", 5: "0.5", 6: "0.5", 7: "0.5", 8: "1.25",
                 9: "0.2+0.5", 10: "0.5+1.25", 11: "0"}

nb_to_nb_of_patches = {0: 52, 1: 24, 2: 7, 3: 25, 4: 52, 5: 24, 6: 7, 7: 25, 8: 24, 9: 24, 10: 24, 11: 24}

name_to_nb_list = {"close": [0, 4], "med": [1, 5, 8, 9, 10, 11], "far": [2, 6], "cluster": [3, 7], "0.2": [0, 1, 2, 3],
                   "0.5": [4, 5, 6, 7], "1.25": [8], "control": [11], "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   "close 0.2": [0], "med 0.2": [1], "far 0.2": [2], "cluster 0.2": [3], "close 0.5": [4],
                   "med 0.5": [5], "far 0.5": [6], "cluster 0.5": [7]}

name_to_color = {"close": "darkslategrey", "med": "teal", "far": "darkturquoise", "cluster": "deepskyblue",
                 "0.2": "burlywood", "0.5": "darkgoldenrod", "1.25": "chocolate", "control": "gray", "all": "yellowgreen",
                 "close 0.2": "darkslategrey", "med 0.2": "teal", "far 0.2": "darkturquoise", "cluster 0.2": "deepskyblue",
                 "close 0.5": "darkslategrey", "med 0.5": "teal", "far 0.5": "darkturquoise", "cluster 0.5": "deepskyblue",
                 "med 0.2+0.5": "brown", "med 0.5+1.25": "orange", "med 1.25": "saddlebrown"}
