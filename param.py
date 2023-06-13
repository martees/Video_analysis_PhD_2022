# General parameters

# Tolerance for the worm exiting a food patch, so when it's radius+tolerance away we still count it as being inside
radial_tolerance = 0
# Just a parameter to toggle extensive printing in the functions (for debugging purposes)
verbose = False
condition_names = ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                   "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25", "control"]

nb_to_name = {0: "close 0.2", 1: "med 0.2", 2: "far 0.2", 3: "cluster 0.2", 4: "close 0.5", 5: "med 0.5", 6: "far 0.5", 7: "cluster 0.5", 8: "med 1.25", 9: "med 0.2+0.5", 10: "med 0.5+1.25", 11: "control"}
nb_to_distance = {0: "close", 1: "med", 2: "far", 3: "cluster", 4: "close", 5: "med", 6: "far", 7: "cluster", 8: "med", 9: "med", 10: "med"}
nb_to_density = {0: "0.2", 1: "0.2", 2: "0.2", 3: "0.2", 4: "0.5", 5: "0.5", 6: "0.5", 7: "0.5", 8: "1.25", 9: "0.2+0.5", 10: "0.5+1.25", 11: "0"}
condition_to_nb = {"close": [0, 4], "med": [1, 5, 8, 9, 10], "far": [2, 6], "cluster": [3, 7], "0.2": [0, 1, 2, 3], "0.5": [5, 6, 7, 8], "1.25": [8], "control": [11]}
nb_to_nb_of_patches = {0: 52, 1: 24, 2: 7, 3: 25, 4: 52, 5: 24, 6: 7, 7: 25, 8: 24, 9: 24, 10: 24}
