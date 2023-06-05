# General parameters

# Tolerance for the worm exiting a food patch, so when it's radius+tolerance away we still count it as being inside
radial_tolerance = 0
# Just a parameter to toggle extensive printing in the functions (for debugging purposes)
verbose = False
condition_names = ["close 0.2", "med 0.2", "far 0.2", "cluster 0.2", "close 0.5", "med 0.5", "far 0.5",
                   "cluster 0.5", "med 1.25", "med 0.2+0.5", "med 0.5+1.25"]
