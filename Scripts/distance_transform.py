# Distance transform algorithm based on https://pure.rug.nl/ws/files/3059926/2002CompImagVisMeijster.pdf
# Computes distance transform running on each column / line separately, allowing for parallelization
import numpy as np
import multiprocessing as mp
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Generating_data_tables import generate_trajectories as gt
from Parameters import parameters as param
import find_data as fd
import analysis as ana


def minimal_y_distance_to_a_patch(in_patch_map):
    """
    This function will return a numpy array, with each column containing cells with the distance between the
    cell and the closest patch pixel IN THIS COLUMN.
    In order to do so, I first fill the array with either 0 (is in a patch) or a big value (frame size squared).
    Then, I scan each column from top (starting at index 1) to bottom, computing the distance to the closest patch
        ABOVE each point, by either leaving 0 or incrementing by 1 the value in the cell just above.
    Then, I scan the column bottom to top, and replace each value with the pixel below + 1 if it's smaller (closest
        patch BELOW each point).
    Example with patch in coordinates 0 and 3:
    [0]                         [0]                         [0]
    [0]                         [1]                         [1]
    [0] ==> After 1st round ==> [2] ==> After 2nd round ==> [1] (because 0+1 < 2, we replace 2 by 1)
    [0]                         [0]                         [0]
    [0]                         [1]                         [1]

    @param in_patch_map: numpy array with, in each cell, whether it belongs to a patch (value >= 0) or not (-1).
    """
    nb_of_lines = len(in_patch_map)
    nb_of_columns = len(in_patch_map[0])
    column_distance_array = np.zeros((nb_of_lines, nb_of_columns))

    for i_col in range(nb_of_columns):
        # First, leave 0's inside food patches, big value outside
        for i_row in range(nb_of_lines):
            if in_patch_map[i_col, i_row] == -1:  # inverted because it's an image
                column_distance_array[i_col, i_row] = nb_of_lines * nb_of_columns
        # Then, top to bottom run, if smaller, replace by distance to the closest patch pixel ABOVE the current cell
        for i_row in range(1, nb_of_lines):
            column_distance_array[i_col, i_row] = min(column_distance_array[i_col, i_row],
                                                      column_distance_array[i_col, i_row - 1] + 1)
        # Then, bottom to top, if smaller, replace by distance to the closest patch pixel BELOW the current cell
        for i_row in range(nb_of_lines - 2, -1, -1):
            column_distance_array[i_col, i_row] = min(column_distance_array[i_col, i_row],
                                                      column_distance_array[i_col, i_row + 1] + 1)

    return column_distance_array


def function_to_minimize(x, i, column_distance_matrix):
    return (x - i)**2 + column_distance_matrix[i]**2


def sep(i, u, min_y_distance_current_row):
    """
    I don't understand what this is but it's useful for the optimization algorithm xD
    Incomplete explanation:
    Returns the first integer that is larger or equal than the horizontal coordinate of the intersection
    between F_u and F_i, where:
    F_i is equal to the distance between the current point's min_y_distance and its horizontal neighbors'
    """
    return (u**2 - i**2 + min_y_distance_current_row[u]**2 - min_y_distance_current_row[i]**2) // max(1, 2*(u-i))


def distance_transform(in_patch_map):
    """
    Function that returns the distance of each pixel to the closest pixel inside a food patch.
    @param in_patch_map:
    @return:
    """
    nb_of_lines = len(in_patch_map)
    nb_of_columns = len(in_patch_map[0])
    distance_transform_array = np.zeros((nb_of_lines, nb_of_columns))
    min_y_distance_array = minimal_y_distance_to_a_patch(in_patch_map)
    for y in range(nb_of_lines):
        q = 0
        s = [0 for _ in range(nb_of_columns*nb_of_lines)]
        t = [0 for _ in range(nb_of_columns*nb_of_lines)]
        min_y_distance_this_row = min_y_distance_array[:, y]
        for u in range(nb_of_columns):
            while q >= 0 and function_to_minimize(t[q], s[q], min_y_distance_this_row) > function_to_minimize(t[q], u, min_y_distance_this_row):
                q = q - 1
            if q < 0:
                q = 0
                s[0] = u
            else:
                w = 1 + sep(s[q], u, min_y_distance_this_row)
                if w < nb_of_columns:
                    q = q + 1
                    s[q] = u
                    t[q] = int(w)
        for u in range(nb_of_lines - 1, -1, -1):
            min_y_distance_this_row = min_y_distance_array[:, u]
            distance_transform_array[u, y] = function_to_minimize(u, s[q], min_y_distance_this_row)
            if u == t[q]:
                q = q - 1
    plt.imshow(distance_transform_array)
    plt.show()


# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=False)
results = pd.read_csv(path + "clean_results.csv")
plate = results["folder"][12]
in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
in_patch_matrix = pd.read_csv(in_patch_matrix_path)

# distance_transform(in_patch_matrix.to_numpy().transpose())

in_patch_matrix = in_patch_matrix.to_numpy()
for i in range(len(in_patch_matrix)):
    for j in range(len(in_patch_matrix[i])):
        if in_patch_matrix[i, j] == -1:
            in_patch_matrix[i, j] = 1
        else:
            in_patch_matrix[i, j] = 0

from scipy import ndimage
img = ndimage.distance_transform_edt(in_patch_matrix)
plt.imshow(img)
plt.show()

