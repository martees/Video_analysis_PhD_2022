import os

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
import pandas as pd
import random
import time
from scipy import ndimage
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import copy

from sympy.physics.units import speed

from Generating_data_tables import main as gen
import ReferencePoints
import analysis as ana
import find_data as fd
from Parameters import parameters as param
from Parameters import colored_line_plot as colored_lines_script
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script


def generate_environment_matrices(results_path, xp_plate_list):
    """
    Function that will generate idealized matrices of how the patches should be in our experiments, for each distance.
    Will save them in a folder and reload them when they already exist.
    Will take the patch positions from the Parameters folder. Will take the average patch radius from our experimental
    plates. Will generate a circular mask around the plates using the average reference points distance from our
    experimental plates.
    :param results_path: path where all of our experimental plates are.
    :param xp_plate_list: a list of paths to traj.csv data in experimental subfolders.
    :return: a list of pandas dataframes. Each dataframe contains the patch to which each pixel belongs for 1 distance
    """
    # if not os.path.isdir(results_path + "perfect_patch_maps"):
    #     os.mkdir(results_path + "perfect_patch_maps")

    print("Generating environmental matrices...")

    plate_size = 1847
    #patch_radius = heatmap_script.generate_average_patch_radius_each_condition(results_path, xp_plate_list)
    #print(patch_radius)
    patch_radius = 42.114
    #ref_point_distance = heatmap_script.compute_average_ref_points_distance(results_path, xp_plate_list)
    #print(ref_point_distance)
    ref_point_distance = 987.21

    environment_matrices = []
    all_distances = ["close", "med", "far", "superfar"]
    for distance in all_distances:
        print(">>> Distance "+distance+"...")
        # Initialize the patch map
        # -1 is the default value, will be left in the pixels which are outside any food patch
        patch_map_this_distance = -1 * np.ones((plate_size, plate_size))

        # Load patch centers
        # First find the ideal reference points positions
        margin = (plate_size - ref_point_distance) / 2
        bottom_left = [margin, margin]
        bottom_right = [plate_size - margin, margin]
        top_left = [margin, plate_size - margin]
        top_right = [plate_size - margin, plate_size - margin]
        # Then use these perfect reference points to convert the patch centers using the ReferencePoints class
        robot_xy = np.array(param.distance_to_xy[distance])
        small_ref_points = ReferencePoints.ReferencePoints([[-16, 16], [16, 16], [16, -16], [-16, -16]])
        big_ref_points = ReferencePoints.ReferencePoints([bottom_left, bottom_right, top_left, top_right])
        robot_xy[:, 0] = - robot_xy[:, 0]
        patch_centers = big_ref_points.mm_to_pixel(small_ref_points.pixel_to_mm(robot_xy))

        # For each patch, create a map with, for each pixel, the distance to the center of the closest patch
        # And in patch_map_this_distance, set the pixels that are closer than patch_radius to the current patch number
        for i_patch in range(len(patch_centers)):
            current_patch_x, current_patch_y = patch_centers[i_patch]
            distance_map = np.ones((plate_size, plate_size))
            distance_map[int(np.rint(current_patch_x)), int(np.rint(current_patch_y))] = 0
            distance_map = ndimage.distance_transform_edt(distance_map)
            indices_inside_patch = np.where(distance_map <= patch_radius)
            patch_map_this_distance[indices_inside_patch] = i_patch

        # Add it to the list
        environment_matrices.append(patch_map_this_distance)

    # Add the mask to each distance: -2 outside the plate (in experiments, the worm is not tracked outside of this area
    # Also do a distance transform, but this time add -1 to the pixels that are further away than the plate radius
    # which is defined as half of the diagonal of the reference square (so half of sqrt(2) * the side of the square)
    plate_radius = (np.sqrt(2)/2)*ref_point_distance
    distance_map = np.ones((plate_size, plate_size))
    distance_map[plate_size // 2, plate_size // 2] = 0
    distance_map = ndimage.distance_transform_edt(distance_map)
    indices_outside_plate = np.where(distance_map >= plate_radius)
    for i_distance in range(len(all_distances)):
        environment_matrices[i_distance][indices_outside_plate] = -2

    return environment_matrices


def return_patch(environment, x, y):
    """
    Returns the value of environment in line x, column y, taking the boundary if x and y exceed it.
    :param environment: 2-dimensional pandas dataframe
    :param x: a number
    :param y: a number
    :return: a number
    """
    return environment[int(np.clip(y, 0, len(environment[0]) - 1))][int(np.clip(x, 0, len(environment) - 1))]


def random_walk(sim_length, speed_inside, speed_outside, environment_matrix):
    """
    Function that takes two model parameters (speed inside food patches and
    speed outside food patches), an environment (with -2 outside the
    environment, -1 outside the food patches, and integers >=0 for the
    food patches).
    Will create a perfectly random walk, with two step-lengths, one inside patches and one outside.
    Will run a simulation for sim_length time steps, and return:
        - The total time spent inside food patches
        - The total time spent outside food patches
          (should be sim_length - total time inside)
        - The number of visited patches
        - The number of visits
    """
    time_list = list(range(0, sim_length))
    x_list = [0 for _ in range(sim_length)]
    y_list = [0 for _ in range(sim_length)]
    speed_list = [0 for _ in range(sim_length)]
    total_time_inside = 0
    total_time_outside = 0
    list_of_visited_patches = []  # Will add patches as they get visited
    # Starting point
    plate_size = len(environment_matrix)
    x_list[0] = random.uniform(plate_size/4, 3*plate_size/4)
    y_list[0] = random.uniform(plate_size/4, 3*plate_size/4)
    current_patch = return_patch(environment_matrix, x_list[0], y_list[0])
    if current_patch == -1:
        speed_list[0] = speed_outside
    else:
        speed_list[1] = speed_inside
    # Simulation loop
    for i_time in range(1, sim_length):
        previous_patch = current_patch
        current_patch = return_patch(environment_matrix, x_list[i_time - 1], y_list[i_time - 1])
        current_heading = np.random.rand() * 2 * np.pi  # choose a random angle
        # If worm is inside
        if current_patch >= 0:
            x_list[i_time] = x_list[i_time - 1] + speed_inside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_inside * np.sin(current_heading)
            total_time_inside += 1
            speed_list[i_time] = speed_inside
            if current_patch != previous_patch:
                list_of_visited_patches.append(current_patch)
        # If worm is outside
        elif current_patch == -1:
            x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
            speed_list[i_time] = speed_outside
            # This is a trick for the plotting to work better
            speed_list[i_time - 1] = speed_outside
            total_time_outside += 1
        # If worm is escaping the plate (current_patch == -2)
        else:
            # While it's escaped, draw a new direction by progressively rotating
            # the current_heading
            i_while = 0
            # Store those to avoid having to access the table a bazillion time
            previous_x = x_list[i_time - 1]
            previous_y = y_list[i_time - 1]
            while current_patch == -2 and i_while < 7:
                current_heading += 1
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 70:
                current_heading += 0.1
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 700:
                current_heading += 0.01
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            total_time_outside += 1
            speed_list[i_time] = speed_outside

            x_list[i_time] = new_x
            y_list[i_time] = new_y

    # plt.imshow(environment_matrix, cmap="plasma")
    # plt.plot(x_list, y_list, color="orange")
    # plt.show()

    return time_list, x_list, y_list, speed_list, total_time_inside, total_time_outside, len(np.unique(list_of_visited_patches)), len(list_of_visited_patches)


def correlated_walk(sim_length, speed_inside, speed_outside, max_turning_angle, environment_matrix, sharp_turn_probability, turn_probability_factor):
    """
    Function that takes three model parameters (speed inside food patches, speed outside food patches and
    a parameter for the level of correlation between time steps), an environment (with -2 outside the
    environment, -1 outside the food patches, and integers >=0 for the
    food patches).
    Will create a correlated random walk, with two step-lengths, one inside patches and one outside.
    The max_turning_angle argument determines how much the animal can turn between two consecutive steps. It is
    in radians, so if it's 2pi, the walk is random, if it's pi, the animal can turn by at most 90° between two
    time steps.
    Will run a simulation for sim_length time steps, and return:
        - The total time spent inside food patches
        - The total time spent outside food patches
          (should be sim_length - total time inside)
        - The number of visited patches
        - The number of visits
    """
    time_list = list(range(0, sim_length))
    x_list = [0 for _ in range(sim_length)]
    y_list = [0 for _ in range(sim_length)]
    speed_list = [0 for _ in range(sim_length)]
    total_time_inside = 0
    total_time_outside = 0
    list_of_visited_patches = []  # Will add patches as they get visited
    # Remember when was the last time the agent was in the patch, to compute the sharp turning angle probability
    time_since_patch_exit = 0
    # Starting point
    plate_size = len(environment_matrix)
    x_list[0] = random.uniform(plate_size / 4, 3 * plate_size / 4)
    y_list[0] = random.uniform(plate_size / 4, 3 * plate_size / 4)
    current_patch = return_patch(environment_matrix, x_list[0], y_list[0])
    if current_patch == -1:
        speed_list[0] = speed_outside
    else:
        speed_list[1] = speed_inside
    current_heading = np.random.rand() * 2 * np.pi  # choose a random first angle
    # Simulation loop
    for i_time in range(1, sim_length):
        previous_patch = current_patch
        current_patch = return_patch(environment_matrix, x_list[i_time - 1], y_list[i_time - 1])
        # Heading is modified by a random value between the extrema defined by max_turning_angle
        # There is a probability of doing a sharp turn (turning of more than max_turning_angle)
        # This probability is multiplied by 10 if you have exited a patch not long ago
        p = random.uniform(0, 1)
        #if time_since_patch_exit < 100:
        #if time_since_patch_exit < 100 and current_patch == -1 :
        if abs(time_since_patch_exit) < 100:
            if p < turn_probability_factor*sharp_turn_probability:
                current_heading += np.pi
            else:
                current_heading += random.uniform(-max_turning_angle, max_turning_angle)
        else:
            if p < sharp_turn_probability:
                current_heading += np.pi
            else:
                current_heading += random.uniform(-max_turning_angle, max_turning_angle)

        # If worm is inside
        if current_patch >= 0:
            if current_patch != previous_patch:
                # Set the time since patch exit to 0, as a new patch has now been found
                time_since_patch_exit = 0
                list_of_visited_patches.append(current_patch)
            x_list[i_time] = x_list[i_time - 1] + speed_inside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_inside * np.sin(current_heading)
            total_time_inside += 1
            time_since_patch_exit -= 1  # this keeps decrementing as long as the worm is inside a food patch
            speed_list[i_time] = speed_inside
        # If worm is outside
        elif current_patch == -1:
            if current_patch != previous_patch:
                # Set the time since patch exit to 0, as the patch has now been exited
                time_since_patch_exit = 0
            x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
            speed_list[i_time] = speed_outside
            total_time_outside += 1
            time_since_patch_exit += 1  # this keeps incrementing as long as no patch has been encountered
        # If worm is escaping the plate (current_patch == -2)
        else:
            # While it's escaped, draw a new direction by progressively rotating
            # the current_heading
            i_while = 0
            # Store those to avoid having to access the table a bazillion time
            previous_x = x_list[i_time - 1]
            previous_y = y_list[i_time - 1]
            while current_patch == -2 and i_while < 7:
                current_heading += 1
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 70:
                current_heading += 0.1
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 700:
                current_heading += 0.01
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            total_time_outside += 1
            time_since_patch_exit += 1  # this keeps incrementing as long as no patch has been encountered
            speed_list[i_time] = speed_outside
            x_list[i_time] = new_x
            y_list[i_time] = new_y

    # plt.imshow(environment_matrix, cmap="plasma")
    # plt.plot(x_list, y_list, color="orange")
    # plt.show()

    return time_list, x_list, y_list, speed_list, total_time_inside, total_time_outside, len(np.unique(list_of_visited_patches)), len(list_of_visited_patches)


def auto_correlated_walk(sim_length, speed_inside, speed_outside, max_turning_angle, environment_matrix):
    """
    Function that takes three model parameters (speed inside food patches, speed outside food patches and
    a parameter for the level of correlation between time steps), an environment (with -2 outside the
    environment, -1 outside the food patches, and integers >=0 for the
    food patches).
    Will create a correlated random walk, with two step-lengths, one inside patches and one outside.
    The max_turning_angle argument determines how much the animal can turn between two consecutive steps. It is
    in radians, so if it's 2pi, the walk is random, if it's pi, the animal can turn by at most 90° between two
    time steps.
    Will run a simulation for sim_length time steps, and return:
        - The total time spent inside food patches
        - The total time spent outside food patches
          (should be sim_length - total time inside)
        - The number of visited patches
        - The number of visits
    """
    time_list = list(range(0, sim_length))
    x_list = [0 for _ in range(sim_length)]
    y_list = [0 for _ in range(sim_length)]
    speed_list = [0 for _ in range(sim_length)]
    total_time_inside = 0
    total_time_outside = 0
    list_of_visited_patches = []  # Will add patches as they get visited
    # Remember when was the last time the agent was in the patch, to compute the sharp turning angle probability
    time_since_patch_exit = 0
    # Starting point
    plate_size = len(environment_matrix)
    x_list[0] = random.uniform(plate_size / 4, 3 * plate_size / 4)
    y_list[0] = random.uniform(plate_size / 4, 3 * plate_size / 4)
    current_patch = return_patch(environment_matrix, x_list[0], y_list[0])
    if current_patch == -1:
        speed_list[0] = speed_outside
    else:
        speed_list[1] = speed_inside
    current_heading = np.random.rand() * 2 * np.pi  # choose a random first angle
    current_heading_difference = 0
    # Simulation loop
    for i_time in range(1, sim_length):
        previous_patch = current_patch
        current_patch = return_patch(environment_matrix, x_list[i_time - 1], y_list[i_time - 1])
        # Heading DIFFERENCE is modified by a random value between the extrema defined by max_turning_angle
        # There is a probability of doing a sharp turn (turning of more than max_turning_angle)
        # This probability is multiplied by 10 if you have exited a patch not long ago
        current_heading_difference += random.uniform(-max_turning_angle/2, max_turning_angle/2)
        current_heading += current_heading_difference

        # If worm is inside
        if current_patch >= 0:
            x_list[i_time] = x_list[i_time - 1] + speed_inside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_inside * np.sin(current_heading)
            total_time_inside += 1
            speed_list[i_time] = speed_inside
            if current_patch != previous_patch:
                # Set the time since patch exit to 0, as a new patch has now been found
                time_since_patch_exit = 0
                list_of_visited_patches.append(current_patch)
        # If worm is outside
        elif current_patch == -1:
            x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
            speed_list[i_time] = speed_outside
            total_time_outside += 1
            time_since_patch_exit += 1  # this keeps incrementing as long as no patch has been encountered
        # If worm is escaping the plate (current_patch == -2)
        else:
            # While it's escaped, draw a new direction by progressively rotating
            # the current_heading
            i_while = 0
            # Store those to avoid having to access the table a bazillion time
            previous_x = x_list[i_time - 1]
            previous_y = y_list[i_time - 1]
            while current_patch == -2 and i_while < 7:
                current_heading += 1
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 70:
                current_heading += 0.1
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 700:
                current_heading += 0.01
                new_x = previous_x + speed_outside * np.cos(current_heading)
                new_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, new_x, new_y)
                i_while += 1
            current_heading_difference = 0  # reset it to help the walker escape from the border
            total_time_outside += 1
            time_since_patch_exit += 1  # this keeps incrementing as long as no patch has been encountered
            speed_list[i_time] = speed_outside
            x_list[i_time] = new_x
            y_list[i_time] = new_y

    # plt.imshow(environment_matrix, cmap="plasma")
    # plt.plot(x_list, y_list, color="orange")
    # plt.show()

    return time_list, x_list, y_list, speed_list, total_time_inside, total_time_outside, len(np.unique(list_of_visited_patches)), len(list_of_visited_patches)


# def dynamic_speed(speed_inside, speed_outside, half_speed_life, time_since_patch_exit):
#    """
#    Will return the current speed of an agent that has left the patch for "time_since_patch_exit" time steps.
#    The function increases in 1/t, and should go from speed_inside (in t=0) to speed_outside (in t=infinite).
#    It reaches speed_inside + speed_outside/2 at half_speed_life
#    """
#    return speed_outside - (speed_outside-speed_inside)/(1 + time_since_patch_exit * ((1 - 2 * (speed_inside/speed_outside))/half_speed_life))


def dynamic_speed(speed_inside, speed_outside, half_speed_life, time_since_patch_exit):
    """
    Will return the current speed of an agent that has left the patch for "time_since_patch_exit" time steps.
    The function increases in 1/exp(t), and should go from speed_inside (in t=0) to speed_outside (in t=infinite).
    It reaches speed_outside/2 at half_speed_life
    """
    return speed_outside - (speed_outside-speed_inside)*np.exp((1/half_speed_life) * np.log(speed_outside / (2*speed_outside - 2*speed_inside)) * time_since_patch_exit)


def dynamic_speed_table(speed_inside, speed_outside, half_speed_life, t_max):
    """
    Returns a list with the values of dynamic_speed() as a function of time, from 1 to t_max.
    """
    speed_values = [0 for _ in range(t_max)]
    for i in range(len(speed_values)):
        speed_values[i] = dynamic_speed(speed_inside, speed_outside, half_speed_life, i)
    return speed_values


def dynamic_speed_walk(sim_length, speed_inside, speed_outside, speed_table, environment_matrix):
    """
        Function that takes three model parameters (speed inside/outside food patches and the dynamic of speed as time
        since exit increases), an environment (with -2 outside the environment, -1 outside the food patches,
        and integers >=0 for the food patches).
        Will run a random walk with a step-length inside food patches (speed_inside), and a speed that progressively
        increases from speed_inside to speed_outside once the agent leaves a patch. It takes half_speed_life for the
        speed to recover half of speed outside.
        Will run a simulation for sim_length time steps, and return:
            - The total time spent inside food patches
            - The total time spent outside food patches
              (should be sim_length - total time inside)
            - The number of visited patches
            - The number of visits
        """
    time_list = list(range(0, sim_length))
    x_list = [0 for _ in range(sim_length)]
    y_list = [0 for _ in range(sim_length)]
    speed_list = [0 for _ in range(sim_length)]
    total_time_inside = 0
    total_time_outside = 0
    list_of_visited_patches = []  # Will add patches as they get visited
    # Starting point
    x_list[0] = len(environment_matrix) / 2
    y_list[0] = len(environment_matrix[0]) / 2
    speed_list[0] = speed_outside
    current_patch = return_patch(environment_matrix, x_list[0], y_list[0])
    # Remember when was the last time the agent was in the patch, to compute the speed
    time_since_patch_exit = 0
    # Also, in the beginning of the simulation, the worm should just go at speed_outside (no past patch)
    beginning_of_sim = True
    # Simulation loop
    for i_time in range(1, sim_length):
        previous_patch = current_patch
        current_patch = return_patch(environment_matrix, x_list[i_time - 1], y_list[i_time - 1])
        current_heading = np.random.rand() * 2 * np.pi  # choose a random angle
        # If worm is inside
        if current_patch >= 0:
            x_list[i_time] = x_list[i_time - 1] + speed_inside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_inside * np.sin(current_heading)
            speed_list[i_time] = speed_inside
            total_time_inside += 1
            if current_patch != previous_patch:
                list_of_visited_patches.append(current_patch)
                # Set the time since patch exit to 0, as a new patch has now been found
                time_since_patch_exit = 0
                if beginning_of_sim:  # if first patch encounter, then set beginning to False
                    beginning_of_sim = False
        # If worm is outside
        elif current_patch == -1:
            if beginning_of_sim:  # In the beginning of the sim just go at speed_outside
                current_speed = speed_outside
            else:
                current_speed = speed_table[time_since_patch_exit]
            x_list[i_time] = x_list[i_time - 1] + current_speed * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + current_speed * np.sin(current_heading)
            speed_list[i_time] = current_speed
            time_since_patch_exit += 1  # this keeps incrementing as long as no patch has been encountered
            total_time_outside += 1
        # If worm is escaping the plate (current_patch == -2)
        else:
            # While it's escaped, draw a new direction by progressively rotating
            # the current_heading
            i_while = 0
            # Store those to avoid having to access the table a bazillion time
            previous_x = x_list[i_time - 1]
            previous_y = y_list[i_time - 1]
            while current_patch == -2 and i_while < 7:
                current_heading += 1
                if beginning_of_sim:  # In the beginning of the sim just go at speed_outside
                    current_speed = speed_outside
                else:
                    current_speed = speed_table[time_since_patch_exit]
                x_list[i_time] = previous_x + current_speed * np.cos(current_heading)
                y_list[i_time] = previous_y + current_speed * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, x_list[i_time], y_list[i_time])
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 70:
                current_heading += 0.1
                if beginning_of_sim:  # In the beginning of the sim just go at speed_outside
                    current_speed = speed_outside
                else:
                    current_speed = speed_table[time_since_patch_exit]
                x_list[i_time] = previous_x + current_speed * np.cos(current_heading)
                y_list[i_time] = previous_y + current_speed * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, x_list[i_time], y_list[i_time])
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 700:
                current_heading += 0.01
                if beginning_of_sim:  # In the beginning of the sim just go at speed_outside
                    current_speed = speed_outside
                else:
                    current_speed = speed_table[time_since_patch_exit]
                x_list[i_time] = previous_x + current_speed * np.cos(current_heading)
                y_list[i_time] = previous_y + current_speed * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, x_list[i_time], y_list[i_time])
                i_while += 1
            time_since_patch_exit += 1  # this keeps incrementing as long as no patch has been encountered
            speed_list[i_time] = current_speed
            total_time_outside += 1

    # plt.imshow(environment_matrix, cmap="plasma")
    # plt.plot(x_list, y_list, color="orange")
    # plt.show()

    return time_list, x_list, y_list, speed_list, total_time_inside, total_time_outside, len(np.unique(list_of_visited_patches)), len(list_of_visited_patches)


def values_one_type_of_walk(map_each_distance, nb_of_walkers, type_of_walk, sim_length, parameters):
    times_outside_each_distance = [[] for _ in range(len(map_each_distance))]
    times_inside_each_distance = [[] for _ in range(len(map_each_distance))]
    nb_of_visits_each_distance = [[] for _ in range(len(map_each_distance))]
    avg_visit_each_distance = [[] for _ in range(len(map_each_distance))]
    if type_of_walk == "random":
        speed_inside, speed_outside = parameters
        for i_map, current_map in enumerate(map_each_distance):
            print("Generating random walks for distance", list(param.distance_to_xy.keys())[i_map], "...")
            for i_walk in range(nb_of_walkers):
                if i_walk % (nb_of_walkers // 4) == 0:
                    print(">>> Walker ", i_walk, " / ", nb_of_walkers, "...")
                # Run simulation
                _, _, _, _, time_in, time_out, nb_visited, nb_visits = random_walk(sim_length, speed_inside, speed_outside, current_map)
                # Compute the relevant variables
                if nb_visited > 0:
                    times_outside_each_distance[i_map].append(time_out / nb_visited)
                    times_inside_each_distance[i_map].append(time_in / nb_visited)
                    avg_visit_each_distance[i_map].append(time_in / nb_visits)
                    nb_of_visits_each_distance[i_map].append(nb_visits / nb_visited)

    if type_of_walk == "dynamic_speed":
        speed_inside, speed_outside, half_speed_life = parameters
        speed_table = dynamic_speed_table(speed_inside, speed_outside, half_speed_life, sim_length)
        for i_map, current_map in enumerate(map_each_distance):
            print("Generating dynamic speed walks for distance", list(param.distance_to_xy.keys())[i_map], "...")
            for i_walk in range(nb_of_walkers):
                if i_walk % (nb_of_walkers // 4) == 0:
                    print(">>> Walker ", i_walk, " / ", nb_of_walkers, "...")
                # Run simulation
                _, _, _, _, time_in, time_out, nb_visited, nb_visits = dynamic_speed_walk(sim_length, speed_inside, speed_outside, speed_table, current_map)
                # Compute the relevant variables
                if nb_visited > 0:
                    times_outside_each_distance[i_map].append(time_out / nb_visited)
                    times_inside_each_distance[i_map].append(time_in / nb_visited)
                    avg_visit_each_distance[i_map].append(time_in / nb_visits)
                    nb_of_visits_each_distance[i_map].append(nb_visits / nb_visited)

    if type_of_walk == "correlated":
        speed_inside, speed_outside, max_turning_angle = parameters
        for i_map, current_map in enumerate(map_each_distance):
            print("Generating correlated walks for distance", list(param.distance_to_xy.keys())[i_map], "...")
            for i_walk in range(nb_of_walkers):
                if i_walk % (nb_of_walkers // 4) == 0:
                    print(">>> Walker ", i_walk, " / ", nb_of_walkers, "...")
                # Run simulation
                _, _, _, _, time_in, time_out, nb_visited, nb_visits = correlated_walk(sim_length, speed_inside, speed_outside, max_turning_angle, current_map, sharp_turn_probability=0, turn_probability_factor=0)
                # Compute the relevant variables
                if nb_visited > 0:
                    times_outside_each_distance[i_map].append(time_out / nb_visited)
                    times_inside_each_distance[i_map].append(time_in / nb_visited)
                    avg_visit_each_distance[i_map].append(time_in / nb_visits)
                    nb_of_visits_each_distance[i_map].append(nb_visits / nb_visited)

    if type_of_walk == "auto_correlated":
        speed_inside, speed_outside, max_turning_angle = parameters
        for i_map, current_map in enumerate(map_each_distance):
            print("Generating auto-correlated walks for distance", list(param.distance_to_xy.keys())[i_map], "...")
            for i_walk in range(nb_of_walkers):
                if i_walk % (nb_of_walkers // 4) == 0:
                    print(">>> Walker ", i_walk, " / ", nb_of_walkers, "...")
                # Run simulation
                _, _, _, _, time_in, time_out, nb_visited, nb_visits = auto_correlated_walk(sim_length, speed_inside, speed_outside, max_turning_angle, current_map)
                # Compute the relevant variables
                if nb_visited > 0:
                    times_outside_each_distance[i_map].append(time_out / nb_visited)
                    times_inside_each_distance[i_map].append(time_in / nb_visited)
                    avg_visit_each_distance[i_map].append(time_in / nb_visits)
                    nb_of_visits_each_distance[i_map].append(nb_visits / nb_visited)

    if type_of_walk == "correlated_sharp_turns":
        speed_inside, speed_outside, max_turning_angle, sharp_turn_probability, turn_probability_factor = parameters
        for i_map, current_map in enumerate(map_each_distance):
            print("Generating correlated walks with sharp turns for distance", list(param.distance_to_xy.keys())[i_map], "...")
            for i_walk in range(nb_of_walkers):
                if i_walk % (nb_of_walkers // 4) == 0:
                    print(">>> Walker ", i_walk, " / ", nb_of_walkers, "...")
                # Run simulation
                _, _, _, _, time_in, time_out, nb_visited, nb_visits = correlated_walk(sim_length, speed_inside, speed_outside, max_turning_angle, current_map, sharp_turn_probability=sharp_turn_probability, turn_probability_factor=turn_probability_factor)
                # Compute the relevant variables
                if nb_visited > 0:
                    times_outside_each_distance[i_map].append(time_out / nb_visited)
                    times_inside_each_distance[i_map].append(time_in / nb_visited)
                    avg_visit_each_distance[i_map].append(time_in / nb_visits)
                    nb_of_visits_each_distance[i_map].append(nb_visits / nb_visited)

    # Then bootstraaaaaap
    time_outside_per_patch_avg = [0 for _ in range(len(map_each_distance))]
    nb_of_visits_per_patch_avg = [0 for _ in range(len(map_each_distance))]
    time_inside_per_patch_avg = [0 for _ in range(len(map_each_distance))]
    avg_visit_avg = [0 for _ in range(len(map_each_distance))]
    time_outside_per_patch_errors_inf = [0 for _ in range(len(map_each_distance))]
    time_outside_per_patch_errors_sup = [0 for _ in range(len(map_each_distance))]
    nb_of_visits_per_patch_errors_inf = [0 for _ in range(len(map_each_distance))]
    nb_of_visits_per_patch_errors_sup = [0 for _ in range(len(map_each_distance))]
    time_inside_per_patch_errors_inf = [0 for _ in range(len(map_each_distance))]
    time_inside_per_patch_errors_sup = [0 for _ in range(len(map_each_distance))]
    avg_visit_errors_inf = [0 for _ in range(len(map_each_distance))]
    avg_visit_errors_sup = [0 for _ in range(len(map_each_distance))]
    # Average for the current condition
    for i in range(len(map_each_distance)):
        if len(times_outside_each_distance[i]) > 0 and len(nb_of_visits_each_distance[i]) > 0:
            # Time out per patch
            time_outside_per_patch_avg[i] = np.nanmean(times_outside_each_distance[i])
            bootstrap_ci = ana.bottestrop_ci(times_outside_each_distance[i], 1000)
            time_outside_per_patch_errors_inf[i] = time_outside_per_patch_avg[i] - bootstrap_ci[0]
            time_outside_per_patch_errors_sup[i] = bootstrap_ci[1] - time_outside_per_patch_avg[i]
            # Nb of visits
            nb_of_visits_per_patch_avg[i] = np.nanmean(nb_of_visits_each_distance[i])
            bootstrap_ci = ana.bottestrop_ci(nb_of_visits_each_distance[i], 1000)
            nb_of_visits_per_patch_errors_inf[i] = nb_of_visits_per_patch_avg[i] - bootstrap_ci[0]
            nb_of_visits_per_patch_errors_sup[i] = bootstrap_ci[1] - nb_of_visits_per_patch_avg[i]
            # Time inside per patch
            time_inside_per_patch_avg[i] = np.nanmean(times_inside_each_distance[i])
            bootstrap_ci = ana.bottestrop_ci(times_inside_each_distance[i], 1000)
            time_inside_per_patch_errors_inf[i] = time_inside_per_patch_avg[i] - bootstrap_ci[0]
            time_inside_per_patch_errors_sup[i] = bootstrap_ci[1] - time_inside_per_patch_avg[i]
            # Average visit
            avg_visit_avg[i] = np.nanmean(avg_visit_each_distance[i])
            bootstrap_ci = ana.bottestrop_ci(avg_visit_each_distance[i], 1000)
            avg_visit_errors_inf[i] = avg_visit_avg[i] - bootstrap_ci[0]
            avg_visit_errors_sup[i] = bootstrap_ci[1] - avg_visit_avg[i]

    return (time_outside_per_patch_avg, nb_of_visits_per_patch_avg, time_inside_per_patch_avg, avg_visit_avg,
            [time_outside_per_patch_errors_inf, time_outside_per_patch_errors_sup],
            [nb_of_visits_per_patch_errors_inf, nb_of_visits_per_patch_errors_sup],
            [time_inside_per_patch_errors_inf, time_inside_per_patch_errors_sup],
            [avg_visit_errors_inf, avg_visit_errors_sup])


def experimental_values(distance_list, xp_table, list_xp_folders):
    time_outside_per_patch_avg = [0 for _ in range(len(distance_list))]
    time_outside_per_patch_errors_inf = [0 for _ in range(len(distance_list))]
    time_outside_per_patch_errors_sup = [0 for _ in range(len(distance_list))]
    nb_of_visits_per_patch_avg = [0 for _ in range(len(distance_list))]
    nb_of_visits_per_patch_errors_inf = [0 for _ in range(len(distance_list))]
    nb_of_visits_per_patch_errors_sup = [0 for _ in range(len(distance_list))]
    time_inside_per_patch_avg = [0 for _ in range(len(distance_list))]
    time_inside_per_patch_errors_inf = [0 for _ in range(len(distance_list))]
    time_inside_per_patch_errors_sup = [0 for _ in range(len(distance_list))]
    avg_visit_avg = [0 for _ in range(len(distance_list))]
    avg_visit_errors_inf = [0 for _ in range(len(distance_list))]
    avg_visit_errors_sup = [0 for _ in range(len(distance_list))]
    for i_distance, distance in enumerate(distance_list):
        time_out = []
        time_in = []
        avg_visit = []
        nb_of_visits = []
        # folder_list = fd.return_folders_condition_list(list_xp_folders, [param.name_to_nb[distance + " 0.2"], param.name_to_nb[distance + " 0.5"], param.name_to_nb[distance + " 1.25"]])
        folder_list = fd.return_folders_condition_list(list_xp_folders, [param.name_to_nb[distance + " 0.5"]])
        for i_folder, folder in enumerate(folder_list):
            current_results = xp_table[xp_table["folder"] == folder]

            # FOR AVG VISIT TIME, ONLY UNCENSORED VISITS, STARTING BEFORE 16 000 seconds
            current_visits = fd.load_list(current_results, "uncensored_visits")
            for visit in current_visits:
                if param.times_to_cut_videos[0] > visit[0] > param.times_to_cut_videos[1]:
                    current_visits.remove(visit)
            if len(current_visits) > 0:
                avg_visit.append(np.mean(ana.convert_to_durations(current_visits)))

            # FOR NB OF VISITS, TIME INSIDE AND TIME OUTSIDE PER PATCH, TAKE THE FIRST 16 000 seconds OF ALL EVENTS
            current_visits = fd.load_list(current_results, "no_hole_visits")
            current_transits = fd.load_list(current_results, "aggregated_raw_transits")
            # Mix visits and transits and sort them by beginning
            current_events = copy.deepcopy(current_visits) + copy.deepcopy(current_transits)
            current_events = sorted(current_events, key=lambda x: x[0])
            # Loop through them and stop when it's reached time to cut!!!
            cumulated_duration_of_events = 0
            i_event = 0
            new_list_of_events = []
            first_event_found = False
            while i_event < len(current_events) and cumulated_duration_of_events < (
                    param.times_to_cut_videos[1] - param.times_to_cut_videos[0]):
                current_event = current_events[i_event]
                if current_event[0] >= param.times_to_cut_videos[0]:
                    if not first_event_found:
                        first_event_found = True
                        # If this is the first event that starts after time_to_cut[0] and not the first of the video, add the previous one
                        if i_event > 0:
                            previous_event = current_events[i_event - 1]
                            previous_event[0] = param.times_to_cut_videos[0]  # but set it to start at time_to_cut[0]
                            cumulated_duration_of_events += previous_event[1] - previous_event[0]
                            new_list_of_events.append(previous_event)
                            # If this previous event does not exceed time_to_cut[1], then you can add the current one
                            if cumulated_duration_of_events < (
                                    param.times_to_cut_videos[1] - param.times_to_cut_videos[0]):
                                cumulated_duration_of_events += current_event[1] - current_event[0]
                                new_list_of_events.append(current_event)
                        # If this is the first event, just add it! and start it at time_to_cut[0]
                        else:
                            current_event[0] = param.times_to_cut_videos[0]  # but set it to start at time_to_cut[0]
                            cumulated_duration_of_events += current_event[1] - current_event[0]
                            new_list_of_events.append(current_event)

                    else:
                        cumulated_duration_of_events += current_event[1] - current_event[0]
                        new_list_of_events.append(current_event)
                i_event += 1
            # In the end of the loop, if we have reached the cut parameter, cut the last event
            if cumulated_duration_of_events > (param.times_to_cut_videos[1] - param.times_to_cut_videos[0]):
                new_list_of_events[-1][1] -= cumulated_duration_of_events - (
                            param.times_to_cut_videos[1] - param.times_to_cut_videos[0])
            # Then, sort the events back to visits!
            current_visits = [event for event in new_list_of_events if event[2] != -1]
            if len(current_visits) > 0:
                nb_visited_patches = len(np.unique(np.array(current_visits)[:, 2]))
                time_out.append(np.sum(ana.convert_to_durations(current_transits))/nb_visited_patches)
                time_in.append(np.sum(ana.convert_to_durations(current_visits))/nb_visited_patches)
                nb_of_visits.append(len(current_visits) / nb_visited_patches)

        # Time outside per patch
        time_outside_per_patch_avg[i_distance] = np.nanmean(time_out)
        bootstrap_ci = ana.bottestrop_ci(time_out, 1000)
        time_outside_per_patch_errors_inf[i_distance] = time_outside_per_patch_avg[i_distance] - bootstrap_ci[0]
        time_outside_per_patch_errors_sup[i_distance] = bootstrap_ci[1] - time_outside_per_patch_avg[i_distance]
        # Nb of visits per patch
        nb_of_visits_per_patch_avg[i_distance] = np.nanmean(nb_of_visits)
        bootstrap_ci = ana.bottestrop_ci(nb_of_visits, 1000)
        nb_of_visits_per_patch_errors_inf[i_distance] = nb_of_visits_per_patch_avg[i_distance] - bootstrap_ci[0]
        nb_of_visits_per_patch_errors_sup[i_distance] = bootstrap_ci[1] - nb_of_visits_per_patch_avg[i_distance]
        # Time inside per patch
        time_inside_per_patch_avg[i_distance] = np.nanmean(time_in)
        bootstrap_ci = ana.bottestrop_ci(time_in, 1000)
        time_inside_per_patch_errors_inf[i_distance] = time_inside_per_patch_avg[i_distance] - bootstrap_ci[0]
        time_inside_per_patch_errors_sup[i_distance] = bootstrap_ci[1] - time_inside_per_patch_avg[i_distance]
        # Average visit
        avg_visit_avg[i_distance] = np.nanmean(avg_visit)
        bootstrap_ci = ana.bottestrop_ci(avg_visit, 1000)
        avg_visit_errors_inf[i_distance] = avg_visit_avg[i_distance] - bootstrap_ci[0]
        avg_visit_errors_sup[i_distance] = bootstrap_ci[1] - avg_visit_avg[i_distance]

    return (time_outside_per_patch_avg, nb_of_visits_per_patch_avg, time_inside_per_patch_avg, avg_visit_avg,
            [time_outside_per_patch_errors_inf, time_outside_per_patch_errors_sup],
            [nb_of_visits_per_patch_errors_inf, nb_of_visits_per_patch_errors_sup],
            [time_inside_per_patch_errors_inf, time_inside_per_patch_errors_sup],
            [avg_visit_errors_inf, avg_visit_errors_sup])


def effect_of_length(speed_in, speed_out):
    # Prepare background of plot with Alfonso's equation
    plt.title("Speed inside=" + str(speed_in) + ", Speed outside=" + str(speed_out))
    plt.xlabel("Log of number of visits")
    plt.ylabel("Log of effective travel time")
    plt.xscale("log")
    plt.yscale("log")
    visit_values = np.logspace(0, 4, 100)
    travel_values = np.logspace(0, 6, 100)
    visit_values, time_out_values = np.meshgrid(visit_values, travel_values)
    equation_values = time_out_values / visit_values
    plt.contourf(visit_values, time_out_values, equation_values, levels=np.logspace(-6, 6, 20), cmap="plasma", norm="log")  # contour colors
    plt.contour(visit_values, time_out_values, equation_values, levels=np.logspace(-6, 6, 20), colors="black", norm="log", linewidths=0.5)  # contour lines

    # Plot simulation results
    # Length = 10 000
    envt_matrices = generate_environment_matrices("", [])
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 100, "random", 10000, [speed_in, speed_out])
    plt.scatter(nb_of_visits, time_out, color="white", label="Random walk, time = 10 000")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="white", linewidth=2)
    # Length = 100 000
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", 100000, [speed_in, speed_out])
    plt.scatter(nb_of_visits, time_out, color="yellow", label="Random walk, time = 100 000")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="yellow", linewidth=2)
    # Length = 500 000
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", 500000, [speed_in, speed_out])
    plt.scatter(nb_of_visits, time_out, color="chartreuse", label="Random walk, time = 500 000")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="chartreuse", linewidth=2)
    # Length = 1 000 000
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", 1000000, [speed_in, speed_out])
    plt.scatter(nb_of_visits, time_out, color="turquoise", label="Random walk, time = 1 000 000")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="turquoise", linewidth=2)

    plt.legend()

    plt.show()


def effect_of_speed_out(length, speed_in):
    # Prepare background of plot with Alfonso's equation
    plt.title("Sim length=" + str(length) + ", Speed inside=" + str(speed_in))
    plt.xlabel("Log of number of visits")
    plt.ylabel("Log of effective travel time")
    plt.xscale("log")
    plt.yscale("log")
    visit_values = np.logspace(0, 4, 100)
    travel_values = np.logspace(0, 6, 100)
    visit_values, time_out_values = np.meshgrid(visit_values, travel_values)
    equation_values = time_out_values / visit_values
    plt.contourf(visit_values, time_out_values, equation_values, levels=np.logspace(-6, 6, 20), cmap="plasma",
                 norm="log")  # contour colors
    plt.contour(visit_values, time_out_values, equation_values, levels=np.logspace(-6, 6, 20), colors="black",
                norm="log", linewidths=0.5)  # contour lines

    # Plot simulations
    envt_matrices = generate_environment_matrices("", [])
    # Speed outside = 1
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [speed_in, 1])
    plt.scatter(nb_of_visits, time_out, color="white", label="Random walk, speed out = 1")
    # x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    # plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="white", linewidth=2)
    # Speed outside = 10
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [speed_in, 10])
    plt.scatter(nb_of_visits, time_out, color="yellow", label="Random walk, speed out = 10")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="yellow", linewidth=2)
    # Speed outside = 20
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [speed_in, 20])
    plt.scatter(nb_of_visits, time_out, color="chartreuse", label="Random walk, speed out = 20")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="chartreuse", linewidth=2)
    # Speed outside = 100
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [speed_in, 40])
    plt.scatter(nb_of_visits, time_out, color="turquoise", label="Random walk, speed out = 100")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="turquoise", linewidth=2)

    plt.legend()

    plt.show()


def effect_of_speed_in(length, speed_out):
    # Prepare background of plot with Alfonso's equation
    plt.title("Sim length=" + str(length) + ", Speed outside=" + str(speed_out))
    plt.xlabel("Log of number of visits")
    plt.ylabel("Log of effective travel time")
    plt.xscale("log")
    plt.yscale("log")
    visit_values = np.logspace(0, 4, 100)
    travel_values = np.logspace(0, 6, 100)
    visit_values, time_out_values = np.meshgrid(visit_values, travel_values)
    equation_values = time_out_values / visit_values
    plt.contourf(visit_values, time_out_values, equation_values, levels=np.logspace(-6, 6, 20), cmap="plasma",
                 norm="log")  # contour colors
    plt.contour(visit_values, time_out_values, equation_values, levels=np.logspace(-6, 6, 20), colors="black",
                norm="log", linewidths=0.5)  # contour lines

    # Plot simulations
    envt_matrices = generate_environment_matrices("", [])
    # Speed outside = 1
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [0.6, speed_out])
    plt.scatter(nb_of_visits, time_out, color="white", label="Random walk, speed in = 0.6")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="white", linewidth=2)
    # Speed outside = 10
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [1.6, speed_out])
    plt.scatter(nb_of_visits, time_out, color="yellow", label="Random walk, speed in = 1.6")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="yellow", linewidth=2)
    # Speed outside = 20
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [3.2, speed_out])
    plt.scatter(nb_of_visits, time_out, color="chartreuse", label="Random walk, speed in = 3.2")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="chartreuse", linewidth=2)
    # Speed outside = 100
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [12, speed_out])
    plt.scatter(nb_of_visits, time_out, color="turquoise", label="Random walk, speed in = 12")
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="turquoise", linewidth=2)

    plt.legend()

    plt.show()


def effect_of_walk_type(distance_list, length, nb_of_walkers, speed_in, speed_out, half_life_speed,
                        max_turning_angle, sharp_angle_probability, turn_probability_factor, xp_path, xp_table, list_xp_folders,
                        what_to_plot=None):
    # Plots on the left a comparison of our three walk types in the toy model of Alfonso, with
    # effective time outside (time outside per patch) and number of visits per patch
    # On the right, two plots, one with the avg visit time for all walks + xp, and one with the time inside per patch
    fig, [ax0, ax1] = plt.subplots(1, 2)
    fig.suptitle("Sim length=" + str(length) + ", speed inside="+str(speed_in)+", speed outside=" + str(speed_out)+", tau="+str(half_life_speed))
    fig.set_size_inches(18, 5)

    # Prepare the two rightmost plots
    #ax1.set_ylabel("Average visit duration (hours)", fontsize=16)
    ax1.set_ylabel("Total time inside per patch (hours)", fontsize=16)
    #ax1.set_yscale("log")  # for easier comparison with the results of the colormap which is in log!
    # Set the x labels to the distance icons!
    # Stolen from https://stackoverflow.com/questions/8733558/how-can-i-make-the-xtick-labels-of-a-plot-be-simple-drawings
    for i in range(len(distance_list)):
        #ax1.set_xticks([])
        ax1.set_xticks([])
        # Image to use
        arr_img = plt.imread(os.getcwd().replace("\\", "/")[:-len("Scripts_models/")] + "/Parameters/icon_" + distance_list[i] + '.png')

        # Image box to draw it!
        #imagebox1 = OffsetImage(arr_img, zoom=0.6)
        #imagebox1.image.axes = ax1
        #x_annotation_box1 = AnnotationBbox(imagebox1, (i, 0),
        #                                   xybox=(0, -8),
        #                                   # that's the shift that the image will have compared to (i, 0)
        #                                   xycoords=("data", "axes fraction"),
        #                                   boxcoords="offset points",
        #                                   box_alignment=(.5, 1),
        #                                   bboxprops={"edgecolor": "none"})
        #ax1.add_artist(x_annotation_box1)

        # Image box to draw it!
        imagebox2 = OffsetImage(arr_img, zoom=0.6)
        imagebox2.image.axes = ax1
        x_annotation_box2 = AnnotationBbox(imagebox2, (i, 0),
                                           xybox=(0, -8),
                                           # that's the shift that the image will have compared to (i, 0)
                                           xycoords=("data", "axes fraction"),
                                           boxcoords="offset points",
                                           box_alignment=(.5, 1),
                                           bboxprops={"edgecolor": "none"})

        ax1.add_artist(x_annotation_box2)

    # Plot background of left plot with Alfonso's equation
    ax0.set_xlabel("Number of visits per patch (log)", fontsize=16)
    ax0.set_ylabel("Effective travel time (hours)", fontsize=16)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    visit_values_list = np.logspace(0, 2.8, 100)
    travel_values_list = np.logspace(-2, 1, 100)
    visit_values, time_out_values = np.meshgrid(visit_values_list, travel_values_list)
    equation_values = time_out_values / visit_values
    contours = ax0.contourf(visit_values, time_out_values, equation_values, levels=np.logspace(-4.8, 1, 10),
                            cmap="plasma", norm="log")  # contour colors
    ax0.contour(visit_values, time_out_values, equation_values, levels=np.logspace(-4.8, 1, 10), colors="black",
                norm="log", linewidths=0.5)  # contour lines
    fig.colorbar(contours, ticks=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])

    # plt.show()

    # Plot simulation values
    envt_matrices = generate_environment_matrices(xp_path, list_xp_folders)

    # Classical random walk
    time_out, nb_of_visits, time_in, avg_visit, time_out_errors, nb_of_visits_errors, time_in_errors, avg_visit_errors = (
        values_one_type_of_walk(envt_matrices, nb_of_walkers, "random", length, [speed_in, speed_out]))
    ax0.errorbar(nb_of_visits, np.array(time_out)/3600, xerr=nb_of_visits_errors, yerr=np.array(time_out_errors)/3600, color="yellow", capsize=5, linewidth=3, label="Random walk", marker="o")
    if what_to_plot == "nb_of_visits":
        ax1.errorbar(range(len(nb_of_visits)), np.array(nb_of_visits), yerr=np.array(nb_of_visits_errors), color="gold", capsize=5, linewidth=3, label="Random walk", marker="o")
    if what_to_plot == "total_time":
        ax1.errorbar(range(len(time_in)), np.array(time_in)/3600, yerr=np.array(time_in_errors)/3600, color="gold", capsize=5, linewidth=3, label="Random walk", marker="o")
    if what_to_plot == "avg_visit_time":
        ax1.errorbar(range(len(avg_visit)), np.array(avg_visit)/3600, yerr=np.array(avg_visit)/3600, color="gold", capsize=5, linewidth=3, label="Random walk", marker="o")
    # Plot the last point of each line (most distant) as a star
    ax0.scatter(nb_of_visits[-1], np.array(time_out)[-1]/3600, color="goldenrod", marker="*", s=200, zorder=10)
    # ax1.scatter(range(len(time_in))[-1], np.array(time_in)[-1]/3600, color="goldenrod", marker="*", s=200, zorder=10)

    # Walk that speeds up when leaving the patch
    # time_out, nb_of_visits, time_in, avg_visit, time_out_errors, nb_of_visits_errors, time_in_errors, avg_visit_errors = (
    #     values_one_type_of_walk(envt_matrices, nb_of_walkers, "dynamic_speed", length, [speed_in, speed_out, half_life_speed]))
    # ax0.errorbar(nb_of_visits, np.array(time_out)/3600, xerr=nb_of_visits_errors, yerr=np.array(time_out_errors)/3600, color="chartreuse", capsize=5, linewidth=3, label="Dynamic speed walk", marker="o")
    # ax1.errorbar(range(len(avg_visit)), np.array(avg_visit)/3600, yerr=np.array(avg_visit_errors)/3600, color="chartreuse", capsize=5, linewidth=3, label="Dynamic speed walk", marker="o")
    # ax2.errorbar(range(len(time_in)), np.array(time_in)/3600, yerr=np.array(time_in_errors)/3600, color="chartreuse", capsize=5, linewidth=3, label="Dynamic speed walk", marker="o")
    # # Plot the last point of each line (most distant) as a star
    # ax0.scatter(nb_of_visits[-1], np.array(time_out)[-1]/3600, color="limegreen", marker="*", s=200, zorder=10)
    # ax1.scatter(range(len(avg_visit))[-1], np.array(avg_visit)[-1]/3600, color="limegreen", marker="*", s=200, zorder=10)
    # ax2.scatter(range(len(time_in))[-1], np.array(time_in)[-1]/3600, color="limegreen", marker="*", s=200, zorder=10)

    # Correlated walk
    time_out, nb_of_visits, time_in, avg_visit, time_out_errors, nb_of_visits_errors, time_in_errors, avg_visit_errors = (
        values_one_type_of_walk(envt_matrices, nb_of_walkers, "auto_correlated", length, [speed_in, speed_out, max_turning_angle]))
    ax0.errorbar(nb_of_visits, np.array(time_out)/3600, xerr=nb_of_visits_errors, yerr=np.array(time_out_errors)/3600, color="chartreuse", capsize=5, linewidth=3, label="Correlated walk", marker="o")
    if what_to_plot == "nb_of_visits":
        ax1.errorbar(range(len(nb_of_visits)), np.array(nb_of_visits), yerr=np.array(nb_of_visits_errors), color="chartreuse",
                     capsize=5, linewidth=3, label="Correlated walk", marker="o")
    if what_to_plot == "total_time":
        ax1.errorbar(range(len(time_in)), np.array(time_in) / 3600, yerr=np.array(time_in_errors) / 3600, color="chartreuse",
                     capsize=5, linewidth=3, label="Correlated walk", marker="o")
    if what_to_plot == "avg_visit_time":
        ax1.errorbar(range(len(avg_visit)), np.array(avg_visit) / 3600, yerr=np.array(avg_visit) / 3600, color="chartreuse",
                     capsize=5, linewidth=3, label="Correlated walk", marker="o")
    # Plot the last point of each line (most distant) as a star
    ax0.scatter(nb_of_visits[-1], np.array(time_out)[-1]/3600, color="limegreen", marker="*", s=200, zorder=10)
    # ax1.scatter(range(len(time_in))[-1], np.array(time_in)[-1]/3600, color="limegreen", marker="*", s=200, zorder=10)

    # Correlated walk that sometimes does sharp turns
    time_out, nb_of_visits, time_in, avg_visit, time_out_errors, nb_of_visits_errors, time_in_errors, avg_visit_errors = (
        values_one_type_of_walk(envt_matrices, nb_of_walkers, "correlated_sharp_turns", length, [speed_in, speed_out, max_turning_angle, sharp_angle_probability, turn_probability_factor]))
    ax0.errorbar(nb_of_visits, np.array(time_out)/3600, xerr=nb_of_visits_errors, yerr=np.array(time_out_errors)/3600, color="turquoise", capsize=5, linewidth=3, label="Correlated walk + sharp turns", marker="o")
    if what_to_plot == "nb_of_visits":
        ax1.errorbar(range(len(nb_of_visits)), np.array(nb_of_visits), yerr=np.array(nb_of_visits_errors), color="turquoise",
                     capsize=5, linewidth=3, label="Correlated walk + sharp turns", marker="o")
    if what_to_plot == "total_time":
        ax1.errorbar(range(len(time_in)), np.array(time_in) / 3600, yerr=np.array(time_in_errors) / 3600, color="turquoise",
                     capsize=5, linewidth=3, label="Correlated walk + sharp turns", marker="o")
    if what_to_plot == "avg_visit_time":
        ax1.errorbar(range(len(avg_visit)), np.array(avg_visit) / 3600, yerr=np.array(avg_visit) / 3600, color="turquoise",
                     capsize=5, linewidth=3, label="Correlated walk + sharp turns", marker="o")
    # Plot the last point of each line (most distant) as a star
    ax0.scatter(nb_of_visits[-1], np.array(time_out)[-1]/3600, color="lightseagreen", marker="*", s=200, zorder=10)
    # ax1.scatter(range(len(time_in))[-1], np.array(time_in)[-1]/3600, color="lightseagreen", marker="*", s=200, zorder=10)

    # # Autocorrelated walk
    # time_out, nb_of_visits, time_in, avg_visit, time_out_errors, nb_of_visits_errors, time_in_errors, avg_visit_errors = (
    #     values_one_type_of_walk(envt_matrices, nb_of_walkers, "auto_correlated", length, [speed_in, speed_out, max_turning_angle]))
    # ax0.errorbar(nb_of_visits, np.array(time_out)/3600, xerr=nb_of_visits_errors, yerr=np.array(time_out_errors)/3600, color="deepskyblue", capsize=5, linewidth=3, label="Correlated walk + sharp turns", marker="o")
    # #ax1.errorbar(range(len(avg_visit)), np.array(avg_visit)/3600, yerr=np.array(avg_visit_errors)/3600, color="turquoise", capsize=5, linewidth=3, label="Correlated walk + sharp turns", marker="o")
    # ax1.errorbar(range(len(time_in)), np.array(time_in)/3600, yerr=np.array(time_in_errors)/3600, color="deepskyblue", capsize=5, linewidth=3, label="Correlated walk + sharp turns", marker="o")
    # # Plot the last point of each line (most distant) as a star
    # ax0.scatter(nb_of_visits[-1], np.array(time_out)[-1]/3600, color="dodgerblue", marker="*", s=200, zorder=10)
    # #ax1.scatter(range(len(avg_visit))[-1], np.array(avg_visit)[-1]/3600, color="lightseagreen", marker="*", s=200, zorder=10)
    # ax1.scatter(range(len(time_in))[-1], np.array(time_in)[-1]/3600, color="dodgerblue", marker="*", s=200, zorder=10)

    # Plot C. elegans values
    time_out, nb_of_visits, time_in, avg_visit, time_out_errors, nb_of_visits_errors, time_in_errors, avg_visit_errors = experimental_values(distance_list, xp_table, list_xp_folders)
    ax0.errorbar(nb_of_visits, np.array(time_out)/3600, xerr=nb_of_visits_errors, yerr=np.array(time_out_errors)/3600, color="black", capsize=5, linewidth=3, label="Experimental data", marker="o")
    if what_to_plot == "nb_of_visits":
        ax1.errorbar(range(len(nb_of_visits)), np.array(nb_of_visits), yerr=np.array(nb_of_visits_errors), color="black",
                     capsize=5, linewidth=3, label="Experimental data", marker="o")
    if what_to_plot == "total_time":
        ax1.errorbar(range(len(time_in)), np.array(time_in) / 3600, yerr=np.array(time_in_errors) / 3600, color="black",
                     capsize=5, linewidth=3, label="Experimental data", marker="o")
    if what_to_plot == "avg_visit_time":
        ax1.errorbar(range(len(avg_visit)), np.array(avg_visit) / 3600, yerr=np.array(avg_visit) / 3600, color="black",
                     capsize=5, linewidth=3, label="Experimental data", marker="o")
    # Plot the last point of each line (most distant) as a star
    ax0.scatter(nb_of_visits[-1], np.array(time_out)[-1]/3600, color="grey", marker="*", s=200, zorder=10)
    # ax1.scatter(range(len(time_in))[-1], np.array(time_in)[-1]/3600, color="grey", marker="*", s=200, zorder=10)

    # Plot empty stars to add it to the legend
    ax0.scatter([], [], color="black", marker="*", label="Highest inter-patch distance", s=200)
    # ax1.scatter([], [], color="black", marker="*", label="Highest inter-patch distance", s=200)

    ax1.legend(fontsize=11)

    # Reset axis limits because sometimes they get weird
    ax0.set_xlim(np.min(visit_values_list), np.max(visit_values_list))
    ax0.set_ylim(np.min(travel_values_list), np.max(travel_values_list))

    plt.show()


def show_each_walk_type(length, speed_in, speed_out, half_life_speed, max_turning_angle, sharp_turn_probability, turn_probability_factor, zoom_in):

    fig, [ax0, ax1, ax2] = plt.subplots(1, 3)

    if zoom_in:
        vmax = 0
    else:
        vmax = None

    # Plot simulations
    envt_matrices = generate_environment_matrices("", [])

    # Classical random walk
    #t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = random_walk(length, speed_in, speed_out, envt_matrices[1])
    t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = random_walk(length, speed_in, speed_out, envt_matrices[1])
    ax0.imshow(envt_matrices[1], vmax=vmax)
    colored_lines_script.colored_line_between_pts(x, y, c=speeds, ax=ax0, cmap='hot')
    ax0.set_title("Random walk", fontsize=20)
    ax0.set_xlabel("Parameters = "+str([length, speed_in, speed_out]))
    if zoom_in:
        ax0.set_xlim(380, 740)
        ax0.set_ylim(600, 960)

    # # Dynamic speed walk
    # speed_table = dynamic_speed_table(speed_in, speed_out, half_life_speed, length)
    # t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = dynamic_speed_walk(length, speed_in, speed_out, speed_table, envt_matrices[1])
    # ax1.imshow(envt_matrices[1])
    # # ax1.plot(x, y, color="orange")
    # # normalize = mplcolors.Normalize(vmin=6, vmax=30)
    # # ax1.scatter(x, y, c=speeds, cmap="hot", norm=normalize)
    # colored_lines_script.colored_line(x, y, c=speeds, ax=ax1, cmap='hot')
    # ax1.set_title("Dynamic speed walk", fontsize=20)
    # ax1.set_xlabel("Parameters = "+str([length, speed_in, speed_out, half_life_speed]))
    # if zoom_in:
    #     ax1.set_xlim(720, 1030)
    #     ax1.set_ylim(720, 1030)

    # Correlated walk
    t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = correlated_walk(length, speed_in, speed_out, max_turning_angle, envt_matrices[1], sharp_turn_probability=0, turn_probability_factor=turn_probability_factor)
    ax1.imshow(envt_matrices[1], vmax=vmax)
    colored_lines_script.colored_line(x, y, c=speeds, ax=ax1, cmap='hot')
    ax1.set_title("Correlated walk", fontsize=20)
    ax1.set_xlabel("Parameters = "+str([length, speed_in, speed_out, max_turning_angle]))
    if zoom_in:
        ax1.set_xlim(380, 740)
        ax1.set_ylim(600, 960)

    # Correlated walk with sharp turns
    t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = correlated_walk(length, speed_in, speed_out, max_turning_angle, envt_matrices[1], sharp_turn_probability=sharp_turn_probability, turn_probability_factor=turn_probability_factor)
    ax2.imshow(envt_matrices[1], vmax=vmax)
    lines = colored_lines_script.colored_line(x, y, c=speeds, ax=ax2, cmap='hot')
    ax2.set_title("Correlated walk + sharp turns", fontsize=20)
    ax2.set_xlabel("Parameters = "+str([length, speed_in, speed_out, max_turning_angle, sharp_turn_probability, turn_prob_factor]))
    if zoom_in:
        ax2.set_xlim(380, 740)
        ax2.set_ylim(600, 960)

    # Autocorrelated walk
    #t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = auto_correlated_walk(length, speed_in, speed_out, max_turning_angle, envt_matrices[1])
    #ax3.imshow(envt_matrices[1], vmax=vmax)
    #colored_lines_script.colored_line(x, y, c=speeds, ax=ax3, cmap='hot')
    #ax3.set_title("Autocorrelated walk", fontsize=20)
    #ax3.set_xlabel("Parameters = "+str([length, speed_in, speed_out, max_turning_angle]))
    #if zoom_in:
    #    ax3.set_xlim(720, 1030)
    #    ax3.set_ylim(720, 1030)

    plt.show()


def show_one_walk_type(length, speed_in, speed_out, half_life_speed, max_turning_angle, sharp_turn_probability, turn_probability_factor, zoom_in):
    fig, [ax0, ax1, ax2, ax3] = plt.subplots(1, 4)

    # Plot simulations
    envt_matrices = generate_environment_matrices("", [])

    time_out_list = []
    time_in_list = []
    nb_visited_list = []
    nb_visits_list = []

    # Correlated walk
    t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = correlated_walk(length, speed_in, speed_out, max_turning_angle, envt_matrices[0], sharp_turn_probability=0, turn_probability_factor=turn_probability_factor)
    ax0.imshow(envt_matrices[0])
    colored_lines_script.colored_line(x, y, c=speeds, ax=ax0, cmap='hot')
    ax0.set_title("Correlated walk", fontsize=20)
    ax0.set_xlabel("Parameters = "+str([length, speed_in, speed_out, half_life_speed]))
    if zoom_in:
        ax0.set_xlim(720, 1030)
        ax0.set_ylim(720, 1030)
    time_out_list.append(time_out)
    time_in_list.append(time_in)
    nb_visited_list.append(nb_visited)
    nb_visits_list.append(nb_visits)

    # Correlated walk
    t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = correlated_walk(length, speed_in, speed_out, max_turning_angle, envt_matrices[1], sharp_turn_probability=0, turn_probability_factor=turn_probability_factor)
    ax1.imshow(envt_matrices[1])
    colored_lines_script.colored_line(x, y, c=speeds, ax=ax1, cmap='hot')
    ax1.set_title("Correlated walk", fontsize=20)
    ax1.set_xlabel("Parameters = "+str([length, speed_in, speed_out, half_life_speed]))
    if zoom_in:
        ax1.set_xlim(720, 1030)
        ax1.set_ylim(720, 1030)
    time_out_list.append(time_out)
    time_in_list.append(time_in)
    nb_visited_list.append(nb_visited)
    nb_visits_list.append(nb_visits)

    # Correlated walk
    t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = correlated_walk(length, speed_in, speed_out, max_turning_angle, envt_matrices[2], sharp_turn_probability=0, turn_probability_factor=turn_probability_factor)
    ax2.imshow(envt_matrices[2])
    colored_lines_script.colored_line(x, y, c=speeds, ax=ax2, cmap='hot')
    ax2.set_title("Correlated walk", fontsize=20)
    ax2.set_xlabel("Parameters = "+str([length, speed_in, speed_out, half_life_speed]))
    if zoom_in:
        ax2.set_xlim(720, 1030)
        ax2.set_ylim(720, 1030)
    time_out_list.append(time_out)
    time_in_list.append(time_in)
    nb_visited_list.append(nb_visited)
    nb_visits_list.append(nb_visits)

    # Correlated walk
    t, x, y, speeds, time_in, time_out, nb_visited, nb_visits = correlated_walk(length, speed_in, speed_out, max_turning_angle, envt_matrices[3], sharp_turn_probability=0, turn_probability_factor=turn_probability_factor)
    ax3.imshow(envt_matrices[3])
    colored_lines_script.colored_line(x, y, c=speeds, ax=ax3, cmap='hot')
    ax3.set_title("Correlated walk", fontsize=20)
    ax3.set_xlabel("Parameters = "+str([length, speed_in, speed_out, half_life_speed]))
    if zoom_in:
        ax3.set_xlim(720, 1030)
        ax3.set_ylim(720, 1030)
    time_out_list.append(time_out)
    time_in_list.append(time_in)
    nb_visited_list.append(nb_visited)
    nb_visits_list.append(nb_visits)

    plt.show()

    print("Time out list: ", time_out_list)
    print("Time in list: ", time_in_list)
    print("Nb patches list: ", nb_visited_list)
    print("Nb visits list: ", nb_visits_list)


def plot_dynamic_speed_function(speed_inside, speed_outside, sim_length, inverse=True):
    speed_table_1 = dynamic_speed_table(speed_inside, speed_outside, 1, sim_length)
    speed_table_10 = dynamic_speed_table(speed_inside, speed_outside, 10, sim_length)
    speed_table_100 = dynamic_speed_table(speed_inside, speed_outside, 100, sim_length)
    speed_table_1000 = dynamic_speed_table(speed_inside, speed_outside, 1000, sim_length)
    speed_table_2000 = dynamic_speed_table(speed_inside, speed_outside, 2000, sim_length)
    speed_table_4000 = dynamic_speed_table(speed_inside, speed_outside, 4000, sim_length)
    speed_table_6000 = dynamic_speed_table(speed_inside, speed_outside, 6000, sim_length)
    speed_table_10000 = dynamic_speed_table(speed_inside, speed_outside, 10000, sim_length)
    speed_table_20000 = dynamic_speed_table(speed_inside, speed_outside, 20000, sim_length)
    if inverse:
        plt.ylabel("Turning rate (per min)")
        plt.xlabel("Time post patch exit")
        plt.xlim(0, 3000)
        plt.ylim(0, 2)
        plt.plot(range(sim_length), 1 / (np.array(speed_table_1)), label="half_life = 1")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_10)), label="half_life = 10")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_100)), label="half_life = 100")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_1000)), label="half_life = 1000")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_2000)), label="half_life = 2000")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_4000)), label="half_life = 4000")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_6000)), label="half_life = 6000")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_10000)), label="half_life = 10000")
        plt.plot(range(sim_length), 1 / (np.array(speed_table_20000)), label="half_life = 20000")
    else:
        plt.ylabel("Speed")
        plt.xlabel("Time post patch exit")
        plt.plot(range(sim_length), np.array(speed_table_1), label="half_life = 1")
        plt.plot(range(sim_length), np.array(speed_table_10), label="half_life = 10")
        plt.plot(range(sim_length), np.array(speed_table_100), label="half_life = 100")
        plt.plot(range(sim_length), np.array(speed_table_1000), label="half_life = 1000")
        plt.plot(range(sim_length), np.array(speed_table_6000), label="half_life = 6000")
        plt.plot(range(sim_length), np.array(speed_table_10000), label="half_life = 10000")
        plt.plot(range(sim_length), np.array(speed_table_20000), label="half_life = 20000")
    plt.legend()
    plt.show()


# Turns / minute in [Klein 2017]
exit_turn_per_min = 2
long_term_turn_per_min = 0.4

# Speeds in mm / s from [Iwanir 2016]
speed_inside_mm_s = 0.02
speed_outside_mm_s = 0.1

# Convert those to our units
speed_inside_pixel_s = speed_inside_mm_s * (1/param.one_pixel_in_mm)
speed_outside_pixel_s = speed_outside_mm_s * (1/param.one_pixel_in_mm)
exit_turn_per_s = exit_turn_per_min / 60
long_term_turn_per_s = long_term_turn_per_min / 60
run_length_inside = speed_inside_pixel_s / exit_turn_per_s
run_length_outside = speed_outside_pixel_s / long_term_turn_per_s

print("From literature values, it looks like our run lengths should be ", run_length_inside, " pixels inside food, and ", run_length_outside, " pixels outside.")
print("From literature values, it looks like the speed should be ", speed_inside_pixel_s, " pixels/s inside food, and ", speed_outside_pixel_s, " pixels/s outside.")

# effect_of_length(1.6, 30)
# effect_of_speed_out(500000, 1.6)
# effect_of_speed_in(500000, 16)
# plot_dynamic_speed_function(0.6, 3.1, 50000)
# show_each_walk_type(100000, 0.6, 3.1, 1000, np.pi/2)

path = gen.generate("")
results = pd.read_csv(path + "clean_results.csv")
speed_in = 1.5
speed_out = 11
# speed_in = 6
# speed_out = 6
max_turn = np.pi/4
turn_prob = 0.01
turn_prob_factor = 50
sim_length = 25000
# show_one_walk_type(sim_length, speed_in, speed_out, 1000, max_turn, turn_prob, turn_prob_factor, zoom_in=False)

# effect_of_walk_type(["close", "med", "far", "superfar"], sim_length, 100, speed_in, speed_out,
#                     1000, max_turn, turn_prob, turn_prob_factor, path, results, results["folder"],
#                     "nb_of_visits")
show_each_walk_type(sim_length, speed_in, speed_out, 1000, max_turn, turn_prob, turn_prob_factor, zoom_in=True)

