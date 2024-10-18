import numpy as np
import pandas as pd
import random

from Parameters import parameters as param
import find_data as fd


def return_patch(environment, x, y):
    """
    Returns the value of environment in line x, column y, taking the boundary if x and y exceed it.
    :param environment: 2-dimensional pandas dataframe
    :param x: a number
    :param y: a number
    :return: a number
    """
    return environment[str(int(np.clip(x, 0, len(environment["0"]) - 1)))][int(np.clip(y[0], 0, len(environment) - 1))]


def random_walk(sim_length, speed_inside, speed_outside, environment_matrix):
    """
    Function that takes two model parameters (speed inside food patches and
    speed outside food patches), an environment (with -2 outside of the
    environment, -1 outside the food patches, and integers >=0 for the
    food patches). Will run a simulation for sim_length time steps, and return:
        - The total time spent inside food patches
        - The total time spent outside food patches
          (should be sim_length - total time inside)
        - The number of visited patches
        - The number of visits
    """
    time_list = list(range(0, sim_length))
    x_list = [0 for _ in range(sim_length)]
    y_list = [0 for _ in range(sim_length)]
    total_time_inside = 0
    total_time_outside = 0
    list_of_visited_patches = []  # Will add patches as they get visited
    # Starting point
    x_list[0] = len(environment_matrix) / 2
    y_list[0] = len(environment_matrix[0]) / 2
    current_patch = return_patch(environment_matrix, x_list[0], y_list[0])
    # Simulation loop
    for time in range(1, sim_length+1):
        previous_patch = current_patch
        current_patch = return_patch(environment_matrix, x_list[time-1], y_list[time-1])
        current_heading = np.random.rand() * 2 * np.pi  # choose a random angle
        # If worm is inside
        if current_patch >= 0:
            x_list[time] = x_list[time-1] + speed_inside * np.cos(current_heading)
            y_list[time] = y_list[time-1] + speed_inside * np.sin(current_heading)
            total_time_inside += 1
            if current_patch != previous_patch:
                list_of_visited_patches.append(current_patch)
        # If worm is outside
        elif current_patch == -1:
            x_list[time] = x_list[time-1] + speed_outside * np.cos(current_heading)
            y_list[time] = y_list[time-1] + speed_outside * np.sin(current_heading)
            total_time_outside += 1
        # If worm is escaping the plate (current_patch == -2)
        else:
            # While it's escaped, draw a new direction by progressively rotating
            # the current_heading
            i_while = 0
            # Store those to avoid having to access the table a bazillion time
            previous_x = x_list[time-1]
            previous_y = y_list[time-1]
            while current_patch == -2 and i_while < 70:
                current_heading += 0.1
                previous_x = previous_x + speed_outside * np.cos(current_heading)
                previous_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, previous_x, previous_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 700:
                current_heading += 0.01
                previous_x = previous_x + speed_outside * np.cos(current_heading)
                previous_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, previous_x, previous_y)
                i_while += 1
            total_time_outside +=1
            x_list[time] = previous_x
            y_list[time] = previous_y

    return time_list, x_list, y_list, total_time_inside, total_time_outside, len(np.unique(list_of_visited_patches)), len(list_of_visited_patches)


def values_one_type_of_walk(map_each_distance, nb_of_walkers, type_of_walk, sim_length, parameters):
    time_outside_per_patch = []
    nb_of_visits_per_patch = []
    if type_of_walk == "random":
        speed_inside, speed_outside = parameters
        for current_map in map_each_distance:
            for i_walk in range(nb_of_walkers):
                # Run simulation
                _, _, _, time_in, time_out, nb_visited, nb_visits = random_walk(sim_length, speed_inside, speed_outside, current_map)
                # Compute the relevant variables
                time_outside_per_patch.append(time_out / nb_visited)
                nb_of_visits_per_patch.append(nb_visits / nb_visited)                                        
    return time_outside_per_patch, nb_of_visits_per_patch
