import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
from scipy import ndimage

from Generating_data_tables import main as gen
import ReferencePoints
import analysis as ana
from Parameters import parameters as param
from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap_script
import find_data as fd


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
    # patch_radius = heatmap_script.generate_average_patch_radius_each_condition(results_path, xp_plate_list)
    patch_radius = 41
    # ref_point_distance = heatmap_script.compute_average_ref_points_distance(results_path, xp_plate_list)
    ref_point_distance = 985

    environment_matrices = []
    all_distances = param.distance_to_xy.keys()
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
        small_ref_points = ReferencePoints.ReferencePoints([[-20, 20], [20, 20], [20, -20], [-20, -20]])
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
    total_time_inside = 0
    total_time_outside = 0
    list_of_visited_patches = []  # Will add patches as they get visited
    # Starting point
    x_list[0] = len(environment_matrix) / 2
    y_list[0] = len(environment_matrix[0]) / 2
    current_patch = return_patch(environment_matrix, x_list[0], y_list[0])
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
            if current_patch != previous_patch:
                list_of_visited_patches.append(current_patch)
        # If worm is outside
        elif current_patch == -1:
            x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
            y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
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
                previous_x = previous_x + speed_outside * np.cos(current_heading)
                previous_y = previous_y + speed_outside * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, previous_x, previous_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
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
            total_time_outside += 1
            x_list[i_time] = previous_x
            y_list[i_time] = previous_y

    # plt.imshow(environment_matrix, cmap="plasma")
    # plt.plot(x_list, y_list, color="white")
    # plt.show()

    return time_list, x_list, y_list, total_time_inside, total_time_outside, len(np.unique(list_of_visited_patches)), len(list_of_visited_patches)


def dynamic_speed(speed_inside, speed_outside, half_speed_life, time_since_patch_exit):
    """
    Will return the current speed of an agent that has left the patch for "time_since_patch_exit" time steps.
    The function increases in 1/t, and should go from speed_inside (in t=0) to speed_outside (in t=infinite).
    It reaches speed_inside + speed_outside/2 at half_speed_life
    """
    return speed_outside - (speed_outside-speed_inside)/(1 + time_since_patch_exit * ((1 - 2 * (speed_inside/speed_outside))/half_speed_life))


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
    total_time_inside = 0
    total_time_outside = 0
    list_of_visited_patches = []  # Will add patches as they get visited
    # Starting point
    x_list[0] = len(environment_matrix) / 2
    y_list[0] = len(environment_matrix[0]) / 2
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
                x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
                y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
            else:
                current_speed = speed_table[time_since_patch_exit]
                x_list[i_time] = x_list[i_time - 1] + current_speed * np.cos(current_heading)
                y_list[i_time] = y_list[i_time - 1] + current_speed * np.sin(current_heading)
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
                    x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
                    y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
                else:
                    current_speed = speed_table[time_since_patch_exit]
                    x_list[i_time] = x_list[i_time - 1] + current_speed * np.cos(current_heading)
                    y_list[i_time] = y_list[i_time - 1] + current_speed * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, previous_x, previous_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 70:
                current_heading += 0.1
                if beginning_of_sim:  # In the beginning of the sim just go at speed_outside
                    x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
                    y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
                else:
                    current_speed = speed_table[time_since_patch_exit]
                    x_list[i_time] = x_list[i_time - 1] + current_speed * np.cos(current_heading)
                    y_list[i_time] = y_list[i_time - 1] + current_speed * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, previous_x, previous_y)
                i_while += 1
            # In case the previous, coarse-grained loop didn't work
            i_while = 0
            while current_patch == -2 and i_while < 700:
                current_heading += 0.01
                if beginning_of_sim:  # In the beginning of the sim just go at speed_outside
                    x_list[i_time] = x_list[i_time - 1] + speed_outside * np.cos(current_heading)
                    y_list[i_time] = y_list[i_time - 1] + speed_outside * np.sin(current_heading)
                else:
                    current_speed = speed_table[time_since_patch_exit]
                    x_list[i_time] = x_list[i_time - 1] + current_speed * np.cos(current_heading)
                    y_list[i_time] = y_list[i_time - 1] + current_speed * np.sin(current_heading)
                current_patch = return_patch(environment_matrix, previous_x, previous_y)
                i_while += 1
            time_since_patch_exit += 1  # this keeps incrementing as long as no patch has been encountered
            total_time_outside += 1
            x_list[i_time] = previous_x
            y_list[i_time] = previous_y

    # plt.imshow(environment_matrix, cmap="plasma")
    # plt.plot(x_list, y_list, color="white")
    # plt.show()

    return time_list, x_list, y_list, total_time_inside, total_time_outside, len(np.unique(list_of_visited_patches)), len(list_of_visited_patches)


def values_one_type_of_walk(map_each_distance, nb_of_walkers, type_of_walk, sim_length, parameters):
    time_outside_per_patch = []
    nb_of_visits_per_patch = []
    if type_of_walk == "random":
        speed_inside, speed_outside = parameters
        for i_map, current_map in enumerate(map_each_distance):
            print("Generating random walks for distance", list(param.distance_to_xy.keys())[i_map], "...")
            for i_walk in range(nb_of_walkers):
                if i_walk % (nb_of_walkers // 4) == 0:
                    print(">>> Walker ", i_walk, " / ", nb_of_walkers, "...")
                # Run simulation
                _, _, _, time_in, time_out, nb_visited, nb_visits = random_walk(sim_length, speed_inside, speed_outside, current_map)
                # Compute the relevant variables
                if nb_visited > 0:
                    time_outside_per_patch.append(time_out / nb_visited)
                    nb_of_visits_per_patch.append(nb_visits / nb_visited)
                else:
                    time_outside_per_patch.append(np.nan)
                    nb_of_visits_per_patch.append(np.nan)

    if type_of_walk == "dynamic_speed":
        speed_inside, speed_outside, half_speed_life = parameters
        speed_table = dynamic_speed_table(speed_inside, speed_outside, half_speed_life, sim_length)
        for i_map, current_map in enumerate(map_each_distance):
            print("Generating dynamic speed walks for distance", list(param.distance_to_xy.keys())[i_map], "...")
            for i_walk in range(nb_of_walkers):
                if i_walk % (nb_of_walkers // 4) == 0:
                    print(">>> Walker ", i_walk, " / ", nb_of_walkers, "...")
                # Run simulation
                _, _, _, time_in, time_out, nb_visited, nb_visits = dynamic_speed_walk(sim_length, speed_inside, speed_outside, speed_table, current_map)
                # Compute the relevant variables
                if nb_visited > 0:
                    time_outside_per_patch.append(time_out / nb_visited)
                    nb_of_visits_per_patch.append(nb_visits / nb_visited)
                else:
                    time_outside_per_patch.append(np.nan)
                    nb_of_visits_per_patch.append(np.nan)

    return time_outside_per_patch, nb_of_visits_per_patch


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

    # Plot C. elegans values
    xp_data = pd.read_csv(gen.generate("", shorten_traj=True) + "model_parameters_from_alid.csv")
    for distance in param.distance_to_xy.keys():
        print("hehehe")

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


def effect_of_walk_type(length, speed_in, speed_out, half_life_speed):
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

    # Walk that speeds up when leaving the patch
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "dynamic_speed", length, [speed_in, speed_out, half_life_speed])
    plt.scatter(nb_of_visits, time_out, color="yellow", label="Dynamic walk, speed in =" + str(speed_in) + ", speed out = " + str(speed_out) + ", half life = " + str(half_life_speed))
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="turquoise", linewidth=2)

    # Classical random walk
    time_out, nb_of_visits = values_one_type_of_walk(envt_matrices, 10, "random", length, [speed_in, speed_out])
    plt.scatter(nb_of_visits, time_out, color="white", label="Random walk, speed in ="+str(speed_in)+", speed out = "+str(speed_out))
    x_extent_points = np.linspace(min(nb_of_visits), max(nb_of_visits), 10)
    plt.plot(x_extent_points, ana.log_regression(nb_of_visits, time_out, x_extent_points), color="chartreuse", linewidth=2)

    plt.legend()

    plt.show()


# effect_of_length(1.6, 30)
# effect_of_speed_out(500000, 1.6)
# effect_of_speed_in(500000, 16)
effect_of_walk_type(50000, 1.6, 30, 1000)

