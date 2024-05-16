import numpy as np
import pandas as pd

from Parameters import parameters as param
import find_data as fd
import matplotlib.pyplot as plt
import copy
import ReferencePoints


def spline_value(angular_position, spline_breaks, spline_coefs):
    """
    Matlab splines are curves divided in subsections, each subsection defined by a polynomial.

    ::spline_breaks: the limits between the different polynomials
    :spline_coefs: the coefficients for each subsection
    """
    i = 0
    while i < len(spline_breaks) - 1 and angular_position >= spline_breaks[i]:
        i += 1
    # invert coefficient order (matlab to numpy format)
    coefficients = [spline_coefs[i - 1][j] for j in range(len(spline_coefs[i - 1]) - 1, -1, -1)]
    local_polynomial = np.polynomial.polynomial.Polynomial(coefficients)
    return local_polynomial(angular_position - spline_breaks[i - 1])


def in_patch(position, patch_center, spline_breaks, spline_coefs):
    """
    returns True if position = [x,y] is inside the patch
    YET TO IMPLEMENT: uses general parameter radial_tolerance: the worm is still considered inside the patch when its center is sticking out by that distance or less
    """
    # Compute radial coordinates
    # Compute the angle in radians measured counterclockwise from the positive x-axis, returns the angle t in the range (-π, π].
    angular_position = np.arctan2((position[1] - patch_center[1]), (position[0] - patch_center[0]))
    distance_from_center = np.sqrt((position[0] - patch_center[0]) ** 2 + (position[1] - patch_center[1]) ** 2)

    # Compute the local radius of the patch spline
    local_radius = spline_value(angular_position, spline_breaks, spline_coefs)

    return distance_from_center < local_radius


def in_patch_silhouette(silhouette_x, silhouette_y, patch_map):
    """
    Takes a list of x and y coordinates for the worm in one frame, and the label matrix outputted by in_patch_all_pixels.
    If any pixel of worm silhouette is inside a food patch, returns the number of that food patch.
    Otherwise, returns -1.
    """
    # First check if rectangle in which the worm is inscribed intersects with the patch
    min_x, max_x, min_y, max_y = np.min(silhouette_x), np.max(silhouette_x), np.min(silhouette_y), np.max(silhouette_y)
    nb_of_corners_inside_patch = 0
    for corner in [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]:
        # To call the corresponding patch map pixel, we inverse x and y coordinates, because y = line and x = column
        current_corner_patch = patch_map[corner[1], corner[0]]
        if current_corner_patch != -1:
            nb_of_corners_inside_patch += 1
        # If at least two corners are inside, we can safely say that it is inside (?? I think ??)
        if nb_of_corners_inside_patch == 2:
            return int(current_corner_patch)
    # If no corner of the rectangle hangs inside the food patch, then we can safely say that the worm is not inside
    if nb_of_corners_inside_patch == 0:
        return -1

    # In case there is only one corner inside, then... well check the whole silhouette for any point inside
    i_point = 0
    # note: put i_point condition first to avoid list index out of range bugs (python checks conditions sequentially)
    while i_point < len(silhouette_y) and patch_map[silhouette_y[i_point], silhouette_x[i_point]] == -1:
        i_point += 1
    if i_point == len(silhouette_y):  # if the while went all the way through the list, there was no point inside
        return -1
    else:  # otherwise it stopped before and one point was inside
        return int(patch_map[silhouette_y[i_point], silhouette_x[i_point]])


def map_from_boundaries(folder, boundary_position_list, patch_centers):
    # Load image size
    _, _, frame_size = fd.load_silhouette(folder)

    # Table that will be filled with a number indicating whether a point is outside patches (-1) or inside (patch nb)
    plate_map = -1 * np.ones((frame_size[0], frame_size[1]))  # for now only -1

    # Keep only unique boundary positions, and then sort them (pixels read in occidental reading order)
    boundary_position_list = list(set(boundary_position_list))  # shady method to keep only unique tuples
    boundary_position_list = sorted(boundary_position_list, key=lambda b: b[0])
    boundary_position_list = sorted(boundary_position_list, key=lambda b: b[1])
    # For each line and each patch, keep only the min and max points (to have a single boundary line)
    cleaned_boundaries_list = []
    for line in range(frame_size[1]):
        if boundary_position_list:  # while there are points to look at!
            # Find the points that sit on the current line
            current_line_nb = boundary_position_list[-1][1]
            current_line = []
            while boundary_position_list and boundary_position_list[-1][1] == current_line_nb:
                current_line.append(boundary_position_list.pop())  # remove them from boundary_position_list on the go
            for i_patch in range(len(patch_centers)):
                mini = 10000000
                maxi = 0
                for point in current_line:  # for each point of the line
                    if point[2] == i_patch:
                        if point[0] < mini:
                            mini = point[0]
                        if point[0] > maxi:
                            maxi = point[0]
                if mini != 10000000 and maxi != 0:
                    cleaned_boundaries_list.append((mini, current_line_nb, i_patch))
                    cleaned_boundaries_list.append((maxi, current_line_nb, i_patch))
    # Re-sort it because
    cleaned_boundaries_list = sorted(cleaned_boundaries_list, key=lambda b: b[0])
    cleaned_boundaries_list = sorted(cleaned_boundaries_list, key=lambda b: b[1])

    # Color the map :D
    # In this algorithm, we go through each line, and fill the map depending on the boundaries that we encounter.
    # When we encounter a boundary, it means that we either enter or exit a patch.
    # In order to know whether we should do one or the other, we record in the :open_patch: variable the patch that is
    # currently open (if open_patch is -1, all patches are currently closed).
    # If we encounter a boundary while open_patch is -1, we are entering, so we set all values of the line after the
    # boundary to the patch value.
    # If we encounter a boundary while open_patch is not -1, we are exiting the patch, so we set all values of the line
    # after the current boundary to -1.
    open_patch = -1
    # Bool that we set to True if there are overlapping patches (which shouldn't happen + is badly handled by algorithm)
    is_bad = False
    for boundary in cleaned_boundaries_list:
        current_patch = boundary[2]  # patch number of the patch we're looking at
        if open_patch == -1:  # if the current open_patch is -1, we should open the patch
            plate_map[boundary[1]][
            boundary[0]:] = current_patch  # on the matrix line y, fill all points after x with patch nb
            open_patch = current_patch
        elif open_patch == current_patch:  # if the current open_patch is the current patch, we should close the patch
            plate_map[boundary[1]][boundary[0]:] = -1
            open_patch = -1
        else:  # any other case would be a bug (trying to open a new patch while a different one was not closed)
            is_bad = True
            # Still handle this case, which happens because of bad patch tracking
            # Open the current patch even if previous one was not closed
            plate_map[boundary[1]][boundary[0]:] = current_patch
            open_patch = current_patch

    # Additional loop to check if the folder has 4 reference points: if it doesn't it should be excluded
    # In order to do that, load the position of the four reference points at each corner of the plate
    source_folder_metadata = fd.folder_to_metadata(folder)
    source_xy_holes = source_folder_metadata["holes"][0]
    source_reference_points = ReferencePoints.ReferencePoints(source_xy_holes)
    if len(source_reference_points.xy_holes) < 4:
        is_bad = True

    return plate_map, is_bad


def in_patch_all_pixels(folder):
    """
    Will return a table containing the patch in which each pixel of the video from folder is.
    (so if line 5 column 7 there is a -1, it means the point with x=7, y=5 is outside any food patch)
    """
    # In order to avoid having to classify all pixels one by one, we use an algorithm that's more fun :-)
    # # First we generate a list of patch boundary points, enough of them so that for each patch there's at least one
    #   of those points for each pixel row (patches are around 80 px in diameter, so let's do 480 to be sure)
    # # Then, we convert these patch boundary points to integers => any pixel crossed by the spline becomes a boundary pixel
    # # Finally, we go through every row of boundary pixels, and we fill the map using parity rules (first boundary px
    #   encountered converts rest of the line for its belonging patch, then second boundary px of this patch converts px
    #   after it back to -1 (outside patches)).

    # Load metadata for the plate
    metadata = fd.folder_to_metadata(folder)
    patch_centers = metadata["patch_centers"]
    patch_spline_breaks = metadata["spline_breaks"]
    patch_spline_coefs = metadata["spline_coefs"]

    # List of discrete positions for the patch boundaries
    boundary_position_list = []

    # For each patch
    for i_patch in range(len(patch_centers)):
        # For a range of 480 angular positions
        angular_pos = np.linspace(-np.pi, np.pi, 480)
        radiuses = np.zeros(len(angular_pos))
        # Compute the local spline value for each of those radiuses
        for i_angle in range(len(angular_pos)):
            radiuses[i_angle] = spline_value(angular_pos[i_angle], patch_spline_breaks[i_patch],
                                             patch_spline_coefs[i_patch])
        # Add to position list discrete (int) cartesian positions
        # (due to discretization, positions will be added multiple times, but we don't care)
        for point in range(len(angular_pos)):
            x = int(patch_centers[i_patch][0] + (radiuses[point] * np.cos(angular_pos[point])))
            y = int(patch_centers[i_patch][1] + (radiuses[point] * np.sin(angular_pos[point])))
            boundary_position_list.append((x, y, i_patch))  # 3rd tuple element is patch number

    # Compute the plate map, which also returns the plates with bad patches (overlapping patches or missing ref points)
    plate_map, is_bad = map_from_boundaries(folder, boundary_position_list, patch_centers)
    pd.DataFrame(plate_map).to_csv(folder[:-len(folder.split("/")[-1])] + "in_patch_matrix.csv", index=False)

    return plate_map, is_bad


def in_patch_list(traj, using):
    """
    Function that takes in our trajectories dataframe, and returns a column with the patch where the worm is at
    each time step.
    :using: if equal to "centroid", worm will be considered to be inside a food patch if its centroid is inside.
            if equal to "silhouette", worm will be considered to be inside a food patch if any of its pixels is inside.
    """
    list_of_plates = pd.unique(traj["folder"])
    nb_of_plates = len(list_of_plates)

    # List where we'll put the patch where the worm is at each timestep
    list_of_patches = [-1 for i in range(len(traj["x"]))]
    i = 0  # global counter, because we output a single list for all the plates in the trajectory

    # List where we record plates that have bad patch tracking
    is_bad = pd.DataFrame()
    is_bad["overlapping_patches"] = [False for _ in range(nb_of_plates)]
    is_bad["folder"] = list_of_plates

    for i_plate, current_plate in enumerate(list_of_plates):  # for every plate
        # Handmade progress bar
        if param.verbose and (i_plate % 20 == 0 or i_plate == nb_of_plates):
            print("Computing patch_position_" + using + " for plate ", i_plate, "/", nb_of_plates)

        # Extract worm positions
        current_data = traj[traj["folder"] == current_plate].reset_index(drop=True)
        list_x_centroid = current_data["x"]
        list_y_centroid = current_data["y"]
        current_silhouettes, _, frame_size = fd.load_silhouette(current_plate)
        if using == "silhouette":
            current_silhouettes = fd.reindex_silhouette(current_silhouettes, frame_size)
            list_x_silhouette = [[] for _ in range(len(current_silhouettes))]
            list_y_silhouette = [[] for _ in range(len(current_silhouettes))]
            for i_frame in range(len(current_data)):
                list_x_silhouette[i_frame] = current_silhouettes[i_frame][0]
                list_y_silhouette[i_frame] = current_silhouettes[i_frame][1]

        # Generate pixel map of the plate
        in_patch_map, is_bad.loc[i_plate, "overlapping_patches"] = in_patch_all_pixels(current_plate)

        # Fill the table
        # We go through the whole trajectory
        # and register for each time step the patch that contains centroid / intersects with silhouette
        for time in range(len(list_x_centroid)):
            if using == "centroid":
                list_of_patches[i] = int(in_patch_map[int(np.clip(list_x_centroid[time], 0, len(in_patch_map[0]) - 1))][
                                             int(np.clip(list_y_centroid[time], 0, len(in_patch_map) - 1))])
            if using == "silhouette":
                if list_x_silhouette[time]:
                    list_of_patches[i] = in_patch_silhouette(list_x_silhouette[time], list_y_silhouette[time],
                                                             in_patch_map)
            i = i + 1

    return list_of_patches, is_bad


def trajectory_distances(traj):
    """
    Function that takes in our trajectories dataframe, and returns a column with the distance covered by the worm since
    last timestep, for each time step. It should put 0 for the first point of every folder, but record distance even
    when there's a tracking hole.
    """
    # Slice the traj file depending on the folder, because we only want to compare one worm to itself
    # We use pd.unique because it doesn't sort outputs, otherwise the output won't be a column that we can add to traj
    folder_list = pd.unique(traj["folder"])
    list_of_distances = []

    # For each folder, array operation to compute distance
    for folder in folder_list:
        current_traj = traj[traj["folder"] == folder].reset_index()

        # Generate shifted versions of our position columns, either shifted leftwards or rightwards
        array_x_r = np.array(current_traj["x"].iloc[1:])
        array_y_r = np.array(current_traj["y"].iloc[1:])
        array_x_l = np.array(current_traj["x"].iloc[:-1])
        array_y_l = np.array(current_traj["y"].iloc[:-1])
        # Do the computation
        current_list_of_distances = np.sqrt((array_x_l - array_x_r) ** 2 + (array_y_l - array_y_r) ** 2)
        # Add 0 in the beginning because the first point has no speed
        current_list_of_distances = np.insert(current_list_of_distances, 0, 0)
        # Fill the lists that are to be returned
        list_of_distances += list(current_list_of_distances)

    return list_of_distances


def trajectory_speeds(traj):
    """
    Requires the distances to have been computed!
    Will compute speeds from the distances and frame numbers in the trajectories columns
    """
    # Slice the traj file depending on the folder, because we only want to compare one worm to itself
    # We use pd.unique because it doesn't sort outputs, otherwise the output won't be a column that we can add to traj
    folder_list = pd.unique(traj["folder"])
    list_of_speeds = []
    # For each folder, array operation to compute speed
    # NOTE: distance and speed should be equal almost all the time, exceptions = points where the tracking is
    #       interrupted, because there can be multiple frames between to consecutive tracked positions
    for folder in folder_list:
        current_traj = traj[traj["folder"] == folder].reset_index()
        current_list_of_distances = current_traj["distances"]

        # Generate shifted versions of our position columns, either shifted leftwards or rightwards
        array_frame_r = np.array(current_traj["frame"].iloc[1:])
        array_frame_l = np.array(current_traj["frame"].iloc[:-1])

        # Compute the number of frames elapsed between each two lines
        current_list_of_time_steps = array_frame_r - array_frame_l
        # Add 1 in the beginning because the first point isn't relevant (not zero to avoid division issues)
        current_list_of_time_steps = np.insert(current_list_of_time_steps, 0, 1)

        if param.verbose:
            nb_double_frames = np.count_nonzero(current_list_of_time_steps - np.maximum(current_list_of_time_steps,
                                                                                        0.1 * np.ones(
                                                                                            len(current_list_of_time_steps))))
            if nb_double_frames > 0:
                print("number of double frames:", str(nb_double_frames))

        # Remove the zeros and replace them by 0.1
        current_list_of_time_steps = np.maximum(current_list_of_time_steps,
                                                0.1 * np.ones(len(current_list_of_time_steps)))

        # Compute speeds
        list_of_speeds += list(current_list_of_distances / current_list_of_time_steps)

    return list_of_speeds


def smooth_trajectory(trajectory, radius):
    """
    Function that takes a trajectory and samples it spatially with a certain radius: the worm is only considered to have
    moved to a different position if its displacement since last trajectory point is > radius.
    @param trajectory: a trajectory in a dataframe format, with an "x" column and a "y" column
    @param radius: a number
    @return: a new trajectory with fewer points, resampled as described above
    """
    trajectory = trajectory.reset_index(drop=True)
    # Create a column with whether this time point was "smoothed out" (its coordinate were changed during smoothing)
    trajectory["is_smoothed"] = np.array([False for _ in range(len(trajectory))])
    current_circle_center_x = trajectory["x"][0]
    current_circle_center_y = trajectory["y"][0]
    for i_time in range(1, len(trajectory)):
        if i_time % 100000 == 0:
            print("Computing which time points should be smoothed ", i_time, " / ", len(trajectory))
        current_x = trajectory["x"][i_time]
        current_y = trajectory["y"][i_time]
        # If the point is far enough, it's kept in the trajectory, and we set it as the reference center for the next point
        if np.sqrt((current_x - current_circle_center_x) ** 2 + (current_y - current_circle_center_y) ** 2) > radius:
            current_circle_center_x = current_x
            current_circle_center_y = current_y
        else:
            # trajectory["x"][i_time] = np.nan
            trajectory.loc[i_time, "is_smoothed"] = True  # we mark as True points that are not far enough

    # Make a copy of pre-smoothing x and y coordinates, for verification purposes
    trajectory["pre_smoothing_x"] = copy.copy(trajectory["x"])
    trajectory["pre_smoothing_y"] = copy.copy(trajectory["y"])

    # At this point we have the trajectory with points that have to be smoothed out marked as True in "is_smoothed" col
    # For each of those points, we interpolate linearly between the previous and next False points to redefine them
    # So if in three points A, B, C, B has to be smoothed out, their new coordinates become
    # A: (xA, yA), B: ((xA+xC)/2, (yA+yC)/2), C: (xC, yC)
    false_indices = np.where(trajectory["is_smoothed"] == False)[0]
    for i_gap in range(len(false_indices) - 1):
        if i_gap % 50000 == 0:
            print("Smoothing time point ", i_gap, " / ", len(false_indices))
        previous_false_index = false_indices[i_gap]
        next_false_index = false_indices[i_gap + 1]
        nb_of_points_to_smooth = next_false_index - previous_false_index - 1
        # If smoothing needs to be done, define coordinates of the points between which to smooth
        if nb_of_points_to_smooth > 0:
            x1 = trajectory.loc[previous_false_index, "x"]
            y1 = trajectory.loc[previous_false_index, "y"]
            x2 = trajectory.loc[next_false_index, "x"]
            y2 = trajectory.loc[next_false_index, "y"]
        # Loop runs only if there is at least one index between the two current false indices
        for i_point in range(previous_false_index + 1, next_false_index):
            # We interpolate linearly between the two points
            # i_point is the index of the current point being smoothed, and its value depends on its rank among the
            # points to smooth (so if we smooth between index 20 and 24, we will smooth point 21, 22 and 23, and their
            # new coordinates are based on the fact that they are the 1st, 2nd and 3rd points on the interpolation).
            rank_of_point = i_point - previous_false_index
            trajectory.loc[i_point, "x"] = x1 + rank_of_point * (x2 - x1) / (nb_of_points_to_smooth + 1)
            trajectory.loc[i_point, "y"] = y1 + rank_of_point * (y2 - y1) / (nb_of_points_to_smooth + 1)

    return trajectory


def plot_smoothed_traj(traj, t1, t2, radius1, radius2, radius3):
    """
    Crappy function for tests.
    Will take the trajectories.csv folder, and smooth the lines from t1 to t2, with three different smoothing radii.
    It will plot 4 subplots, one with the original trajectory, and the others with smoothing.
    ! Does not work if the folder changes between t1 and t2 !
    """
    traj = traj[t1:t2].reset_index()
    traj_smooth0 = smooth_trajectory(traj, radius1)
    traj_smooth1 = smooth_trajectory(traj, radius2)
    traj_smooth2 = smooth_trajectory(traj, radius3)

    pixels, intensities, frame_size = fd.load_silhouette(traj["folder"][0])
    pixels = fd.reindex_silhouette(pixels, frame_size)

    fig, axs = plt.subplots(2, 2)

    for i in range(t1, t2 - 1, 200):
        axs[0, 0].plot(pixels[i][0], pixels[i][1])
    axs[0, 0].plot(traj["x"], traj["y"], color="black", marker="x")
    axs[0, 0].set_title('Original trajectory, length=' + str(len(traj)))

    for i in range(t1, t2 - 1, 210):
        axs[0, 1].plot(pixels[i][0], pixels[i][1])
    axs[0, 1].plot(traj_smooth0["x"], traj_smooth0["y"], color="black", marker="x")
    axs[0, 1].set_title('Smoothing radius' + str(radius1) + ', length=' + str(len(traj_smooth0)))

    for i in range(t1, t2 - 1, 190):
        axs[1, 0].plot(pixels[i][0], pixels[i][1])
    axs[1, 0].plot(traj_smooth1["x"], traj_smooth1["y"], color="black", marker="x")
    axs[1, 0].set_title('Smoothing radius' + str(radius2) + ', length=' + str(len(traj_smooth1)))

    for i in range(t1, t2 - 1, 215):
        axs[1, 1].plot(pixels[i][0], pixels[i][1])
    axs[1, 1].plot(traj_smooth2["x"], traj_smooth2["y"], color="black", marker="x")
    axs[1, 1].set_title('Smoothing radius' + str(radius3) + ', length=' + str(len(traj_smooth2)))

    plt.show()

# Example call
# path = gr.generate(starting_from="")
# trajectories = pd.read_csv(path + "clean_trajectories.csv")
# results = pd.read_csv(path + "clean_results.csv")
# plot_smoothed_traj(trajectories, 29000, 30000, 2, 4, 6)
