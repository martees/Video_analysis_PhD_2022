import numpy as np
import pandas as pd

from Parameters import parameters as param
import find_data as fd


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


def in_patch_silhouette(silhouette_x, silhouette_y, patch_center, spline_breaks, spline_coefs):
    """
    Takes a list of x and y coordinates for the worm in one frame, and the coordinates of a patch and its spline contour.
    Return True if any pixel of worm silhouette is inside the food patch.
    """
    # First check if rectangle in which the worm is inscribed intersects with the patch
    min_x, max_x, min_y, max_y = np.min(silhouette_x), np.max(silhouette_x), np.min(silhouette_y), np.max(silhouette_y)
    nb_of_corners_inside_patch = 0
    for corner in [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]:
        if in_patch(corner, patch_center, spline_breaks, spline_coefs):
            nb_of_corners_inside_patch += 1
        # If at least two corners are inside, we can safely say that it is inside (?? I think ??)
        if nb_of_corners_inside_patch == 2:
            return True
    # If no corner of the rectangle hangs inside the food patch, then we can safely say that the worm is not inside
    if nb_of_corners_inside_patch == 0:
        return False

    # In case there is only one corner inside, then... well check the whole silhouette for any point inside
    i_point = 0
    # note: put i_point condition first to avoid list index out of range bugs (python checks conditions sequentially)
    while i_point < len(silhouette_y) and not in_patch([silhouette_x[i_point], silhouette_y[i_point]], patch_center,
                                                       spline_breaks,
                                                       spline_coefs):
        i_point += 1
    if i_point == len(silhouette_y):  # if the while went all the way through the list, there was no point inside
        return False
    else:  # otherwise it stopped before and one point was inside
        return True


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
    i = 0  # global counter

    for i_plate in range(nb_of_plates):  # for every plate
        # Handmade progress bar
        if param.verbose and (i_plate % 20 == 0 or i_plate == nb_of_plates):
            print("patch_position_"+using+" for plate ", i_plate, "/", nb_of_plates)
        # print("plate name: "+list_of_plates[i_plate])
        # Extract patch information
        current_plate = list_of_plates[i_plate]
        current_metadata = fd.folder_to_metadata(current_plate)
        patch_centers = current_metadata["patch_centers"]
        patch_spline_breaks = current_metadata["spline_breaks"]
        patch_spline_coefs = current_metadata["spline_coefs"]

        # Extract positions
        current_data = traj[traj["folder"] == current_plate]
        list_x_centroid = current_data["x"]
        list_y_centroid = current_data["y"]
        if using == "silhouette":
            current_silhouettes, _, frame_size = fd.load_silhouette(current_plate)
            current_silhouettes = fd.reindex_silhouette(current_silhouettes, frame_size)
            list_x_silhouette = [[] for _ in range(len(current_silhouettes))]
            list_y_silhouette = [[] for _ in range(len(current_silhouettes))]
            for i_frame in range(len(list_x_silhouette)):
                list_x_silhouette[i_frame] = current_silhouettes[i_frame][0]
                list_y_silhouette[i_frame] = current_silhouettes[i_frame][1]

        # Analyze
        # Here we choose to iterate on time and not on patches, two reasons:
        # First, like that we just have to detect patch changes, and it's mostly bad when worm is outside and all patches have to be checked at each time step
        # Second, worms spend most of their time inside food patches, so it's okay that it's worse outside
        # However, it might as well be terrible because it requires loading patch polynomials over and over again
        patch_where_it_is = -1  # initializing variable with index of patch where the worm currently is
        # We go through the whole trajectory
        for time in range(len(list_x_centroid)):
            if param.verbose and time % 100 == 0:
                print(time, "/", len(list_x_centroid))

            # First we figure out where the worm is
            patch_where_it_was = patch_where_it_is  # index of the patch where it is
            patch_where_it_is = -1  # resetting the variable to "worm is out"

            # Check if there is a silhouette for this time
            if using == "silhouette":
                no_silhouette_this_time = list_x_silhouette[time] == []

            # In case the worm is in the same patch, don't try all the patches (doesn't work if worm is out):
            if patch_where_it_was != -1:
                if using == "centroid" or no_silhouette_this_time:
                    if in_patch([list_x_centroid[time], list_y_centroid[time]], patch_centers[patch_where_it_was],
                                patch_spline_breaks[patch_where_it_was], patch_spline_coefs[patch_where_it_was]):
                        patch_where_it_is = patch_where_it_was
                elif using == "silhouette":
                    if in_patch_silhouette(list_x_silhouette[time], list_y_silhouette[time], patch_centers[patch_where_it_was],
                                           patch_spline_breaks[patch_where_it_was],
                                           patch_spline_coefs[patch_where_it_was]):
                        patch_where_it_is = patch_where_it_was

            # If the worm is out or changed patch, then look for it
            else:
                if using == "centroid" or no_silhouette_this_time:
                    for i_patch in range(len(patch_centers)):  # for every patch
                        if in_patch([list_x_centroid[time], list_y_centroid[time]], patch_centers[i_patch], patch_spline_breaks[i_patch],
                                    patch_spline_coefs[i_patch]):  # check if the worm is in it:
                            patch_where_it_is = i_patch  # if it's in it, keep that in mind

                elif using == "silhouette":
                    for i_patch in range(len(patch_centers)):  # for every patch
                        if in_patch_silhouette(list_x_silhouette[time], list_y_silhouette[time], patch_centers[i_patch],
                                               patch_spline_breaks[i_patch],
                                               patch_spline_coefs[i_patch]):  # check if the worm is in it:
                            patch_where_it_is = i_patch  # if it's in it, keep that in mind

            # Update list accordingly
            list_of_patches[i] = patch_where_it_is  # still -1 if patch wasn't found
            i = i + 1
    return list_of_patches


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


