# This is a script that I start with the intention of connecting more precisely pixel-level depletion (using overlap with
# worm silhouette as a proxy) and instantaneous behavior (looking at whether the worm moves away from a pixel).
import numpy as np
import find_data as fd


def pixel_wise_leaving_delay(folder):
    """
    Function that takes a folder containing a time series of silhouettes, and returns:
        - a list of delay before leaving for every visited pixel
        - a list of corresponding times already spent in each visited pixel
    """
    # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    pixels, intensities, frame_size = fd.load_silhouette(folder)
    pixels = fd.reindex_silhouette(pixels, frame_size)

    # Create a table with a list containing, for each pixel in the image, a sublist with the duration of visits
    # to this pixel. In the following algorithm, when the last element of a sublist is -1, it means that the pixel
    # was not being visited at the previous time point.
    # We start by creating an array with one sublist per pixel, each sublist only containing -1 in the beginning
    visit_times_each_pixel = np.array([[[-1] for _ in range(frame_size[0])] for _ in range(frame_size[1])])
    # For each time point, create visits in pixels that just started being visited, continue those that have already
    # started, and end those that are finished
    for i_time in range(len(pixels)):
        current_visited_pixels = pixels[i_time]
        for i_pixel in range(len(current_visited_pixels)):
            current_pixel = current_visited_pixels[i_pixel]
            # If visit just started, start it
            if visit_times_each_pixel[current_pixel[0], current_pixel[1]][-1] == -1:
                visit_times_each_pixel[current_pixel[0], current_pixel[1]][-1] = 1
            # If visit is continuing, increment time spent
            else:
                visit_times_each_pixel[current_pixel[0], current_pixel[1]][-1] += 1
        # Then, close the visits of the previous time step that are not being continued
        if i_time > 0:
            previous_visited_pixels = pixels[i_time - 1]
            for i_pixel in range(len(previous_visited_pixels)):
                if previous_visited_pixels[i_pixel] not in current_visited_pixels:
                    previous_visited_pixels.append(-1)

    # Remove the -1 because they were only useful for the previous algorithm
    for i_line in range(len(visit_times_each_pixel)):
        for i_column in range(len(visit_times_each_pixel[i_line])):
            if visit_times_each_pixel[i_line, i_column][-1] == -1:
                visit_times_each_pixel[i_line, i_column] = visit_times_each_pixel[i_line, i_column][:-1]

    # Then, go through the list of visits to pixel, and create a list of "delay before leaving" and the corresponding
    # list of "time already spent in pixel"
    delays_before_leaving = []
    time_already_spent_in_pixel = []
    for i_line in range(len(visit_times_each_pixel)):
        for i_column in range(len(visit_times_each_pixel[i_line])):
            current_list_of_visits = visit_times_each_pixel[i_line, i_column]
            # In time already spent in patch we have a range equal to total time spent inside
            # (so if worm spent a visit of length 4, then one of length 2, we get [0, 1, 2, 3, 4, 5])
            time_already_spent_in_pixel.append(range(np.sum(current_list_of_visits)))
            # In delays before leaving, we put a range going from visit length to zero, for each visit
            # (so if worm spent a visit of length 4, then one of length 2, then we get [4, 3, 2, 1, 2, 1])
            delays_before_leaving.append(np.array([range(visit, 0, -1) for visit in current_list_of_visits]))








