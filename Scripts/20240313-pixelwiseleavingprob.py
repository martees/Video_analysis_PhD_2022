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

    # Create a table with a list containing -1 for each pixel of the image. In the following algorithm, -1 will mean
    # that there is no ongoing visit in the current pixel
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









