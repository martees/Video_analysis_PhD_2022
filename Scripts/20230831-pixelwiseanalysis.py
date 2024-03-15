# The goal of this script is to look at the dynamics of patch depletion by the worms, both spatially and temporally

# In order to do so, this script creates a new analysis file in each video folder: worm_pixels_traj.csv, containing
# info about which pixel was visited at which time frame by the worm.
# I won't centralize it in one file like trajectories.csv because I'm afraid it will be too huge of a file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mplcolors

from Generating_data_tables import main as gen
from Generating_data_tables import generate_trajectories as gt
import find_data as fd


def in_patch_silhouette_each_pixel(silhouette_x, silhouette_y, patch_center, spline_breaks, spline_coefs, in_patch_map):
    """
    Takes a list of x and y coordinates for the worm silhouette in one frame, and the coordinates of a patch and its spline contour.
    Return a list of bools that corresponds to which pixels are inside.
    """
    # We look at intersects between patch boundary and the corners of the rectangle in which the worm is inscribed
    min_x, max_x, min_y, max_y = np.min(silhouette_x), np.max(silhouette_x), np.min(silhouette_y), np.max(silhouette_y)
    nb_of_corners_inside_patch = 0
    for corner in [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]:
        if int(in_patch_map[str(corner[0])][corner[1]]) != -1:
            nb_of_corners_inside_patch += 1
        # If all four corners are inside, we can safely say that all pixels are inside
        if nb_of_corners_inside_patch == 4:
            return [True for _ in range(len(silhouette_x))]
    # There shouldn't be any case where no corner is inside (because we only look at cases so defined)
    if nb_of_corners_inside_patch == 0:
        print("There's a bug in the in_patch_silhouette function in generate_results")

    # In case there is 1, 2 or 3 corners inside, then... well check the whole silhouette hehe
    in_patch_list = []
    for i_point in range(len(silhouette_x)):
        in_patch_list.append(int(in_patch_map[str(silhouette_y[i_point])][silhouette_x[i_point]]))  # line of the matrix is the y (vertical) coord
        # str() around y coord because in_patch_map is now a dataframe, and column names are strings
    return in_patch_list


def pixelwise_one_traj(traj):
    """
    Runs on the data for ONE PLATE.
    Function that returns a dataframe with the list of the pixels visited by any part of the worm, at each (tracked) frame.
    (frames where the tracking failed have no info)
    @param traj: traj.csv dataframe, containing (x,y) coordinates for a plate.
    @return: pixelwise.csv dataframe, containing (x,y) coordinates visited by any silhouette pixel, the frame at which
    it was visited, and the patch where it is.
    """
    # Extract necessary data
    folder = traj["folder"][0]
    current_metadata = fd.folder_to_metadata(folder)
    patch_centers = current_metadata["patch_centers"]
    spline_breaks = current_metadata["spline_breaks"]
    spline_coefs = current_metadata["spline_coefs"]
    in_patch_map = pd.read_csv(folder[:-len("traj.csv")]+"in_patch_matrix.csv")

    # Get silhouette table, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    silhouettes, _, frame_size = fd.load_silhouette(folder)
    silhouettes = fd.reindex_silhouette(silhouettes, frame_size)

    # Lists to fill the final table
    # (I just copy them in a pandas dataframe in the end because I can't figure out how to fill a pandas on the go)
    frame_col = []
    folder_col = []
    patch_col = []
    frame_size_col = []
    x_col = []
    y_col = []

    # For every time step of the video
    for time_index in range(len(silhouettes)):
        # Load frame, patch, and compute number of pixels in the silhouette for this frame
        current_frame = traj["frame"][time_index]
        current_frame_nb_of_pixels = len(silhouettes[time_index][0])
        # Fill the pixels table accordingly
        frame_col += [current_frame] * current_frame_nb_of_pixels
        folder_col += [folder] * current_frame_nb_of_pixels
        frame_size_col += [frame_size[0]] * current_frame_nb_of_pixels
        x_col += silhouettes[time_index][0]
        y_col += silhouettes[time_index][1]

        # Some work for the patch column
        current_patch = traj["patch_silhouette"][time_index]
        # We can profit from the fact that we have already computed when the worm was outside.
        if current_patch == -1:
            patch_col += [current_patch for _ in range(current_frame_nb_of_pixels)]
        # And otherwise we can... suffer? :D
        else:
            # We have a silhouette, and need to classify pixels between those that are inside and those that are outside
            x_list = silhouettes[time_index][0]
            y_list = silhouettes[time_index][1]
            current_patch_center = patch_centers[current_patch]
            current_spline_breaks = spline_breaks[current_patch]
            current_spline_coefs = spline_coefs[current_patch]
            in_patch_each_pixel = in_patch_silhouette_each_pixel(x_list, y_list, current_patch_center, current_spline_breaks, current_spline_coefs, in_patch_map)
            for i in range(len(in_patch_each_pixel)):
                if in_patch_each_pixel[i] == False:
                    patch_col.append(-1)
                else:
                    patch_col.append(current_patch)

    # Our final pixel table should contain the same thing but with actual frame as a column.
    pixels = pd.DataFrame()
    pixels["frame"] = frame_col
    pixels["patch_silhouette"] = patch_col
    pixels["folder"] = folder_col
    pixels["frame_size"] = frame_size_col
    pixels["x"] = x_col
    pixels["y"] = y_col

    return pixels


def generate_pixelwise(traj, do_regenerate=True):
    """
    Takes a trajectories.csv dataframe as outputted by generate_results (containing xy coordinates for all worms), and
    saves in each worm folder the corresponding pixelwise table.
    @param traj: full trajectories.csv dataframe
    @param do_regenerate: if TRUE will regenerate all files. If FALSE will only generate worm_pixels_traj for the folders that don't already have it.
    @return: saves its output
    """
    folder_list = pd.unique(traj["folder"])
    for i_folder in range(len(folder_list)):
        print("Generating pixelwise trajectory for folder ", i_folder," / ", len(folder_list))
        folder = folder_list[i_folder]
        # If we should regenerate existing files, OR if the file is missing, generate it:
        if do_regenerate or not os.path.isfile(folder[:-len("traj.csv")]+"in_patch_matrix.csv"):
            in_patch_all_pixels(folder)
        if do_regenerate or not os.path.isfile(folder[:-len("traj.csv")]+"worm_pixels_traj.csv"):
            current_traj = traj[traj["folder"] == folder].reset_index()
            pixelwise = pixelwise_one_traj(current_traj)
            pixelwise.to_csv(folder[:-len("traj.csv")]+"worm_pixels_traj.csv")


def plot_in_patch_silhouette(folder):
    """
    Just a function to check that the in_patch_silhouette function is working.
    """
    # Load pixelwise trajectory from current folder
    pixels_path = folder[:-len("traj.csv")] + "worm_pixels_traj.csv"
    if not os.path.isfile(pixels_path):  # Error message
        print("Pixelwise trajectory file has not been generated yet. Run generate_pixelwise function in the current path.")
    pixels = pd.read_csv(pixels_path)

    # Show background and patches
    # patches_x, patches_y = plots.patches(folder, is_plot=False)
    # plt.scatter(patches_x, patches_y, color="blue", s=2)
    composite = plt.imread(fd.load_image_path(folder, "composite_patches.tif"))
    plt.imshow(composite)

    pixels_in_patch = pixels[pixels["patch_silhouette"]!=-1]
    pixels_out_of_patch = pixels[pixels["patch_silhouette"]==-1]
    plt.scatter(pixels_in_patch["x"], pixels_in_patch["y"], color="yellow", s=.5)
    plt.scatter(pixels_out_of_patch["x"], pixels_out_of_patch["y"], color="black", s=.5)

    plt.show()


def plot_depletion(folder, frame, depletion_rate):
    """
    Plots the depletion patterns of the patches of a plate at a given frame.
    @param folder: folder in which to analyse
    @param frame: frame at which to stop, if set to 100,000 or more it will just take the whole thing
    @param depletion_rate: how much a pixel is depleted every time a worm pixel visits it (values from 0 to 1)
    """
    # Load pixelwise trajectory from current folder
    pixels_path = folder[:-len("traj.csv")]+"worm_pixels_traj.csv"
    if not os.path.isfile(pixels_path):  # Error message
        print("Pixelwise trajectory file has not been generated yet. Run generate_pixelwise function in the current path.")
    pixels = pd.read_csv(pixels_path)

    # If necessary, slice
    if frame < 100000:
        # Find at which index in the table we need to stop to exclude frames after :frame:
        stop_index = 0
        while pixels["frame"][stop_index] <= frame:  # linear search because dichotomy is tiring (one day I'll write a function)
            stop_index += 1
        pixels = pixels[:-stop_index]  # slice pixel visits to keep only previous frames

    # Create the depletion matrix and go through the pixelwise matrix to empty its pixels
    depletion_matrix = np.ones((pixels["frame_size"][0], pixels["frame_size"][1]))
    for i_pixel in range(len(pixels)):  # for each visited pixel
        if i_pixel % 200000 == 0:
            print("Depleting the matrix for pixel ", i_pixel, " / ", len(pixels))
        depletion_matrix[pixels["x"][i_pixel]][pixels["y"][i_pixel]] -= depletion_rate

    # Create a color map that becomes transparent when depletion value is 1
    nb_of_colors = 256
    color_array = plt.get_cmap('viridis_r')(range(nb_of_colors))
    # change alpha values (one for every value except the last)
    color_array[:, -1] = np.append(np.ones(nb_of_colors - 1), 0)
    # create a colormap object
    map_object = mplcolors.LinearSegmentedColormap.from_list(name='viridis_alpha', colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

    # Create norm to better see depleted places
    normalize = mplcolors.Normalize(vmin=1-100*depletion_rate, vmax=1)

    # Show background and patches
    # patches_x, patches_y = plots.patches(folder, is_plot=False)
    # plt.scatter(patches_x, patches_y, color="blue", s=2)
    composite = plt.imread(fd.load_image_path(folder, "composite_patches.tif"))
    plt.imshow(composite)

    # Show depletion
    dep = plt.imshow(np.transpose(depletion_matrix), cmap='viridis_alpha', norm=normalize)
    # # Show depletion EVERYWHEEERE
    # for i_x in range(len(depletion_matrix)):
    #     for i_y in range(len(depletion_matrix[i_x])):
    #         plt.scatter(i_x, i_y, color="yellow", alpha=depletion_matrix[i_x][i_y])

    plt.title("Plate " + folder[-48:-9] + ", depletion_rate = " + str(depletion_rate))
    plt.colorbar(dep, cmap="viridis_alpha")
    plt.show()


def patch_depletion_evolution(folder, nb_of_frames, depletion_rate):
    """
    Function that plots the level of depletion of the patches visited by the worm, as a function of time.
    IMPORTANT NOTE: This level of depletion is dependent on the number of pixels in each patch, which currently I didn't
    compute. All of this is based on visual estimates for now (5701 pixels per patch).
    @param folder: a folder where generate_pixelwise was already run
    @param nb_of_frames: frame at which to stop
    @param depletion_rate: rate at which each pixel is depleted every time a worm pixel overlaps with it (pixel fullness ranges from 1 to 0)
    @return: plots the overall level of depletion of the patches visited by the worm, as a function of time
    """
    # Badly measured parameter
    nb_of_pixels_per_patch = 5701

    # Load pixelwise trajectory from current folder
    pixels_path = folder[:-len("traj.csv")] + "worm_pixels_traj.csv"
    if not os.path.isfile(pixels_path):  # Error message
        print(
            "Pixelwise trajectory file has not been generated yet. Run generate_pixelwise function in the current path.")
    pixels = pd.read_csv(pixels_path)

    # List of visited patches
    visited_patch_list = pd.unique(pixels["patch_silhouette"])
    # Color map
    colors = plt.cm.jet(np.linspace(0,1, len(visited_patch_list)))

    for i_patch in range(1):
        # Initialization
        patch = visited_patch_list[i_patch]
        depletion_curve = [1]  # curve that we want to fill up (patch starts full, at 1)

        # Retrieve information about the pixels visited by the worm
        pixel_visits = pixels[pixels["patch_silhouette"] == patch]  # pixels that were counted as inside this patch
        pixels_visited = pixel_visits.drop_duplicates()  # list of unique pixels visited by the worm
        nb_of_visited_pixels = len(pixels_visited)
        list_of_visit_frames = pd.unique(pixel_visits["frame"])  # frames at which those visits happened

        # Initialize a column to record depletion level of all visited pixels (one item per unique visited pixel)
        pixels_visited["fullness"] = [1 for _ in range(len(pixels_visited))]

        for frame in range(nb_of_frames):
            if frame % 1000 == 0:
                print("Computing depletion for ", frame," / ", nb_of_frames)
            # For each frame with a visit, reduce the fullness of relevant pixels
            if frame in list_of_visit_frames:
                visited_pixels_this_frame = pixel_visits[pixel_visits["frame"] == frame].reset_index()
                for i in range(len(visited_pixels_this_frame)):
                    pixel = visited_pixels_this_frame.iloc[i,]
                    # This big line is just to adjust the fullness of the right pixel in the pixels_visited dataframe
                    pixels_visited.loc[(pixels_visited["x"]==pixel["x"])&(pixels_visited["y"]==pixel["y"]), "fullness"] -= depletion_rate
                # Set any fullness value < 0 to 0
                pixels_visited["fullness"] = np.where(pixels_visited["fullness"] > 0, pixels_visited["fullness"], 0)
                # Patch fullness is equal to fullness of all pixels (visited + unvisited) divided by total number of pixels
                patch_fullness = (np.sum(pixels_visited["fullness"]) + (nb_of_pixels_per_patch - nb_of_visited_pixels)) / nb_of_pixels_per_patch
                # Update patch depletion accordingly
                depletion_curve.append(patch_fullness)
            else:  # if no new visit, depletion level is constant
                depletion_curve.append(depletion_curve[-1])
        plt.plot(range(nb_of_frames + 1), depletion_curve, color=colors[i_patch], label=str(patch))

    plt.title("Plate "+folder[-48:-9]+ ", depletion_rate = "+str(depletion_rate))
    plt.ylabel("Food level in each patch")
    plt.xlabel("Frame")
    plt.legend()
    plt.show()

# Load existing data
path = gen.generate(test_pipeline=True)
trajectories = pd.read_csv(path + "clean_trajectories.csv")

# Run this line only to regenerate all pixelwise tables (I guess it might take a while)
generate_pixelwise(trajectories, do_regenerate=True)

plot_in_patch_silhouette("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_subset_for_tests/20221011T111213_SmallPatches_C1-CAM3/traj.csv")
#plot_depletion("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221014T191303_SmallPatches_C2-CAM4/traj.csv",100000, 0.01)
#patch_depletion_evolution("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_subset_for_tests/20221011T111213_SmallPatches_C1-CAM3/traj.csv", 33000, 0.001)

#in_patch_all_pixels("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_subset_for_tests/20221011T111213_SmallPatches_C1-CAM3/traj.csv")

















