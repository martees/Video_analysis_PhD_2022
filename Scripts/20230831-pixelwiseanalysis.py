# The goal of this script is to look at the dynamics of patch depletion by the worms, both spatially and temporally

# In order to do so, this script creates a new analysis file in each video folder: worm_pixels_traj.csv, containing
# info about which pixel was visited at which time frame by the worm.
# I won't centralize it in one file like trajectories.csv because I'm afraid it will be too huge of a file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

import generate_results as gr
import find_data as fd
import plots

def pixelwise_one_traj(traj):
    """
    Function that returns a dataframe with the list of the pixels visited by any part of the worm, at each (tracked) frame.
    (frames where the tracking failed have no info)
    @param traj: traj.csv dataframe, containing (x,y) coordinates for a plate.
    @return: pixelwise.csv dataframe, containing (x,y) coordinates visited by any silhouette pixel, the frame at which
    it was visited, and the patch where it is.
    """
    folder = traj["folder"][0]
    # Get silhouette table, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    silhouettes, _, frame_size = fd.load_silhouette(folder)
    silhouettes = fd.reindex_silhouette(silhouettes, frame_size)

    # Lists to fill the final table
    frame_col = []
    folder_col = []
    patch_col = []
    frame_size_col = []
    x_col = []
    y_col = []

    for time_index in range(len(silhouettes)):
        # Load frame, patch, and compute number of pixels in the silhouette for this frame
        current_frame = traj["frame"][time_index]
        current_patch = traj["patch_silhouette"][time_index]
        current_frame_nb_of_pixels = len(silhouettes[time_index][0])
        # Fill the pixels table accordingly
        frame_col += [current_frame] * current_frame_nb_of_pixels
        folder_col += [folder] * current_frame_nb_of_pixels
        frame_size_col += [frame_size[0]] * current_frame_nb_of_pixels
        x_col += silhouettes[time_index][0]
        y_col += silhouettes[time_index][1]

        # Some work for the patch column
        # We use in_patch_list instead of at the end, like this we can profit from the fact that we have already computed
        # when the worm was outside. It requires pretending the silhouette is a traj hehe.
        if current_patch == -1:
            patch_col += [current_patch] * current_frame_nb_of_pixels
        else:
            curr_silhouette = pd.DataFrame({"x": , "y": ,})
            patch_list = gr.in_patch_list()


    # Our final pixel table should contain the same thing but with actual frame as a column.
    pixels = pd.DataFrame()
    pixels["frame"] = frame_col
    pixels["patch_silhouette"] = patch_col
    pixels["folder"] = folder_col
    pixels["frame_size"] = frame_size_col
    pixels["x"] = x_col
    pixels["y"] = y_col

    return pixels


def generate_pixelwise(traj):
    """
    Takes a trajectories.csv dataframe as outputted by generate_results (containing xy coordinates for all worms), and
    saves in each worm folder the corresponding pixelwise table.
    @param traj: full trajectories.csv dataframe
    @return: saves its output
    """
    folder_list = pd.unique(traj["folder"])
    for i_folder in range(len(folder_list)):
        print("Generating pixelwise trajectory for folder ", i_folder," / ", len(folder_list))
        folder = folder_list[i_folder]
        current_traj = traj[traj["folder"] == folder].reset_index()
        pixelwise = pixelwise_one_traj(current_traj)
        pixelwise.to_csv(folder[:-len("traj.csv")]+"worm_pixels_traj.csv")


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
    color_array = plt.get_cmap('hot_r')(range(nb_of_colors))
    # change alpha values (one for every value except the last)
    color_array[:, -1] = np.append(np.ones(nb_of_colors - 1), 0)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='hot_alpha', colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

    # Show background and patches
    # patches_x, patches_y = plots.patches(folder, is_plot=False)
    # plt.scatter(patches_x, patches_y, color="blue", s=2)
    composite = plt.imread(fd.load_image_path(folder, "composite_patches.tif"))
    plt.imshow(composite)

    # Show depletion
    dep = plt.imshow(np.transpose(depletion_matrix), cmap='hot_alpha')
    # # Show depletion EVERYWHEEERE
    # for i_x in range(len(depletion_matrix)):
    #     for i_y in range(len(depletion_matrix[i_x])):
    #         plt.scatter(i_x, i_y, color="yellow", alpha=depletion_matrix[i_x][i_y])
    plt.colorbar(dep, cmap="hot_alpha")
    plt.show()


def patch_depletion_evolution(folder, nb_of_frames, depletion_rate):
    # Load pixelwise trajectory from current folder
    pixels_path = folder[:-len("traj.csv")] + "worm_pixels_traj.csv"
    if not os.path.isfile(pixels_path):  # Error message
        print(
            "Pixelwise trajectory file has not been generated yet. Run generate_pixelwise function in the current path.")
    pixels = pd.read_csv(pixels_path)

    visited_patch_list = pd.unique(pixels["patch_silhouette"])
    colors = plt.cm.jet(np.linspace(0,1, len(visited_patch_list)))

    # depletion_curve = [np.ones(nb_of_frames) for _ in range(len(visited_patch_list))]  # one curve for each visited patch
    # for i_patch in visited_patch_list:
    #     patch = visited_patch_list[i_patch]
    #     visited_pixels = pixels[pixels["patch"] == patch]
    #     for frame in visited_pixels["frame"]:  # for each frame with a visit, deplete all subsequent points in the curve
    #         nb_of_pixels_visited = len(visited_pixels[visited_pixels["frame"] == frame])
    #         depletion_curve[i_patch][:-frame] -= nb_of_pixels_visited * depletion_rate

    for i_patch in range(len(visited_patch_list)):
        depletion_curve = []
        patch = visited_patch_list[i_patch]
        visited_pixels = pixels[pixels["patch_silhouette"] == patch]
        list_of_visit_frames = pd.unique(visited_pixels["frame"])
        depletion_value = 1
        for frame in range(nb_of_frames):  # for each frame with a visit, reduce the depletion value of the patch
            if frame in list_of_visit_frames:
                nb_of_pixels_visited = len(visited_pixels[visited_pixels["frame"] == frame])
                depletion_value -= nb_of_pixels_visited * depletion_rate
            depletion_curve.append(depletion_value)
        plt.plot(range(nb_of_frames), depletion_curve, color=colors[i_patch], label=str(patch))

    plt.title("Plate "+folder[-48:-9]+ ", depletion_rate = "+str(depletion_rate))
    plt.ylabel("Food level in each patch")
    plt.xlabel("Frame")
    plt.legend()
    plt.show()

path = gr.generate(test_pipeline=True)
trajectories = pd.read_csv(path + "clean_trajectories.csv")
# Run this line only to regenerate all pixelwise tables (I guess it might take a while)
#generate_pixelwise(trajectories)

#plot_depletion("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_subset_for_tests/20221011T111213_SmallPatches_C1-CAM2/traj.csv",100000, 0.001)
patch_depletion_evolution("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_subset_for_tests/20221011T111213_SmallPatches_C1-CAM2/traj.csv", 33000, 0.00001)


















