# In this script, I want to look at the number of explored pixels during each visit to a food patch

from Generating_data_tables import main as gen
from Generating_data_tables import generate_results as gr
from Parameters import parameters as param

import matplotlib.pyplot as plt
import time
import find_data as fd
import pandas as pd
import os
import numpy as np


def nb_of_pixels_explored(trajectories, folder_list, per_visit_or_per_patch="per_visit"):
    """
    Function that will return a list of number of pixels explored, either during each visit, or during each patch, in
    each of the folders (one sublist per folder, and inside one value per visit, or per patch).
    @param trajectories: a table with the format of "clean_trajectories.csv". See readme.txt.
    @param folder_list: a list of paths to "traj.csv" files
    @param per_visit_or_per_patch: if == "per_visit", the function will return the number of explored pixels in each visit.
    if == "per_patch", the function will return the number of explored pixels in each patch.
    @return: returns a list of numbers.
    """
    nb_explored_pixels = [[] for _ in range(len(folder_list))]
    for i_folder, folder in enumerate(folder_list):
        print(">>> Folder ", i_folder, " / ", len(folder_list))

        # If it's not already done, compute the pixel visit durations
        pixelwise_visits_path = folder[:-len("traj.csv")] + "pixelwise_visits.npy"
        if not os.path.isfile(pixelwise_visits_path):
            gr.generate_pixelwise_visits(trajectories, folder)
        # In all cases, load it from the .npy file, so that the format is always the same (recalculated or loaded)
        pixel_wise_visits = np.load(pixelwise_visits_path, allow_pickle=True)

        print(">>>>>> Loaded pixel visits ")

        # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
        silhouettes, _, frame_size = fd.load_silhouette(folder)
        silhouettes = fd.reindex_silhouette(silhouettes, frame_size)

        if per_visit_or_per_patch == "per_visit":
            # Remove empty lists, and make it a list of visits (instead of a list of lists)
            pixel_wise_visits = [pixel_wise_visits[i][j]
                                 for i in range(len(pixel_wise_visits))
                                 for j in range(len(pixel_wise_visits[i]))
                                 if len(pixel_wise_visits[i][j]) > 0]
            # Make a list of silhouette pixels for every visit (taking silhouettes from start to end)
            silhouettes_each_visit = [
                silhouettes[pixel_wise_visits[i_pixel][i_visit][0]:pixel_wise_visits[i_pixel][i_visit][1] + 1] for
                i_pixel in range(len(pixel_wise_visits)) for i_visit in range(len(pixel_wise_visits[i_pixel]))]
            # Right now, format is for each visit [[silhouette0], [silhouette1], ...]
            #   with silhouette0 = [x0, x1, ...], [y0, y1, ...]
            # With this loop I merge the silhouettes together to create for each visit a list of x, and a list of y coordinates
            #   So the full table becomes [[[x00 x01 ...], [y00 y01 ...]], [[x10 x11 ...], [y10 y11 ...], ...]
            #   with y01 being the second y coordinate of the silhouettes covered during visit 0
            nb_of_visits = len(silhouettes_each_visit)
            for i_visit in range(nb_of_visits):
                if i_visit % (nb_of_visits // 5) == 0:
                    print(">>>>>>>>> Computing for visit... ", i_visit, " / ", nb_of_visits)
                silhouettes_x_this_patch = [silhouettes_each_visit[i_visit][i_silhouette][0][i_pixel] for i_silhouette
                                            in range(len(silhouettes_each_visit[i_visit])) for i_pixel in
                                            range(len(silhouettes_each_visit[i_visit][i_silhouette][0]))]
                silhouettes_y_this_patch = [silhouettes_each_visit[i_visit][i_silhouette][1][i_pixel] for i_silhouette
                                            in range(len(silhouettes_each_visit[i_visit])) for i_pixel in
                                            range(len(silhouettes_each_visit[i_visit][i_silhouette][1]))]
                nb_of_unique_pixels_this_visit = len(np.unique([silhouettes_x_this_patch, silhouettes_y_this_patch],
                                                               axis=1)[0])
                nb_explored_pixels[i_folder].append(
                    nb_of_unique_pixels_this_visit)  # nb of different xy coordinates visited

        if per_visit_or_per_patch == "per_patch":
            # Load to which patch each pixel belongs
            in_patch_matrix_path = folder[:-len("traj.csv")] + "in_patch_matrix.csv"
            in_patch_matrix = pd.read_csv(in_patch_matrix_path)
            in_patch_matrix = in_patch_matrix.to_numpy()

            nb_of_patches = np.unique(
                in_patch_matrix_path) - 1  # remove 1 because one value (-1, outside) is not a patch
            for i_patch in range(len(nb_of_patches)):
                indexes_this_patch = np.where(in_patch_matrix == i_patch)
                visits_this_patch = pixel_wise_visits[indexes_this_patch]
                # Remove empty lists, and make it a list of visits (instead of a list of lists)
                visits_this_patch = [visits_this_patch[i][j]
                                     for i in range(len(visits_this_patch))
                                     for j in range(len(visits_this_patch[i]))
                                     if len(visits_this_patch[i][j]) > 0]
                # Make a list of silhouette pixels for every visit (taking silhouettes from start to end)
                silhouettes_this_patch = [silhouettes[visits_this_patch[i_visit][0]:visits_this_patch[i_visit][1] + 1]
                                          for i_visit in range(len(visits_this_patch))]
                # Right now, format is for each visit [[silhouette0], [silhouette1], ...]
                #   with silhouette0 = [x0, x1, ...], [y0, y1, ...]
                # With this loop I merge the silhouettes together to create for each visit a list of x, and a list of y coordinates
                #   So the full table becomes [[[x00 x01 ...], [y00 y01 ...]], [[x10 x11 ...], [y10 y11 ...], ...]
                #   with y01 being the second y coordinate of the silhouettes covered during visit 0
                nb_of_visits = len(silhouettes_this_patch)
                for i_visit in range(nb_of_visits):
                    if i_visit % (nb_of_visits // 5) == 0:
                        print(">>>>>>>>> Computing for visit... ", i_visit, " / ", nb_of_visits)
                    silhouettes_x_this_patch += [silhouettes_this_patch[i_visit][i_silhouette][0][i_pixel] for
                                                 i_silhouette in range(len(silhouettes_this_patch[i_visit])) for i_pixel
                                                 in range(len(silhouettes_this_patch[i_visit][i_silhouette][0]))]
                    silhouettes_y_this_patch += [silhouettes_this_patch[i_visit][i_silhouette][1][i_pixel] for
                                                 i_silhouette in range(len(silhouettes_this_patch[i_visit])) for i_pixel
                                                 in range(len(silhouettes_this_patch[i_visit][i_silhouette][1]))]
                nb_of_unique_pixels_this_patch = len(np.unique([silhouettes_x_this_patch, silhouettes_y_this_patch],
                                                               axis=1)[0])
                nb_explored_pixels[i_folder].append(
                    nb_of_unique_pixels_this_patch)  # nb of different xy coordinates visited

    return nb_explored_pixels


def plot_nb_of_explored_pixels(full_folder_list, trajectories, bar_list, bar_names, per_visit_or_per_patch="per_visit"):
    tic = time.time()
    avg_each_folder_each_curve = [[] for _ in range(len(bar_list))]
    avg_each_curve = np.zeros(len(bar_list))
    for i_curve, curve in enumerate(bar_list):
        print(int(time.time() - tic), "s: Curve ", i_curve + 1, " / ", len(bar_list))
        folder_list = fd.return_folders_condition_list(full_folder_list, curve)
        explored_pixels_each_folder = nb_of_pixels_explored(trajectories, folder_list,
                                                            per_visit_or_per_patch=per_visit_or_per_patch)
        avg_each_folder_each_curve[i_curve] = [np.mean(explored_pixels_each_folder[i_folder]) for i_folder in
                                               range(len(explored_pixels_each_folder))]
        avg_each_curve[i_curve] = np.mean(avg_each_folder_each_curve[i_curve])

    print("Total time: ", int((time.time() - tic) // 60), "min")

    # Plotttt
    plt.title("Average nb of pixels explored " + per_visit_or_per_patch + " in " + str(bar_names))
    fig = plt.gcf()
    ax = fig.gca()
    fig.set_size_inches(5, 6)

    plt.ylabel("Average nb of pixels explored " + per_visit_or_per_patch)
    ax.bar(range(len(bar_list)), avg_each_curve,
           color=[param.name_to_color[bar_names[i]] for i in range(len(bar_list))])
    ax.set_xticks(range(len(bar_list)))
    ax.set_xticklabels(bar_names, rotation=45)
    ax.set(xlabel="Condition")

    plt.show()


# Load path and clean_results.csv, because that's where the list of folders we work on is stored
path = gen.generate(test_pipeline=True)
resultos = pd.read_csv(path + "clean_results.csv")
traj = pd.read_csv(path + "clean_trajectories.csv")

# plot_nb_of_explored_pixels(resultos["folder"], traj, [[0, 1, 2, 3], [4, 5, 6, 7], [8], [9, 10], [12, 13, 14, 15]], ["0.2", "0.5", "1.25", "mixed", "control"], "per_visit")
plot_nb_of_explored_pixels(resultos["folder"], traj, [[0], [1], [2]], ["close 0.2", "med 0.2", "far 0.2"], "per_patch")
plot_nb_of_explored_pixels(resultos["folder"], traj, [[4], [5], [6]], ["close 0.5", "med 0.5", "far 0.5"], "per_patch")
