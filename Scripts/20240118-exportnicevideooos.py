# This is a script to export nice .mp4 sped up videos of the worms in our experiments :-)
# I'm coding this for obvious scientific reasons and not just because it's fun :-))))

import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import time

# My scripts
from main import *
import find_data as fd
from Generating_data_tables import generate_trajectories as gt


# Create a folder to save the videos :3
path = gr.generate(starting_from="")
path_for_videos = path + "video_exports"
if not os.path.isdir(path_for_videos):
    os.mkdir(path_for_videos)


def export_one_video(folder, export_path):
    """
    Takes the path to a folder containing tracking info, and exports a video to export_path.
    """
    # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    pixels, intensities, frame_size = fd.load_silhouette(folder)
    pixels = fd.reindex_silhouette(pixels, frame_size)

    # Define figure
    fig, ax1 = plt.subplots()
    fig.suptitle(folder)
    ax2 = ax1.twinx()
    ax2.set_xlim(0, int(frame_size[0] * 1.15))
    ax2.set_ylim(frame_size[1], 0)
    normalize = mplcolors.Normalize(vmin=0, vmax=3.5)  # colormap normalization for speeds

    # Plot background
    composite = plt.imread(folder[:-len('traj.csv')] + "composite_patches.tif")
    #ax1.imshow(composite, extent=(0, int(frame_size[0] * 1.15), frame_size[1], 0))

    # Plot patches
    patches_x, patches_y = plots.patches([folder], show_composite=False, is_plot=False)
    for i_patch in range(len(patches_x)):
        ax1.plot(patches_x[i_patch], patches_y[i_patch], color="black")

    # Load centers of mass from the tracking
    centers_of_mass = fd.trajcsv_to_dataframe([folder])
    # Get speeds
    centers_of_mass["distances"] = gt.trajectory_distances(centers_of_mass)
    speed_list = gt.trajectory_speeds(centers_of_mass)

    # Define video output
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output_video = cv2.VideoWriter(export_path + "/" + folder[-45:-9] + ".avi", fourcc, 1, (int(frame_size[0]*1.15), frame_size[1]))

    # Define initial lists for plotting purposes
    current_list_x = []
    current_list_y = []
    current_speed_list = []

    # Runtime optim
    t0 = time.time()
    last_time = t0

    step1 = 0
    step2 = 0
    step3 = 0
    step4 = 0

    # For each frame
    for i_frame in range(0, 10000, 50):
        if i_frame % 20 == 0:
            print("Frame ", i_frame, " / ", len(centers_of_mass))

        step1 += time.time() - last_time
        last_time = time.time()

        # Plot the worm in some shade of grey
        ax2.scatter([pixels[i_frame][0]], [pixels[i_frame][1]], color=plt.cm.Greys(np.linspace(0, 1, 256))[80], marker="o", s=1.3)

        step2 += time.time() - last_time
        last_time = time.time()

        # Plot the past "ghost" frames of the center of mass
        # Instead of calling all the past 100 frames, we add the current one, and remove the first when there are > 100
        if len(current_list_x) <= 100:
            current_list_x.append(centers_of_mass["x"][i_frame])
            current_list_y.append(centers_of_mass["y"][i_frame])
            current_speed_list.append(speed_list[i_frame])
        if len(current_list_x) > 100:
            current_list_x = current_list_x[1:] + [centers_of_mass["x"][i_frame]]
            current_list_y = current_list_y[1:] + [centers_of_mass["y"][i_frame]]
            current_speed_list = current_speed_list[1:] + [speed_list[i_frame]]
        ax2.scatter(current_list_x, current_list_y, c=current_speed_list, cmap="hot", norm=normalize, s=1, zorder=1.3)

        step3 += time.time() - last_time
        last_time = time.time()

        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        figure = data.reshape((int(h), int(w), -1))

        figure = cv2.resize(figure, (int(frame_size[0]*1.15), frame_size[1]))
        figure = cv2.cvtColor(figure, cv2.COLOR_RGBA2BGR)  # convert it to BGR for cv2 outputting
        output_video.write(figure)

        plt.cla()
        ax2.set_xlim(0, int(frame_size[0] * 1.15))
        ax2.set_ylim(frame_size[1], 0)

        step4 += time.time() - last_time
        last_time = time.time()

    print(step1)
    print(step2)
    print(step3)
    print(step4)
    output_video.release()


export_one_video("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111213_SmallPatches_C1-CAM1/traj.csv", path_for_videos)
#export_one_video("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111213_SmallPatches_C1-CAM2/traj.csv", path_for_videos)

