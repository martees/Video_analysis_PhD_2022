import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
import pandas as pd

# My code
from Parameters import parameters as param
import find_data as fd
import plots
from Generating_data_tables import generate_trajectories as gt
from Generating_data_tables import main as gr


def show_frame(folder, frame, is_plot=True):
    """
    Should return an array with the worm silhouette at frame "frame" in folder "folder".
    :folder: folder path
    :frame: frame to be shown
    """

    pixels, intensities, frame_size = fd.load_silhouette(folder)

    # Because of tracking holes, frames and traj.csv indexes are not the same
    frame_index = fd.load_frame(folder, frame)
    pixels = pixels[frame_index]
    intensities = intensities[frame_index]

    # For each pixel of the silhouette
    nb_of_pixels = len(pixels)
    silhouette_x = np.zeros(nb_of_pixels)
    silhouette_y = np.zeros(nb_of_pixels)
    for px in range(nb_of_pixels):
        silhouette_x[px], silhouette_y[px] = np.unravel_index(pixels[px], frame_size, order="F")

    if is_plot:
        colors = plt.cm.Greys(np.linspace(0, 1, 256))
        fig = plt.gcf()
        fig.set_size_inches(8, 8)

        for px in range(len(silhouette_y)):
            plt.scatter(silhouette_x[px], silhouette_y[px], color=colors[intensities[px]], s=1)

        plt.show()

    return silhouette_x, silhouette_y, intensities


def show_frames(folder, trajectories, first_frame):
    """
    Starts by showing first_frame of folder. Then, user can scroll to go through frames.
    """
    # Define figure and axes
    global top_ax
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    top_ax = fig.gca()

    # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    pixels, _, frame_size = fd.load_silhouette(folder)
    pixels = fd.reindex_silhouette(pixels, frame_size)

    # Load centers of mass from the tracking
    centers_of_mass = trajectories[trajectories["folder"] == folder].reset_index()

    # Plot the background
    composite = plt.imread(fd.load_file_path(folder, "composite.tif"))
    top_ax.imshow(composite, extent=(0, frame_size[0], frame_size[1], 0))

    # Plot the patches
    fig = plt.gcf()
    top_ax = fig.gca()
    fig.suptitle(folder)

    patches_x, patches_y = plots.patches([folder], show_composite=False, is_plot=False)
    for i_patch in range(len(patches_x)):
        top_ax.plot(patches_x[i_patch], patches_y[i_patch], color="black")
    # Plot patch centers
    current_metadata = fd.folder_to_metadata(folder)
    patch_centers = current_metadata["patch_centers"]
    plt.scatter(np.transpose(patch_centers)[0], np.transpose(patch_centers)[1])

    # Get patch info
    #patch_list = gt.in_patch_list(centers_of_mass, using="silhouette")
    patch_list = centers_of_mass["patch_silhouette"]
    # Get speeds
    centers_of_mass["distances"] = gt.trajectory_distances(centers_of_mass)
    speed_list = gt.trajectory_speeds(centers_of_mass)

    # Make a copy of full image limits, for zoom out purposes
    global img_xmin, img_xmax, img_ymin, img_ymax
    img_xmin, img_xmax, img_ymin, img_ymax = plt.axis()
    # Define axes limits that will be modified to follow the worm as it moves
    global xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = img_xmin, img_xmax, img_ymin, img_ymax

    # Define frame index counter (will be incremented/decremented depending on what user does)
    global curr_index
    curr_index = fd.load_index(trajectories, folder, first_frame)

    # Initialize the plot objects
    global worm_plot  # worm silhouette
    global center_of_mass_plot
    global center_to_center_line  # a line between the center of the patch and the center of the worm
    worm_plot = top_ax.plot([], [], color=plt.cm.Greys(np.linspace(0, 1, 256))[60], marker="o", markersize=5, linewidth=0)
    center_of_mass_plot = top_ax.scatter([], [], zorder=3, color="orange")
    center_to_center_line = top_ax.plot([], [], color="white")
    # Call the update frame once to initialize the plot
    update_frame(trajectories, folder, curr_index, pixels, centers_of_mass, patch_list, speed_list)

    # Make the plot scrollable
    def scroll_event(event):
        global curr_index
        if event.button == "up":
            curr_index = curr_index + 1
        elif event.button == "down":
            curr_index = max(0, curr_index - 1)
        else:
            return
        update_frame(trajectories, folder, curr_index, pixels, centers_of_mass, patch_list, speed_list)

    fig.canvas.mpl_connect('scroll_event', scroll_event)

    # Create a slider
    global bottom_ax
    fig = plt.gcf()
    bottom_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(bottom_ax, 'Index', 0, 30000, valinit=first_frame)
    plt.subplots_adjust(left=0.25, bottom=.2, right=None, top=.9, wspace=.2, hspace=.2)
    # Slider update function

    def slider_update(val):  # val argument gives "unused" warning in PyCharm but is necessary
        global curr_index
        global bottom_ax
        curr_index = int(frame_slider.val)
        update_frame(trajectories, folder, curr_index, pixels, centers_of_mass, patch_list, speed_list, zoom_out=True)

    # Slider function update call
    frame_slider.on_changed(slider_update)

    print("=== INSTRUCTION GUIDE! :-) ===")
    print("In order to move through the trajectory slowly, use the mouse scroll. If it freezes, just wait, it's loading.")
    print("In order to jump to a faraway timestep, use the slider below the graph.")

    plt.show()


def update_frame(trajectories, folder, index, pixels, centers_of_mass, patch_list, speed_list, zoom_out=False):
    """
    Function that is called every time the frame number has to be updated to a new index.
    @param folder: path of the current video
    @param index: new frame number
    @param pixels: dataframe containing one line per frame, and in each frame a list of x,y coordinates for worm silhouette
    @param centers_of_mass: center of mass of the worm
    @param patch_list: list of patches where the worm is at each frame (for checking purposes, displayed in title)
    @param speed_list: list of speeds at which the worm is going at each frame (for checking purposes, displayed as x legend)
    @param zoom_out: if True, xlim and ylim will be set to image limits, otherwise narrow around worm
    """
    global worm_plot
    global center_of_mass_plot
    global center_to_center_line
    global top_ax
    global xmin, xmax, ymin, ymax

    print(len(pixels[index][0]))
    if len(pixels[index][0]) > param.invalid_silhouette_threshold:
        worm_plot[0].set_color("red")
    else:
        worm_plot[0].set_color("gray")
    worm_plot[0].set_data([pixels[index][0]], [pixels[index][1]])
    center_of_mass_plot.set_offsets([centers_of_mass["x"][index], centers_of_mass["y"][index]])

    curr_x = centers_of_mass["x"][index]
    curr_y = centers_of_mass["y"][index]
    curr_patch = patch_list[index]
    top_ax.scatter(curr_x, curr_y, s=4)

    if zoom_out:
        top_ax.set_xlim(img_xmin, img_xmax)
        top_ax.set_ylim(img_ymin, img_ymax)
    else:
        xmin, xmax, ymin, ymax = curr_x - 30, curr_x + 30, curr_y - 30, curr_y + 30
        top_ax.set_xlim(xmin, xmax)
        top_ax.set_ylim(ymax, ymin)  # min and max values reversed because in our background image y-axis is reversed

    # Write as x label the speed of the worm
    top_ax.set_xlabel("Speed of the worm: "+str(speed_list[index]))

    top_ax.set_title("Frame: " + str(fd.load_frame(trajectories, folder, index)) + ", patch: " + str(curr_patch)+", worm xy: "+str(curr_x)+", "+str(curr_y))

    curr_fig = plt.gcf()
    curr_fig.canvas.draw()


if __name__ == "__main__":
    path = gr.generate(starting_from="", test_pipeline=False)
    traj = pd.read_csv(path + "clean_trajectories.csv")
    show_frames(path + '20221011T191645_SmallPatches_C3-CAM7/traj.csv', traj, 31917)
