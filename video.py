import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# My code
import find_data as fd
import plots
import generate_results as gr


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


def unravel_index_list(index_list, frame_size):
    unraveled = []
    for i in range(len(index_list)):
        unraveled.append(np.unravel_index(index_list[i], frame_size))
    return unraveled


def show_frames(folder, first_frame):
    """
    Starts by showing first_frame of folder. Then, user can scroll to go through frames.
    """
    fig = plt.gcf()
    fig.set_size_inches(15, 15)

    # Plot the background
    composite = plt.imread(folder[:-len('traj.csv')] + "composite_patches.tif")
    plt.imshow(composite)

    # Plot the patches
    global patches_ax
    fig = plt.gcf()
    patches_ax = fig.gca()
    patches_x, patches_y = plots.patches([folder], show_composite=False, is_plot=False)
    for i_patch in range(len(patches_x)):
        patches_ax.plot(patches_x[i_patch], patches_y[i_patch], color="black")

    # Get silhouette and intensity tables, and reindex pixels (from MATLAB linear indexing to (x,y) coordinates)
    pixels, intensities, frame_size = fd.load_silhouette(folder)
    for frame in range(len(pixels)):
        pixels[frame] = unravel_index_list(pixels[frame], frame_size)
    reformatted_pixels = []
    for frame in range(len(pixels)):
        x_list = []
        y_list = []
        for pixel in range(len(pixels[frame])):
            x_list.append(pixels[frame][pixel][0])
            y_list.append(pixels[frame][pixel][1])
        reformatted_pixels.append([x_list, y_list])
    pixels = reformatted_pixels

    # Load centers of mass from the tracking
    center_of_mass = fd.trajmat_to_dataframe([folder])
    # Get patch info
    patch_list = gr.in_patch_list(center_of_mass)

    colors = plt.cm.Greys(np.linspace(0, 1, 256))
    global curr_index
    global xmin, xmax, ymin, ymax
    curr_index = fd.load_index(folder, first_frame)
    xmin, xmax, ymin, ymax = plt.axis()

    global worm_pixels
    global ax
    fig = plt.gcf()
    ax = plt.gca()
    # Removing 60 to the intensities to make the worm lighter
    worm_pixels = ax.scatter(pixels[curr_index][0], pixels[curr_index][1], color=colors[np.array(intensities[curr_index]) - 80], s=1)
    ax.scatter(center_of_mass["x"][curr_index], center_of_mass["y"][curr_index])
    ax.annotate(str(patch_list[curr_index]), [center_of_mass["x"][curr_index] + 10, center_of_mass["y"][curr_index] + 10])
    ax.set_title("Frame: " + str(fd.load_frame(folder, curr_index)))

    # Make the plot scrollable
    def key_event(e):
        global curr_index
        global curr_fig

        if e.button == "up":
            curr_index = curr_index + 1
        elif e.button == "down":
            curr_index = max(0, curr_index - 1)
        else:
            return
        curr_fig = plt.gcf()
        update_frame(folder, colors, curr_index, pixels, intensities, center_of_mass, patch_list)
    fig.canvas.mpl_connect('scroll_event', key_event)

    # Create a slider
    fig = plt.gcf()
    slider_axis = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(slider_axis, 'Index', 0, 10000, valinit=first_frame)
    plt.subplots_adjust(left=0.25, bottom=.2, right=None, top=.9, wspace=.2, hspace=.2)

    def slider_update():
        global curr_index
        global curr_fig

        curr_index = int(frame_slider.val)
        curr_fig = plt.gcf()
        update_frame(folder, colors, curr_index, pixels, intensities, center_of_mass, patch_list)
    frame_slider.on_changed(slider_update)

    plt.show()


def update_frame(folder, colors, index, pixels, intensities, center_of_mass, patch_list):
    global worm_pixels
    global ax
    global patches_ax
    global curr_fig
    global xmin, xmax, ymin, ymax

    worm_pixels.set_offsets([[pixels[index][0][i], pixels[index][1][i]] for i in range(len(pixels[index]))])
    worm_pixels.set_array(colors[np.array(intensities[index]) - 80])
    ax.annotate(str(patch_list[curr_index]), [center_of_mass["x"][curr_index] + 10, center_of_mass["y"][curr_index] + 10])

    curr_x = center_of_mass["x"][curr_index]
    curr_y = center_of_mass["y"][curr_index]
    ax.scatter(curr_x, curr_y, s=4)

    xmin, xmax, ymin, ymax = curr_x - 100, curr_x + 100, curr_y - 100, curr_y + 100

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    patches_ax.set_xlim(xmin, xmax)
    patches_ax.set_ylim(ymin, ymax)
    patches_ax.set_xlim(xmin, xmax)
    patches_ax.set_ylim(ymin, ymax)

    ax.set_title("Frame: " + str(fd.load_frame(folder, index)))

    curr_fig.canvas.draw()


show_frames("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111213_SmallPatches_C1-CAM3/traj.csv", 612)

