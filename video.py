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
    # Define figure and axes
    global top_ax
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    top_ax = fig.gca()

    # Plot the background
    composite = plt.imread(folder[:-len('traj.csv')] + "composite_patches.tif")
    top_ax.imshow(composite)

    # Plot the patches
    fig = plt.gcf()
    top_ax = fig.gca()
    patches_x, patches_y = plots.patches([folder], show_composite=False, is_plot=False)
    for i_patch in range(len(patches_x)):
        top_ax.plot(patches_x[i_patch], patches_y[i_patch], color="black")
    # Plot patch centers
    current_metadata = fd.folder_to_metadata(folder)
    patch_centers = current_metadata["patch_centers"]
    plt.scatter(np.transpose(patch_centers)[0], np.transpose(patch_centers)[1])

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
    centers_of_mass = fd.trajmat_to_dataframe([folder])
    # Get patch info
    patch_list = gr.in_patch_list(centers_of_mass)

    # Make a copy of full image limits, for zoom out purposes
    global img_xmin, img_xmax, img_ymin, img_ymax
    img_xmin, img_xmax, img_ymin, img_ymax = plt.axis()
    # Define axes limits that will be modified to follow the worm as it moves
    global xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = img_xmin, img_xmax, img_ymin, img_ymax

    # Define frame index counter (will be incremented/decremented depending on what user does)
    global curr_index
    curr_index = fd.load_index(folder, first_frame)

    # Initialize the plot objects
    global worm_plot  # worm silhouette
    global center_of_mass_plot
    global center_to_center_line  # a line between the center of the patch and the center of the worm
    worm_plot = top_ax.plot([], [], color=plt.cm.Greys(np.linspace(0, 1, 256))[80], marker="o", linewidth=0)
    center_of_mass_plot = top_ax.scatter([], [], zorder=3, color="orange")
    center_to_center_line = top_ax.plot([], [], color="white")

    update_frame(folder, curr_index, pixels, centers_of_mass, patch_list, patch_centers, current_metadata)

    # Make the plot scrollable
    def key_event(e):
        global curr_index
        if e.button == "up":
            curr_index = curr_index + 1
        elif e.button == "down":
            curr_index = max(0, curr_index - 1)
        else:
            return
        update_frame(folder, curr_index, pixels, centers_of_mass, patch_list, patch_centers, current_metadata)

    fig.canvas.mpl_connect('scroll_event', key_event)

    # Create a slider
    global bottom_ax
    fig = plt.gcf()
    bottom_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(bottom_ax, 'Index', 0, 30000, valinit=first_frame)
    plt.subplots_adjust(left=0.25, bottom=.2, right=None, top=.9, wspace=.2, hspace=.2)

    # Slider update function
    def slider_update(val):
        global curr_index
        global bottom_ax
        curr_index = int(frame_slider.val)
        update_frame(folder, curr_index, pixels, centers_of_mass, patch_list, patch_centers, current_metadata, zoom_out=True)

    # Slider function update call
    frame_slider.on_changed(slider_update)

    plt.show()


def update_frame(folder, index, pixels, centers_of_mass, patch_list, patch_centers, current_metadata, zoom_out=False):
    global worm_plot
    global center_of_mass_plot
    global center_to_center_line
    global top_ax
    global xmin, xmax, ymin, ymax

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
        xmin, xmax, ymin, ymax = curr_x - 100, curr_x + 100, curr_y - 100, curr_y + 100
        top_ax.set_xlim(xmin, xmax)
        top_ax.set_ylim(ymax, ymin)  # min and max values reversed because in our background image y-axis is reversed

    # If worm is in a patch
    if patch_list[curr_index] != -1:
        # Draw a line between the patch center and the worm center of mass
        curr_patch_center = patch_centers[curr_patch]
        curr_worm_pos = [curr_x, curr_y]
        center_to_center_line[0].set_data([curr_patch_center[0], curr_worm_pos[0]], [curr_patch_center[1], curr_worm_pos[1]])

        # As a y-label, put distance of worm to center computed geometrically and spline value
        worm_to_center = np.sqrt((curr_patch_center[0] - curr_worm_pos[0])**2 + (curr_patch_center[1] - curr_worm_pos[1])**2)
        angular_position = np.arctan2((curr_worm_pos[1] - curr_patch_center[1]), (curr_worm_pos[0] - curr_patch_center[0]))
        spline_value = gr.spline_value(angular_position, current_metadata["spline_breaks"][curr_patch], current_metadata["spline_coefs"][curr_patch])
        # Plot spline value
        top_ax.plot(patch_centers[curr_patch][0] + (spline_value * np.cos(angular_position)), patch_centers[curr_patch][1] + (spline_value * np.sin(angular_position)), marker="x", color="yellow")

        top_ax.annotate(str(patch_centers[curr_patch][0])+", "+str(patch_centers[curr_patch][1]), [patch_centers[curr_patch][0], patch_centers[curr_patch][1]])

        top_ax.set_xlabel("Worm_distance: "+str(worm_to_center)+", spline value: "+str(spline_value))

    top_ax.set_title("Frame: " + str(fd.load_frame(folder, index)) + ", patch: " + str(curr_patch)+", worm xy: "+str(curr_x)+", "+str(curr_y))

    curr_fig = plt.gcf()
    curr_fig.canvas.draw()



