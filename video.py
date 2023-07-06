import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# My code
import find_data as fd
import plots


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
    # Plot the background
    composite = plt.imread(folder[:-len('traj.csv')] + "composite_patches.tif")
    plt.imshow(composite)

    # Plot the patches
    patches_x, patches_y = plots.patches([folder], show_composite=False, is_plot=False)
    for i_patch in range(len(patches_x)):
        plt.plot(patches_x[i_patch], patches_y[i_patch], color="white")

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

    colors = plt.cm.Greys(np.linspace(0, 1, 256))
    # now the real code :)
    global curr_pos
    curr_pos = first_frame
    global xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = plt.axis()

    def key_event(e):
        global curr_pos
        global worm_ax

        if e.button == "up":
            curr_pos = curr_pos + 1
        elif e.button == "down":
            curr_pos = max(0, curr_pos - 1)
        else:
            return
        #curr_pos = curr_pos % len(plots)

        curr_fig = plt.gcf()
        worm_ax.cla()
        worm_ax.set_xlim(xmin, xmax)
        worm_ax.set_ylim(ymin, ymax)
        worm_ax.scatter(pixels[curr_pos][0], pixels[curr_pos][1], color=colors[np.array(intensities[curr_pos])-60], s=1)
        worm_ax.set_title("Frame: "+str(curr_pos))
        curr_fig.canvas.draw()

    global worm_ax
    fig = plt.gcf()
    patches_ax = fig.gca()
    worm_ax = patches_ax.twinx()
    worm_ax.set_xlim(xmin, xmax)
    worm_ax.set_ylim(ymin, ymax)
    fig.canvas.mpl_connect('scroll_event', key_event)
    # Removing 60 to the intensities to make the worm lighter
    worm_ax.scatter(pixels[curr_pos][0], pixels[curr_pos][1], color=colors[np.array(intensities[curr_pos])-60], s=1)

    # Set the slider for frequency and amplitude
    fig = plt.gcf()
    slider_axis = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(slider_axis, 'Frame', 0, 10000, valinit=first_frame)
    plt.subplots_adjust(left=0.25, bottom=.2, right=None, top=.9, wspace=.2, hspace=.2)

    # Update() function to change the graph when the
    # slider is in use
    def update(val):
        global curr_pos
        global worm_ax

        curr_pos = int(frame_slider.val)
        curr_fig = plt.gcf()
        worm_ax.cla()
        worm_ax.set_xlim(xmin, xmax)
        worm_ax.set_ylim(ymin, ymax)
        worm_ax.scatter(pixels[curr_pos][0], pixels[curr_pos][1], color=colors[np.array(intensities[curr_pos]) - 60],
                        s=1)
        worm_ax.set_title("Frame: " + str(curr_pos))
        curr_fig.canvas.draw()

    # update function called using on_changed() function
    # for both frequency and amplitude
    frame_slider.on_changed(update)

    plt.show()


# def onclick(event):
#     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           ('double' if event.dblclick else 'single', event.button,
#            event.x, event.y, event.xdata, event.ydata))
#
#
# fig = plt.gcf()
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
#

show_frames("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111213_SmallPatches_C1-CAM4/traj.csv", 1000)
show_frames("/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T112411_SmallPatches_C5-CAM1/traj.csv", 600)

