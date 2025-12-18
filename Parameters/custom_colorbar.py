import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# Adapted from: https://stackoverflow.com/a/14779462/21414975


def discrete_rainbow(nb_of_rainbow_points, nb_of_grey_points, plot=False):
    # Will take two numbers R and G,
    # will create the colormap and norm necessary to make plots, with values [0;R] that are associated to a color
    # from the rainbow, and values [-G;-1] that are associated to a color picked from a gray scale (lowest = darkest)

    nb_of_points = nb_of_grey_points + nb_of_rainbow_points

    cmap = plt.cm.jet(np.linspace(0, 1, nb_of_rainbow_points+1))  # define the colormap
    # extract needed colors from the .jet map
    rainbow_colors = [cmap[i] for i in range(nb_of_rainbow_points)]
    # force the first color entry to be grey
    grey_colors = []
    for g in range(nb_of_grey_points):
        grey = g * (0.84/nb_of_grey_points)
        grey_colors.append((grey, grey, grey, 1))

    cmaplist = grey_colors + rainbow_colors

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, nb_of_points)

    if plot:
        plt.subplots(1, 1, figsize=(6, 6))  # setup the plot
        x = np.random.rand(nb_of_points)  # define the data
        y = np.random.rand(nb_of_points)  # define the data
        tag = np.random.randint(0, nb_of_points, nb_of_points)
        tag[0:3] = 0  # make sure there are some 0 values to show up as grey
        # make the scatter
        plt.scatter(x, y, c=tag, s=np.random.randint(100, 500, nb_of_points),
                          cmap=cmap, vmin=np.min(tag) - 0.5, vmax=np.max(tag) + 0.5)
        plt.colorbar(ticks=np.arange(np.min(tag), np.max(tag) + 1))
        plt.show()

    return cmap

if __name__ == "__main__":
    discrete_rainbow(10, 2, plot=True)