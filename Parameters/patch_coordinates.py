import numpy as np

# x-y coordinates of the patches in the reference points system
xy_patches_far = [
    [-9.0, -15.59],
    [9.0, -15.59],
    [-18.0, 0.0],
    [0.0, 0.0],
    [18.0, 0.0],
    [-9.0, 15.59],
    [9.0, 15.59]
]
#Invert the y coordinate because images have y axis reversed in matlab. Invert x coordinate because were looking from below.
xy_patches_far = [[xy_patches_far[i][0], -xy_patches_far[i][1]] for i in range(len(xy_patches_far))]

xy_patches_med = [
    [-13.5, -15.59],
    [-4.5, -15.59],
    [4.5, -15.59],
    [13.5, -15.59],
    [-18.0, -7.79],
    [-9.0, -7.79],
    [0.0, -7.79],
    [9.0, -7.79],
    [18.0, -7.79],
    [-22.5, 0.0],
    [-13.5, 0.0],
    [-4.5, 0.0],
    [4.5, 0.0],
    [13.5, 0.0],
    [22.5, 0.0],
    [-18.0, 7.79],
    [-9.0, 7.79],
    [0.0, 7.79],
    [9.0, 7.79],
    [18.0, 7.79],
    [-13.5, 15.59],
    [-4.5, 15.59],
    [4.5, 15.59],
    [13.5, 15.59]
]

#alpha = -15 / 180 * np.pi
#mediumSpaceListOrig = xy_patches_med.copy()
#for iPatch in range(len(xy_patches_med)):
#    xy = xy_patches_med[iPatch]
#    xy_patches_med[iPatch] = [xy[0] * np.cos(alpha) - xy[1] * np.sin(alpha),
#                              xy[0] * np.sin(alpha) + xy[1] * np.cos(alpha)]

#Invert the y coordinate because images have y axis reversed in matlab. Invert x coordinate because were looking from below.
xy_patches_med = [[xy_patches_med[i][0], -xy_patches_med[i][1]] for i in range(len(xy_patches_med))]


xy_patches_close = [
    [-15.75, -11.69],
    [-11.25, -11.69],
    [-6.75, -11.69],
    [-2.25, -11.69],
    [2.25, -11.69],
    [6.75, -11.69],
    [11.25, -11.69],
    [15.75, -11.69],
    [-13.5, -7.79],
    [-9.0, -7.79],
    [-4.5, -7.79],
    [0.0, -7.79],
    [4.5, -7.79],
    [9.0, -7.79],
    [13.5, -7.79],
    [-15.75, -3.90],
    [-11.25, -3.90],
    [-6.75, -3.90],
    [-2.25, -3.90],
    [2.25, -3.90],
    [6.75, -3.90],
    [11.25, -3.90],
    [15.75, -3.90],
    [-13.5, 0.0],
    [-9.0, 0.0],
    [-4.5, 0.0],
    [4.5, 0.0],
    [9.0, 0.0],
    [13.5, 0.0],
    [-15.75, 3.90],
    [-11.25, 3.90],
    [-6.75, 3.90],
    [-2.25, 3.90],
    [2.25, 3.90],
    [6.75, 3.90],
    [11.25, 3.90],
    [15.75, 3.90],
    [-13.5, 7.79],
    [-9.0, 7.79],
    [-4.5, 7.79],
    [0.0, 7.79],
    [4.5, 7.79],
    [9.0, 7.79],
    [13.5, 7.79],
    [-15.75, 11.69],
    [-11.25, 11.69],
    [-6.75, 11.69],
    [-2.25, 11.69],
    [2.25, 11.69],
    [6.75, 11.69],
    [11.25, 11.69],
    [15.75, 11.69]
]
#Invert the y coordinate because images have y-axis reversed in matlab. Invert x coordinate because were looking from below.
xy_patches_close = [[xy_patches_close[i][0], -xy_patches_close[i][1]] for i in range(len(xy_patches_close))]

xy_patches_cluster = [
    [-16.14, -9.6],
    [-12.1, -9.96],
    [-18.94, -6.54],
    [-14.48, -5.68],
    [-20.09, -11.42],
    [-2.71, 12.51],
    [-4.6, 7.81],
    [0.47, 15.37],
    [0.65, 10.3],
    [-6.82, 11.76],
    [5.97, -16.54],
    [7.33, -11.5],
    [9.23, -18.94],
    [2.66, -14.34],
    [5.56, -20.51],
    [17.69, 4.04],
    [15.78, 8.05],
    [20.36, 6.92],
    [13.3, 3.47],
    [16.1, 0.44],
    [-19.21, 8.89],
    [-22.63, 6.53],
    [-15.15, 8.95],
    [-18.35, 5.02]
]
#Invert the y coordinate because images have y axis reversed in matlab. Invert x coordinate because were looking from below.
xy_patches_cluster = [[xy_patches_cluster[i][0], -xy_patches_cluster[i][1]] for i in range(len(xy_patches_cluster))]

mediumSpaceHighDensityMask = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]