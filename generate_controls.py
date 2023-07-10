import os
import ReferencePoints
import scipy

# My code
from parameters import *
import find_data as fd


def generate_controls(path):
    """
    Takes a path prefix, finds the files of all control conditions inside.
    Will create subfolders each containing:
        - a copy of the traj.csv file from the parent folder
        - a foodpatches.mat folder containing a new condition number
        - a foodpatches_new
    """

    # Full list of paths for the traj.csv files that can be found in the arborescence starting in path
    folder_list = fd.path_finding_traj(path)
    # Select folders that correspond to a control condition (11)
    folder_list = fd.return_folders_condition_list(folder_list, 11)

    for folder in folder_list:
        close_patches = return_control_patches(folder, "close")
        med_patches = return_control_patches(folder, "med")
        far_patches = return_control_patches(folder, "far")
        cluster_patches = return_control_patches(folder, "cluster")


def return_control_patches(folder, distance):
    """
    Will take the folder path of a control condition, check that it is indeed one, and then make a foodpatches_distance.csv
    file containing the patch centers and splines for fake patches distanced by distance.
    For example, foodpatches_close or foodpatches_med.
    """

    for i in range(len(nb_to_xy)):
        nb_to_xy[i] = -nb_to_xy[i]  # Invert the y-coordinate and flip the x-coordinate

    # Densities for each condition
    cond2densities = []
    i_cond = 0
    for density in [0.2, 0.5]:
        for i_grid in range(4):
            i_cond += 1
            cond2densities.append(density * np.ones(nb_to_xy[i_cond].shape[0]))

    i_cond += 1
    cond2densities.append(1.25 * np.ones(nb_to_xy[i_cond].shape[0]))

    i_cond += 1
    density0 = 0.2
    density1 = 0.5
    cond2densities.append(density0 * np.ones(nb_to_xy[i_cond].shape[0]))
    cond2densities[i_cond][mediumSpaceHighDensityMask == 1] = density1

    i_cond += 1
    density0 = 1.25
    density1 = 0.5
    cond2densities.append(density0 * np.ones(nb_to_xy[i_cond].shape[0]))
    cond2densities[i_cond][mediumSpaceHighDensityMask == 1] = density1

    i_cond += 1
    cond2densities.append(np.zeros(nb_to_xy[i_cond].shape[0]))

    # Add the info to each video
    for i_folder in range(len(folders)):
        folder = folders[i_folder]
        last_sep = folder.rfind('\\')
        date_exp = folder[last_sep + 1:last_sep + 9]
        if date_exp != '20221007':  # Exclude the pilot study
            if date_exp == '20221011':  # Day when still using the 3x3 code
                rowcol_code = [3, 3]  # Number of rows and columns of the code
            else:
                rowcol_code = [2, 4]  # Number of rows and columns of the code
            if not os.path.isfile(os.path.join(folder, 'holes.mat')):
                print(folder, 'is missing the holes')
            else:
                data = scipy.io.loadmat(os.path.join(folder, 'holes.mat'))
                pointList = data['pointList']
                if folder[-35:] == '20221012T200743_SmallPatches_C2-CAM6':
                    pointList[-1, 2] = 2  # Correct a mistake manually
                if not np.any(pointList[:, 2] == 4):
                    refPoints = ReferencePoints(pointList[pointList[:, 2] == 2, 0:2], 'side_square_mm', 32)
                    if len(refPoints.errors.errorList) == 0:
                        xy_code = refPoints.pixel_to_mm(pointList[pointList[:, 2] == 1, 0:2])
                        code = classCode(xy_code, 'n_row_col', rowcol_code)
                        if len(code.errors.errorList) == 0 and (date_exp != '20221011' or (code.num != 3 and code.num != 7)):
                            centers_patches = nb_to_xy[code.num + 1]
                            centers_patches = refPoints.mm_to_pixel(centers_patches)
                            densities_patches = cond2densities[code.num + 1]
                            densities_patches = densities_patches[~np.isnan(densities_patches)]
                            if len(densities_patches) > 0:
                                # Process the data further as needed
                                pass
                            else:
                                print('No density values for', folder)
                        else:
                            print('Code error in', folder)
                    else:
                        print('RefPoints error in', folder)
                else:
                    print('PointList error in', folder)
        else:
            print('Pilot study excluded:', folder)
    return 0