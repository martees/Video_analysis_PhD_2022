import os
import ReferencePoints
import scipy

# My code
from param import *


def generate_controls(folder):
    """
    Will take the folder path of a control condition, check that it is indeed one, and then make a foodpatches_control.csv
    file containing the patch centers and splines
    """

    # Define folders
    general_folder = 'H:\\Results_minipatches_20221108'
    folders_dir = os.listdir(general_folder)
    folders = [os.path.join(general_folder, folder) for folder in folders_dir]

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
                    if len(refPoints.error.errorList) == 0:
                        xy_code = refPoints.pixel2mm(pointList[pointList[:, 2] == 1, 0:2])
                        code = classCode(xy_code, 'n_row_col', rowcol_code)
                        if len(code.error.errorList) == 0 and (date_exp != '20221011' or (code.num != 3 and code.num != 7)):
                            centers_patches = nb_to_xy[code.num + 1]
                            centers_patches = refPoints.mm2pixel(centers_patches)
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