# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from datetime import datetime, date, time
import pandas as pd


def mat_to_pandas(path_of_mat):
    """
    Takes the path of a .mat object, and returns a pandas dataframe containing the same data
    """
    mat = loadmat(path_of_mat)  # load mat-file
    mdata = mat['measuredData']  # variable in mat file
    mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
    # * SciPy reads in structures as structured NumPy arrays of dtype object
    # * The size of the array is the size of the structure array, not the number
    #   elements in any particular field. The shape defaults to 2-dimensional.
    # * For convenience make a dictionary of the data using the names from dtypes
    # * Since the structure has only one element, but is 2-D, index it at [0, 0]
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}
    # Reconstruct the columns of the data table from just the time series
    # Use the number of intervals to test if a field is a column or metadata
    columns = [n for n, v in ndata.iteritems() if v.size == ndata['numIntervals']]
    # now make a data frame, setting the time stamps as the index
    return pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1),
                        index=[datetime(*ts) for ts in ndata['timestamps']], columns=columns)


path = "/home/admin/Desktop/Camera_setup_analysis/"
traj = mat_to_pandas(path + "Tracking_Video/traj.mat")
vt_b = mat_to_pandas(path + "Tracking_Video/traj.mat")
vt_background = mat_to_pandas(path + "Tracking_Video/vt_background.mat")
vt_br = mat_to_pandas(path + "Tracking_Video/vt_br.mat")
vt_mask = mat_to_pandas(path + "Tracking_Video/vt_mask.mat")
vt = mat_to_pandas(path + "Tracking_Video/vt.mat")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/