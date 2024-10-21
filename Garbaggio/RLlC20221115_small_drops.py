# coding: utf-8

# Core code updates: 
# - Dec 7, 2021: Fix multi-transfers, and add report about time spent in each type of process
# - Jan 18, 2022: Sort drops from the same source by volume, so that multitransfers work well even when we mix different volume
# - Jul 13, 2022: Change protocol so that each condition always falls at the same place, and also to put the drops in two different days (food the first day, salt the second)
# - Oct 8, 2022: Add labware definition for when the plates are in the corners of the space
# - Oct 10, 2022: Add possibility of having several different drop patterns

# # 1. Imports and general definitions
# If you touch this, you will probably die

# In[ ]:


#from opentrons import labware, instruments, robot
import time
import numpy as np
import copy
import numbers
import math
import datetime
import pickle


def triangular_grid(origin, size, distance, shifted_center=True, patch_in_center=False):
    # arguments:
    # origin: [x,y] coordinates of the first point, central to the grid
    # size: maximal vertical and horizontal distance to origin
    # distance: distance between points of the grid / length of triangle side
    # rotation: clockwise rotation in degrees
    #
    # returns:
    # list of [x,y] coordinates for points of a triangular grid
    # respecting the constraints of the arguments

    list_of_coordinates = []  # that's like the final output
    # rotated_size = size / math.cos(rotation)  # that's the maximal offset from origin once you rotate the grid
    # rotated_distance = distance * math.cos(rotation)
    # size being the vertical and horizontal distance to origin

    # First we build a list of y coordinates for the rows
    nb_of_rows = int(size // distance) + 2  # number of rows you can fit in the grid ABOVE or BELOW origin
    distance_between_rows = (math.tan(
        math.radians(60)) * distance) / 2  # geometry for row distance, height of equilateral triangle
    # Fit rows above then below
    list_y_rows_above = [origin[1] + i * distance_between_rows for i in
                         range(nb_of_rows + 1)]  # adding one more in case it fits
    list_y_rows_below = [origin[1] - i * distance_between_rows for i in range(1, nb_of_rows + 1)]
    list_y_rows = list_y_rows_above + list_y_rows_below
    if shifted_center:
        print('shifting them centers')
        list_y_rows = [i - distance / (2 * math.sqrt(3)) for i in list_y_rows]
    list_y_rows.sort()

    # Then a list of x coordinates for the columns in even rows (where first point is vertically aligned w/ origin)
    nb_of_col_even = int(size // distance) + 2
    # number of columns you can fit in the grid LEFT or RIGHT of origin
    list_x_col_even_right = [origin[0] + i * distance for i in range(nb_of_col_even)]
    list_x_col_even_left = [origin[0] - i * distance for i in range(1, nb_of_col_even + 1)]
    list_x_col_even = list_x_col_even_left + list_x_col_even_right
    list_x_col_even.sort()

    # Then a list of x coordinates for the column in uneven rows
    nb_of_col_odd = int(
        (size - (distance / 2)) // distance) + 2  # number of columns you can horizontally fit in the grid
    # after shifting it from half a distance (x shift between rows)
    list_x_col_odd_right = [origin[0] + distance / 2 + i * distance for i in range(nb_of_col_odd)]
    list_x_col_odd_left = [origin[0] - distance / 2 - i * distance for i in range(nb_of_col_odd)]
    list_x_col_odd = list_x_col_odd_left + list_x_col_odd_right
    list_x_col_odd.sort()

    # This determines the parity of the center row, where the origin is
    if not patch_in_center:
        center_parity = (nb_of_rows - 3) % 2
    if patch_in_center:
        center_parity = int(not (nb_of_rows - 3) % 2)

    # Then we put everything together and it's great
    # We do check whether points are out of bound to exclude the ones that are out of the square built from origin
    for i_row in range(len(list_y_rows)):
        if i_row % 2 == center_parity:  # for even rows (whose first point is aligned to the origin) that are in fact odd rows xD
            for i_col in range(len(list_x_col_even)):
                current_x = list_x_col_even[i_col]
                current_y = list_y_rows[i_row]
                if abs(current_x - origin[0]) < size and abs(current_y - origin[1]) < size:
                    list_of_coordinates.append([current_x, current_y])
        else:
            for i_col in range(len(list_x_col_odd)):
                current_x = list_x_col_odd[i_col]
                current_y = list_y_rows[i_row]
                if abs(current_x - origin[0]) < size and abs(current_y - origin[1]) < size:
                    list_of_coordinates.append([current_x, current_y])

    return [list(i) for i in np.round(list_of_coordinates, 3)]


spacing = 18
grid = triangular_grid([0, 0], 20, spacing, shifted_center=False, patch_in_center=True)
print(grid)
print(len(grid))
grid2 = triangular_grid([0, 0], 23, spacing / 2, shifted_center=False)
print(grid2)
print(len(grid2))
grid3 = triangular_grid([0, 0], 16, spacing / 4, shifted_center=False, patch_in_center=True)
print(grid3)
print(len(grid3))

# In[ ]:


superDistantSpaceList = [
    [0, 16], [16 * np.cos(30 * np.pi / 180), -16 * np.sin(30 * np.pi / 180)],
    [-16 * np.cos(30 * np.pi / 180), -16 * np.sin(30 * np.pi / 180)]
]
alpha = 40 / 180 * np.pi
superDistantSpaceListOrig = superDistantSpaceList.copy()
for iPatch in range(len(superDistantSpaceList)):
    xy = superDistantSpaceList[iPatch].copy()
    superDistantSpaceList[iPatch] = [xy[0] * math.cos(alpha) - xy[1] * math.sin(alpha),
                                     xy[0] * math.sin(alpha) + xy[1] * math.cos(alpha)]

distantSpaceList = [
    [-9.0, -15.59], [9.0, -15.59],
    [-18.0, 0.0], [0.0, 0.0], [18.0, 0.0],
    [-9.0, 15.59], [9.0, 15.59]]

mediumSpaceList = [
    [-13.5, -15.59], [-4.5, -15.59], [4.5, -15.59], [13.5, -15.59],
    [-18.0, -7.79], [-9.0, -7.79], [0.0, -7.79], [9.0, -7.79], [18.0, -7.79],
    [-22.5, 0.0], [-13.5, 0.0], [-4.5, 0.0], [4.5, 0.0], [13.5, 0.0], [22.5, 0.0],
    [-18.0, 7.79], [-9.0, 7.79], [0.0, 7.79], [9.0, 7.79], [18.0, 7.79],
    [-13.5, 15.59], [-4.5, 15.59], [4.5, 15.59], [13.5, 15.59]]

alpha = -15 / 180 * np.pi
mediumSpaceListOrig = mediumSpaceList.copy()
for iPatch in range(len(mediumSpaceList)):
    xy = mediumSpaceList[iPatch].copy()
    mediumSpaceList[iPatch] = [xy[0] * math.cos(alpha) - xy[1] * math.sin(alpha),
                               xy[0] * math.sin(alpha) + xy[1] * math.cos(alpha)]

#removed points = [-22.5, -15.59], [22.5, -15.59], [-22.5, 15.59], [22.5, 15.59]

mediumSpaceHighDensityMask = [
    0, 1, 0, 1,
    0, 1, 0, 1, 0,
    0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
    0, 1, 0, 1]

closeSpaceList = [
    [-15.75, -11.69], [-11.25, -11.69], [-6.75, -11.69], [-2.25, -11.69], [2.25, -11.69], [6.75, -11.69],
    [11.25, -11.69], [15.75, -11.69],
    [-13.5, -7.79], [-9.0, -7.79], [-4.5, -7.79], [0.0, -7.79], [4.5, -7.79], [9.0, -7.79], [13.5, -7.79],
    [-15.75, -3.90], [-11.25, -3.90], [-6.75, -3.90], [-2.25, -3.90], [2.25, -3.90], [6.75, -3.90], [11.25, -3.90],
    [15.75, -3.90],
    [-13.5, 0.0], [-9.0, 0.0], [-4.5, 0.0], [4.5, 0.0], [9.0, 0.0], [13.5, 0.0],
    [-15.75, 3.90], [-11.25, 3.90], [-6.75, 3.90], [-2.25, 3.90], [2.25, 3.90], [6.75, 3.90], [11.25, 3.90],
    [15.75, 3.90],
    [-13.5, 7.79], [-9.0, 7.79], [-4.5, 7.79], [0.0, 7.79], [4.5, 7.79], [9.0, 7.79], [13.5, 7.79],
    [-15.75, 11.69], [-11.25, 11.69], [-6.75, 11.69], [-2.25, 11.69], [2.25, 11.69], [6.75, 11.69], [11.25, 11.69],
    [15.75, 11.69],
]

# removed points: [-13.5, -15.59], [-9.0, -15.59], [-4.5, -15.59], [0.0, -15.59], [4.5, -15.59], [9.0, -15.59], [13.5, -15.59],
# [-13.5, 15.59], [-9.0, 15.59], [-4.5, 15.59], [0.0, 15.59], [4.5, 15.59], [9.0, 15.59], [13.5, 15.59]


# In[ ]:


# **************
# PIPETTE TO USE
# **************
namePipette = 'P10'

# ******************************
# PLATES AND SUBPLATES AVAILABLE
# ******************************
nPetriPerPlate = 2  # Number of petri dishes that fit in each multiplate
nSemiRound = 1  # Set to 1 if you don't know what this is. If this is greater than 1, the robot will do only part of the petris each time we execute the next cell. So if it's 2, we need to execute twice to do all petris, and so on. The total number of petris must be divisible by this number

# **************************************************
# POSITION FOR EACH SOURCE (SALT, FOOD, OR WHATEVER)
# **************************************************
# Each element of the list sources should be like this: [density,position], where 'density' is the density of the source, and 'position' its position in the robot, in the usual format [plate,subplate,well]
# So for example, [0.1,[10,0,'A1']] means that the source with density 0.1 is at well A1 of the plate at position 10, subposition 0.
# The densities are here so that the robot can always work from low to high density. This is always good, and is very important if we don't change tips between sources
# Densities must have unique numbers, so if you're using different species you need to add some number to distinguish them.
# Redundantly, the third element of each source must be the species name.
sourceList = [[0, [10, 0, 'C1'], 'OP50'],
              [.2, [10, 0, 'C2'], 'OP50'],
              [.5, [10, 0, 'C3'], 'OP50'],
              [1.25, [10, 0, 'C4'], 'OP50'],
              ]
# sourceList = [sourceList[0]]

# ****************************
# DEFINITION OF THE CONDITIONS
# ****************************
shiftCodeCondition = 0
# shiftCodeCondition = 4

# To be used to distinguish between different runs
# conditionList should be a list, and each element should be like this:
# [number_of_plates_for_this_condition,[[density_drop1, volume_drop1], [density_drop2, volume_drop2],..., [density_dropN, volume_dropN]], "type_of_drop_pattern"]

# conditionList=[]
# nPlatePerCondition=13
# for densityRef in [10**-2, 10**-1.5, 10**-1, 10**-.5, 10**0, 10**.5]:
#     conditionList.append([nPlatePerCondition,[0, densityRef]])
#     conditionList.append([nPlatePerCondition,[densityRef, 0]])
#     conditionList.append([nPlatePerCondition,[densityRef/10, densityRef]])
#     conditionList.append([nPlatePerCondition,[densityRef, densityRef/10]])

# densityListNow = [x[0] for x in sourceList]
# conditionList=[]
# nPlatePerCondition=11
# for density in densityListNow:
#     conditionList.append([nPlatePerCondition,[density]])

# DEFINE POSITIONS OF DROPS
xyDropList = {}
xyDropList["triang1"] = closeSpaceList
xyDropList["triang2"] = mediumSpaceList
xyDropList["triang3"] = distantSpaceList
xyDropList["clusters"] = [[-16.14, -9.6], [-12.1, -9.96], [-18.94, -6.54], [-14.48, -5.68], [-20.09, -11.42],
                          [-2.71, 12.51], [-4.6, 7.81], [0.47, 15.37], [0.65, 10.3], [-6.82, 11.76], [5.97, -16.54],
                          [7.33, -11.5], [9.23, -18.94], [2.66, -14.34], [5.56, -20.51], [17.69, 4.04], [15.78, 8.05],
                          [20.36, 6.92], [13.3, 3.47], [16.1, 0.44], [-19.21, 8.89], [-22.63, 6.53], [-15.15, 8.95],
                          [-18.35, 5.02]]
xyDropList["triang4"] = superDistantSpaceList

# DEFINE CONDITIONS
volume_drop = 0.75  # ROGER
nPlatePerCondition = 10
conditionList = []
# Densities 0.2 and 0.5: All distances and clusters
for density in [0.2, 0.5]:
    for name_pattern in ["triang1", "triang2", "triang3", "clusters"]:
        conditionList.append(
            [nPlatePerCondition, [[density, volume_drop] for _ in xyDropList[name_pattern]], name_pattern])

# Intermediate distance at density 1.25
name_pattern = "triang2"
density = 1.25
conditionList.append([nPlatePerCondition, [[density, volume_drop] for _ in xyDropList[name_pattern]], name_pattern])

# Intermediate distance with two mixed densities: 0.2 and 0.5
dataDropList = []
name_pattern = "triang2"
density0 = .2
density1 = .5
for iDrop in range(len(xyDropList[name_pattern])):
    if mediumSpaceHighDensityMask[iDrop] == 1:
        dataDropList.append([density1, volume_drop])
    else:
        dataDropList.append([density0, volume_drop])
conditionList.append([nPlatePerCondition, dataDropList, name_pattern])

# Intermediate distance with two mixed densities: 0.5 and 1.25
dataDropList = []
name_pattern = "triang2"
density0 = 1.25
density1 = .5
for iDrop in range(len(xyDropList[name_pattern])):
    if mediumSpaceHighDensityMask[iDrop] == 1:
        dataDropList.append([density1, volume_drop])
    else:
        dataDropList.append([density0, volume_drop])
conditionList.append([nPlatePerCondition, dataDropList, name_pattern])

# Control: Intermediate distance with buffer
name_pattern = "triang2"
density = 0
conditionList.append([nPlatePerCondition, [[density, volume_drop] for _ in xyDropList[name_pattern]], name_pattern])

# Small distance at density 1.25
name_pattern = "triang1"
density = 1.25
conditionList.append([nPlatePerCondition, [[density, volume_drop] for _ in xyDropList[name_pattern]], name_pattern])

# Large distance at density 1.25
name_pattern = "triang3"
density = 1.25
conditionList.append([nPlatePerCondition, [[density, volume_drop] for _ in xyDropList[name_pattern]], name_pattern])

# Super large distances at the three densities
for density in [0.2, 0.5, 1.25]:
    name_pattern = "triang4"
    conditionList.append([nPlatePerCondition, [[density, volume_drop] for _ in xyDropList[name_pattern]], name_pattern])

print("conditionList")
for i in range(len(conditionList)):
    print(str(i)+": "+str(conditionList[i])+"\n")


volumeDrop = np.nan  # Volume of the drops. IMPORTANT: This only matters if we don't specify volumes in conditionList. If we do specify them, this will be ignored
rateDispense = 2  # Speed at which the drop will be dispensed. For P50 pipette, 1 is good. For P300, 0.5 is good.

# **************************
# REFERENCE POINTS AND CODES
# **************************
# To switch off reference points, set distRefPoint=[]. To switch off codes, set xyCode=[]
xyCode = [0,
          20]  # (For 3x3 codes) Position of the codes, in mm, with respect to the center of the plate (For 88 mm plates use [0,37], for 55 mm plates use [0,20])
xyCode = [0,
          21]  # (For 2x4 codes) Position of the codes, in mm, with respect to the center of the plate (For 88 mm plates use [0,37], for 55 mm plates use [0,20])
xyShiftRefPoint = []  # Position of the reference points for the drops, with respect to the center of each drop, in milimiters
xyRefPoint = [[16, 16],
              [16, -16],
              [-16, -16],
              [-16,
               16]]  # Position of the reference points, with respect to the center of the plate, in milimeters. Leave empty to have no reference points.
depthHole = 3.5  # Depth of the holes for the codes and reference points, starting from the height specified by heightAgar. 3 mm is a good standard value. If you set heightAgar well above the actual agar surface, increase this value accordingly.

# ****************
# MULTI-TRANSFERS?
# ****************
# Multi-transfer means that for drops of the same source the robot will fill as much volume as it can, and dispense
# several drops without coming back. Faster than coming back, but probably less accurate (not thoroughly validated)
doMultiTransfer = True  # Whether multitransfers are done
volumeAfterMultiTransfer = 6.25  #6.25 is optimial value for 0.75 micrl drops (Roger on 14/10/2022) # Volume that will remain in the pipette after the transfer, if doing multitransfers, in microliters

# ******
# MIXING
# ******
nMixFirsttime = 0  # Number of mixes the first time we go to a new density
nMixEachTime = 1  # Number of mixes each time we take new volume
volumeMix = math.inf  # In microliters. Set to math.inf to fill the pipette
shiftVerticalMix = 10

# Big Mix
namePipetteBigMix = 'P50'
nBigMix = 3
volumeBigMix = 50
shiftVerticalBigMix = 10
doDropTipBigMix = 'never'  # 'auto' to change tip. 'never' to not change tip
nRoundsPerBigMix = 2  # It will do a big mix every nRoundsPerBigMix rounds
