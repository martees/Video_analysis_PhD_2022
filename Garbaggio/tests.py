import pandas as pd
from scipy.spatial import distance
import numpy as np

traj = pd.DataFrame()
traj["x"] = [1, 1, 10]
traj["y"] = [1, 1, 10]

array_x_r = np.array(traj["x"][1:])
array_y_r = np.array(traj["y"][1:])
array_x_l = np.array(traj["x"][:-1])
array_y_l = np.array(traj["y"][:-1])

distances = np.sqrt((array_x_l - array_x_r)**2 + (array_y_l-array_y_r)**2)


import time
traj = pd.DataFrame()
traj["x"] = pd.DataFrame(np.ones(1000000))
traj["y"] = pd.DataFrame(np.zeros(1000000))
start = time.time()
traj.apply(lambda t: t[0] - t[1], axis = 1)
end = time.time()
print(end - start)
3.8984646797180176
start = time.time()
ntraj = [traj["x"][i] - traj["y"][i] for i in range(len(traj["x"]))]
end = time.time()
print(end - start)
6.51120400428772
start = time.time()
traj.apply(lambda t: t[0] - t[1], axis = 1)
end = time.time()
print(end - start)
3.9597415924072266


##
import matplotlib.pyplot as plt
import numpy as np

x_coords = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_coords = [3, 5, 5, 8, 9, 5, 4, 6, 7, 9]

# Determine unique x values
unique_x = np.unique(x_coords)

# Calculate the y coordinate frequency for each x coordinate
y_freq = [y_coords.count(y) for y in unique_x]

# Create the bar plot
plt.bar(unique_x, y_freq)

# Customize the plot with labels and a title
plt.xlabel('X Coordinates')
plt.ylabel('Frequency')
plt.title('Distribution of Y Coordinates for each X Coordinate')

# Display the plot
plt.show()