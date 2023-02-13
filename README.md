# Trajectory analysis of _C. elegans_ tracks collected in our lab setup
This project contains code that allows us to analyze tracking from our experimental setup, that consists in looking at
_C. elegans_ foraging behavior in various landscapes. It is specifically designed for our setup and data format.

## Requirements



## Project structure
### find_data.py
Contains the auxiliary functions related to finding the data in a specific path, and reformatting it to dataframes.  
The output data should be a pandas dataframe, containing tracking ID, trajectory, and folder name. 
Trajectories are in the following format: [[x0, x1, ..., xN], [y0, y1, ..., yN]] (xi and yi being respectively the x and y coordinates 
of the individual at time i).
Find_data then also contains a function that takes a folder name, and returns "metadata" found in that folder:
condition number, patch positions, patch densities. This allows to not have this info copied in every line of the data
table, which would make it uselessly large.

### generate_results.py
This contains trajectory analysis functions (compute number of visited patches, stuff like that), will run them and 
save them in a table in the original path inputted by the user, named "results.csv".

### param.py
Contains global parameters, for easier editing.

### main.py 
Contains functions to:
- derive statistics from the results returned by generate_results.py
- plot these statistics using mostly badly written functions


