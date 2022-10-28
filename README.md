# Trajectory analysis of _C. elegans_ tracks collected in our lab setup
This project contains code that allows us to analyze tracking from our experimental setup, that consists in looking at
_C. elegans_ foraging behavior in various landscapes. It is specifically designed for our setup and data format.

## Requirements



## Project structure
### main.py 
Contains global parameters, and example of function calls and figure plotting. 

### define_data.py
Contains the auxiliary functions related to finding the data in a specific path, and reformatting it.  
The output should be a pandas dataframe, containing a column with the trajectories, 
and columns with the associated experimental conditions.  
Trajectories are in the following format: [[x0,y0], [x1, y1], ..., [xN,yN]] (xi and yi being respectively the x and y coordinates 
of the individual at time i).