# Trajectory analysis of _C. elegans_ tracks collected in our lab setup
This project contains code that allows us to analyze tracking from our experimental setup, that consists in looking at
_C. elegans_ foraging behavior in various landscapes. It is specifically designed for our setup and data format.

## Requirements
I am running the code in both Windows 10 and Ubuntu 20.04.  
I am using Python 3.8, and the following libraries are installed:  
DateTime 4.7  	
Pillow	9.2.0  
contourpy	1.0.5  
cycler	0.11.0  
fonttools	4.37.4  
importlib-metadata	5.0.0  
kiwisolver	1.4.4  
llvmlite	0.39.1  
matplotlib	3.6.1  
numba	0.56.4  
numpy	1.23.3  
packaging	21.3  
pandas	1.5.0  
pip	21.3.1  
pyparsing	3.0.9  
python-dateutil	2.8.2  
pytz	2022.2.1  
scipy	1.9.1  
setuptools	60.2.0  
six	1.16.0  
wheel	0.37.1  
zipp	3.10.0  
zope.interface	5.4.0  

## Project structure
### param.py
Contains global parameters, for easier editing.

### find_data.py
Contains the auxiliary functions related to finding the data in a specific path, and reformatting it to dataframes.  
The output data should be a pandas dataframe, containing tracking ID, trajectory, and folder name.  
Trajectories are in the following format: [[x0, x1, ..., xN], [y0, y1, ..., yN]] (xi and yi being respectively the x and y coordinates 
of the individual at time i).  
Also contains a function that takes a folder name, and returns "metadata" found in that folder:
condition number, patch positions, patch densities. This allows to not have this info copied in every line of the data
table, which would make it uselessly large.

### generate_results.py
This contains trajectory analysis functions (compute number of visited patches, stuff like that). Contains the generate_and_save
function, that will:
- Extract and save the trajectories in "trajectories.csv"
- Run analyses on these trajectories, and save the results in a table in the original path inputted by the user, named "results.csv".

### main.py 
Contains functions to:
- derive statistics from the results returned by generate_results.py
- plot these statistics using mostly badly written functions


