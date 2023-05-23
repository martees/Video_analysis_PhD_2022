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
seaborn	0.12.2  
setuptools	60.2.0  
six	1.16.0  
wheel	0.37.1  
zipp	3.10.0  
zope.interface	5.4.0

## Data structure

Each replicate of our experiment is stored in one folder, and our tracking software saves the 
following information in this folder:
- a "traj.csv" file containing the following columns:
  - id_conservative: one id for every different object that the tracking detected. When the tracking lost an object and 
caught it again, this id will have been incremented. This means we have to do some work to put those different tracks back together.
  - frame: the video frame at which the object was detected by the tracking (note: in our case 1 frame = 0.8 sec)
  - x: column with x coordinates
  - y: column with y coordinates
- a "foodpatches.mat" which contains information about the density of each patch, and the experimental condition
- a "foodpatches_new.mat" which contains a spline 


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

### Output structure
#### results_per_id.csv

- **folder** = folder from which the worm comes (so plate identifier)  
- **condition** = condition written on the plate of the worm  
- **track_id** = number of the track (100 times the file number + id attributed by tracking algorithm)  
- **total_time** = total number of frames for this worm  
- **raw_visits** = list outputed by patch_visits_single_traj (see its description)  
- **order_of_visits** = list of order of visits [2 3 0 1] = first patch 2, then patch 3, etc  
- **total_visit_time** = total duration of visits for each worm  
- **nb_of_visits** = nb of visits to patches this worm did  
- **nb_of_visited_patches** = nb of different patches it visited  
- **furthest_patch_distance** = furthest patch visited  
- **total_transit_time** = total time spent outside of patches (same as total_time - total_visit_time)  
- **adjusted_raw_visits** = adjusted: consecutive visits to the same patch are counted as one  
- **adjusted_total_visit_time** = should be the same as duration sum (did this to check)  
- **adjusted_nb_of_visits** = nb of adjusted visits  

