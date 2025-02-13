# Trajectory analysis of _C. elegans_ tracks collected in our lab setup
This project contains code that allows us to analyze tracking from our experimental setup, that consists in looking at
_C. elegans_ foraging behavior in various landscapes. It is specifically designed for our setup and data format.

## Requirements
I am running the code in both Windows 10 and Ubuntu 20.04.  
I am using Python 3.8, and have installed the following libraries:
contourpy	1.2.1  
datatable	1.1.0  
json5	0.9.28  
jsonpointer	3.0.0  
jsonschema	4.23.0  
jsonschema-specifications	2024.10.1  
kiwisolver	1.4.5  
matplotlib	3.9.0  
numpy	1.26.4 
opencv-python	4.8.1.78  
pandas	2.2.2  
pillow	10.4.0  
pip	21.1.2  
scipy	1.14.1  
seaborn	0.13.2  
setuptools	57.0.0  
sympy	1.13.3  


## Data structure

Each replicate of our experiment is stored in one folder, and our tracking software saves the 
following information in this folder:
- a "traj.csv" file containing the following columns:
  - id_conservative: one id for every different object that the tracking detected. When the tracking lost an object and 
caught it again, this id will have been incremented. This means we have to do some work to put those different tracks back together.
  - frame: the video frame at which the object was detected by the tracking.
  - time: the corresponding time, in seconds, with a bazillion decimals. To check frame to second distribution, see the script Scripts_sanity_checks/interframe_times.py. 
  - x: column with x coordinate of the object centroid, in pixels. 
  - y: column with y coordinate of the object centroid, in pixels.
- a "foodpatches.mat" which contains information about the density of each patch, and the experimental condition
- a "foodpatches_new.mat" which contains spline information (stored in matlab format) for exact patch contours. 
- a "foodpatches_reviewed.mat" which contains spline information reviewed by hand for the patch contours.

## Project structure

### Generating_data_tables directory
> #### generate_trajectories.py
> _Functions to make trajectories.csv in path_  
This contains function to preanalyze our trajectories.  
Goes into our data folders, takes the "traj.csv", mixes them all
into too big of a table, and computes in which patch the worm is
> at each time step.

> #### generate_controls.py
> _Functions to make control subfolders inside the folders 
> corresponding to control conditions_  
> Will take our existing control plates, and make new controls out of it, one corresponding to each patch
layout used in our experiments. Uses one good plate from each condition as a patch layout.

>#### generate_results.py
> _Functions to generate results_per_id, results_per_plate and
> results.csv in path_  
> This contains trajectory analysis functions (compute number of visited patches, stuff like that). 
Will extract and save the trajectories in "trajectories.csv"
Run analyses on these trajectories, and save the results in a table in the original path inputted by the user, named "results.csv".

> #### main.py
> _Go there to actually generate the data tables_  
> Will orchestrate all previously described "generate_XXX.py" 
> functions, curate the data and save everything in
> clean_trajectories.csv and clean_results.csv.

### Parameters directory
> #### param.py
> Contains global parameters, for easier editing, as well as a pipeline
to generate a lot of condition and graphical-related dictionaries 
(to go from condition number to distance, density, color, etc.)

> #### patch_coordinates
>_Contains the x,y coordinates of our patches as extracted from 
our pipetting robot scripts_

### Scripts directory
Contains scripts for performing specific analyses.
They call functions from other files, but are not meant to be called
for (if a script is that useful, it's integrated in the main pipeline).

### Main directory
>#### find_data.py
>_Contains the auxiliary functions related to finding the data in a specific path, and reformatting it to dataframes._
Converts .mat files to a pandas dataframe, containing tracking ID, trajectory, and folder name.  
Trajectories are in the following format: [[x0, x1, ..., xN], [y0, y1, ..., yN]] (xi and yi being respectively the x and y coordinates 
of the individual at time i).  
Also contains a function that takes a folder name, and returns "metadata" found in that folder:
condition number, patch positions, patch densities, splines. This allows to not have this info copied in every line of the data
table, which would make it uselessly large.

> #### analysis.py
> _Contains functions to extract statistics from the data, perform statistical tests, etc._

> #### plots.py
> _Contains plot functions._

> #### main.py 
> _Contains code to perform routine analyses with the right titles / colors / condition slicing:_
> - Commented lines to load results and trajectories.
> - The plot_graphs() function, which basically takes a string and calls the 
    > corresponding functions in plots or other scripts to make the plot appear.

### Output tables structure
#### trajectories.csv

- **id_conservative** = ID of the current tracked object (there's usually multiple of those for each worm)
- **frame** = current video frame
- **time** = corresponding time (unfilled for now)
- **x** = current x position of the object
- **y** = current y position
- **quality** = some column returned by Alfonso's tracking (unfilled)
- **folder** = path to the folder containing all the data about this point
- **patch_silhouette** = patch where the worm is (if any pixel of the worm touches a patch, it is considered inside it) 
- **distances** = distance since last point (0 for first point of the video)
- **speeds** = distance covered / frames elapsed (same as distance except when there are missing frames in tracking)

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

#### clean_results.csv

- **folder** = folder from which the worm comes (so plate identifier)  
- **condition** = condition written on the plate of the worm
- **total_video_time** = last tracked time - first tracked time (seconds)
- **total_tracked_time** = number of tracked frames (frames)
- **nb_of_holes** = number of different tracks in the plate
- **nb_all_bad_holes** = number of times a hole starts inside + ends outside, or vice versa
- **length_all_bad_holes** = cumulated length of these bad holes in seconds
- **nb_long_bad_holes** = number of bad holes whose length is higher than the parameter defined in Parameters/parameters.py
- **length_long_bad_holes** = cumulated length of these long bad holes in seconds
- **nb_of_teleportations** = number of times the worm jumps from a patch to another
- **avg_proportion_double_frames** = proportion of frames which have two entries in trajectories.csv
- **total_visit_time** = cumulated duration of all visits in seconds
- **total_transit_time** = cumulated duration of all transits in seconds
- **no_hole_visits** = time stamps of visits ([[t0, t1, patch id],...], with holes shorter than threshold that are filled in.
- **aggregated_raw_transits** = same but for transits ([[t0, t1, -1], ...])
- **uncensored_visits** = same but for only visits that are not interrupted by a hole (start of video does not count)
- **uncensored_transits** = same but for transits
- **visits_to_uncensored_patches** = same but with only visits to patches which contain no censored event
- **nb_of_visits** = what it says lol
- **average_speed_inside** = average speed of the worm inside patches. 
Computed from the average speed inside for each track, and then weighted by the
time spent inside during that track divided by the total time spent inside in the
video.
- **average_speed_outside** = same but for speed outside.
      

