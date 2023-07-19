
##
from main import *

plots.patches([
    "/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111213_SmallPatches_C1-CAM5/traj.csv"])

## Plot distribution of visit + same transits + cross transits
from main import *

plot_graphs(what_to_plot="distribution", curve_list=param.name_to_nb_list["close"])
plot_graphs(what_to_plot="distribution", curve_list=param.name_to_nb_list["med"])
plot_graphs(what_to_plot="distribution", curve_list=param.name_to_nb_list["far"])
plot_graphs(what_to_plot="distribution", curve_list=param.name_to_nb_list["cluster"])
plot_graphs(what_to_plot="distribution", curve_list=param.name_to_nb_list["0.2"])
plot_graphs(what_to_plot="distribution", curve_list=param.name_to_nb_list["0.5"])
plot_graphs(what_to_plot="distribution", curve_list=param.name_to_nb_list["all"])

## Plot distribution of visit + same transits + cross transits
from main import *

plot_graphs(what_to_plot="distribution_aggregated", curve_list=param.name_to_nb_list["close"])
plot_graphs(what_to_plot="distribution_aggregated", curve_list=param.name_to_nb_list["med"])
plot_graphs(what_to_plot="distribution_aggregated", curve_list=param.name_to_nb_list["far"])
plot_graphs(what_to_plot="distribution_aggregated", curve_list=param.name_to_nb_list["cluster"])
plot_graphs(what_to_plot="distribution_aggregated", curve_list=param.name_to_nb_list["0.2"])
plot_graphs(what_to_plot="distribution_aggregated", curve_list=param.name_to_nb_list["0.5"])
plot_graphs(what_to_plot="distribution_aggregated", curve_list=param.name_to_nb_list["all"])

## Plot distribution of visit + same transits + cross transits
from main import *

plot_graphs(what_to_plot="leaving_events", curve_list=param.name_to_nb_list["close"])
plot_graphs(what_to_plot="leaving_events", curve_list=param.name_to_nb_list["med"])
plot_graphs(what_to_plot="leaving_events", curve_list=param.name_to_nb_list["far"])
plot_graphs(what_to_plot="leaving_events", curve_list=param.name_to_nb_list["cluster"])
plot_graphs(what_to_plot="leaving_events", curve_list=param.name_to_nb_list["0.2"])
plot_graphs(what_to_plot="leaving_events", curve_list=param.name_to_nb_list["0.5"])
plot_graphs(what_to_plot="leaving_events", curve_list=param.name_to_nb_list["all"])

## Plot duration of aggregated visits (see thresholds in param.py file)
from main import *

plot_graphs("aggregated_visit_duration", "all")

## Leaving probability
from main import *

plot_graphs("leaving_probability", [["close 0"], ["med 0"], ["far 0"]])
plot_graphs("leaving_probability", [["close 0.2"], ["med 0.2"], ["far 0.2"]])
plot_graphs("leaving_probability", [["close 0.5"], ["med 0.5"], ["far 0.5"]])
plot_graphs("leaving_probability", [["med 0"], ["med 0.2"], ["med 0.5"], ["med 1.25"]])
plot_graphs("leaving_probability", [["far"], ["med"], ["close"]])
plot_graphs("leaving_probability", ["cluster"])
plot_graphs("leaving_probability", [["0"], ["0.2"], ["0.5"], ["1.25"]])

## Effect of food on distribution of visits and transits durations
from main import *

plots.plot_variable_distribution(results, condition_list=param.name_to_nb_list["close"], effect_of="density")
##
plots.plot_variable_distribution(results, condition_list=param.name_to_nb_list["med"], effect_of="density")
##
plots.plot_variable_distribution(results, condition_list=param.name_to_nb_list["far"], effect_of="density")
##
plots.plot_variable_distribution(results, condition_list=param.name_to_nb_list["cluster"], effect_of="density")



## Basura
# plot_patches(fd.path_finding_traj(path))
# plot_avg_furthest_patch()
# plot_data_coverage(trajectories)
# plots.plot_traj(trajectories, 11, n_max=4, is_plot_patches=True, show_composite=False, plot_in_patch=True, plot_continuity=True, plot_speed=True, plot_time=False)
# plot_graphs(plot_visit_duration_vs_visit_start=True)
# plot_graphs(plot_visit_duration_vs_previous_transit=True)
# plot_graphs(plot_visit_duration_vs_entry_speed=True)
# plot_speed_time_window_list(trajectories, [1, 100, 1000], 1, out_patch=True)
# plot_speed_time_window_continuous(trajectories, 1, 120, 1, 100, current_speed=False, speed_history=False, past_speed=True)
# binned_speed_as_a_function_of_time_window(trajectories, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 100, 1000], [0, 1], 1)


# plot_graphs(plot="distribution", densities_list=param.condition_to_nb["close"])
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["med"])
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["far"])
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["cluster"])
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["0.2"])
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["0.5"])

# plots.trajectories_1condition(trajectories, 2, plate_list=["/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111318_SmallPatches_C2-CAM2/traj.csv"])
# gr.add_aggregation(path, [10, 100, 100000])
# plot_graphs(plot="distribution aggregated", raw_condition_list="all")
# plots.plot_test(results)
