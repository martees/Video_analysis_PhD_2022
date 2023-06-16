
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


# Possible arguments for plot_graphs:
#               - "double_frames"
#               - "bad_events"
#               - "speed"
#               - "visit_duration"
#               - "visit_duration_mvt"
#               -
#               - visit
#               - visit
#               - visit
#               - "distribution"
#               - "transit_properties"
#               - "aggregated_visits"

# plot_graphs(plot="distribution", densities_list=param.condition_to_nb["close"], include_control=False)
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["med"])
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["far"], include_control=False)
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["cluster"], include_control=False)
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["0.2"], include_control=False)
# plot_graphs(plot=["distribution"], raw_condition_list=param.condition_to_nb["0.5"], include_control=False)

# plots.trajectories_1condition(trajectories, 2, plate_list=["/home/admin/Desktop/Camera_setup_analysis/Results_minipatches_20221108_clean_fp/20221011T111318_SmallPatches_C2-CAM2/traj.csv"])
# gr.add_aggregation(path, [10, 100, 100000])
# plot_graphs(plot="distribution aggregated", raw_condition_list="all")
plots.plot_variable_distribution(results, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], effect_of="food",
                                 variable_list=["aggregated_visits_thresh_10"], convert_to_duration=False)
plots.plot_variable_distribution(results, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], effect_of="food",
                                 variable_list=["aggregated_visits_thresh_100"], convert_to_duration=False)
plots.plot_variable_distribution(results, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], effect_of="food",
                                 variable_list=["aggregated_visits_thresh_100000"], convert_to_duration=False)
plots.plot_variable_distribution(results, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], effect_of="food",
                                 variable_list=["aggregated_visits_thresh_10_include_transits"], convert_to_duration=True)
plots.plot_variable_distribution(results, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], effect_of="food",
                                 variable_list=["aggregated_visits_thresh_100_include_transits"], convert_to_duration=True)
plots.plot_variable_distribution(results, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], effect_of="food",
                                 variable_list=["aggregated_visits_thresh_100000_include_transits"], convert_to_duration=True)

# plots.plot_test(results)