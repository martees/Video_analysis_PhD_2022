from Generating_data_tables import main as gen
from Parameters import parameters as param
import pandas as pd
import datatable as dt


path = gen.generate(starting_from="", test_pipeline=False)
results = pd.read_csv(path + "clean_results.csv")
trajectories = dt.fread(path + "clean_trajectories.csv")
full_list_of_folders = list(results["folder"])


to_regenerate = ["2G"]

if to_regenerate == "all":
    to_regenerate = ["1A", "1B", "1C", "1D", "1E", "1F",
                     "2B", "2C", "2D", "2E", "2G",
                     "3B", "3C", "3D", "3E", "3G",
                     "4A", "4B", "4C",
                     "5F"]

is_recompute=True

# Figure 1
if "1A" in to_regenerate:
    from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap
    heatmap.plot_existing_heatmap(path, [1], "pixel_visits", v_max=0.00002)
if "1B" in to_regenerate:
    from Scripts_analysis import s20240606_distancetoedgeanalysis as dist2edge
    list_of_distance_bins_mm = [-1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.7, 1.5, 2.4]
    list_of_distance_bins = [b/param.one_pixel_in_mm for b in list_of_distance_bins_mm]
    dist2edge.plot_variable_vs_distance(full_list_of_folders, trajectories,
                              ['med 0', 'med 0.2', 'med 0.5', 'med 1.25'], list_of_distance_bins,
                              variable="time_avg_visited_patch", only_show_density=True, recompute=is_recompute)
if "1C" in to_regenerate:
    from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap
    heatmap.plot_existing_heatmap(path, [1], "speed", v_max=0.2)
if "1D" in to_regenerate:
    from Scripts_analysis import s20240606_distancetoedgeanalysis as dist2edge
    list_of_distance_bins_mm = [-1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.7, 1.5, 2.4]
    list_of_distance_bins = [b/param.one_pixel_in_mm for b in list_of_distance_bins_mm]
    dist2edge.plot_variable_vs_distance(full_list_of_folders, trajectories,
                              ['med 0', 'med 0.2', 'med 0.5', 'med 1.25'], list_of_distance_bins,
                              variable="speed", only_show_density=True, recompute=is_recompute)
    # Inset
    from Scripts_analysis import s20230725_speed_at_patch_border as speedborder
    speedborder.condition_pool(["med 0", "med 0.2", "med 0.5", "med 1.25"], show_only_density=True)
if "1E" in to_regenerate:
    import plots
    plots.trajectories_1condition(path, trajectories, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
                                   show_composite=False, is_plot_patches=True, plot_continuity=False, plot_in_patch=True,
                                   plot_speed=False, is_plot=False, save_fig=True, plot_lines=False)
    # And then you can see all trajectories in the trajectory_plot folder (inside results folder)
if "1F" in to_regenerate:
    from Scripts_analysis import s20241022_visit_duration_vs_time_in_patch as tvtp
    bins = [100, 400, 1000, 1900, 3100, 4600, 6400, 8500, 10900, 13600]
    tvtp.plot_visit_duration_vs_time_in_patch(results, ["med 0", "med 0.2", "med 0.5", "med 1.25"], bins, 20, only_show_density=True)


# Figure 2
if "2B" in to_regenerate:
    from Scripts_analysis import s20240605_global_presence_heatmaps as heatmap
    heatmap.plot_existing_heatmap(path, [4], "pixel_visits", v_max=0.00002)
    heatmap.plot_existing_heatmap(path, [5], "pixel_visits", v_max=0.00002)
    heatmap.plot_existing_heatmap(path, [6], "pixel_visits", v_max=0.00002)
    heatmap.plot_existing_heatmap(path, [15], "pixel_visits", v_max=0.00002)
if "2C" in to_regenerate:
    from Scripts_analysis import s20240606_distancetoedgeanalysis as dist2edge
    dist2edge.plot_variable_vs_distance(full_list_of_folders, trajectories,
                              ['close 0.5', 'med 0.5', 'far 0.5', 'superfar 0.5'], list_of_distance_bins,
                              variable="time_avg_visited_patch", only_show_density=False, recompute=is_recompute)
if "2D" in to_regenerate:
    from Scripts_analysis import s20240606_distancetoedgeanalysis as dist2edge
    dist2edge.plot_variable_vs_distance(full_list_of_folders, trajectories,
                              ['close 0.5', 'med 0.5', 'far 0.5', 'superfar 0.5'], list_of_distance_bins,
                              variable="speed", only_show_density=False, recompute=is_recompute)
    # Inset
    from Scripts_analysis import s20230725_speed_at_patch_border as speedborder
    speedborder.condition_pool(["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], show_only_density=False)
if "2E" in to_regenerate:
    from Scripts_analysis import s20241022_visit_duration_vs_time_in_patch as tvtp
    bins = [100, 400, 1000, 1900, 3100, 4600, 6400, 8500, 10900, 13600]
    tvtp.plot_visit_duration_vs_time_in_patch(results, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], bins, 20)
if "2G" in to_regenerate:
    import main
    main.plot_graphs(results, "total_visit_time", [["close 0", "med 0", "far 0", "superfar 0"],
                                                                  ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
                                                                  ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
                                                                  ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])


# Figure 3
if "3B" in to_regenerate:
    import main
    main.plot_graphs(results, "number_of_visits_per_visited_patch", [["close 0", "med 0", "far 0", "superfar 0"],
                                                                                  ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
                                                                                  ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
                                                                                  ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])
if "3C" in to_regenerate:
    import main
    main.plot_graphs(results, "visit_duration", [["close 0", "med 0", "far 0", "superfar 0"],
                                                                                  ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
                                                                                  ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
                                                                                  ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])
if "3D" in to_regenerate:
    from Scripts_analysis import s20240918_total_time_vs_nb_of_visits as tpnv
    tpnv.plot_Tp_vs_Nv(["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], True, linear_or_log="log", mix_plates=False)
    tpnv.plot_Tp_vs_Nv(["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], True, linear_or_log="log", mix_plates=False)
    tpnv.plot_Tp_vs_Nv(["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], True, linear_or_log="log", mix_plates=False)
if "3F" in to_regenerate:
    import main
    main.plot_graphs(results, "first_visit_duration_patch", [["close 0", "med 0", "far 0", "superfar 0"],
                                                                          ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"],
                                                                          ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"],
                                                                          ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"]])
if "3G" in to_regenerate:
    from Scripts_analysis import s20240902_visitvsprevioustravel as prevtrav
    bin_list = [0, 0.01 * 60, 0.1 * 60, 1 * 60, 10 * 60]
    only_first_visit = False
    prevtrav.visit_versus_previous_travel(results, [[param.name_to_nb["close 0.2"]], [param.name_to_nb["med 0.2"]],
                                           [param.name_to_nb["far 0.2"]],
                                           [param.name_to_nb["superfar 0.2"]]], travel_time_bins=bin_list,
                                            only_first=only_first_visit, min_nb_points=10, show_counts=False)
    prevtrav.visit_versus_previous_travel(results, [[param.name_to_nb["close 0.5"]], [param.name_to_nb["med 0.5"]],
                                           [param.name_to_nb["far 0.5"]],
                                           [param.name_to_nb["superfar 0.5"]]], travel_time_bins=bin_list,
                                            only_first=only_first_visit, min_nb_points=10, show_counts=False)
    prevtrav.visit_versus_previous_travel(results, [[param.name_to_nb["close 1.25"]], [param.name_to_nb["med 1.25"]],
                                           [param.name_to_nb["far 1.25"]],
                                           [param.name_to_nb["superfar 1.25"]]], travel_time_bins=bin_list,
                                           only_first=only_first_visit, min_nb_points=10, show_counts=False)


# Figure 4
if "4A" in to_regenerate:
    from Scripts_analysis import s20240804_displacement_since_patch_exit as msd
    bin_list = [10, 20, 35, 55, 75, 100, 190, 470, 770]
    msd.msd_analysis(results, trajectories, ["close 0", "med 0", "far 0", "superfar 0"], bin_list, 1, 4, False, "probability", True)
    msd.msd_analysis(results, trajectories, ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], bin_list, 1, 4, False, "probability", True)
    msd.msd_analysis(results, trajectories, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], bin_list, 1, 4, False, "probability", True)
    msd.msd_analysis(results, trajectories, ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], bin_list, 1, 4, False, "probability", True)
if "4B" in to_regenerate:
    from Scripts_analysis import s20240804_displacement_since_patch_exit as msd
    bin_list = [10, 20, 35, 55, 75, 100, 190, 470, 770]
    msd.msd_analysis(results, trajectories, ["close 0", "med 0", "far 0", "superfar 0"], bin_list, 1, 4, False, "time", True)
    msd.msd_analysis(results, trajectories, ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"], bin_list, 1, 4, False, "time", True)
    msd.msd_analysis(results, trajectories, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"], bin_list, 1, 4, False, "time", True)
    msd.msd_analysis(results, trajectories, ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"], bin_list, 1, 4, False, "time", True)
if "4C" in to_regenerate:
    from Scripts_models import s20260325_roger_model_analysis as roger
    results_of_model = dt.fread(path + "random_walk_results.csv")
    results_of_xp = results
    roger.nb_of_visits_model_xp(results_of_xp, results_of_model, ["close 0.2", "med 0.2", "far 0.2", "superfar 0.2"])
    roger.nb_of_visits_model_xp(results_of_xp, results_of_model, ["close 0.5", "med 0.5", "far 0.5", "superfar 0.5"])
    roger.nb_of_visits_model_xp(results_of_xp, results_of_model, ["close 1.25", "med 1.25", "far 1.25", "superfar 1.25"])


# Figure 5
if "5F" in to_regenerate:
    from Scripts_models import s20240626_transitionmatrix as trans
    trans.behavior_vs_geometry(path, results, "close 0.2", 4000, 25000, visit_order=True, plot_tpnv=False, mix_xp_plates=True)
    trans.behavior_vs_geometry(path, results, "close 0.5", 4000, 25000, visit_order=True, plot_tpnv=False, mix_xp_plates=True)
    trans.behavior_vs_geometry(path, results, "close 1.25", 4000, 25000, visit_order=True,
                         plot_1stvisit=False, plot_tpnv=False, mix_xp_plates=False)




