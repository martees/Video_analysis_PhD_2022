from Parameters import parameters as param
from Scripts_analysis.s20240605_global_presence_heatmaps import *

results_path = gen.generate("")
results = pd.read_csv(results_path + "clean_results.csv")
plate_list = results["folder"]

frame_size = 1847

condition_list = param.name_to_nb_list["close 0"]
condition_plates = fd.return_folders_condition_list(plate_list, condition_list)

# Plot a heatmap of experimental patch coverage
heatmap = np.zeros((1944, 1944))
for i_plate, plate in enumerate(condition_plates):
    in_patch_matrix_path = plate[:-len("traj.csv")] + "in_patch_matrix.csv"
    if not os.path.isfile(in_patch_matrix_path):
        gt.in_patch_all_pixels(in_patch_matrix_path)
    in_patch_matrix = pd.read_csv(in_patch_matrix_path).to_numpy()
    if len(heatmap) != len(in_patch_matrix):
        heatmap[:len(in_patch_matrix), :len(in_patch_matrix)] += in_patch_matrix
    else:
        heatmap += in_patch_matrix
plt.imshow(heatmap)
plt.colorbar()

# Plot the perfect food patches
# Compute the idealized patch positions by converting the robot xy data to mm in a "perfect" reference frame
ideal_patch_centers_each_cond = idealized_patch_centers_mm(results_path, plate_list, frame_size)
# Load the average patch radius
average_patch_radius_each_cond = pd.read_csv(
    results_path + "perfect_heatmaps/average_patch_radius_each_condition.csv")
average_radius = np.mean(average_patch_radius_each_cond["avg_patch_radius"])
for i_cond, cond in enumerate(condition_list):
    ideal_patch_centers = ideal_patch_centers_each_cond[cond]
    plt.scatter(ideal_patch_centers[:, 0], ideal_patch_centers[:, 1],
                color=param.name_to_color[param.nb_to_name[cond]],
                label="ideal patches")
    for i_patch in range(len(ideal_patch_centers)):
        circle = plt.Circle(ideal_patch_centers[i_patch], average_radius,
                            color=param.name_to_color[param.nb_to_name[cond]],
                            fill=False)
        plt.gca().add_patch(circle)

# Plot the original food patches from the robot script
plt.scatter([-16, -16, 16, 16], [-16, 16, -16, 16], color="gray", marker="x")
coord = param.patch_coordinates.xy_patches_super_far
plt.scatter([xy[0] for xy in coord], [xy[1] for xy in coord], color="gray")

# Plot the average refpoints positions
# compute_average_ref_points_distance(results_path, plate_list)
ref_points_table = pd.read_csv(results_path + "perfect_heatmaps/average_reference_points_distance_each_condition.csv")
avg_point_1_x, avg_point_1_y = ref_points_table["point_1_x"], ref_points_table["point_1_y"]
avg_point_2_x, avg_point_2_y = ref_points_table["point_2_x"], ref_points_table["point_2_y"]
avg_point_3_x, avg_point_3_y = ref_points_table["point_3_x"], ref_points_table["point_3_y"]
avg_point_4_x, avg_point_4_y = ref_points_table["point_4_x"], ref_points_table["point_4_y"]
plt.scatter([avg_point_1_x, avg_point_2_x, avg_point_3_x, avg_point_4_x],
            [avg_point_1_y, avg_point_2_y, avg_point_3_y, avg_point_4_y],
            marker="x", label="Average reference points in the plates", color="orange")
# Plot the reference points position used to build idealized landscapes
average_ref_points_distance = np.mean(ref_points_table["avg_ref_points_distance"])
# Deduce reference points from that
margin = (frame_size - average_ref_points_distance) / 2
bottom_left = [margin, margin]
bottom_right = [frame_size - margin, margin]
top_left = [margin, frame_size - margin]
top_right = [frame_size - margin, frame_size - margin]
plt.scatter([bottom_left[0], bottom_right[0], top_left[0], top_right[0]],
            [bottom_left[1], bottom_right[1], top_left[1], top_right[1]],
            marker="*", label="Ref points used for perfect plates", color="cornflowerblue")

# Plot the plate size that goes with this
# Add circle for plate border
heatmap_size = 1847
plate_radius = (np.sqrt(2)/2)*987.21
plate = plt.Circle((heatmap_size/2, heatmap_size/2), radius=plate_radius, color="gray", fill=False)
plt.gcf().gca().add_patch(plate)

# Add dashed circle for tracking area border
tracking_radius = 0.86*plate_radius
tracking_area = plt.Circle((heatmap_size/2, heatmap_size/2), radius=tracking_radius, color="gray", fill=False, linestyle= '--')
plt.gcf().gca().add_patch(tracking_area)

plt.xlim(-20, 1600)
plt.ylim(-20, 1600)

plt.legend()
plt.show()

