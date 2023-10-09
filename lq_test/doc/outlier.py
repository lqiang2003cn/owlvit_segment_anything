import open3d as o3d

print("Load a ply point cloud, print it, and render it")
# sample_pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud("resources/colored_points_20231009_102104.ply")

o3d.visualization.draw(
    geometry=[pcd],
    show_ui=False,
    lookat=[0, 0, 1],
    # eye=[0, 0, 0],
    eye=[0, 0, 0],
    up=[0, -1, 0],
    intrinsic_matrix=[
        [519.32470703125, 0.0, 323.80621337890625],
        [0.0, 519.1089477539062, 236.90045166015625],
        [0.0, 0.0, 1.0]
    ],
    point_size=3

)

# o3d.visualization.draw_geometries(
#     [pcd],
# zoom=0.6,
# front=[0.4257, -0.2125, -0.8795],
# lookat=[2.6172, 2.0475, 1.532],
# up=[-0.0694, -0.9768, 0.2024]
# )

# print("Downsample the point cloud with a voxel of 0.02")
# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
# o3d.visualization.draw_geometries([voxel_down_pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])
