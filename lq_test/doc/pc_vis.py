# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud('/home/lq/.ros/point_cloud/points_20231006_225450.ply')
    mask = np.load('resources/masks.npy')[0]
    hwd = np.asarray(pcd.points)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(hwd)

    # hwd = hwd.reshape((480, 640, 3))
    # filtered = hwd[mask]
    # pcd.points = o3d.utility.Vector3dVector(filtered)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying Open3D pointcloud made using numpy array ...")
    o3d.visualization.draw([pcd])
