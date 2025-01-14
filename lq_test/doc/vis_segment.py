# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud('resources/colored_points_20231009_203407.ply')

    mask = np.load('resources/masks.npy')[0]
    filtered_points = np.asarray(pcd.points).reshape((480, 640, 3))[mask]
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    filtered_colors = np.asarray(pcd.colors).reshape((480, 640, 3))[mask]
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    o3d.visualization.draw(
        geometry=[pcd],
        lookat=[0, 0, 1],
        eye=[0, 0, 0],
        up=[0, -1, 0],
        intrinsic_matrix=[
            [519.32470703125, 0.0, 323.80621337890625],
            [0.0, 519.1089477539062, 236.90045166015625],
            [0.0, 0.0, 1.0]
        ]
    )

    # mask = np.load('resources/masks.npy')[0]
    # hwd = np.asarray(pcd.points)
    # hwd = hwd.reshape((480, 640, 3))
    # hwd[mask] = np.array([0, 0,  0])
    # hwd = hwd.reshape((-1,3))
    # pcd.points = o3d.utility.Vector3dVector(hwd)
    # o3d.visualization.draw([pcd])
