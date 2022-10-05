import os
import re
import math
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
import open3d.core as o3c


class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj



def generate_point_cloud_with_rgbd(pic):
    color_raw = pic.color
    depth_raw = pic.depth
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def generate_point_cloud_with_rgbd_and_matrix(pic, M):
    color_raw = pic.color
    depth_raw = pic.depth
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        M
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def merge_pcds(pcd_list):
    tmp_points = np.concatenate(tuple([i.points for i in pcd_list]), axis=0)
    tmp_colors = np.concatenate(tuple([i.colors for i in pcd_list]), axis=0)
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(tmp_points)
    tmp.colors = o3d.utility.Vector3dVector(tmp_colors)
    return tmp