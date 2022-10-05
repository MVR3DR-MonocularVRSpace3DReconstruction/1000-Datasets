
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import open3d as o3d

import os

from utils import *

DATA_DIR = Path('datasets/redwood-livingroom/')
OUT_DIR = Path('outputs/')
POSE_FILE = OUT_DIR / "traj.log"

COLOR_LIST = sorted(os.listdir(DATA_DIR / 'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR / 'depth/'))

STEP = 1

print(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
camera_poses = read_trajectory("{}".format(POSE_FILE))
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

count = 0
pcd = o3d.geometry.PointCloud()
for i in [0, 50, 100, 150, 200]: #range(0,len(camera_poses),STEP)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    print("Integrate {:d}-th image into the volume.".format(i))
    # print("{}image/{}".format(DATA_DIR, COLOR_LIST[i]))
    color = o3d.io.read_image(str(DATA_DIR / "image/" / COLOR_LIST[i]))
    depth = o3d.io.read_image(str(DATA_DIR / "depth/" / DEPTH_LIST[i]))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    # tmp = generate_point_cloud_with_rgbd_and_matrix(rgbd,camera_poses[count].pose)
    # pcd = merge_pcds([pcd, tmp])
    # o3d.visualization.draw_geometries([pcd])
    # volume.integrate(
    #     rgbd,
    #     o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
    #     camera_poses[count].pose) #np.linalg.inv(camera_poses[count].pose)
    
    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525, 525, 320, 240),  # o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        camera_poses[count].pose) #np.linalg.inv(camera_poses[count].pose)
    tmp = volume.extract_point_cloud()
    tmp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    tmp.estimate_normals()
    # o3d.visualization.draw_geometries([pcd])
    pcd = merge_pcds([pcd, tmp])

    count+=1

# pcd = volume.extract_point_cloud()
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud( OUT_DIR / "tmp_camera_pose.ply", pcd)
