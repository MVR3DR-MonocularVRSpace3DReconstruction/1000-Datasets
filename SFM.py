import os
import glob
from collections import defaultdict
from copy import deepcopy
from turtle import color
import numpy as np
from matplotlib import cm
import open3d as o3d
import open3d.visualization as vis
import pycolmap

def pcd_from_colmap(rec, min_track_length=4, max_reprojection_error=8):
    points = []
    colors = []
    for p3D in rec.points3D.values():
        if p3D.track.length() < min_track_length:
            continue
        if p3D.error > max_reprojection_error:
            continue
        points.append(p3D.xyz)
        colors.append(p3D.color/255.)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(points))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors))
    print(len(points), len(rec.points3D))
    return pcd

def add_point(pcd, point):
    pcd.points = o3d.utility.Vector3dVector(np.stack([point]))
    pcd.colors = o3d.utility.Vector3dVector(np.stack([[0, 0, 1]]))

    return pcd
    

blocks_dir = sorted(glob.glob('outputs/22-10-27_01-50-20/*'))

for dir in blocks_dir:
    app = vis.gui.Application.instance
    app.initialize()
    w = vis.O3DVisualizer(width=2048, height=1024)
    w.show_ground = False
    w.show_axes = False
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    rec = pycolmap.Reconstruction(dir+'/sfm')
    # print(rec.summary())

    # Add sparse point cloud
    pcd = pcd_from_colmap(rec)
    w.add_geometry('pcd', pcd, mat)

    # Define the camera frustums as lines
    camera_lines = {}
    for camera in rec.cameras.values():
        camera_lines[camera.camera_id] = o3d.geometry.LineSet.create_camera_visualization(
            camera.width, camera.height, camera.calibration_matrix(), np.eye(4), scale=1)
    
    
    # Draw the frustum for each image
    for image in rec.images.values():
        T = np.eye(4)
        T[:3, :4] = image.inverse_projection_matrix()
        cam = deepcopy(camera_lines[image.camera_id]).transform(T)
        
        p = np.array(cam.points)
        cam_p = o3d.geometry.PointCloud()
        cam_p.points = o3d.utility.Vector3dVector([p[0]])
        cam_p.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
        print(image.name)
        
        w.add_geometry(image.name+"p", cam_p)
        cam.paint_uniform_color([1.0, 0.0, 0.0])  # red
        # w.add_geometry(image.name, cam)

    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.point_size = 5
    w.enable_raw_mode(True)
    app.add_window(w)
    app.run()