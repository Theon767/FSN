###Used to check wheter the gt_occ and the extrinsics are correct


# import os, sys
# import mayavi.mlab as mlab
# import numpy as np
# import glob
# from natsort import natsorted
# import argparse, torch, os, json
# import shutil
# import numpy as np
# from mmcv import Config
# from mmcv.parallel import collate
# from collections import OrderedDict
# from projects.mmdet3d_plugin.datasets.builder import build_dataloader
# from mmcv import Config
# from natsort import natsorted
# from mmdet3d.datasets import build_dataset
# from torch.utils.data import DataLoader
# from functools import partial
# # from pyvirtualdisplay import Display
# # display = Display(visible=False, size=(2560, 1440))
# # display.start()
import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch
import glob
from natsort import natsorted
from mmcv import Config
from natsort import natsorted
# from mmdet3d.datasets import build_dataset
# from mmcv.utils import Registry, build_from_cfg
# from projects.mmdet3d_plugin.datasets.builder import build_dataloader

def world2lidar(points_world, lidar_to_world_extrinsics_R, lidar_to_world_extrinsics_T):
        world_to_lidar_extrinsics_R = np.linalg.inv(lidar_to_world_extrinsics_R)
        world_to_lidar_extrinsics_T=-lidar_to_world_extrinsics_T
        points_lidar=points_world
        points_lidar[:,:3] = np.dot(world_to_lidar_extrinsics_R, points_world[:,:3].T+world_to_lidar_extrinsics_T[:,np.newaxis]).T
        return points_lidar


colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

#mlab.options.offscreen = True

voxel_size = 0.05
pc_range = [0, -2, -1, 2, 2 , 1]


gt_visual_path='new_data/train_data/1/gt_labels'
extrinsics_R_path='new_data/train_data/1/extrinsics_R'
extrinsics_T_path='new_data/train_data/1/extrinsics_T'
gt_visual_paths=glob.glob(os.path.join(gt_visual_path, '**')) # fov_voxel (W,H,Z)
gt_visual_paths=natsorted(gt_visual_paths)
extrinsics_R_paths=glob.glob(os.path.join(extrinsics_R_path, '**')) # fov_voxel (W,H,Z)
extrinsics_R_paths=natsorted(extrinsics_R_paths)
extrinsics_T_paths=glob.glob(os.path.join(extrinsics_T_path, '**')) # fov_voxel (W,H,Z)
extrinsics_T_paths=natsorted(extrinsics_T_paths)
# cfg = Config.fromfile('test_config.py')
# dataset = build_dataset(cfg.data.train)
samples_per_gpu=1
for gt_visual_dir,extrinsic_R_dir,extrinsic_T_dir in zip(gt_visual_paths,extrinsics_R_paths,extrinsics_T_paths):
    print(gt_visual_dir)
    ###Load data###
    fov_voxels = np.load(gt_visual_dir) # fov_voxels are in lidar ego coordinate
    print(max(fov_voxels[:,0]),min(fov_voxels[:,0]))
    print(max(fov_voxels[:,1]),min(fov_voxels[:,1]))
    print(max(fov_voxels[:,2]),min(fov_voxels[:,2]))
    print(type(fov_voxels[0,0]))
    extrinsics_R=np.load(extrinsic_R_dir)
    extrinsics_R=extrinsics_R.reshape(3,3)
    extrinsics_T=np.load(extrinsic_T_dir)
    ###Build Scene and set Camera control###
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = extrinsics_R
    extrinsic_matrix[:3, 3] = extrinsics_T
    camera_pos_ego = np.array([-1, 0, 0.5])  # 在 ego 坐标系下的相机位置
    camera_view_ego = np.array([0, 0, 0])  # 在 ego 坐标系下的视角方向
    camera_up_ego = np.array([0, 0, 1])    # 在 ego 坐标系下的上方向

    # 将 ego 坐标系下的相机参数转换为世界坐标系
    # 1. 转换相机位置
    camera_pos_world = np.dot(extrinsic_matrix, np.append(camera_pos_ego, 1))[:3]
    # 2. 转换视角方向
    # 视角方向需要相对于相机位置进行变换，因此需要考虑旋转部分
    camera_view_world = camera_pos_world + np.dot(extrinsics_R, camera_view_ego - camera_pos_ego)
    # 3. 转换上方向
    camera_up_world = np.dot(extrinsics_R, camera_up_ego)
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    scene = figure.scene
    scene.camera.position = camera_pos_ego
    scene.camera.focal_point = camera_view_ego
    scene.camera.view_angle = 60.0
    scene.camera.view_up = camera_up_ego

    # for fov_voxel in fov_voxels:
    #       print(fov_voxel)

    ###Draw voxels###
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0] 
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]
    

    #figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    
    # pdb.set_trace() # Draw H W Z
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )
    # 绘制坐标轴
    x_start = 0
    y_start = 0
    z_start = 0
    x_length = 1
    y_length = 1
    z_length = 1

    # 绘制x轴（红色）
    mlab.quiver3d(x_start, y_start, z_start, x_length, 0, 0, color=(1, 0, 0), scale_factor=1)

    # 绘制y轴（绿色）
    mlab.quiver3d(x_start, y_start, z_start, 0, y_length, 0, color=(0, 1, 0), scale_factor=1)

    # 绘制z轴（蓝色）
    mlab.quiver3d(x_start, y_start, z_start, 0, 0, z_length, color=(0, 0, 1), scale_factor=1)

    # # 添加坐标轴标签
    # mlab.text(x_start + x_length, y_start, z_start, 'X', color=(1, 0, 0))
    # mlab.text(x_start, y_start + y_length, z_start, 'Y', color=(0, 1, 0))
    # mlab.text(x_start, y_start, z_start + z_length, 'Z', color=(0, 0, 1))

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    
    #mlab.savefig('temp/mayavi.png')
    scene.render()
    mlab.show()
