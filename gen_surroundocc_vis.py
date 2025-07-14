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

voxel_size = 0.5
pc_range = [-50, -50, -5.0, 50, 50, 3.0]


gt_visual_path='/home/wangzc/projects/FSN_base/data/nuscenes_occ/samples'
gt_visual_paths=glob.glob(os.path.join(gt_visual_path, '**')) # fov_voxel (W,H,Z)
gt_visual_paths=natsorted(gt_visual_paths)
# cfg = Config.fromfile('test_config.py')
# dataset = build_dataset(cfg.data.train)
samples_per_gpu=1
for gt_visual_dir in gt_visual_paths:
    print(gt_visual_dir)
    ###Load data###
    fov_voxels = np.load(gt_visual_dir) # fov_voxels are in lidar ego coordinate
    print(max(fov_voxels[:,0]),min(fov_voxels[:,0]))
    print(max(fov_voxels[:,1]),min(fov_voxels[:,1]))
    print(max(fov_voxels[:,2]),min(fov_voxels[:,2]))
    print(type(fov_voxels[0,0]))
    fov_voxels=fov_voxels.astype(np.float64)
    #print(fov_voxels)
    # for fov_voxel in fov_voxels:
    #       print(fov_voxel)

    ###Draw voxels To lidar coordinates###
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
    mlab.show()
