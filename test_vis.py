import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch
import glob
from natsort import natsorted
colors = np.array(
        [
    [255, 120, 50, 255],  # 地毯及垃圾: 橘色
    [0, 0, 255, 255],  # 窗户及镜子: 蓝色
    [128, 128, 128, 255],  # 墙壁: 灰色
    [255, 0, 0, 255],  # 地面: 红色
    [160, 32, 240, 255],  # 橱柜: 紫色
    [255, 255, 0, 255],  # 桌子: 黄色
    [0, 255, 0, 255],  # 椅子: 绿色
    [0, 0, 139, 255],  # 垃圾桶: 深蓝色
    [255, 192, 203, 255],  # 其他: 粉色
    [0, 0, 0, 255],  # 未标注: 黑色
    [135, 60, 0, 255],  # trailer: 棕色
    [160, 32, 240, 255],  # truck: 紫色
    [255, 0, 255, 255],  # driveable_surface: 深粉色
    [139, 137, 137, 255],  # 原列表对应标签，可按需调整: 灰色
    [75, 0, 75, 255],  # sidewalk: 深紫色
    [150, 240, 80, 255],  # terrain: 浅绿色
    [230, 230, 250, 255],  # manmade: 白色
    [0, 175, 0, 255],  # vegetation: 绿色
    [0, 255, 127, 255],  # ego car: 深青色
    [255, 99, 71, 255],  # 原列表对应标签，可按需调整: 番茄色
    [0, 191, 255, 255]  # 原列表对应标签，可按需调整: 天蓝色
]
    
).astype(np.uint8)

#mlab.options.offscreen = True

voxel_size = 0.05
pc_range = [0, -2, -1, 2, 2, 1]

gt_visual_path='FSN_semantic_visual_dir/1'
# gt_visual_path='FSN_gt_save_dir'
#gt_visual_path='new_data/train_data/1/gt_labels'
all_dir_path=glob.glob(os.path.join(gt_visual_path, '**')) # fov_voxel (W,H,Z)
all_dir_path=natsorted(all_dir_path)
for dir in all_dir_path:
    
    print(dir)
    visual_path = os.path.join(dir, 'pred.npy')
    fov_voxels = np.load(visual_path)
    print('pred shape',fov_voxels.shape)
    column4 = fov_voxels[:, 3]  # 提取第四列

    # 统计唯一值和频率
    values, counts = np.unique(column4, return_counts=True)

    # 输出结果
    for value, count in zip(values, counts):
        print(f"{value}: {count}次")
    # for i in range(fov_voxels.shape[0]):
    #     if int(fov_voxels[i,3]) == 2:
    #         fov_voxels[i,3] = 0
    # for i in range(fov_voxels.shape[0]):
    #     print(fov_voxels[i])
    # fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    # fov_voxels[:, 0] += pc_range[0] 
    # fov_voxels[:, 1] += pc_range[1]
    # fov_voxels[:, 2] += pc_range[2]
    

    #figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # pdb.set_trace() # Draw H W Z
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors


    #mlab.savefig('temp/mayavi.png')
    mlab.show()
    visual_path = os.path.join(dir, 'gt_occ.npy')
    fov_voxels = np.load(visual_path)
    print('gt_occ shape',fov_voxels.shape)
    # for i in range(fov_voxels.shape[0]):
    #     if int(fov_voxels[i,3]) == 2:
    #         fov_voxels[i,3] = 0
    # for i in range(fov_voxels.shape[0]):
    #     print(fov_voxels[i])
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0] 
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]
    
    
    #figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # pdb.set_trace() # Draw H W Z
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors


    #mlab.savefig('temp/mayavi.png')
    mlab.show()
