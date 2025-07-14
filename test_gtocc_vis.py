import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch
import glob
from natsort import natsorted
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


gt_visual_path='semantic_data/train_data/1/gt_labels'
all_dir_path=glob.glob(os.path.join(gt_visual_path, '**')) # fov_voxel (W,H,Z)
all_dir_path=natsorted(all_dir_path)
for dir in all_dir_path:
    print(dir)
    fov_voxels = np.load(dir)
    column4 = fov_voxels[:, 3]  # 提取第四列

    # 统计唯一值和频率
    values, counts = np.unique(column4, return_counts=True)

    # 输出结果
    for value, count in zip(values, counts):
        print(f"{value}: {count}次")
    # fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    # fov_voxels[:, 0] += pc_range[0] 
    # fov_voxels[:, 1] += pc_range[1]
    # fov_voxels[:, 2] += pc_range[2]
    

    # #figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    # figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # # pdb.set_trace() # Draw H W Z
    # plt_plot_fov = mlab.points3d(
    #     fov_voxels[:, 0],
    #     fov_voxels[:, 1],
    #     fov_voxels[:, 2],
    #     fov_voxels[:, 3],
    #     colormap="viridis",
    #     scale_factor=voxel_size - 0.05*voxel_size,
    #     mode="cube",
    #     opacity=1.0,
    #     vmin=0,
    #     vmax=19,
    # )


    # plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    # plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    
    # #mlab.savefig('temp/mayavi.png')
    # mlab.show()
