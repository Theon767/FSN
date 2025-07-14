import argparse, torch, os, json
import shutil
import numpy as np
import mmcv
from mmcv import Config
from mmcv.parallel import collate
from collections import OrderedDict
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
import glob
from mmcv import Config
from natsort import natsorted
from mmdet3d.datasets import build_dataset
from torch.utils.data import DataLoader
from functools import partial
# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(2560, 1440))
# display.start()

from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw(
    voxels,          # semantic occupancy predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dirs=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
):
    w, h, z = voxels.shape
    grid = grid.astype(np.int)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    grid_coords[grid_coords[:, 3] == 17, 3] = 20
    


    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] < 20)]
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19, # 16
    )

    colors = np.array(
        [
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            [  0, 255, 127, 255],       # ego car              dark cyan
            [255,  99,  71, 255],       # ego car
            [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)
    
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene

    for i, save_dir in enumerate(save_dirs):
        if i < 6:
            scene.camera.position = cam_positions[i] - np.array([0.7, 1.3, 0.])
            scene.camera.focal_point = focal_positions[i] - np.array([0.7, 1.3, 0.])
            scene.camera.view_angle = 35
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()
        elif i == 6:
            # scene.camera.position = [-4.69302904, -52.74874688, 19.16181492]
            # scene.camera.focal_point = [-4.52985313, -51.8233303, 18.81979477]
            # scene.camera.view_angle = 40.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [0.01, 300.]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
            scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        else:
            # scene.camera.position = [91.84365261779985, 87.2356528161641, 86.90232146965226]
            # scene.camera.focal_point = [4.607997894287109, -1.9073486328125e-06, -0.33333325386047363]
            # scene.camera.view_angle = 30.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [33.458354318473965, 299.5433372220855]
            # scene.camera.compute_view_plane_normal()
            # scene.render()
            scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
            scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [0., 1., 0.]
            scene.camera.clipping_range = [0.01, 400.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        mlab.savefig(os.path.join(save_dir, f'vis_{timestamp}.png'))
    mlab.close()


if __name__ == "__main__":
    ### TO DO: Using TPVformer style video generation to generate frames of gt_occ and pred_occ to check the correctness of gt,
    ### Then using hooks to check the structure of the model and backpropagation.
    voxel_origin=[0,0,0]
    voxel_max = [80,40,40]
    grid_size = [40,80,40]
    num_workers=1
    cfg = Config.fromfile('test_config.py')
    dataset = build_dataset(cfg.data.train)
    samples_per_gpu=1
    data_loader = [build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)]
    
    for index in range(len(dataset)):
        print(dataset[index].keys())
        # resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]
        # gt_visual_path='FSN_gt_save_dir'
        # #gt_visual_path='new_data/train_data/1/gt_labels'
        # all_dir_path=glob.glob(os.path.join(gt_visual_path, '**')) # fov_voxel (W,H,Z)
        # all_dir_path=natsorted(all_dir_path)
        # for voxel_path in all_dir_path:
        #     fov_voxels=np.load(voxel_path)
        #     print(fov_voxels.shape)
            # draw(fov_voxels, 
            #         voxel_origin, 
            #         resolution, 
            #         [0,0,0], 
            #         pt_label.squeeze(-1),
            #         clip_dirs,
            #         img_metas['cam_positions'],
            #         img_metas['focal_positions'],
            #         timestamp=timestamp,)