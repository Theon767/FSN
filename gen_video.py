import os, os.path as osp
import PIL.Image as Image
import cv2
import json
import argparse
from glob import glob
from natsort import ns, natsorted
import numpy as np
from mayavi import mlab
colors=[
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
    [255, 120, 50, 255],  # 地毯及垃圾: 橘色
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


def visualize_semantic_image(rgb_image, semantic_mask, color_map):
    """
    Visualize the semantic mask on top of the original RGB image.

    Args:
    rgb_image (numpy.ndarray): The original RGB image to overlay the semantic mask.
    semantic_mask (numpy.ndarray): The semantic mask image.
    color_map (numpy.ndarray): The color map for the semantic labels.

    Returns:
    numpy.ndarray: The blended image showing both the original and colored semantic mask.
    """
    # Apply the color map to the semantic mask to create a colored semantic image
    colored_semantic = color_map[semantic_mask]

    # Set the alpha blending value (0 for fully transparent, 1 for fully opaque)
    alpha = 0.5
    # Blend the original image with the colored semantic mask using cv2.addWeighted
    blended = cv2.addWeighted(rgb_image, 1 - alpha, colored_semantic, alpha, 0)

    # Display the combined image window
    # cv2.imshow('Masked Semantic Image', blended)
    # cv2.waitKey(0)  # Wait for user input to close the window
    # cv2.destroyAllWindows()  # Close all OpenCV windows

    return blended


def cat_images(dataset_dir, cam_img_size, pred_img_size):
    rgb_img_list = []
    color_semantic_list = []
    ego_img_list = []
    bev_img_list = []
    voxel_size=0.05
    img_paths = []
    fcclip_paths = []
    pred_occ_paths=[]

    for i in range(3):
        img_paths.append(glob(os.path.join(dataset_dir,'**',str(i)+ ".jpg")))
        img_paths[i] = natsorted(img_paths[i], alg=ns.IGNORECASE)

    pred_occ_paths = natsorted(glob(os.path.join(dataset_dir, "**" ,"pred.npy"), recursive=True))
    print(pred_occ_paths)
    print(len(pred_occ_paths))
    print(len(img_paths[0]))
    #bev_paths = natsorted(glob(os.path.join(dataset_dir, "bev_temp", "*.png"), recursive=True))
    # load all images
    for idx in range(0, len(img_paths[0])-1):
        print(idx)
        rgb_img_cam_list = []

        for i in range(3):
            rgb_image = cv2.imread(img_paths[i][idx])
            rgb_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)).resize(cam_img_size)

            rgb_img_cam_list.append(rgb_image)
        rgb_img_list.append(rgb_img_cam_list)
        visual_path=pred_occ_paths[idx]
        # plotting pred_occ using mayavi
        fov_voxels = np.load(visual_path)
        # pdb.set_trace() # Draw H W Z
        mlab.options.offscreen = True
        mlab.view(azimuth=90, elevation=30, distance=5)  # 设置视角
        mlab.gcf().scene.camera.position = [1, 0, 10]   # 设置相机位置
        mlab.gcf().scene.camera.focal_point = [1, 0, 0]   # 设置焦点
        mlab.gcf().scene.camera.view_up = [0, 1, 0]  # 设置 view-up 向量为 y 轴向上
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
        mlab.draw()

        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
        #mlab.show()
        
        img_name=str(idx)+'.png'
        save_img_path=os.path.join(dataset_dir,'mayavi_view')
        os.makedirs(save_img_path,exist_ok=True)
        print(img_name)
        
        mlab.savefig(os.path.join(save_img_path,img_name))
        mlab.close()
        
        #Setting cameras
        bev_img=cv2.imread(os.path.join(save_img_path,img_name))
        bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bev_img=Image.fromarray(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)).resize(pred_img_size)
        bev_img_list.append(bev_img)
    cam_w, cam_h = cam_img_size
    pred_w, pred_h = pred_img_size
    result_w = 2*pred_w
    result_h = cam_h+pred_h

    results = []
    for i in range(len(rgb_img_list)):
        print(i)
        result = Image.new('RGB', (result_w, result_h), (0, 0, 0))
        result.paste(rgb_img_list[i][0], box=(cam_w+pred_w//2, 0))
        result.paste(rgb_img_list[i][1], box=(0, cam_h))
        result.paste(rgb_img_list[i][2], box=(cam_w+pred_w, 1 * cam_h))
        # result.paste(rgb_img_list[i][3], box=(1 * cam_w + 80, 1 * cam_h))

        # idx2 = (i - 46) // 2
        #result.paste(ego_img_list[i], box=(0, 2 * cam_h))
        result.paste(bev_img_list[i], box=(cam_w, cam_h))

        results.append(result)

    result_path = osp.join(dataset_dir, 'merge_img')
    os.makedirs(result_path, exist_ok=True)
    for i, result in enumerate(results):
        result.save(osp.join(result_path, f'{i}.png'))
    return results


def get_video(img_path, video_path, fps, size):
    video = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size
    )

    num_imgs = len(os.listdir(img_path))
    # img_files = [osp.join(img_path, fn) for fn in img_files]
    # img_files = sorted(img_files)
    for i in range(num_imgs):
        fn = osp.join(img_path, f'{i}.png')
        img = cv2.imread(fn)
        video.write(img)

    video.release()
    #cv2.destroyAllWindows()


if __name__ == "__main__":

    cam_img_size = [640, 360]
    pred_img_size = [1440, 1280]
    # spacing = 10
    dataset_dir = "/home/wangzc/projects/FSN_base/FSN_semantic_visual_dir_test"
    instance_data_dir=glob(osp.join(dataset_dir,"**"))
    instance_data_dir=natsorted(instance_data_dir)
    save_video_dir='/home/wangzc/projects/FSN_base/video'
    for i,instance in enumerate(instance_data_dir):
        results = cat_images(instance, cam_img_size, pred_img_size)

        get_video(
            osp.join(instance, 'merge_img'),
            osp.join(save_video_dir, 'demo'+str(i+10)+'.avi'),
            10,
            [2880, 360+1280]
        )
