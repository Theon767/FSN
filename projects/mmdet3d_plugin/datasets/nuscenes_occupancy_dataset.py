import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
import random
import pdb, os
from mmdet.datasets.pipelines import Compose
import glob
from collections import defaultdict
import json
from natsort import natsorted
from .ray import generate_voxel_rays
from scipy.spatial import KDTree
import cv2
@DATASETS.register_module()
class CustomNuScenesOccDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self._set_group_flag()
        
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            occ_path=info['occ_path'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range)
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix

                if 'lidar2cam' in cam_info.keys():
                    lidar2cam_rt = cam_info['lidar2cam'].T
                else:
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                

                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            info = self.data_infos[idx]
            
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {0: 'IoU'}
            class_num = len(self.class_names) + 1
            for i, name in enumerate(self.class_names):
                class_names[i + 1] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0)
            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])


        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict={'Acc':results[0],
                          'Comp':results[1],
                          'CD':results[2],
                          'Prec':results[3],
                          'Recall':results[4],
                          'F-score':results[5]}

        return results_dict

@DATASETS.register_module()
class CustomFSNDataset(Dataset):
    def __init__(self, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, pipeline=None,train_data_path=None,test_data_path=None,parameters_path=None,test_mode=False,
                 modality=None,*args, **kwargs):
        super().__init__()
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.modality=modality
        self.use_semantic = use_semantic
        self.class_names = classes
        self.test_mode=test_mode
        self.data_list=[]
        self.parameters_path=parameters_path
        if not self.test_mode:
            self._set_group_flag()
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        if self.test_mode:
            self.gen_info(test_data_path)
        else: 
            self.gen_info(train_data_path)
    def gen_info(self,data_path):
        data_dict=defaultdict(list)
        data_folds=glob.glob(data_path+'/*')
        for data_fold in data_folds:
            folders = {
            'extrinsics_T': os.path.join(data_fold, 'extrinsics_T'),
            'extrinsics_R': os.path.join(data_fold, 'extrinsics_R'),
            'occ_path': os.path.join(data_fold, 'gt_labels'),
            'pointclouds': os.path.join(data_fold, 'pointclouds'),
            'image1': os.path.join(data_fold, 'image1'),
            'image3': os.path.join(data_fold, 'image3'),
            'image4': os.path.join(data_fold, 'image4'),
            'cam_intrinsic':os.path.join(self.parameters_path,'camera_parameters'),
            'lidar2cam':os.path.join(self.parameters_path,'lidar_camera_extrinsics'),
            'lidar2img':os.path.join(self.parameters_path,'lidar_camera_extrinsics')
            }
            for folder_name,folder_path in folders.items():
                files=glob.glob(folder_path+'/*')
                files=natsorted(files)
                for file in files:
                    data_dict[folder_name].append(file)
        # for key,values in data_dict.items():
        #     print(key,len(values))
        # for i in range(len(data_dict['extrinsics_R'])):
        #     for key,values in data_dict.items():
        #         if key=='image1' or key=='image3' or key=='image4':
        #             keystr=list(key)
        #             intrinsic_json=data_dict['cam_intrinsic'][int(keystr[-1])-1]
        #             extrinsic_json=data_dict['lidar2cam'][int(key[-1])-1]
        #             print(data_dict['occ_path'][i],data_dict['extrinsics_R'][i],data_dict['extrinsics_T'][i],key,intrinsic_json,extrinsic_json)
        #print(data_dict['cam_intrinsic'])
        rotation_m = np.array([[0.7071067811865476, 0, 0.7071067811865475,0],
                        [0,1, 0,0],
                        [-0.7071067811865475, 0, 0.7071067811865476,0],
                        [0, 0,0  ,1]])
        for i in range(len(data_dict['extrinsics_R'])):
            new_data_dict={}
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for key,values in data_dict.items():
                if key=='image1' or key=='image3' or key=='image4':
                    #print(i,len(values),key)
                    value=values[i]
                    image_paths.append(value)
                    keystr=list(key)
                    #print(data_dict['cam_intrinsic'][int(keystr[-1])])
                    json_name=os.path.join(data_dict['cam_intrinsic'][int(keystr[-1])-1])
                    #print('Key json correspond',key,json_name)
                    with open(json_name) as f:
                        cam_intrinsic=json.load(f)
                    viewpad = np.eye(4)
                    intrinsic=np.array(cam_intrinsic['camera_matrix']).reshape(3,3)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    cam_intrinsics.append(viewpad)
                    json_name=os.path.join(data_dict['lidar2cam'][int(key[-1])-1])
                    #print('Key json correspond',key,json_name)
                   # print('Key json correspond',key,json_name)
                    with open(json_name) as f:
                        lidar2cam=json.load(f)
                    R=np.array(lidar2cam['rotation_matrix'])

                    t=np.array(lidar2cam['translation_vector'])
                    t = t[:, np.newaxis]
                    cat_lidar2cam= np.hstack((R, t))  # 水平拼接R和t
                    cat_lidar2cam = np.vstack((cat_lidar2cam, [0, 0, 0, 1]))  # 垂直拼接最后一行
                    cat_lidar2cam=cat_lidar2cam @ rotation_m.T 
                    lidar2img_rt = (viewpad @ cat_lidar2cam)
                    lidar2img_rts.append(np.array(lidar2img_rt,dtype=np.float32))
                    lidar2cam_rts.append(np.array(cat_lidar2cam,dtype=np.float32))
                    
                    
                elif key=='occ_path' or key=='pointclouds' or key=='extrinsics_T' or key=='extrinsics_R':
                    value=values[i]
                    new_data_dict[key]=value
                
                new_data_dict['filename']=image_paths
                new_data_dict['img_filename']=image_paths
                new_data_dict['cam_intrinsic']=cam_intrinsics
                new_data_dict['lidar2img']=lidar2img_rts
                new_data_dict['lidar2cam']=lidar2cam_rts
                new_data_dict['occ_size']=self.occ_size
                new_data_dict['pc_range']=self.pc_range
            self.data_list.append(new_data_dict)
    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        return example


    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        return example


        

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_list[index]

        
        return info



    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def __len__(self):
        return len(self.data_list)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

def world2lidar(points_world, lidar_to_world_extrinsics_R, lidar_to_world_extrinsics_T):
        world_to_lidar_extrinsics_R = np.linalg.inv(lidar_to_world_extrinsics_R)
        world_to_lidar_extrinsics_T=-lidar_to_world_extrinsics_T
        points_lidar=points_world
        points_lidar[:,:3] = np.dot(world_to_lidar_extrinsics_R, points_world[:,:3].T+world_to_lidar_extrinsics_T[:,np.newaxis]).T
        return points_lidar

def gen_voxel_centers(pc_range, occ_size, ray_ratio, lidar2cam, lidar_points, 
                     cam_intrinsic, lidar_world2ego=None, type='rand'):
    """
    生成带插值的体素中心点，并投影到图像坐标
    :param pc_range: ego坐标系下的点云范围 [x_min,y_min,z_min,x_max,y_max,z_max]
    :param occ_size: 体素网格尺寸 [W,H,Z]
    :param ray_ratio: 每个体素内的插值倍数
    :param lidar2cam: LiDAR到相机的4x4变换矩阵
    :param lidar_points: 世界坐标系下的原始点云 (N,3)
    :param cam_intrinsic: 相机内参矩阵3x3
    :param lidar_world2ego: 世界坐标系到ego坐标系的变换矩阵（若lidar_points为世界坐标需转换）
    :param type: 插值类型 'rand'或'uniform'
    :return: 
        interpolated_voxel_centers: 图像坐标(u,v)数组 (M,2)
        corresponding_depths: 对应的深度值 (M,)
    """
    # 坐标转换：世界坐标系 → ego坐标系
    if lidar_world2ego is not None:
        ones = np.ones((lidar_points.shape[0], 1))
        points_homo = np.hstack([lidar_points, ones])
        lidar_points_ego = (lidar_world2ego @ points_homo.T).T[:, :3]
    else:
        lidar_points_ego = lidar_points  # 假设已经是ego坐标

    # 生成基础体素中心 ----------------------------------------------------------
    x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]
    x_max, y_max, z_max = pc_range[3], pc_range[4], pc_range[5]
    W, H, Z = occ_size
    
    dx = (x_max - x_min) / W
    dy = (y_max - y_min) / H
    dz = (z_max - z_min) / Z
    
    # 基础中心点生成
    x_centers = np.linspace(x_min + dx/2, x_max - dx/2, W)
    y_centers = np.linspace(y_min + dy/2, y_max - dy/2, H)
    z_centers = np.linspace(z_min + dz/2, z_max - dz/2, Z)
    grid_x, grid_y, grid_z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    voxel_centers = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    # 生成插值点 --------------------------------------------------------------
    interpolated_points = []
    for center in voxel_centers:
        # 根据插值类型生成额外点
        if type == 'rand':
            offsets = np.random.uniform(-0.5, 0.5, (ray_ratio, 3)) * [dx, dy, dz]
        elif type == 'uniform':
            x = np.linspace(-0.5, 0.5, int(ray_ratio**(1/3))+2)[1:-1]
            offsets = np.stack(np.meshgrid(x, x, x)).T.reshape(-1, 3) * [dx, dy, dz]
        else:
            raise ValueError("Unsupported interpolation type")
        
        # 生成插值点坐标
        new_points = center + offsets
        interpolated_points.append(new_points)
    
    interpolated_voxel_centers_ego = np.concatenate(interpolated_points, axis=0)

    # 深度关联：寻找最近点 -----------------------------------------------------
    n_neighbors = 5  # 选择最近的5个点

    kdtree = KDTree(lidar_points_ego)
    dists, idxs = kdtree.query(interpolated_voxel_centers_ego, k=n_neighbors)  # 获取k个最近邻

    # 方法1：简单平均（推荐）
    neighbor_depths = np.linalg.norm(lidar_points_ego[idxs], axis=2)  # 形状 (M, n_neighbors)
    corresponding_depths = np.mean(neighbor_depths, axis=1)            # 沿邻居维度平均

    # 坐标投影：ego → LiDAR → Camera → Image -----------------------------------
    # ego到LiDAR坐标系（假设ego与LiDAR坐标系一致，若不一致需额外变换）
    points_lidar = interpolated_voxel_centers_ego  # 此处假设ego与LiDAR坐标系相同
    
    # LiDAR到Camera坐标系 (齐次坐标变换)
    ones = np.ones((points_lidar.shape[0], 1))
    points_lidar_homo = np.hstack([points_lidar, ones])
    points_cam = (lidar2cam @ points_lidar_homo.T).T  # (N,4)
    
    # 相机坐标系到图像坐标系
    points_cam = points_cam[:, :3] / points_cam[:, 3:]  # 归一化 (x,y,z)
    cam_intrinsic=cam_intrinsic[:3,0:3]
    uv = (cam_intrinsic @ points_cam.T).T                # (u,v) = K*(x,y,z)
    uv = uv[:, :2] / uv[:, 2:]                           # 齐次坐标除法
    
    # 过滤超出图像范围的点（假设图像尺寸为[width, height]）
    image_width, image_height =  719 ,1279  # 需根据实际相机参数修改
    valid_mask = (uv[:,0] >= 0) & (uv[:,0] < image_width) & (uv[:,1] >= 0) & (uv[:,1] < image_height)
    uv = uv[valid_mask]
    corresponding_depths = corresponding_depths[valid_mask]

    return uv, corresponding_depths



@DATASETS.register_module()
class CustomFSNRENDORDataset(Dataset):
    def __init__(self, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, pipeline=None,train_data_path=None,test_data_path=None,parameters_path=None,test_mode=False,
                 modality=None,aux_frame=None,*args, ray_ratio=1,**kwargs):
        super().__init__()
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.modality=modality
        self.use_semantic = use_semantic
        self.class_names = classes
        self.test_mode=test_mode
        self.data_list=[]
        self.parameters_path=parameters_path
        self.aux_frame=aux_frame
        self.pc_range=pc_range
        self.occ_size=occ_size
        self.ray_ratio=ray_ratio
        ###All voxel_centers and their augumentation in image coords
        
        if not self.test_mode:
            self._set_group_flag()
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        if self.test_mode:
            self.gen_info(test_data_path)
        else: 
            self.gen_info(train_data_path)
    def get_rays(self, index, data_dict):
        info = data_dict

        lidar2imgs = []
        lidar2cams = []
        intrins = []
        seg_masks=[]
        images=[]
        time_ids = {}
        idx = 0

        # for time_id in [0] + self.aux_frames:
        #     time_ids[time_id] = []
        #     select_id = max(index + time_id, 0)
        #     if select_id>=len(self.data_infos) or self.data_infos[select_id]['scene_token'] != info['scene_token']:
        #         select_id = index  # out of sequence
        #     info = self.data_infos[select_id]
        #print('data_dict[pointclouds]',data_dict['pointclouds'])
        lidar_points=np.load(data_dict['pointclouds'])
        extrinsics_R=data_dict['extrinsics_R']
        extrinsics_T=data_dict['extrinsics_T']
        for cam_num in range(len(data_dict['cam_intrinsic'])):
            intrin = torch.Tensor(data_dict['cam_intrinsic'][cam_num])
            lidar2img, lidar2cam = data_dict['lidar2img'][cam_num],data_dict['lidar2cam'][cam_num]
            #image = cv2.imread(data_dict['filename'][cam_num])
            # images.append(image)
            # load seg/depth GT of rays
            seg_mask = cv2.imread(data_dict['semantic_mask'][cam_num],cv2.IMREAD_GRAYSCALE) 
            seg_masks.append(seg_mask)
            lidar2imgs.append(lidar2img)
            lidar2cams.append(lidar2cam)
            intrins.append(intrin)
            idx += 1
            self.voxel_centers,self.corresponding_depth=gen_voxel_centers(self.pc_range,self.occ_size,self.ray_ratio,
                                                                      lidar2cam=lidar2cam,lidar_points=lidar_points,
                                                                      cam_intrinsic=intrin)
        rays = generate_voxel_rays(
            self.voxel_centers, self.corresponding_depth, intrins,
            lidar2cams,lidar2imgs,seg_masks,extrinsics_R,extrinsics_T)
        rays=torch.cat(rays,dim=0)
        # weights = []
        # if balance_weight is None:  # use batch data to compute balance_weight ( rather than the total dataset )
        #     classes = torch.cat([ray[:,3] for ray in rays])
        #     class_nums = torch.Tensor([0]*17)
        #     for class_id in range(17): 
        #         class_nums[class_id] += (classes==class_id).sum().item()
        #     balance_weight = torch.exp(0.005 * (class_nums.max() / class_nums - 1))

        # for i in range(len(rays)):
        #     # wrs-a
        #     ans = 1.0 if ids[i]==0 else weight_adj
        #     weight_t = torch.full((rays[i].shape[0],), ans)
        #     if ids[i]!=0:
        #         mask_dynamic = (dynamic_class == rays[i][:, 3, None]).any(dim=-1)
        #         weight_t[mask_dynamic] = weight_dyn
        #     # wrs-b
        #     weight_b = balance_weight[rays[i][..., 3].long()]

        #     weight = weight_b * weight_t
        #     weights.append(weight)

        # rays = torch.cat(rays, dim=0)
        # weights = torch.cat(weights, dim=0)
        # if max_ray_nums!=0 and rays.shape[0]>max_ray_nums:
        #     sampler = WeightedRandomSampler(weights, num_samples=max_ray_nums, replacement=False)
        #     rays = rays[list(sampler)]
        
        return rays

    def gen_info(self,data_path):
        data_dict=defaultdict(list)
        data_folds=glob.glob(data_path+'/*')
        mask_base_path=os.path.join(data_path,'..','..','results','mask')
        for data_fold in data_folds:
            folders = {
            'extrinsics_T': os.path.join(data_fold, 'extrinsics_T'),
            'extrinsics_R': os.path.join(data_fold, 'extrinsics_R'),
            'occ_path': os.path.join(data_fold, 'gt_labels'),
            'pointclouds': os.path.join(data_fold, 'pointclouds'),
            'image1': os.path.join(data_fold, 'image1'),
            'image3': os.path.join(data_fold, 'image3'),
            'image4': os.path.join(data_fold, 'image4'),
            'cam_intrinsic':os.path.join(self.parameters_path,'camera_parameters'),
            'lidar2cam':os.path.join(self.parameters_path,'lidar_camera_extrinsics'),
            'lidar2img':os.path.join(self.parameters_path,'lidar_camera_extrinsics')
            }
            for folder_name,folder_path in folders.items():
                files=glob.glob(folder_path+'/*')
                files=natsorted(files)
                for file in files:
                    data_dict[folder_name].append(file)
        for key,values in data_dict.items():
            print(key,len(values))
        # for i in range(len(data_dict['extrinsics_R'])):
        #     for key,values in data_dict.items():
        #         if key=='image1' or key=='image3' or key=='image4':
        #             keystr=list(key)
        #             intrinsic_json=data_dict['cam_intrinsic'][int(keystr[-1])-1]
        #             extrinsic_json=data_dict['lidar2cam'][int(key[-1])-1]
        #             print(data_dict['occ_path'][i],data_dict['extrinsics_R'][i],data_dict['extrinsics_T'][i],key,intrinsic_json,extrinsic_json)
        #print(data_dict['cam_intrinsic'])
        rotation_m = np.array([[0.7071067811865476, 0, 0.7071067811865475,0],
                        [0,1, 0,0],
                        [-0.7071067811865475, 0, 0.7071067811865476,0],
                        [0, 0,0  ,1]])
        for i in range(len(data_dict['extrinsics_R'])):
            new_data_dict={}
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            clip_mask_paths=[]
            cam_intrinsics = []
            for key,values in data_dict.items():
                if key=='image1' or key=='image3' or key=='image4':
                    #print(i,len(values),key)
                    value=values[i]
                    image_paths.append(value)
                    #print('origin image_path',value)
                    clip_mask_path=value.replace('semantic_data',mask_base_path)
                    #print('Clip mask path',clip_mask_path)
                    clip_mask_paths.append(clip_mask_path)
                    keystr=list(key)
                    #print(data_dict['cam_intrinsic'][int(keystr[-1])])
                    json_name=os.path.join(data_dict['cam_intrinsic'][int(keystr[-1])-1])
                    #print('Key json correspond',key,json_name)
                    with open(json_name) as f:
                        cam_intrinsic=json.load(f)
                    viewpad = np.eye(4)
                    intrinsic=np.array(cam_intrinsic['camera_matrix']).reshape(3,3)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    cam_intrinsics.append(viewpad)
                    json_name=os.path.join(data_dict['lidar2cam'][int(key[-1])-1])
                    #print('Key json correspond',key,json_name)
                   # print('Key json correspond',key,json_name)
                    with open(json_name) as f:
                        lidar2cam=json.load(f)
                    R=np.array(lidar2cam['rotation_matrix'])

                    t=np.array(lidar2cam['translation_vector'])
                    t = t[:, np.newaxis]
                    cat_lidar2cam= np.hstack((R, t))  # 水平拼接R和t
                    cat_lidar2cam = np.vstack((cat_lidar2cam, [0, 0, 0, 1]))  # 垂直拼接最后一行
                    cat_lidar2cam=cat_lidar2cam @ rotation_m.T 
                    lidar2img_rt = (viewpad @ cat_lidar2cam)
                    lidar2img_rts.append(np.array(lidar2img_rt,dtype=np.float32))
                    lidar2cam_rts.append(np.array(cat_lidar2cam,dtype=np.float32))
                    
                    
                elif key=='occ_path' or key=='pointclouds' or key=='extrinsics_T' or key=='extrinsics_R':
                    value=values[i]
                    new_data_dict[key]=value
                
                new_data_dict['filename']=image_paths
                new_data_dict['img_filename']=image_paths
                new_data_dict['cam_intrinsic']=cam_intrinsics
                new_data_dict['lidar2img']=lidar2img_rts
                new_data_dict['lidar2cam']=lidar2cam_rts
                new_data_dict['occ_size']=self.occ_size
                new_data_dict['pc_range']=self.pc_range
                new_data_dict['semantic_mask']=clip_mask_paths
            
            self.data_list.append(new_data_dict)
    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        return example


    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        return example


        

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_list[index]
        rays=self.get_rays(index,info)
        info['ray_infos']=rays
        info['rays']=rays
        
        return info



    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def __len__(self):
        return len(self.data_list)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)