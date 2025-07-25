import os
import cv2
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import segment_coo
from torch_efficient_distloss import flatten_eff_distloss

#from .utils import Raw2Alpha, Alphas2Weights, ub360_utils_cuda, silog_loss
from .utils import silog_loss

## OpenOccupancy
# nusc_class_frequencies = np.array([2242961742295, 25985376, 1561108, 28862014, 196106643, 15920504,
#                 2158753, 26539491, 4004729, 34838681, 75173306, 2255027978, 50959399, 646022466, 869055679,
#                 1446141335, 1724391378])

## occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

def print_memory(msg=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{msg}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def cumdist_thres(dist: torch.Tensor, dist_thres: float):
    cumdist = torch.cumsum(dist, dim=1)
    mask = cumdist > dist_thres
    return mask

def alphas_to_weights(alpha: torch.Tensor, ray_id: torch.Tensor, N_ray: int):
    """
    将 alpha 值转换为体积渲染权重，并返回每条射线的最终透射率
    参数：
        alpha:      [n_samples] 每个采样点的不透明度
        ray_id:     [n_samples] 每个采样点所属射线ID
        N_ray:      总射线数量
    返回：
        weights:        [n_samples] 各采样点的渲染权重
        alphainv_last:  [N_ray] 每条射线最后的累积透射率
    """
    # 计算每条射线的累积透射率（1-alpha 的累积乘积）
    one_minus_alpha = 1 - alpha + 1e-11  # 添加极小值防止除零

    # 构建射线分组的重置掩码
    is_ray_start = torch.zeros_like(ray_id, dtype=torch.bool)
    if ray_id.numel() > 0:
        is_ray_start[1:] = ray_id[1:] != ray_id[:-1]
    is_ray_start[0] = True  # 第一个元素总是射线的起点

    # 按射线分组的累积乘积
    cumprod = torch.ones_like(one_minus_alpha)
    current = torch.ones(())
    for i in range(len(one_minus_alpha)):
        if is_ray_start[i]:
            current = one_minus_alpha[i]
        else:
            current *= one_minus_alpha[i]
        cumprod[i] = current

    # 计算各点的前缀透射率（前一项的累积乘积）
    transmittance = torch.cat([torch.ones(1, device=alpha.device), cumprod[:-1]])

    # 计算权重
    weights = alpha * transmittance

    # 提取每条射线最后的透射率值
    _, unique_indices = torch.unique(ray_id, return_inverse=True)
    last_indices = torch.zeros(N_ray, dtype=torch.long, device=alpha.device)
    last_indices.scatter_(0, unique_indices, torch.arange(len(ray_id), device=alpha.device))
    alphainv_last = cumprod[last_indices]

    return weights, alphainv_last

# @functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

def sample_ray(ori_rays_o, ori_rays_d, step_size, scene_center, scene_radius, bg_len, world_len, **render_kwargs):
    print_memory("Generate rays")
    rays_o = (ori_rays_o - scene_center) / scene_radius       # normalization
    rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
    N_inner = int(2 / (2+2*bg_len) * world_len / step_size) + 1
    N_outer = N_inner//15   # hardcode: 15
    b_inner = torch.linspace(0, 2, N_inner+1)
    b_outer = 2 / torch.linspace(1, 1/64, N_outer+1)
    t = torch.cat([
        (b_inner[1:] + b_inner[:-1]) * 0.5,
        (b_outer[1:] + b_outer[:-1]) * 0.5,
    ]).to(rays_o)
    ray_pts = rays_o[:,None,:] + rays_d[:,None,:] * t[None,:,None]
    print_memory("Sampling ray")
    norm = ray_pts.norm(dim=-1, keepdim=True)
    inner_mask = (norm<=1)
    ray_pts = torch.where(
        inner_mask,
        ray_pts,
        ray_pts / norm * ((1+bg_len) - bg_len/norm)
    )
    # ray_pts [N_rays, N_samples, 3] 沿射线的采样点坐标（归一化到场景坐标系）
    # inner_mask [N_rays, N_samples] 布尔掩码，标记内层采样点（True 表示在单位球内）
    # t [N_samples] 采样距离参数（用于后续计算透明度或颜色)
    # reverse bda-aug 
    #ray_pts = bda.matmul(ray_pts.unsqueeze(-1)).squeeze(-1)
    return ray_pts, inner_mask.squeeze(-1), t

def sample_ray_batch(ori_rays_o, ori_rays_d, step_size, scene_center, scene_radius, bg_len, world_len, bda, **render_kwargs):
    rays_o = (ori_rays_o - scene_center) / scene_radius       # normalization
    rays_d = ori_rays_d / ori_rays_d.norm(dim=-1, keepdim=True)
    N_inner = int(2 / (2+2*bg_len) * world_len / step_size) + 1
    N_outer = N_inner//20   # hardcode: 15
    b_inner = torch.linspace(0, 2, N_inner+1)
    b_outer = 2 / torch.linspace(1, 1/64, N_outer+1)
    t = torch.cat([
        (b_inner[1:] + b_inner[:-1]) * 0.5,
        (b_outer[1:] + b_outer[:-1]) * 0.5,
    ]).to(rays_o)
    ray_pts = rays_o[:,None,:] + rays_d[:,None,:] * t[None,:,None]

    norm = ray_pts.norm(dim=-1, keepdim=True)
    inner_mask = (norm<=1)
    ray_pts = torch.where(
        inner_mask,
        ray_pts,
        ray_pts / norm * ((1+bg_len) - bg_len/norm)
    )

    # reverse bda-aug 
    ray_pts = bda.matmul(ray_pts.unsqueeze(-1)).squeeze(-1)
    return ray_pts, inner_mask.squeeze(-1), t

class TimeCounter:
    def __init__(self):
        self.times = [time.time()]
        self.names = []
    
    def clear(self):
        self.times = [time.time()]
        self.names = []

    def add(self, name):
        self.times.append(time.time())
        self.names.append(name)
    
    def print(self):
        times = np.array(self.times)
        times = np.diff(times*1000).astype(np.int16)
        print('> -----Time Cost-----<')
        for i in range(len(self.names)):
            print('%s:  %f'%(self.names[i], times[i]))


from mmdet.models import HEADS
@HEADS.register_module()
class NerfHead(nn.Module):
    def __init__(self, 
            point_cloud_range,
            voxel_size,
            scene_center=None,
            radius=39,
            step_size=0.008, 
            use_depth_sup=True,
            balance_cls_weight=True,
            weight_depth=1.0,
            weight_semantic=1.0,
            weight_entropy_last=0.01,   
            weight_distortion=0.01,
            alpha_init=1e-6,
            fast_color_thres=1e-7,
            ):
        super().__init__()
        self.weight_entropy_last = weight_entropy_last
        self.weight_distortion = weight_distortion

        xyz_min = torch.Tensor(point_cloud_range[:3])
        xyz_max = torch.Tensor(point_cloud_range[3:])
        xyz_range = (xyz_max - xyz_min).float()
        self.bg_len = (xyz_range[0]//2-radius)/radius
        # import ipdb;ipdb.set_trace()
        self.radius = radius
        # init_shift=0.5
        # self.act_shift = nn.Parameter(torch.tensor([init_shift], dtype=torch.float32))

        self.register_buffer('scene_center', ((xyz_min + xyz_max) * 0.5))
        self.register_buffer('scene_radius', torch.Tensor([radius, radius, radius]))

        self.step_size = step_size
        self.use_depth_sup = use_depth_sup
        z_ = xyz_range[2]/xyz_range[0]
        self.register_buffer('xyz_min', torch.Tensor([-1-self.bg_len, -1-self.bg_len, -z_]))
        self.register_buffer('xyz_max', torch.Tensor([1+self.bg_len, 1+self.bg_len, z_]))
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('--> Set density bias shift to', self.act_shift)

        
        self.voxel_size = voxel_size/radius
        self.voxel_size_ratio = torch.tensor(1.0)
        self.world_size = torch.Tensor([40, 80, 40]).long()
        self.world_len = self.world_size[0].item()

        self.fast_color_thres = fast_color_thres
        self.weight_depth = weight_depth
        self.weight_semantic = weight_semantic
        self.depth_loss = silog_loss()

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001))
        else:
            self.class_weights = torch.ones(17)/17 

    def render_one_scene(self,
            rays_o_tr,
            rays_d_tr,
            density,
            semantic,
            mask=None,
        ):
        if mask is not None:
            rays_o = rays_o_tr[mask]
            rays_d = rays_d_tr[mask]
        else:
            rays_o = rays_o_tr.reshape(-1, 3)
            rays_d = rays_d_tr.reshape(-1, 3)
        device = rays_o.device
        # sample points on rays
        print_memory("Before sample ray")
        ray_pts, inner_mask, t = sample_ray(
            ori_rays_o=rays_o, ori_rays_d=rays_d, 
            step_size=self.step_size,
            scene_center=self.scene_center, 
            scene_radius=self.scene_radius, 
            bg_len=self.bg_len, 
            world_len=self.world_len
        )
        print_memory("After sample ray")
        ray_id, step_id = create_full_step_id(ray_pts.shape[:2])

        # skip oversampled points outside scene bbox
        mask = inner_mask.clone()
        # dist_thres = (2+2*self.bg_len) / self.world_len * self.step_size * 0.95
        # dist = (ray_pts[:,1:] - ray_pts[:,:-1]).norm(dim=-1)
        # mask[:, 1:] |= cumdist_thres(dist, dist_thres)
        # ray_pts = ray_pts[mask]
        # inner_mask = inner_mask[mask]

        N_ray = len(rays_o)
        t = t[None].repeat(N_ray,1)[mask]
        ray_id = ray_id[mask.flatten()].to(device)
        step_id = step_id[mask.flatten()].to(device)

        # rays sampling
        shape = ray_pts.shape[:-1]
        xyz = ray_pts.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1

        density = F.grid_sample(density.unsqueeze(0).unsqueeze(1), ind_norm, mode='bilinear', align_corners=True)
        density = density.reshape(1, -1).T.reshape(*shape) 

        
        semantic = semantic.permute(3,0,1,2).unsqueeze(0)
        num_classes = semantic.shape[1]
        semantic = F.grid_sample(semantic, ind_norm, mode='bilinear', align_corners=True)
        semantic = semantic.reshape(num_classes, -1).T.reshape(*shape, num_classes)
        print_memory('Before alpha')
        alpha = self.activate_density(density, interval=self.step_size) 
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            inner_mask = inner_mask[mask]
            t = t[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            semantic = semantic[mask]

        # compute accumulated transmittance
        N_ray = len(rays_o)
        #weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id.to(alpha.device), N_ray)
        print_memory('Before alapha to weights')
        weights, alphainv_last= alphas_to_weights(alpha, ray_id.to(alpha.device), N_ray)
        
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            inner_mask = inner_mask[mask]
            t = t[mask]
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            semantic = semantic[mask]


        s = 1 - 1/(1+t)  # [0, inf] => [0, 1]
        results = {
            'alphainv_last': alphainv_last,
            'weights': weights,
            'ray_id': ray_id,
            's': s,
            't': t,
            'N_ray': N_ray,
            'num_classes': num_classes,
            'density': density,
            'semantic': semantic,
        }
        return results
    
    def compute_loss(self, results):
        losses = {}
        if self.use_depth_sup:
            depth_loss = self.depth_loss(results['render_depth']+1e-7, results['target_depth'])
            losses['loss_render_depth'] = depth_loss * self.weight_depth
        
        target_semantic = results['target_semantic']
        semantic = results['render_semantic']
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights.type_as(semantic), reduction="mean"
        )
        semantic_loss = criterion(semantic, target_semantic.long())
        losses['loss_render_semantic'] = semantic_loss * self.weight_semantic
    

        if self.weight_entropy_last > 0:
            pout = results['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            losses['loss_sdf_entropy'] = self.weight_entropy_last * entropy_last_loss

        if self.weight_distortion > 0:
            n_max = len(results['t'])
            loss_distortion = flatten_eff_distloss(results['weights'], results['s'], 1/n_max, results['ray_id'])
            losses['loss_sdf_distortion'] =  self.weight_distortion * loss_distortion
        return losses

    def render_depth(self, results):
        src = results['weights'] * results['s']
        ray_id = results['ray_id']
        N_ray = results['N_ray']
        device = results['weights'].device

        # 使用 bincount 进行分组求和
        summed = torch.bincount(ray_id, weights=src, minlength=N_ray).to(device)
        depth = summed + 1e-7
        return depth * self.radius  

    def render_semantic(self, results):
        src = results['weights'].unsqueeze(-1) * results['semantic']  # [n_samples, num_classes]
        ray_id = results['ray_id']                                     # 分组索引 [n_samples]
        N_ray = results['N_ray']
        num_classes = results['num_classes']
        device = results['weights'].device

        # 初始化输出张量并分组求和
        semantic = torch.zeros((N_ray, num_classes), device=device)
        semantic.index_add_(dim=0, index=ray_id, source=src)
        return semantic
    
    # def activate_density(self, density, interval=None):
    #     interval = interval if interval is not None else self.voxel_size_ratio
    #     shape = density.shape
    #     return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        
        # 原始实现使用的自定义 autograd 函数
        # return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)
        
        # 新实现（直接数值计算 + 自动微分）
        shifted_density = nn.softplus(density + self.act_shift)
        alpha = 1 - torch.exp(-shifted_density * interval)
        return alpha

    def forward(self, density, semantic, rays=None, **kwargs):
        ###ray meaning, ray.shape(,13) 
        #[0:2] xy coordinates of voxel centers and its sampling points in image/pixel coordinates
        #[2:3] gt depth label, generated by performing knn on its surrounding lidar point
        #[3:4] gt semantic label, genearted by projecting to masked fcclip image
        #[4:7] centre of ray (Optical center of camera) in world coordinate
        #[7:10] directions of ray in world coordinate
        #[10:13] directions of ray in world coordinate (normalized)
        gt_depths = rays[..., 2]
        gt_semantics = rays[..., 3]
        ray_o = rays[..., 4:7]
        ray_d = rays[..., 7:10]

        losses = {}
        #for batch_id in range(rays.shape[0]): ##multi_frame
        torch.cuda.empty_cache()
        rays_o_tr = ray_o
        rays_d_tr = ray_d
        
        ## ================  depth & semantic supervision  ===================
        gt_depth = gt_depths ###generate by fcclip and lidar points interpolation
        gt_semantic = gt_semantics
        gt_depth[gt_depth>3] = 0   
        mask = gt_depth>0 
        target_depth = gt_depth[mask]
        target_semantic = gt_semantic[mask]
        
        results = {}
        results['target_semantic'] = target_semantic
        results['target_depth'] = target_depth
        print_memory('Before renering one scene')
        results.update(
            self.render_one_scene(
                rays_o_tr=rays_o_tr.to(density.device),
                rays_d_tr=rays_d_tr.to(density.device),
                mask = mask.to(density.device),
                density=density,
                semantic=semantic,
            )
        )

        # render depth & semantic
        if self.use_depth_sup:
            results['render_depth'] = self.render_depth(results)
        results['render_semantic'] = self.render_semantic(results)
        # compute loss
        loss_single = self.compute_loss(results)
        for key in loss_single:
            if key in losses:
                losses[key] = losses[key] + loss_single[key]
            else:
                losses[key] = loss_single[key]
        for key in losses:
            losses[key] = losses[key] / density.shape[0]
        return losses

