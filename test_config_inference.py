_base_ = [
    '../mmdetection3d/configs/_base_/default_runtime.py'
]
dataset_type='CustomFSNDataset'
data_root = 'semantic_data'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
point_cloud_range = [0, -2, -1, 2, 2, 1]
occ_size = [40, 80, 40]
use_semantic = True
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names=[ 'windows', 'walls', 'floor', 'cabinets', 'tables', 'chairs', 'trash', 'others', 'unlabeled','carpet_trash']
_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_w_ = [20, 10, 5] #X axis
volume_h_ = [40, 20, 10]#Y axis
volume_z_ = [20, 10, 5] #Z axis
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFSNfiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='FSN_CustomCollect3D', keys=['img', 'gt_occ'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFSNfiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='FSN_CustomCollect3D', keys=['img', 'gt_occ'])
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=32,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        train_data_path='semantic_data/train_data',
        test_data_path='semantic_data/test_data',
        parameters_path='semantic_data/parameters',
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_train.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        train_data_path='semantic_data/train_data',
        test_data_path='semantic_data/test_data',
        parameters_path='semantic_data/parameters',
        test_mode=True,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
seed=42

model = dict(
    type='FSN',
    use_grid_mask=True,
    use_semantic=use_semantic,
    is_vis=True,
    img_backbone=dict(
       type='ResNet',
       depth=101,
       num_stages=4,
       out_indices=(1,2,3),
       frozen_stages=1,
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,
       style='caffe',
       #with_cp=True, # using checkpoint to save GPU memory
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
       stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='FSNHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=13,
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64], # _dim_ = [128, 256, 512] stride=1 input_dim=2*output_dim stride==2: input_dim=output_dim
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1,2,1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            num_cams=3,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                            num_cams=3
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
),

)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 5

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
    )
checkpoint_config = dict(interval=1,file_client_args=dict(backend='disk'))
gpu_ids=[0]
