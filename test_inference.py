from __future__ import division
from mmcv.runner import get_dist_info
from mmcv import Config
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import train_detector
import time
import os
from mmdet3d.utils import collect_env, get_root_logger
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from projects.mmdet3d_plugin.surroundocc.apis.test import custom_multi_gpu_test
import logging
import torch
from projects.mmdet3d_plugin.surroundocc.apis.test import custom_multi_gpu_test,custom_single_gpu_test
from projects.mmdet3d_plugin.surroundocc.apis.mmdet_train import custom_train_detector
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
if __name__ == '__main__':
    distributed = False
    samples_per_gpu = 1
    checkpoint_path='work_dir/latest.pth'
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg = Config.fromfile('test_config_inference.py')
    dataset = build_dataset(cfg.data.train)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(os.path.join(os.getcwd(),'FSN_logs'), f'{timestamp}.log')
    logger = get_root_logger(
        log_file=log_file, log_level=logging.INFO, name='FSN')
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    log_file = os.path.join(os.getcwd(), f'{timestamp}.log')
    
    if cfg.checkpoint_config is not None:
    # save mmdet version, config file content and class names in
    # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.pretty_text)
    cfg.work_dir = os.path.join(os.getcwd(), 'work_dir')
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])

    outputs = custom_single_gpu_test(model, data_loader, tmpdir=None, gpu_collect=True,is_vis=True)
