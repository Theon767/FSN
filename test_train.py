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
import logging
from projects.mmdet3d_plugin.surroundocc.apis.mmdet_train import custom_train_detector
if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg = Config.fromfile('test_config.py')
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
    custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=False,
            validate=False,
            timestamp=timestamp,
            meta=None)
    
