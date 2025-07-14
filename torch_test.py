import pickle
import os
import numpy as np
pkl_path=os.path.join(os.getcwd(),'data','nuscenes_infos_train.pkl')
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
sensor2lidar_r=data['infos'][0]['cams']['CAM_FRONT_RIGHT']['sensor2lidar_rotation']
sensor2lidar_t=data['infos'][0]['cams']['CAM_FRONT_RIGHT']['sensor2lidar_translation']
print('sensor2lidar')
print(sensor2lidar_r)
print(sensor2lidar_t)
lidar2cam_r = np.linalg.inv(sensor2lidar_r)
lidar2cam_t = -lidar2cam_r @ sensor2lidar_t
print('lidar2cam')
print(lidar2cam_r)
print(lidar2cam_t)
lidar2cam_rt = np.eye(4)
lidar2cam_rt[:3, :3] = lidar2cam_r.T
lidar2cam_rt[3, :3] = -lidar2cam_t