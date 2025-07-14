import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import os

################## Load NuScenes data ##################
file_path='data/v1.0-mini'
nusc = NuScenes(version='v1.0-mini',
                        dataroot=file_path,
                        verbose=True)
print(len(nusc.scene))
print(nusc.scene)
my_scene = nusc.scene[0]


################## Process bboxes NuScenes data ##################
sensor = 'LIDAR_TOP'
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

idar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
boxes_token = [box.token for box in boxes]
# for box in boxes:
#     print(type(box))
object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
# for object_token in object_tokens:
#     print(object_tokens)
object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]
# for category in object_category:
#     print(category)
locs = np.array([b.center for b in boxes]).reshape(-1, 3)
dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
rots = np.array([b.orientation.yaw_pitch_roll[0]
                    for b in boxes]).reshape(-1, 1)
gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
gt_bbox_3d[:, 6] += np.pi / 2.
gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1 # Slightly expand the bbox to wrap all object points


#####process lidar points with semantics####
pc_file_name = lidar_data['filename']
print(pc_file_name)
pc0 = np.fromfile(os.path.join(file_path, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4]
print(pc0.shape)
print(pc0[:,3])

