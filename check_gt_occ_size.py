import os
import glob
import numpy as np
dir_path='visual_dir'
train_dirs=glob.glob(os.path.join(dir_path,'**'))
for train_dir in train_dirs:
    pred_occ_path=os.path.join(train_dir,'pred.npy')
    pred_occ=np.load(pred_occ_path)
    print(pred_occ.shape)
    print(max(pred_occ[:,0]),min(pred_occ[:,0]))
    print(max(pred_occ[:,1]),min(pred_occ[:,1]))
    print(max(pred_occ[:,2]),min(pred_occ[:,2]))

