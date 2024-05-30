# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import sys
#from dgp.datasets import SynchronizedSceneDataset
import pickle
import pdb
import cv2

from .mono_dataset import MonoDataset

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import torch

class NuscDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NuscDataset, self).__init__(*args, **kwargs)
        self.split = 'train' if self.is_train else 'val'
        version = 'v1.0-trainval'
        self.data_path = '/mnt/nas/algorithm/xianda.guo/intern/wenjie.yuan/data/nuscenes/raw_data'
        self.nusc = NuScenes(version=version, dataroot=self.data_path, verbose=False)

        # train depth
        # self.truedepth_path = '/mnt/nas/algorithm/xianda.guo/intern/wenjie.yuan/data/nuscenes/depth/depth_train'   # 1帧----------
        self.truedepth_path = '/mnt/nas/algorithm/xianda.guo/intern/wenjie.yuan/data/nuscenes/depth_20f'   # 20帧拼接
        # self.match_path = '/mnt/nas/algorithm/xianda.guo/intern/wenjie.yuan/code/s3depth/data/nuscenes/match'


        # val depth
        self.depth_path = '/mnt/nas/algorithm/xianda.guo/intern/wenjie.yuan/data/nuscenes/depth/depth_val'
        # self.depth_path = '/mnt/nas/algorithm/xianda.guo/intern/wenjie.yuan/data/nuscenes/depth/depth_train'

        # pre and next
        if self.split == 'train':
            # with open('datasets/nusc/depth_train_prenext.txt', 'r') as f:    # 训练的索引--------------
            with open('datasets/nusc/depth_10f.txt', 'r') as f:
                self.filenames = f.readlines()
        else:
            with open('datasets/nusc/{}.txt'.format(self.split), 'r') as f:
            # with open('datasets/nusc/depth_10f.txt'.format(self.split), 'r') as f:
                self.filenames = f.readlines()
        self.camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

    def img2col(self, x, ksize, strid):
        h,w = x.shape # (5, 3, 34, 34)
        img_col = []
        for i in range(0, h-ksize+1, strid):
            for j in range(0, w-ksize+1, strid):
                col = x[i:i+ksize, j:j+ksize].reshape(-1) # (1, 3, 4, 4) # 48
                img_col.append(col)
        return np.array(img_col) # (5, 3, 31, 31, 48)
    
    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []
        if self.is_train:
            if self.opt.use_lidar:
                inputs["lidar_depth"] = []
                
            if self.opt.use_sfm_loss:
                inputs["match_spatial"] = []

            for idx, i in enumerate(self.frame_idxs[1:]):
                inputs[("color", i, -1)] = []
                inputs[("pose_spatial", i)] = []

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = [] 
            
            inputs["pose_spatial"] = []
        else:
            inputs[('K_ori', 0)] = [] 
            inputs['depth'] = []

        inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []
        rec = self.nusc.get('sample', index_temporal)
        pred_gt_name = {}
        for index_spatial in range(6):
            cam_sample = self.nusc.get(
                'sample_data', rec['data'][self.camera_names[index_spatial]])
            pred_gt_name[self.camera_ids[index_spatial]] = cam_sample['filename'][:-4] + '.npy'
            inputs['id'].append(self.camera_ids[index_spatial])
            # print('self.data_path:', os.path.join(self.data_path, cam_sample['filename'])) # ------
            color = self.loader(os.path.join(self.data_path, cam_sample['filename']))
            # color = self.loader('/mnt/goosefs/bjcar01/algorithm/public_data/det3d/nuscenes/origin/sweeps/CAM_FRONT_LEFT/n008-2018-08-01-15-52-19-0400__CAM_FRONT_LEFT__1533153406904799.jpg/mnt/goosefs/bjcar01/algorithm/public_data/det3d/nuscenes/origin/sweeps/CAM_FRONT_LEFT/n008-2018-08-01-15-52-19-0400__CAM_FRONT_LEFT__1533153406904799.jpg')

            inputs['width_ori'].append(color.size[0])
            inputs['height_ori'].append(color.size[1])
            
            if not self.is_train:
                try:
                    depth = np.load(os.path.join(self.depth_path, cam_sample['filename'][:-4] + '.npy'))
                except FileNotFoundError:
                    print(os.path.join(self.depth_path, cam_sample['filename'][:-4] + '.npy'))
                inputs['depth'].append(depth.astype(np.float32))
                #from PIL import Image
                #img = Image.fromarray(depth.astype(np.uint8))
                #img.save("/mnt/cfs/algorithm/wenjie.yuan/visualization-main/nusc_dispToDepth_gt/{}.png".format(index_spatial))
            
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)].append(color)

            ego_spatial = self.nusc.get(
                    'calibrated_sensor', cam_sample['calibrated_sensor_token'])

            if self.is_train:
                pose_0_spatial = Quaternion(ego_spatial['rotation']).transformation_matrix
                pose_0_spatial[:3, 3] = np.array(ego_spatial['translation'])

                inputs["pose_spatial"].append(pose_0_spatial.astype(np.float32))
            else:
                pose_0_spatial = Quaternion(ego_spatial['rotation']).transformation_matrix
                pose_0_spatial[:3, 3] = np.array(ego_spatial['translation'])

            K = np.eye(4).astype(np.float32)
            K[:3, :3] = ego_spatial['camera_intrinsic']
            inputs[('K_ori', 0)].append(K)

            if self.is_train:
                if self.opt.use_lidar:
                    # depth data
                    depth = np.load(os.path.join(self.truedepth_path, cam_sample['filename'][:-4] + '.npy'))
                    inputs['lidar_depth'].append(depth.astype(np.float32))
                
                if self.opt.use_sfm_loss:
                    # match data
                    pkl_path = os.path.join(os.path.join(self.match_path, cam_sample['filename'][:-4] + '.pkl'))
                    with open(pkl_path, 'rb') as f:
                        match_spatial_pkl = pickle.load(f)
                    inputs['match_spatial'].append(match_spatial_pkl['result'].astype(np.float32))

                for idx, i in enumerate(self.frame_idxs[1:]):
                    if i == -1:
                        index_temporal_i = cam_sample['prev']
                    elif i == 1:
                        index_temporal_i = cam_sample['next']
                    try:
                        cam_sample_i = self.nusc.get('sample_data', index_temporal_i)
                    except KeyError:
                        print(index_temporal_i)
                        print((i,cam_sample['prev'],cam_sample['next']))
                    ego_spatial_i = self.nusc.get('calibrated_sensor', cam_sample_i['calibrated_sensor_token'])

                    K = np.eye(4).astype(np.float32)
                    K[:3, :3] = ego_spatial_i['camera_intrinsic']
                    inputs[('K_ori', i)].append(K)

                    color = self.loader(os.path.join(self.data_path, cam_sample_i['filename']))
                    
                    if do_flip:
                        color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
                    inputs[("color", i, -1)].append(color)

                    pose_i_spatial = Quaternion(ego_spatial_i['rotation']).transformation_matrix
                    pose_i_spatial[:3, 3] = np.array(ego_spatial_i['translation'])

    
        if self.is_train:
            for index_spatial in range(6):
                for idx, i in enumerate(self.frame_idxs[1:]):
                    pose_0_spatial = inputs["pose_spatial"][index_spatial]
                    pose_i_spatial = inputs["pose_spatial"][(index_spatial+i)%6]

                    gt_pose_spatial = np.linalg.inv(pose_i_spatial) @ pose_0_spatial
                    inputs[("pose_spatial", i)].append(gt_pose_spatial.astype(np.float32))

            for idx, i in enumerate(self.frame_idxs):
                inputs[('K_ori', i)] = np.stack(inputs[('K_ori', i)], axis=0) 
                if i != 0:
                    inputs[("pose_spatial", i)] = np.stack(inputs[("pose_spatial", i)], axis=0)


            inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)   
        else:
            inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0) 
            inputs['depth'] = np.stack(inputs['depth'], axis=0)
            inputs['name'] = [pred_gt_name]

        for key in ['width_ori', 'height_ori']:
            inputs[key] = np.stack(inputs[key], axis=0)   








