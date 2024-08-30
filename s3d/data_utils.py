import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from utils.utils import *

class S3DDataset(Dataset):

    def __init__(self, dataset_dir, scene_range, depth_suffix="depth40", return_original=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_range = scene_range
        self.depth_suffix = depth_suffix
        self.scene_names = []
        self.scene_start_idx = []
        self.gt_depths = []
        self.gt_poses = []
        self.return_original = return_original
        self.load_scene_start_idx_and_rays()
        self.N = self.scene_start_idx[-1]


    def __len__(self):
        return self.N

    
    def load_scene_start_idx_and_rays(self):
        self.scene_start_idx.append(0)
        start_idx = 0 
        for scene in sorted(os.listdir(self.dataset_dir)):
            if not os.path.isdir(os.path.join(self.dataset_dir, scene)):
                continue
            
            if int(scene[-5:]) > self.scene_range[1] or int(scene[-5:]) < self.scene_range[0]:
                continue
            
            self.scene_names.append(scene)
            # get depth
            depth_file = os.path.join(self.dataset_dir, scene, self.depth_suffix+".txt")
                
            # read depth
            with open(depth_file, "r") as f:
                depth_txt = [line.strip() for line in f.readlines()]
            
            traj_len = len(depth_txt)
            scene_depths = []
            for state_id in range(traj_len):

                # get depth
                depth = depth_txt[state_id].split(" ")
                depth = np.array([float(d) for d in depth])
                scene_depths.append(depth)

            # read pose
            pose_file = os.path.join(self.dataset_dir, scene, "poses_map.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            scene_poses = []
            for state_id in range(traj_len):
                # get pose
                pose = poses_txt[state_id].split(" ")
                pose = np.array([float(s) for s in pose]).astype(np.float32)
                scene_poses.append(pose)


            start_idx += traj_len
            self.scene_start_idx.append(start_idx)
            self.gt_depths.append(scene_depths)
            self.gt_poses.append(scene_poses)
        
        self.gt_depths = np.concatenate([np.stack(scene_depths) for scene_depths in self.gt_depths])
        self.gt_poses = np.concatenate([np.stack(scene_poses) for scene_poses in self.gt_poses])
    
    def __getitem__(self, idx):
        # get the scene name according to the idx
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        # read the image
        image_path = os.path.join(
            self.dataset_dir,
            scene_name,
            "rgb",
            str(idx_within_scene).zfill(3) + ".png",
        )
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # align the image here
        # hardcoded intrinsics from fov
        K = np.array([[320/np.tan(0.698132), 0, 320],
                    [0, 180/np.tan(0.440992), 180],
                    [0, 0, 1]], dtype=np.float32)
        
        # align the image
        if self.return_original:
            original_img = img.copy()
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) / 255.0
        # normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)

        
        # gravity align
        img = gravity_align(img, r=self.gt_poses[idx][-2], p=self.gt_poses[idx][-1], K=K, visualize=False)

        # compute the attention mask
        mask = np.ones(list(img.shape[:2]))
        mask = gravity_align(mask, r=self.gt_poses[idx][-2], p=self.gt_poses[idx][-1], K=K, visualize=False)
        mask[mask < 1] = 0
        mask = mask.astype(np.uint8)
        img = np.transpose(img, [2, 0, 1]).astype(np.float32) # (C, H, W)
       

        data_dict = {"gt_rays": self.gt_depths[idx], "img": img, "mask":mask}
        if self.return_original:
            data_dict["original_img"] = original_img
        return data_dict