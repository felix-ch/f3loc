import numpy as np
from utils.utils import *
import matplotlib.pyplot as plt
import os
import tqdm

# ==================================
# ====  1. set data_dir
# ====  2. set ray_n
# ==================================

resolution = 0.02
dist_max = 20 / resolution
save = True
visualize = False
plt.ion()
data_dir = "~/Downloads/Structured3D"
# get all scene name (dir) in the path
scenes = sorted(os.listdir(data_dir))

# define ray setting
ray_n = 40
F_W = 1/np.tan(0.698132)/2
# define angles
# NOTE: The angles and rays are from left to right, since the image features are from left to right 

# equidistant 40 pixels in radians
center_angs = np.flip(np.arctan2((np.arange(ray_n) - np.arange(ray_n).mean()), ray_n*F_W))
ray_dir = data_dir
if not os.path.exists(ray_dir):
    os.makedirs(ray_dir)
ray_file_suffix = "depth"+str(ray_n)+".txt"


skip = False
continue_at=None
# scenes = scenes[:200]
# scenes = scenes[200:400]
# scenes = scenes[400:600]
# scenes = scenes[600:800]
# scenes = scenes[800:1000]
# scenes = scenes[1000:1200]
# scenes = scenes[1200:1400]
# scenes = scenes[1400:1600]
# scenes = scenes[1600:1800]
# scenes = scenes[1800:2000]
# scenes = scenes[2000:2200]
# scenes = scenes[2200:2400]
# scenes = scenes[2200:2600]
# scenes = scenes[2600:2800]
# scenes = scenes[2800:3000]
# scenes = scenes[3000:]

# iterate though all scenes
for scene in tqdm.tqdm(scenes):
    # print("========== ", scene, " ==========")
    if scene == continue_at:
        skip = False

    if skip:
        continue
    # get map
    map_path = os.path.join(data_dir, scene, 'map.png')
    occ = cv2.imread(map_path)[:,:,0]

    # get poses
    poses_file = os.path.join(data_dir, scene, "poses_map.txt")
    
    # read poses
    with open(poses_file, "r") as f:
        poses_txt = [line.strip() for line in f.readlines()]

    # ray file
    if not os.path.exists(os.path.join(ray_dir, scene)):
        os.makedirs(os.path.join(ray_dir, scene))
    ray_file = os.path.join(ray_dir, scene, ray_file_suffix)
    rays = []
    # iterate through poses
    for pose_txt in poses_txt: 
        pose = pose_txt.split(" ")
        x = float(pose[0])
        y = float(pose[1])
        th = float(pose[2])
        # form the image coordinate of the pose
        pos = np.array([y, x])
        # iterate through viewing angles
        angs = center_angs + th
        ray = []
        for i, ang in enumerate(angs):
            # ray cast
            # this is the depth of the ray
            ray.append(ray_cast(occ, pos, ang, dist_max)*np.cos(center_angs[i])) 

        if visualize:
            plt.figure(1)
            plt.clf()
            plt.imshow(occ, cmap="gray", origin="lower")
            for i, theta in enumerate(angs):
                plt.plot([pos[1], pos[1]+ray[i]/np.cos(center_angs[i])*np.cos(theta)], [pos[0], pos[0]+ray[i]/np.cos(center_angs[i])*np.sin(theta)], "b")
            plt.axis("equal")
            plt.pause(0.1)
        ray = np.array(ray) * resolution
        rays.append(ray)
        
        
    if save: 
        # store the dist
        out_file = open(ray_file, "w")
        for ray in rays:
            line = ''.join([str(dist)+" " for dist in ray]) + '\n'
            out_file.write(line)
        out_file.close()
