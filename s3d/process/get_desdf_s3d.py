from utils.generate_desdf import *
import cv2
import numpy as np
import tqdm
import os

# load test scene
disk_dir = "~/Downloads"
dataset_dir = os.path.join(disk_dir, "Structured3D")
scene_range = [-1, 5000] #NOTE: set this if want to process only a range of scenes
save = True

for scene in tqdm.tqdm(sorted(os.listdir(dataset_dir))):
    if not os.path.isdir(os.path.join(dataset_dir,scene)):
        continue

    if int(scene[-5:]) > scene_range[1] or int(scene[-5:]) < scene_range[0]:
        continue

    map_path = os.path.join(dataset_dir, scene, "map.png")
    occ = cv2.imread(map_path)[:,:,0]
    print("size: ", occ.shape)

    # get boundary
    #   l       r
    # t o -------
    #   |
    #   |
    # b |
    
    l = np.min(np.where(occ == 0)[1]) // 5 *5
    r = (np.max(np.where(occ == 0)[1]) // 5 + 1) * 5
    t = np.min(np.where(occ == 0)[0]) // 5 * 5
    b = (np.max(np.where(occ == 0)[0]) // 5 + 1) * 5

    # cut occ
    occ = occ[t:b, l:r]

    desdf = {}
    desdf["l"] = l
    desdf["t"] = t
    desdf["comment"] = "desdf coordinate to 0.02 map coordinate : x_desdf*5+l, y_desdf*5+t"
    # ray cast desdfsslvpn.ethz.ch/student-net
    desdf["desdf"] = raycast_desdf(occ, orn_slice=36, max_dist=20, original_resolution=0.02)

    if save:
        # save desdf
        scene_dir = os.path.join(disk_dir, "s3d_desdf", scene)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        save_path = os.path.join(scene_dir, "desdf.npy")
        np.save(save_path, desdf)

    # plt.figure(3)
    # for i in range(desdf["desdf"].shape[-1]):
    #     # plt.figure(str(i)+" th rotation")
    #     plt.imshow(desdf["desdf"][:,:,i], origin="lower")
    #     plt.pause(0.1)