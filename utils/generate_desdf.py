"""
Utils and example of how to generate desdf for localization
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils.utils import ray_cast


def raycast_desdf(
    occ, orn_slice=36, max_dist=10, original_resolution=0.01, resolution=0.1
):
    """
    Get desdf from occupancy grid through brute force raycast
    Input:
        occ: the map as occupancy
        orn_slice: number of equiangular orientations
        max_dist: maximum raycast distance, [m]
        original_resolution: the resolution of occ input [m/pixel]
        resolution: output resolution of the desdf [m/pixel]
    Output:
        desdf: the directional esdf of the occ input in meter
    """
    # assume occ resolution is 0.01
    ratio = resolution / original_resolution
    desdf = np.zeros(list((np.array(occ.shape) // ratio).astype(int)) + [orn_slice])
    # iterate through orientations
    for o in tqdm.tqdm(range(orn_slice)):
        theta = o / orn_slice * np.pi * 2
        # iterate through all pixels
        for row in tqdm.tqdm(range(desdf.shape[0])):
            for col in range(desdf.shape[1]):
                pos = np.array([row, col]) * ratio
                desdf[row, col, o] = ray_cast(
                    occ, pos, theta, max_dist / original_resolution
                )

    return desdf * original_resolution


if __name__ == "__main__":
    """
    This is just an example
    """
    # map path
    map_path = os.path.join("./data", "map.png")

    occ = cv2.imread(map_path)[:, :, 0]
    desdf = {}

    # ray cast desdf
    desdf["desdf"] = raycast_desdf(
        occ, orn_slice=36, max_dist=15, original_resolution=0.1, resolution=0.1
    )

    # save desdf
    scene_dir = os.path.join("./data", "scene_name")
    if not os.path.exists(scene_dir):
        os.mkdir(scene_dir)
    save_path = os.path.join(scene_dir, "desdf.npy")
    np.save(save_path, desdf)
