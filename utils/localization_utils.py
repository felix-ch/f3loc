from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import *


def localize(
    desdf: torch.tensor, rays: torch.tensor, orn_slice=36, return_np=True, lambd=40
) -> Tuple[torch.tensor]:
    """
    Localize in the desdf according to the rays
    Input:
        desdf: (H, W, O), counter clockwise
        rays: (V,) from left to right (clockwise)
        orn_slice: number of orientations
        return_np: return as ndarray instead of torch.tensor
        lambd: parameter for likelihood
    Output:
        prob_vol: probability volume (H, W, O), ndarray
        prob_dist: probability distribution, (H, W) maxpool the prob_vol along orientation, ndarray
        orientations: orientation with max likelihood at each position, (H, W), ndarray
        pred: (3, ) predicted state [x,y,theta], ndarray
    """

    # flip the ray, to make rotation direction mathematically positive
    rays = torch.flip(rays, [0])
    O = desdf.shape[2]
    V = rays.shape[0]
    # expand rays to have the same dimension as desdf
    rays = rays.reshape((1, 1, -1))

    # circular pad the desdf
    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # probablility is -l1norm
    prob_vol = torch.stack(
        [
            -torch.norm(pad_desdf[:, :, i : i + V] - rays, p=1.0, dim=2)
            for i in range(O)
        ],
        dim=2,
    )  # (H,W,O)
    prob_vol = torch.exp(prob_vol / lambd)  # NOTE: here make prob positive

    # maxpooling
    prob_dist, orientations = torch.max(prob_vol, dim=2)

    # get the prediction
    pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
    orn = orientations[pred_y, pred_x]
    # from orientation indices to radians
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x, pred_y, orn))
    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )


def get_ray_from_depth(d, V=11, dv=10, a0=None, F_W=3 / 8):
    """
    Shoot the rays to the depths, from left to right
    Input:
        d: 1d depths from image
        V: number of rays
        dv: angle between two neighboring rays
        a0: camera intrisic
        F/W: focal length / image width
    Output:
        rays: interpolated rays
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        # assume a0 is in the middle of the image
        w = np.tan(angles) * W * F_W + (W - 1) / 2  # desired width, left to right
    else:
        w = np.tan(angles) * W * F_W + a0  # left to right

    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)

    return rays


def transit(
    prob_vol,
    transition,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):
    """
    Input:
        prob_vol: torch.tensor(H, W, O), probability volume before the transition
        transition: ego motion
        sig_o: stddev of rotation
        sig_x: stddev in x translation
        sig_w: stddev in y translation
        tsize: translational filter size
        rsize: rotational filter size
        resolution: resolution of the grid [m/pixel]
    """
    H, W, O = list(prob_vol.shape)
    # construction O filters
    filters_trans, filter_rot = get_filters(
        transition,
        O,
        sig_o=sig_o,
        sig_x=sig_x,
        sig_y=sig_y,
        tsize=tsize,
        rsize=rsize,
        resolution=resolution,
    )  # (O, 5, 5), (5,)

    # set grouped 2d convolution, O as channels
    prob_vol = prob_vol.permute((2, 0, 1))  # (O, H, W)

    # convolve with the translational filters
    # NOTE: make sure the filter is convolved correctly need to flip
    prob_vol = F.conv2d(
        prob_vol,
        weight=filters_trans.unsqueeze(1).flip([-2, -1]),
        bias=None,
        groups=O,
        padding="same",
    )  # (O, H, W)

    # convolve with rotational filters
    # reshape as batch
    prob_vol = prob_vol.permute((1, 2, 0))  # (H, W, O)
    prob_vol = prob_vol.reshape((H * W, 1, O))  # (HxW, 1, O)
    prob_vol = F.pad(
        prob_vol, pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)], mode="circular"
    )
    prob_vol = F.conv1d(
        prob_vol, weight=filter_rot.flip(dims=[-1]).unsqueeze(0).unsqueeze(0), bias=None
    )  # TODO (HxW, 1, O)

    # reshape
    prob_vol = prob_vol.reshape([H, W, O])  # (H, W, O)
    # normalize
    prob_vol = prob_vol / prob_vol.sum()

    return prob_vol


def get_filters(
    transition,
    O=36,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):
    """
    Return O different filters according to the ego-motion
    Input:
        transition: torch.tensor (3,), ego motion
    Output:
        filters_trans: torch.tensor (O, 5, 5)
                    each filter is (fH, fW)
        filters_rot: torch.tensor (5)
    """
    # NOTE: be careful about the orienation order, what is the orientation of the first layer?

    # get the filters according to gaussian
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
    )
    # add units
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # calculate center of the gaussian for 36 orientations
    # center for orientation stays the same
    center_o = transition[-1]
    # center_x and center_y depends on the orientation, in total O different, rotate
    orns = (
        torch.arange(0, O, dtype=torch.float32, device=transition.device)
        / O
        * 2
        * torch.pi
    )  # (O,)
    c_th = torch.cos(orns).reshape((O, 1, 1))  # (O, 1, 1)
    s_th = torch.sin(orns).reshape((O, 1, 1))  # (O, 1, 1)
    center_x = transition[0] * c_th - transition[1] * s_th  # (O, 1, 1)
    center_y = transition[0] * s_th + transition[1] * c_th  # (O, 1, 1)

    # add uncertainty
    filters_trans = torch.exp(
        -((grid_x - center_x) ** 2) / (sig_x**2) - (grid_y - center_y) ** 2 / (sig_y**2)
    )  # (O, 5, 5)
    # normalize
    filters_trans = filters_trans / filters_trans.sum(-1).sum(-1).reshape((O, 1, 1))

    # rotation filter
    grid_o = (
        torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=transition.device)
        / O
        * 2
        * torch.pi
    )
    filter_rot = torch.exp(-((grid_o - center_o) ** 2) / (sig_o**2))  # (5)

    return filters_trans, filter_rot
