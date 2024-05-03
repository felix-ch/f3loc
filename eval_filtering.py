import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import tqdm
import yaml
from attrdict import AttrDict
from torch.utils.data import DataLoader

from modules.comp.comp_d_net_pl import *
from modules.mono.depth_net_pl import *
from modules.mv.mv_depth_net_pl import *
from utils.data_utils import *
from utils.localization_utils import *


def evaluate_filtering():
    parser = argparse.ArgumentParser(description="Observation evaluation.")
    parser.add_argument(
        "--net_type",
        type=str,
        default="comp",
        choices=[
            "d",
            "mvd",
            "comp",
            "comp_s",
        ],  # d: monocualr, mvd: multi-view, comp: learned complementary, comp_s: hard threshold complementary
        help="type of the network to evaluate. d: monocualr, mvd: multi-view, comp: learned complementary, comp_s: hard threshold complementary",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/Gibson Floorplan Localization Dataset",
        help="path of the dataset",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./logs", help="path of the checkpoints"
    )
    parser.add_argument(
        "--evol_path",
        type=str,
        default=None,
        help="path to save the tracking evolution figures",
    )
    parser.add_argument(
        "--traj_len", type=int, default=100, help="length of the trajectory"
    )
    args = parser.parse_args()

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("======= USING DEVICE : ", device, " =======")

    # network to evaluate
    net_type = args.net_type

    # paths
    dataset_dir = os.path.join(args.dataset_path, "gibson_t")
    depth_dir = args.dataset_path
    log_dir = args.ckpt_path
    desdf_path = os.path.join(args.dataset_path, "desdf")
    evol_path = args.evol_path

    # instanciate dataset
    traj_l = args.traj_len
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    test_set = TrajDataset(
        dataset_dir,
        split.test,
        L=traj_l,
        depth_dir=depth_dir,
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=True,
    )

    # logs
    log_error = True
    log_timing = True

    # parameters
    L = 3  # number of the source frames
    D = 128  # number of the depth planes
    d_min = 0.1  # minimum depth
    d_max = 15.0  # maximum depth
    d_hyp = -0.2  # depth transform (uniform sampling in d**d_hyp)
    F_W = 3 / 8  # camera intrinsic, focal length / image width
    orn_slice = 36  # number of discretized orientations
    trans_thresh = 0.005  # translation threshold (variance) if using comp_s

    # models
    if net_type == "mvd" or net_type == "comp_s":
        # instaciate model
        mv_net = mv_depth_net_pl.load_from_checkpoint(
            checkpoint_path=os.path.join(log_dir, "mv.ckpt"),
            D=D,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
        ).to(device)
    if net_type == "d" or net_type == "comp_s":
        # instaciate model
        d_net = depth_net_pl.load_from_checkpoint(
            checkpoint_path=os.path.join(log_dir, "mono.ckpt"),
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
        ).to(device)
    if net_type == "comp":
        mv_net_pl = mv_depth_net_pl(D=D, d_hyp=d_hyp, F_W=F_W)
        mono_net_pl = depth_net_pl(d_min=d_min, d_max=d_max, d_hyp=d_hyp, D=D, F_W=F_W)
        comp_net = comp_d_net_pl.load_from_checkpoint(
            checkpoint_path=os.path.join(log_dir, "comp.ckpt"),
            mv_net=mv_net_pl.net,
            mono_net=mono_net_pl.encoder,
            L=L,
            d_min=d_min,
            d_max=d_max,
            d_hyp=d_hyp,
            D=D,
            F_W=F_W,
            use_pred=True,
        ).to(device)
        comp_net.eval()  # this is needed to disable batchnorm

    # get desdf for the scene
    print("load desdf ...")
    desdfs = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        desdfs[scene] = np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        desdfs[scene]["desdf"][desdfs[scene]["desdf"] > 10] = 10  # truncate

    # get the ground truth pose file
    print("load poses and maps ...")
    maps = {}
    gt_poses = {}
    for scene in tqdm.tqdm(test_set.scene_names):
        # load map
        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        maps[scene] = occ
        h = occ.shape[0]
        w = occ.shape[1]

        # single trajectory
        poses = np.zeros([0, 3], dtype=np.float32)
        # get poses
        poses_file = os.path.join(dataset_dir, scene, "poses.txt")

        # read poses
        with open(poses_file, "r") as f:
            poses_txt = [line.strip() for line in f.readlines()]

        traj_len = len(poses_txt)
        traj_len -= traj_len % traj_l
        for state_id in range(traj_len):
            # get pose
            pose = poses_txt[state_id].split(" ")
            x = float(pose[0])
            y = float(pose[1])
            th = float(pose[2])
            # from world coordinate to map coordinate
            x = x / 0.01 + w / 2
            y = y / 0.01 + h / 2

            poses = np.concatenate(
                (poses, np.expand_dims(np.array((x, y, th), dtype=np.float32), 0)),
                axis=0,
            )

        gt_poses[scene] = poses

    # record stats
    RMSEs = []
    success_10 = []  # Success @ 1m
    success_5 = []  # Success @ 0.5m
    success_3 = []  # Success @ 0.3m
    success_2 = []  # Success @ 0.2m

    matching_time = 0
    iteration_time = 0
    feature_extraction_time = 0
    n_iter = 0

    # loop the over scenes
    for data_idx in tqdm.tqdm(range(len(test_set))):

        data = test_set[data_idx]
        # get the scene name according to the data_idx
        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]

        # get idx within scene
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        # get desdf
        desdf = desdfs[scene]

        # get reference pose in map coordinate and in scene coordinate
        poses_map = gt_poses[scene][
            idx_within_scene * traj_l : idx_within_scene * traj_l + traj_l, :
        ]

        # transform to desdf frame
        gt_pose_desdf = poses_map.copy()
        gt_pose_desdf[:, 0] = (gt_pose_desdf[:, 0] - desdf["l"]) / 10
        gt_pose_desdf[:, 1] = (gt_pose_desdf[:, 1] - desdf["t"]) / 10

        imgs = torch.tensor(data["imgs"], device=device).unsqueeze(0)
        poses = torch.tensor(data["poses"], device=device).unsqueeze(0)

        # set prior as uniform distribution
        prior = torch.tensor(
            np.ones_like(desdf["desdf"]) / desdf["desdf"].size, device=imgs.device
        ).to(torch.float32)

        pred_poses_map = []

        # loop over the sequences
        for t in range(traj_l - L):
            start_iter = time.time()
            feature_extraction_start = time.time()
            # form input
            input_dict = {}
            if net_type == "mvd" or net_type == "comp" or net_type == "comp_s":
                input_dict.update(
                    {
                        "ref_img": imgs[:, t + L, :, :, :],
                        "src_img": imgs[:, t : t + L, :, :, :],
                        "ref_pose": poses[:, t + L, :],
                        "src_pose": poses[:, t : t + L, :],
                        "ref_mask": None,  # no masks because the dataset has zero roll pitch
                        "src_mask": None,  # no masks because the dataset has zero roll pitch
                    }
                )
            if net_type == "d" or net_type == "comp_s":
                input_dict.update(
                    {
                        "img": imgs[:, t + L, :, :, :],
                        "mask": None,  # no masks because the dataset has zero roll pitch
                    }
                )
            # check which model to use if hardcoded selection
            if net_type == "comp_s":
                # calculate the relative poses
                pose_var = (
                    torch.cat(
                        (input_dict["ref_pose"].unsqueeze(1), input_dict["src_pose"]),
                        dim=1,
                    )
                    .squeeze(0)
                    .var(dim=0)[:2]
                    .sum()
                )
                if pose_var < trans_thresh:
                    use_mv = False
                    use_mono = True
                else:
                    use_mv = True
                    use_mono = False

            # inference
            if net_type == "mvd" or (net_type == "comp_s" and use_mv):
                pred_dict = mv_net.net(input_dict)
                pred_depths = pred_dict["d"]
            elif net_type == "d" or (net_type == "comp_s" and use_mono):
                pred_depths, attn_2d, prob = d_net.encoder(
                    input_dict["img"], input_dict["mask"]
                )
            elif net_type == "comp":
                pred_dict = comp_net.comp_d_net(input_dict)
                pred_depths = pred_dict["d_comp"]

            pred_depths = pred_depths.squeeze(0).detach().cpu().numpy()

            # get rays from depth
            pred_rays = get_ray_from_depth(pred_depths)
            pred_rays = torch.tensor(pred_rays, device=device)

            feature_extraction_end = time.time()

            matching_start = time.time()
            # use the prediction to localize, produce observation likelihood
            likelihood, likelihood_2d, _, likelihood_pred = localize(
                torch.tensor(desdf["desdf"]).to(prior.device),
                pred_rays.to(prior.device),
                return_np=False,
            )
            matching_end = time.time()

            # multiply with the prior
            posterior = prior * likelihood.to(prior.device)

            # reduce the posterior along orientation for 2d visualization
            posterior_2d, orientations = torch.max(posterior, dim=2)

            # compute prior_2d for visualization
            prior_2d, _ = torch.max(prior, dim=2)

            # maximum of the posterior as result
            pose_y, pose_x = torch.where(posterior_2d == posterior_2d.max())
            if pose_y.shape[0] > 1:
                pose_y = pose_y[0].unsqueeze(0)
                pose_x = pose_x[0].unsqueeze(0)
            orn = orientations[pose_y, pose_x]

            # from orientation indices to radians
            orn = orn / orn_slice * 2 * torch.pi
            pose = torch.cat((pose_x, pose_y, orn)).detach().cpu().numpy()

            pose_in_map = pose.copy()
            pose_in_map[0] = pose_in_map[0] * 10 + desdf["l"]
            pose_in_map[1] = pose_in_map[1] * 10 + desdf["t"]

            pred_poses_map.append(pose_in_map)

            if evol_path is not None:
                # plot posterior 2d
                fig = plt.figure(0, figsize=(20, 20))
                fig.clf()
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(
                    posterior_2d.detach().cpu().numpy(), origin="lower", cmap="coolwarm"
                )
                ax.quiver(
                    pose[0],
                    pose[1],
                    np.cos(pose[2]),
                    np.sin(pose[2]),
                    color="blue",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )
                ax.quiver(
                    gt_pose_desdf[t + L, 0],
                    gt_pose_desdf[t + L, 1],
                    np.cos(gt_pose_desdf[t + L, 2]),
                    np.sin(gt_pose_desdf[t + L, 2]),
                    color="green",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )
                ax.axis("off")
                ax.set_title(str(t) + " posterior")

                ax = fig.add_subplot(1, 2, 1)
                ax.imshow(likelihood_2d, origin="lower", cmap="coolwarm")
                ax.set_title(str(t) + " likelihood")
                ax.axis("off")
                ax.quiver(
                    likelihood_pred[0],
                    likelihood_pred[1],
                    np.cos(likelihood_pred[2]),
                    np.sin(likelihood_pred[2]),
                    color="blue",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )
                ax.quiver(
                    gt_pose_desdf[t + L, 0],
                    gt_pose_desdf[t + L, 1],
                    np.cos(gt_pose_desdf[t + L, 2]),
                    np.sin(gt_pose_desdf[t + L, 2]),
                    color="green",
                    width=0.2,
                    scale_units="inches",
                    units="inches",
                    scale=1,
                    headwidth=3,
                    headlength=3,
                    headaxislength=3,
                    minlength=0.1,
                )

                if not os.path.exists(
                    os.path.join(evol_path, "pretty_filter", str(data_idx))
                ):
                    os.makedirs(os.path.join(evol_path, "pretty_filter", str(data_idx)))
                fig.savefig(
                    os.path.join(
                        evol_path, "pretty_filter", str(data_idx), str(t) + ".png"
                    )
                )

            # transition
            # use ground truth to compute transitions, use relative poses
            if t + L == traj_l - 1:
                continue
            current_pose = poses[0, t + L, :]
            next_pose = poses[0, t + L + 1, :]

            transition = get_rel_pose(current_pose, next_pose)
            prior = transit(
                posterior, transition, sig_o=0.1, sig_x=0.1, sig_y=0.1, tsize=7, rsize=7
            )

            end_iter = time.time()
            matching_time += matching_end - matching_start
            feature_extraction_time += feature_extraction_end - feature_extraction_start
            iteration_time += end_iter - start_iter
            n_iter += 1

        if log_error:
            pred_poses_map = np.stack(pred_poses_map)
            # record success rate, from map to global
            last_errors = (
                ((pred_poses_map[-10:, :2] - poses_map[-10:, :2]) ** 2).sum(axis=1)
                ** 0.5
            ) * 0.01
            # compute RMSE
            RMSE = (
                ((pred_poses_map[-10:, :2] - poses_map[-10:, :2]) ** 2)
                .sum(axis=1)
                .mean()
            ) ** 0.5 * 0.01
            RMSEs.append(RMSE)
            print("last_errors", last_errors)
            if all(last_errors < 1):
                success_10.append(True)
            else:
                success_10.append(False)

            if all(last_errors < 0.5):
                success_5.append(True)
            else:
                success_5.append(False)

            if all(last_errors < 0.3):
                success_3.append(True)
            else:
                success_3.append(False)

            if all(last_errors < 0.2):
                success_2.append(True)
            else:
                success_2.append(False)

    if log_error:
        RMSEs = np.array(RMSEs)
        success_10 = np.array(success_10)
        success_5 = np.array(success_5)
        success_3 = np.array(success_3)
        success_2 = np.array(success_2)

        print("============================================")
        print("1.0 success rate : ", success_10.sum() / len(test_set))
        print("0.5 success rate : ", success_5.sum() / len(test_set))
        print("0.3 success rate : ", success_3.sum() / len(test_set))
        print("0.2 success rate : ", success_2.sum() / len(test_set))
        print("mean RMSE succeeded : ", RMSEs[success_10].mean())
        print("mean RMSE all : ", RMSEs.mean())

    if log_timing:
        feature_extraction_time = feature_extraction_time / n_iter
        matching_time = matching_time / n_iter
        iteration_time = iteration_time / n_iter

        print("============================================")
        print("feature_extraction_time : ", feature_extraction_time)
        print("matching_time : ", matching_time)
        print("iteration_time : ", iteration_time)


if __name__ == "__main__":
    evaluate_filtering()
