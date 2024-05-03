"""
This is module uses multiview and mono as complementary module based on relative poses
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.resnet import *

from modules.mono.depth_net import *
from modules.mv.mv_depth_net import *
from modules.network_utils import *


class comp_d_net(nn.Module):
    def __init__(
        self,
        mv_net: mv_depth_net,
        mono_net: depth_net,
        L=3,
        C=64,
        d_min=0.1,
        d_max=15.0,
        d_hyp=-0.2,
        D=128,
        use_pred=True,
    ) -> None:
        super().__init__()
        self.L = L
        self.C = C
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.use_pred = use_pred
        self.mv_net = mv_net
        self.mono_net = mono_net

        # freeze the weights
        self.mv_net.requires_grad_(False)
        self.mono_net.requires_grad_(False)
        self.nW = 2
        self.selector = selector(L=self.L, C=self.C, nW=self.nW, use_pred=self.use_pred)

    def forward(self, x):
        # compute relative poses
        rel_pose = get_rel_pose(x["ref_pose"], x["src_pose"])  # (N, L, 3)

        # get multi-view prediction
        mv_dict = self.mv_net(x)
        prob_mv = mv_dict["prob"].unsqueeze(1)  # (N, 1, fW, D)
        fW = prob_mv.shape[2]

        # only use single frame mono
        d_mono, _, prob_mono = self.mono_net(
            x["ref_img"], x["ref_mask"]
        )  # d_mono: (N, fWm), prob_mono: (N, fWm, D)
        prob_mono = prob_mono.unsqueeze(1)  # (N, 1, fWm, D)
        d_mono = d_mono.unsqueeze(1)  # (N, 1, fWm)
        mono_dict = {"d_mono": d_mono, "prob_mono": prob_mono}

        if self.use_pred:
            mv_pred_d = mv_dict["d"].mean(dim=-1)  # (N,)
            mono_pred_d = d_mono.mean(dim=-1).squeeze(1)  # (N,)
            comp_w = self.selector(
                rel_pose, d=torch.stack((mv_pred_d, mono_pred_d), dim=-1)
            )
        else:
            # get the weights based on the relative poses
            comp_w = self.selector(rel_pose)  # (N, nW)

        # resize the mono prob
        prob_mono = TF.resize(
            prob_mono, [fW, prob_mono.shape[-1]]
        )  # (N, L or 1, fW, D)

        # fuse the probs
        prob_comp = (
            torch.cat((prob_mv, prob_mono), dim=1) * comp_w.unsqueeze(-1).unsqueeze(-1)
        ).sum(
            dim=1
        )  # (N, fW, D)

        # depth
        d_vals = torch.linspace(
            self.d_min**self.d_hyp,
            self.d_max**self.d_hyp,
            self.D,
            device=prob_comp.device,
        ) ** (
            1 / self.d_hyp
        )  # (D,)
        d = torch.sum(prob_comp * d_vals, dim=-1)  # (N, fW)

        return {
            "d_comp": d,
            "prob_comp": prob_comp,
            "prob_mono": prob_mono,
            "comp_w": comp_w,
            "mv_dict": mv_dict,
            "mono_dict": mono_dict,
        }


class selector(nn.Module):
    def __init__(self, L=3, C=64, nW=4, use_pred=True):
        super().__init__()
        """
        Given L relative poses this generates a set of weights (on the probability volumes)
        Argument:
            use_pred: if True, also use the predicted depths as input
        """
        self.L = L
        self.C = C
        self.nW = nW  # number of weights to output
        self.use_pred = use_pred
        self.in_feat = self.L * 3 if not self.use_pred else self.L * 3 + 2
        self.mlp = nn.Sequential(
            nn.Linear(self.in_feat, self.C),
            nn.BatchNorm1d(self.C),
            nn.ReLU(),
            nn.Linear(self.C, self.C),
            nn.BatchNorm1d(self.C),
            nn.ReLU(),
            nn.Linear(self.C, self.nW),
            nn.Softmax(dim=-1),
        )

    def forward(self, x, d=None):
        """
        Input:
            x: L relative poses, (N, L, 3)
            d: 2 predicted depth, from mv and mono, (N, 2)
        Output:
            x: (N, nW), nW weights
        """
        # stack the pose together
        x = x.reshape((-1, self.L * 3))  # (N, Lx3)
        if self.use_pred:
            x = torch.cat((x, d), dim=-1)  # (N, Lx3+2)
        x = self.mlp(x)  # (N, nW)
        return x
