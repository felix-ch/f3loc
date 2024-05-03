"""
This is module predict the structural ray scan from perspective image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import *

from modules.network_utils import *


class depth_net(nn.Module):
    def __init__(self, d_min=0.1, d_max=15.0, d_hyp=-0.2, D=128) -> None:
        super().__init__()
        """
        depth_feature extracts depth features:
        img (480, 640, 3) => (1, 40, 32)

        FC to predict depth:
        (1, 40, 32) => FC => (1, 40)

        Alternative: directly use the (1, 40, 128) as probability volume, supervise on softmax
        """
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.depth_feature = depth_feature_res()

    def forward(self, x, mask=None):
        # extract depth features
        x, attn = self.depth_feature(x, mask)  # (N, fW, D)

        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=x.device
        ) ** (
            1 / self.d_hyp
        )  # (D,)

        # for probability volume using soft-max
        prob = F.softmax(x, dim=-1)  # (N, fW, D)

        # weighted average
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return d, attn, prob


class depth_feature_res(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        Resnet backbone:
        (480, 640, 3) -> Resnet50(3) -> (30, 40, 1024) # this is the conv feature

        Attn:
        img (480, 640, 3) => ResNet (30, 40, 32) => Avg_Pool (1, 40, 32) ==> Q_proj (1, 40, 32) ==> Attention (1, 40, 32) # this is the attn feature
                            (+ positional encoding)  (+ positional encoding)                             / |
                                            | \=============================> K_proj (30, 40, 32) ======/  |
                                            |===============================> V_proj (30, 40, 32) =========|

        """
        res50 = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False, True]
        )
        self.resnet = nn.Sequential(
            IntermediateLayerGetter(res50, return_layers={"layer4": "feat"})
        )
        self.conv = ConvBnReLU(
            in_channels=2048, out_channels=128, kernel_size=3, padding=1, stride=1
        )

        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )

        # Attention block, attetion to 2D image
        self.q_proj = nn.Linear(160, 128, bias=False)
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        x = self.resnet(x)["feat"]  # (N, 1024, fH, fW)
        x = self.conv(x)  # (N, 32, fH, fW)
        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # reduce vertically
        query = x.mean(dim=2)  # (N, 32, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, 32)

        # reshape from (N, 32, fH, fW) to (N, 32, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 32, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 32)

        # compute 2d positional encoding here
        # todo:
        # Example: for (4, 4) image
        # | (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5)|
        # | (-0.5, -1.5), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.5)|
        # | (0.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 1.5)|
        # | (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)|
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_grid_2d_x, pos_grid_2d_y = torch.meshgrid(pos_x, pos_y)
        pos_grid_2d = torch.stack((pos_grid_2d_x, pos_grid_2d_y), dim=-1)  # (fH, fW, 2)
        pos_enc_2d = self.pos_mlp_2d(pos_grid_2d)  # (fH, fW, 32)
        pos_enc_2d = pos_enc_2d.reshape((1, -1, 32))  # (1, fHxfW, 32)
        pos_enc_2d = pos_enc_2d.repeat((N, 1, 1))
        x = torch.cat((x, pos_enc_2d), dim=-1)  # (N, fHxfW, 32+32)

        # get the 1d positional encoding here
        # todo:
        # Example: for (5, ) ray
        # |-2, -1, 0, 1, 2|
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5  # (fW,)
        pos_enc_1d = self.pos_mlp_1d(pos_v.reshape((-1, 1)))  # (fW, 32)
        pos_enc_1d = pos_enc_1d.reshape((1, -1, 32)).repeat((N, 1, 1))  # (N, fW, 32)
        query = torch.cat((query, pos_enc_1d), dim=-1)  # (N, fW, 32+32)

        # attention
        query = self.q_proj(query)  # (N, fW, 32)
        key = self.k_proj(x)  # (N, fHxfW, 32)
        value = self.v_proj(x)  # (N, fHxfW, 32)

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask as True on valid values
            mask = mask.reshape((mask.shape[0], 1, -1))  # (N, 1, fHxfW)
            # same mask for all fW
            mask = mask.repeat(1, fW, 1)  # (N, fW, fHxfW)
        x, attn_w = self.attn(query, key, value, attn_mask=mask)  # (N, fW, 32)

        return x, attn_w
