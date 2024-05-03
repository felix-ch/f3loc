"""
This is module predict the structural 1d depth from a sequence of perspective images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import *

from modules.network_utils import *


class mv_depth_net(nn.Module):
    def __init__(
        self, D=128, d_min=0.1, d_max=15.0, d_hyp=1.0, C=64, F_W=3 / 8
    ) -> None:
        super().__init__()
        """
        Given L + 1 images
        form (N, L+1, fW, C) depth features

        construct (N, fW, L+1, D, C) feature volume

        construct (N, fW, D, C) cost volume by projection and interpolation

        filter cost volume (N, fW, D)

        D: number of depth hypothesis
        d_hyp: order of the depth, e.g. d_hyp=-1 is inverse depth
        C: feature channels
        F_W: focal_length / width
        """
        self.D = D
        self.C = C
        self.depth_feature = mv_depth_feature_res(C=self.C)
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.F_W = F_W
        self.cost_filter = cost_filer(D=self.D, C=self.C)

    def forward(self, x):
        """
        Input:
            x: "ref_img": (N, 3, H, W)
               "ref_pose": (N, 3)
               "src_img": (N, L, 3, H, W)
               "src_pose": (N, L, 3)
               "ref_mask": (N, H, W)
               "src_mask": (N, L, H, W)
               "ref_depth": (N, W)
        Output:
            d: (N, fW), predicted depth of the reference
            prob: (N, fW, D)
            attn: (N, L+1, fW, fHxfW)
        """
        src_img = x["src_img"]
        ref_img = x["ref_img"]
        src_pose = x["src_pose"]
        ref_pose = x["ref_pose"]
        ref_mask = x["ref_mask"]
        src_mask = x["src_mask"]
        if "ref_intri" in x:
            ref_a0 = x["ref_intri"][:, -2] / 4
            src_a0 = x["src_intri"][:, :, -2] / 4
            ref_F = x["ref_intri"][:, 0] / 4
            src_F = x["src_intri"][:, :, 0] / 4

        else:
            ref_a0 = None  # The a0 in x is the intrinsic of the resolution of input image, should be divided by 4, since the output features are downsampled by 4
            src_a0 = None
            ref_F = None
            src_F = None

        N, L, _, H, W = list(src_img.shape)

        # compute the feature for all NxL images, feat (N, L+1, 32, fW)
        feat, attn = self.depth_feature(
            torch.concat((ref_img.unsqueeze(1), src_img), dim=1).reshape(-1, 3, H, W),
            (
                torch.concat((ref_mask.unsqueeze(1), src_mask), dim=1).reshape(-1, H, W)
                if ref_mask is not None
                else None
            ),
        )
        # WARN: feat is now (Nx(L+1), fW, C) need (N, L+1, C, fW)
        feat = feat.permute(0, 2, 1)  # (Nx(L+1), C, fW)
        feat = feat.reshape((N, L + 1, self.C, -1))  # (N, L+1, C, fW)

        attn = attn.reshape((N, L + 1, feat.shape[-1], -1))
        ref_feat = feat[:, 0, :, :]  # (N, C, fW)
        src_feat = feat[:, 1:, :, :]  # (N, L, C, fW)

        # set depth hypothesis
        d_vals = torch.linspace(
            self.d_min**self.d_hyp, self.d_max**self.d_hyp, self.D, device=feat.device
        ) ** (1 / self.d_hyp)

        # transform the gather the feature volumes to the corresponding reference imgs
        feat_vol, valid_vol = gather_feat_vol(
            ref_pose=ref_pose,
            src_pose=src_pose,
            ref_feat=ref_feat,
            src_feat=src_feat,
            d_vals=d_vals,
            F_W=self.F_W,
            ref_a0=ref_a0,
            src_a0=src_a0,
            ref_F=ref_F,
            src_F=src_F,
        )  # (N, L+1, C, fW, D)

        # compute the cost volume given feature volume (variance)
        cost_vol = get_feat_var(feat_vol, valid_vol)  # (N, C, fW, D)

        # filter the cost_vol
        cost_vol_filtered = self.cost_filter(cost_vol).squeeze(
            1
        )  # (N, fW, D), cost (or probability)

        # soft-argmin to vote
        prob = F.softmax(-cost_vol_filtered, dim=-1)  # (N, fW, D)
        d = torch.sum(prob * d_vals, dim=-1)  # (N, fW)

        return {"d": d, "prob": prob, "attn": attn}


class mv_depth_feature_res(nn.Module):
    def __init__(self, C=64) -> None:
        super().__init__()
        """
        First feature block of resnet50
        (480, 640, 3) -> Resnet50(1) -> conv -> (C, 120, 160) -> avg_pool (C, 1, 160) -> Attn (vertical) -> (C, 1, 160)

        """
        res50 = resnet50(
            pretrained=True, replace_stride_with_dilation=[False, False, False]
        )
        self.resnet = IntermediateLayerGetter(res50, return_layers={"layer1": "feat"})
        self.C = C
        self.conv = ConvBnReLU(
            in_channels=256, out_channels=self.C, kernel_size=3, padding=1, stride=1
        )

        # Attention block, attetion to columns
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.q_proj = nn.Linear(self.C, self.C, bias=False)

        self.k_proj = nn.Linear(self.C + 32, self.C, bias=False)
        self.v_proj = nn.Linear(self.C + 32, self.C, bias=False)
        self.attn = Attention()

    def forward(self, x, mask=None):
        """
        Input:
            x: (N, 3, 480, 640)
            mask: (N, 480, 640)
        Output:
            x: (N, fW, C)
            attn: (N, fW, fH)
        """
        x = self.resnet(x)["feat"]  # (N, 256, fH, fW)
        x = self.conv(x)  # (N, C, fH, fW)
        fH, fW = list(x.shape[2:])
        N = x.shape[0]

        # resize the mask
        if mask is not None:
            mask = fn.resize(mask, (fH, fW), fn.InterpolationMode.NEAREST).type(
                torch.bool
            )  # (N, fH, fW)
            # reduce vertically and be aware of the mask
            query = (x * mask.unsqueeze(1)).sum(-2) / mask.sum(-2).unsqueeze(
                -2
            )  # (N, C, fW)
            query[torch.isnan(query)] = 0
            mask = torch.logical_not(
                mask
            )  # True is not allow to attend, original mask is True on valid values

            # make mask to columns
            mask = mask.permute(0, 2, 1).reshape((-1, 1, fH))  # (N*fW, 1, fH)

        else:
            # reduce vertically
            query = x.mean(dim=2)  # (N, C, fW)

        # channel last
        query = query.permute(0, 2, 1)  # (N, fW, C)

        # reshape from (N, C, fH, fW) to (N, C, fHxfW)
        x = x.view(list(x.shape[:2]) + [-1])  # (N, 32, fHxfW)
        # channel last to cope with fc
        x = x.permute(0, 2, 1)  # (N, fHxfW, 32)

        # only encoding for the height
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        pos_enc_1d = self.pos_mlp_1d(pos_y.unsqueeze(1))  # (fH, 32)
        pos_enc_1d = pos_enc_1d.unsqueeze(1).repeat((1, fW, 1))  # (fH, fW, 32)
        pos_enc_1d = (
            pos_enc_1d.reshape((-1, 32)).unsqueeze(0).repeat((N, 1, 1))
        )  # (N, fHxfW, 32)
        x = torch.cat((x, pos_enc_1d), dim=-1)  # (N, fHxfW, C+32)

        # attention
        query = self.q_proj(query)  # (N, fW, C)
        key = self.k_proj(x)  # (N, fHxfW, C)
        value = self.v_proj(x)  # (N, fHxfW, C)

        # columnwise attention
        # each column is a batch: (N*fW, 1, C), (N*fW, fH, C)
        query = query.reshape((-1, 1, self.C))  # (N*fW, 1, C)
        key = (
            key.reshape((N, fH, fW, self.C))
            .permute(0, 2, 1, 3)
            .reshape((-1, fH, self.C))
        )  # (N*fW, fH, C)
        value = (
            value.reshape((N, fH, fW, self.C))
            .permute(0, 2, 1, 3)
            .reshape((-1, fH, self.C))
        )  # (N*fW, fH, C)

        x, attn_w = self.attn(
            query, key, value, attn_mask=mask
        )  # (N*fW, 1, C), (N*fW, 1, fH)

        # reshape the features and the attn_w
        x = x.reshape((N, fW, -1))
        attn_w = attn_w.reshape((N, fW, -1))

        return x, attn_w


def gather_feat(
    ref_pose,
    src_feat,
    src_pose,
    d,
    ref_a0=None,
    src_a0=None,
    ref_F=None,
    src_F=None,
    F_W=3 / 8,
    return_wsrc=False,
):
    """
    Gather the feature from different source images
    Input:
        ref_pose: torch.tensor (N, 3), [x, y, th]
        src_feat: torch.tensor (N, L, C, W)
        src_pose: torch.tensor (N, L, 3), [x, y, th]
        d: torch.tensor (N, W), depths for each columns
        ref_a0: torch.tensor (N, ) intrinsic a0 of the reference frame
        src_a0: torch.tensor (N, L) intrinsic a0 of the source frame
        ref_F: torch.tensor (N, ) focal length in pixel of the reference frame
        src_F: torch.tensor (N, L) focal length in pixel of the reference frame
        F_W: focal_length / width
    Output:
        ex_feat: torch.tensor (N, L, C, W)
        valid: torch.tensor[bool] (N, L, W), true if the feature is from valid interpolation
    """
    N, L, C, W = list(src_feat.shape)

    # get the points
    w = torch.arange(0, W, dtype=torch.float32, device=src_feat.device)  # (W)
    w = w.repeat(N, 1)  # (N, W)
    p_ref = torch.stack((w, d), dim=-1)  # (N, W, 2)
    feats = []
    valid = []
    w_srcs = []
    for l in range(L):
        w_src = ref2src(
            ref_pose,
            src_pose[:, l, :],
            p_ref,
            ref_a0=ref_a0,
            src_a0=src_a0[:, l] if src_a0 is not None else None,
            ref_F=ref_F,
            src_F=src_F[:, l] if src_F is not None else None,
            W=W,
            F_W=F_W,
            same_in_batch=False,
        )  # (N, W)
        if return_wsrc:
            w_srcs.append(w_src)

        # normalize w_src to sample grid
        w_src = w_src / w_src.shape[1] * 2 - 1  # map [0,1,..., W-1] to [-1, 1]
        # record which features are out of bounds
        valid.append(torch.logical_and(w_src > -1, w_src < 1))

        w_src = w_src.unsqueeze(-1)  # (N, W, 1)

        feat = F.grid_sample(
            src_feat[:, l, :, :].unsqueeze(-1),
            torch.stack((torch.zeros_like(w_src), w_src), dim=-1),
            mode="bilinear",
        )  # (N, C, W, 1)
        feats.append(feat.squeeze(-1))

    ex_feat = torch.stack(feats, dim=1)  # (N, L, C, W)
    valid = torch.stack(valid, dim=1)  # (N, L, W)
    if return_wsrc:
        return ex_feat, valid, torch.stack(w_srcs, dim=1)
    return ex_feat, valid


def gather_feat_vol(
    ref_pose,
    ref_feat,
    src_feat,
    src_pose,
    d_vals,
    ref_a0=None,
    src_a0=None,
    ref_F=None,
    src_F=None,
    F_W=3 / 8,
):
    """
    Gather the features into the reference camera frame, form volumes
    Input:
        ref_pose: torch.tensor (N, 3), [x, y, th]
        ref_feat: torch.tensor (N, C, W)
        src_feat: torch.tensor (N, L, C, W)
        src_pose: torch.tensor (N, L, 3), [x, y, th]
        d_vals: torch.tensor (D,) D depth planes
        ref_a0: torch.tensor (N,) intrinsic a0 of the reference frame
        src_a0: torch.tensor (N, L) intrinsic a0 of the source frame
        ref_F: torch.tensor (N, ) focal length in pixel of the reference frame
        src_F: torch.tensor (N, L) focal length in pixel of the reference frame
        F_W: focal_length / width
    Output:
        feat_vol: torch.tensor (N, L+1, C, W, D)
        valid: torch.tensor (N, L+1, W, D), True if the feature is from a valid interpolation
    """
    N, L, C, W = list(src_feat.shape)
    D = d_vals.shape[0]
    w, d = torch.meshgrid(
        [torch.arange(0, W, dtype=torch.float32, device=d_vals.device), d_vals]
    )
    d, w = d.contiguous(), w.contiguous()
    d, w = d.view(-1), w.view(-1)
    p_ref = torch.stack((w, d), dim=-1)  # (WxD, 2)
    feats = [ref_feat.unsqueeze(-1).repeat(1, 1, 1, D)]
    valid = [torch.ones(N, W, D, dtype=torch.bool, device=ref_feat.device)]
    for l in range(L):
        w_src = ref2src(
            ref_pose,
            src_pose[:, l, :],
            p_ref,
            ref_a0=ref_a0,
            src_a0=src_a0[:, l] if src_a0 is not None else None,
            ref_F=ref_F,
            src_F=src_F[:, l] if src_F is not None else None,
            W=W,
            F_W=F_W,
        ).view(
            N, W, D
        )  # (N, W, D)
        # normalize w_src to sample grid
        w_src = w_src / w_src.shape[1] * 2 - 1  # map [0,1,..., W-1] to [-1, 1]
        valid.append(
            torch.logical_and(w_src < 1, w_src > -1)
        )  # values lie in [-1,1] are valid
        # NOTE: sample coordinate is row first
        feat = F.grid_sample(
            src_feat[:, l, :, :].unsqueeze(-1),
            torch.stack((torch.zeros_like(w_src), w_src), dim=-1),
            mode="bilinear",
        )  # (N, C, W, D)
        feats.append(feat)
    feat_vol = torch.stack(feats, dim=1)  # (N, L+1, C, W, D)
    valid = torch.stack(valid, dim=1)  # (N, L+1, W, D)
    return feat_vol, valid


def get_feat_var(feat_vol, valid_vol):
    """
    Calculate the variance of the feature, invalid features are not considered
    Input:
        feat_vol: torch.tensor (N, L+1, C, fW, D)
        valid_vol: torhc.tensor (N, L+1, fW, D)
    Ouput:
        feat_var: torch.tensor (N, C, fW, D)
    """
    feat_valid = feat_vol * valid_vol.unsqueeze(2)  # (N, L+1, C, fW, D)
    feat_mean = (feat_valid.sum(dim=1) / valid_vol.sum(dim=1).unsqueeze(1)).unsqueeze(
        1
    )  # (N, 1, C, fW, D)
    feat_var = ((feat_valid - feat_mean) ** 2 * valid_vol.unsqueeze(2)).sum(
        dim=1
    ) / valid_vol.sum(dim=1).unsqueeze(
        1
    )  # (N, C, fW, D)
    return feat_var


class cost_filer(nn.Module):
    def __init__(self, D=128, C=32) -> None:
        super().__init__()
        """
        Given cost volume (N, C, fW, D)
        filter the volume (N, fW, D)
        U-Net structure

        (N, C, fW, D)=>cbr1=>(N, 8, fW, D)===============================================================================================================(+)=>(N, 8, fW, D)=>conv=>(N, 1, fW, D)
                                |                                                                                                                         |
                                |=>cbr2=>cbr3=>(N, 16, fW/2, D/2) ==========================================(+)=>(N, 16, fW/2, D/2)=>ctbr2=>(N, 8, fW, D)=|
                                                |                                                            |
                                                |=>cbr4=>cbr5=>(N, 32, fW/4, D/4)=>ctbr1=>(N, 16, fW/2, D/2)=|

        """
        self.D = D
        self.C = C
        self.cbr1 = ConvBnReLU(
            in_channels=self.C, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.down1 = nn.Sequential(
            ConvBnReLU(
                in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1
            ),
            ConvBnReLU(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
        )
        self.down2 = nn.Sequential(
            ConvBnReLU(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            ConvBnReLU(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
        )
        self.down3 = nn.Sequential(
            ConvBnReLU(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            ConvBnReLU(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
        )

        self.up1 = ConvTransBnReLU(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up2 = ConvTransBnReLU(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up3 = ConvTransBnReLU(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.conv = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        out1 = self.cbr1(x)  # (N, 8, fW, D)
        out2 = self.down1(out1)  # (N, 16, fW/2, D/2)
        out3 = self.down2(out2)  # (N, 32, fW/4, D/4)
        x = self.down3(out3)  # (N, 64, fW/8, D/8)
        x = out3 + self.up1(x)  # (N, 32, fW/4, D/4)
        x = out2 + self.up2(x)  # (N, 16, fW/2, D/2)
        x = out1 + self.up3(x)  # (N, 8, fW, D)
        x = self.conv(x)  # (N, 1, fW, D)
        return x
