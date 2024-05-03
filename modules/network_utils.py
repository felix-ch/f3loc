import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, Q, K, V, attn_mask=None):
        """
        one head attention
        Input:
            Q: queries, (N, L, D)
            K: keys, (N, S, D)
            V: values, (N, S, D)
            attn_mask: mask on the KV, (N, L, S)
        Output:
            queried_values: gathered values, (N, L, D)
            attn_weights: weights of the attention, (N, L, S)
        """
        # dot product
        QK = torch.einsum("nld,nsd->nls", Q, K)  # (N, L, S)
        if attn_mask is not None:
            QK[attn_mask] = -torch.inf  # this lead to 0 after softmax

        # softmax
        D = Q.shape[2]
        attn_weights = torch.softmax(QK / (D**2), dim=2)  # (N, L, S)

        # weighted average
        x = torch.einsum("nsd,nls->nld", V, attn_weights)  # (N, L, D)
        return x, attn_weights


class ELU_Plus(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        """
        Make the ELU > 0
        """
        return F.elu(x) + 1


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # conv + bn
        self.convbn = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        x = self.convbn(x)
        return x


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.convbn = ConvBn(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def forward(self, x):
        x = self.convbn(x)
        x = F.relu(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # conv + relu
        self.convrelu = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x


class ConvTransBn(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        # convtrans + bn
        self.convtransbn = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
            ),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        x = self.convtransbn(x)
        return x


class ConvTransBnReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.convtransbn = ConvTransBn(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
        )

    def forward(self, x):
        x = self.convtransbn(x)
        x = F.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True) -> None:
        """
        This is a single residual block
        if downsample:
            x ----> conv(3x3, stride=2, in_ch, out_ch) -> bn -> relu ----> conv(3x3, out_ch, out_ch) -> bn ---(+)---> relu
                |------------------ conv(1x1, stride=2, in_ch, out_ch) -> bn ----------------------------------|
        else:
            x ----> conv(3x3, in_ch, out_ch) -> bn -> relu ----> conv(3x3, out_ch, out_ch) -> bn ---(+)---> relu
                |------------------------------------------------------------------------------------|
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        # first conv
        self.conv1 = ConvBnReLU(
            self.in_channels,
            self.out_channels,
            3,
            stride=2 if self.downsample else 1,
            padding=1,
        )

        # second conv
        self.conv2 = ConvBn(
            self.out_channels, self.out_channels, 3, stride=1, padding=1
        )

        if self.downsample:
            self.conv3 = ConvBn(
                self.in_channels, self.out_channels, 1, stride=2, padding=0
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            out = F.relu(out + self.conv3(x))
        else:
            out = F.relu(out + x)
        return out


def ref2src(
    ref_pose,
    src_pose,
    p_ref,
    W=40,
    ref_a0=None,
    src_a0=None,
    ref_F=None,
    src_F=None,
    F_W=3 / 8,
    same_in_batch=True,
    return_d=False,
):
    """
    Transform the points in scr cam frame to ref cam frame
    Input:
        ref_pose: torch.tensor (N, 3), [x, y, th]
        scr_pose: torch.tensor (N, 3), [x, y, th]
        same_in_batch: bool, whether the points to extracts are same for different batches
        p_ref:
            if same_in_batch: torch.tensor (P, 2) [w, d], points in reference frame
            else: torch.tensor (N, P, 2) [w, d], points in reference frame
        W: width in pixels
        ref_a0: torch.tensor (N, ), the intrinsic parameter a0 of ref frame
        src_a0: torch.tensor (N, ), the intrinsic parameter a0 of src frame
        ref_F: torch.tensot (N, ), the focal length in pixels of ref frame
        src_F: torch.tensot (N, ), the focal length in pixels of src frame
        F_W: focal_length / width torch.tensor (1,) NOTE: if ref_F and src_F is None, use this
        return_d: bool, whether return the depths as well,
            if True, return 2d coordinates (N, P, 2) [w, d]
            else, return 1d coordinates (N, P) [w]
    Output:
        w_src: torch.tensor (N, P) [w], points in respective source frame
        d_src: torch.tensor (N, P) [d], points depth in the respective source frame
    """
    if ref_F is None:
        ref_F = W * F_W  # (1,)
        src_F = W * F_W  # (1,)
    else:
        ref_F = ref_F.unsqueeze(-1)  # (N, 1)
        src_F = src_F.unsqueeze(-1)  # (N, 1)

    th_r = ref_pose[:, -1].unsqueeze(-1)  # (N,1)
    th_s = src_pose[:, -1].unsqueeze(-1)  # (N,1)
    c_r = torch.cos(th_r)  # (N,1)
    s_r = torch.sin(th_r)  # (N,1)
    c_s = torch.cos(th_s)  # (N,1)
    s_s = torch.sin(th_s)  # (N,1)

    # extract the depth and the width of the points
    if same_in_batch:
        w = p_ref[:, 0]  # (P,)
        d = p_ref[:, 1]  # (P,)
    else:
        w = p_ref[:, :, 0]  # (N, P)
        d = p_ref[:, :, 1]  # (N, P)

    # project to the scene
    x_ref = d  # (P,) or (N, P)
    if ref_a0 is None:
        y_ref = (
            -(w - (W - 1) / 2) / ref_F * x_ref
        )  # (P,) or (N, P), NOTE: the negative is because left of the image center is now positive of y axis if the viewing direction is positive x axis
    else:
        y_ref = (
            -(w - ref_a0.unsqueeze(1)) / ref_F * x_ref
        )  # (N, P), NOTE: the negative is because left of the image center is now positive of y axis if the viewing direction is positive x axis

    # x, y in world coordinate w.r.t. source frame
    x_w = (
        x_ref * c_r
        - y_ref * s_r
        + ref_pose[:, 0].unsqueeze(-1)
        - src_pose[:, 0].unsqueeze(-1)
    )  # (N, P)
    y_w = (
        x_ref * s_r
        + y_ref * c_r
        + ref_pose[:, 1].unsqueeze(-1)
        - src_pose[:, 1].unsqueeze(-1)
    )  # (N, P)

    # x, y in source frame
    x_src = x_w * c_s + y_w * s_s  # (N, P)
    y_src = -x_w * s_s + y_w * c_s  # (N, P)

    # to w representation, i.e., cost volume coordinate in source frame
    d_src = x_src  # (N, P)
    if src_a0 is None:
        w_src = (
            -y_src / d_src * src_F + (W - 1) / 2
        )  # (N, P), NOTE: the negative is because left of the image center is now positive of y axis if the viewing direction is positive x axis
    else:
        w_src = -y_src / d_src * src_F + src_a0.unsqueeze(
            1
        )  # (N, P), NOTE: the negative is because left of the image center is now positive of y axis if the viewing direction is positive x axis

    if not return_d:
        return w_src  # (N, P)
    else:
        return torch.stack((w_src, d_src), dim=-1)  # (N, P, 2)


def get_rel_pose(ref_pose, src_pose):
    """
    Input:
        ref_pose: torch.tensor(N, 3)
        src_pose: torch.tensor(N, L, 3)
    Output:
        rel_pose: torch.tensor(N, L, 3)
    """
    # NOTE: the relative pose theta needs to be in -pi/pi
    if ref_pose.dim() == 1 and src_pose.dim() == 1:
        # only compute a single one
        rel_pose = src_pose - ref_pose  # (3)
        cr = torch.cos(ref_pose[-1])
        sr = torch.sin(ref_pose[-1])
        rel_x = cr * rel_pose[0] + sr * rel_pose[1]
        rel_y = -sr * rel_pose[0] + cr * rel_pose[1]
        rel_pose[0] = rel_x
        rel_pose[1] = rel_y
        rel_pose[-1] = (rel_pose[-1] + torch.pi) % (torch.pi * 2) - torch.pi
    else:
        # compute the source pose w.r.t. reference pose
        rel_pose = src_pose - ref_pose.unsqueeze(1)  # (N, L, 3)
        cr = torch.cos(ref_pose[:, -1]).unsqueeze(-1)  # (N, 1)
        sr = torch.sin(ref_pose[:, -1]).unsqueeze(-1)  # (N, 1)
        rel_x = cr * rel_pose[:, :, 0] + sr * rel_pose[:, :, 1]  # (N, L)
        rel_y = -sr * rel_pose[:, :, 0] + cr * rel_pose[:, :, 1]  # (N, L)
        rel_pose[:, :, 0] = rel_x
        rel_pose[:, :, 1] = rel_y
        rel_pose[:, :, -1] = (rel_pose[:, :, -1] + torch.pi) % (torch.pi * 2) - torch.pi

    return rel_pose
