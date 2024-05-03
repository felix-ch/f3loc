import lightning.pytorch as pl
import torch.optim as optim

from modules.mono.depth_net import *
from modules.mv.mv_depth_net import *

from .comp_d_net import *


class comp_d_net_pl(pl.LightningModule):
    """
    lightning wrapper for the complementary network
    """

    def __init__(
        self,
        mv_net: mv_depth_net,
        mono_net: depth_net,
        L=3,
        d_min=0.1,
        d_max=15.0,
        d_hyp=-0.2,
        D=128,
        shape_loss_weight=None,
        F_W=3 / 8,
        use_pred=True,
        lr=1e-3,
    ) -> None:
        super().__init__()
        self.L = L
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.use_pred = use_pred
        self.shape_loss_weight = shape_loss_weight
        self.F_W = F_W
        self.lr = lr
        self.comp_d_net = comp_d_net(
            mv_net=mv_net,
            mono_net=mono_net,
            L=self.L,
            C=64,
            D=self.D,
            d_min=self.d_min,
            d_max=self.d_max,
            d_hyp=self.d_hyp,
            use_pred=self.use_pred,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # train the ray depths
        pred_dict = self.comp_d_net(batch)
        d_comp = pred_dict["d_comp"]
        l1_loss = F.l1_loss(d_comp, batch["ref_depth"])
        self.log("l1_loss-train", l1_loss)

        loss = l1_loss

        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(d_comp, batch["ref_depth"], dim=-1).mean()
            )
            self.log("shape_loss-train", shape_loss)

            loss += shape_loss

        self.log("loss-train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_dict = self.comp_d_net(batch)
        d_comp = pred_dict["d_comp"]
        l1_loss = F.l1_loss(d_comp, batch["ref_depth"])
        self.log("l1_loss-valid", l1_loss)

        loss = l1_loss

        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(d_comp, batch["ref_depth"], dim=-1).mean()
            )
            self.log("shape_loss-valid", shape_loss)

            loss += shape_loss

        self.log("loss-valid", loss)
