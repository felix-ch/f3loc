import lightning.pytorch as pl
import torch.optim as optim

from .mv_depth_net import *


class mv_depth_net_pl(pl.LightningModule):
    """
    lightning wrapper for the multi-view depth prediction
    """

    def __init__(
        self, D=128, d_min=0.1, d_max=15.0, d_hyp=1.0, shape_loss_weight=None, F_W=3 / 8
    ) -> None:
        super().__init__()
        self.D = D
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.F_W = F_W
        self.net = mv_depth_net(
            D=self.D, d_min=self.d_min, d_max=self.d_max, d_hyp=self.d_hyp, F_W=self.F_W
        )
        self.shape_loss_weight = shape_loss_weight

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        pred_dict = self.net(batch)
        depths = pred_dict["d"]

        l1_loss = F.l1_loss(depths, batch["ref_depth"])
        self.log("l1_loss-train", l1_loss)
        loss = l1_loss

        # shape loss
        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(depths, batch["ref_depth"]).mean()
            )
            self.log("shape_loss-train", shape_loss)
            loss += shape_loss

        self.log("loss-train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_dict = self.net(batch)
        depths = pred_dict["d"]

        l1_loss = F.l1_loss(depths, batch["ref_depth"])
        self.log("l1_loss-valid", l1_loss)
        loss = l1_loss

        # shape loss
        if self.shape_loss_weight is not None:
            shape_loss = self.shape_loss_weight * (
                1 - F.cosine_similarity(depths, batch["ref_depth"]).mean()
            )
            self.log("shape_loss-valid", shape_loss)
            loss += shape_loss

        self.log("loss-valid", loss)
