from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Dice
# from src.models.components.segmentation import Seg
import numpy as np 
import segmentation_models_pytorch as smp

from pathlib import Path
from torchmetrics import MetricCollection
from ..utils.metrics import DiceMetric3d

import monai
from monai.networks import nets

class MRIModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: dict,
        configure: torch.optim
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        # loss function
        
        self.sigmoid = torch.nn.Sigmoid()
        
        self.DiceLoss      = monai.losses.DiceLoss(sigmoid=True)
        self.DiceCELoss    = monai.losses.DiceCELoss(sigmoid=True)
        self.DiceFocalLoss = monai.losses.DiceFocalLoss(sigmoid=True)

        self.metrics = self._init_metrics()
        self.class_weight = [1,1,1]

    def criterion(self,y_pred, y_true):
        
        loss = self.DiceCELoss(y_pred,y_true)
        return loss

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def _init_metrics(self):
        
        train_metrics = MetricCollection({"train_dice": DiceMetric3d()})
        val_metrics = MetricCollection({"val_dice": DiceMetric3d()})
        test_metrics = MetricCollection({"test_comp_metric": DiceMetric3d()})

        return torch.nn.ModuleDict(
            {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics
            }
        )

    def step(self, batch: Any):
        
        x, y = batch['image_3d'],batch['mask_3d']
        
        logits = self.forward(x)
        
        loss = self.criterion(logits, y)
        preds = self.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, mask = self.step(batch)

        # log train metrics
        
        metrics = self.metrics[f"train_metrics"](preds,mask)
        
        class_dice = {f'train/dic_score_cl{num}':i for num,i in enumerate(metrics['train_dice'])}
        self.log_dict(class_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return {"loss": loss,"dic_score":metrics['train_dice']}

    # def validation_step(self, batch: Any, batch_idx: int):
    #     loss, preds, mask = self.step(batch)
        
    #     metrics = self.metrics[f"val_metrics"](preds,mask)

    #     class_dice = {f'val/dic_score_cl{num}':i for num,i in enumerate(metrics['val_dice'])}
    #     self.log_dict(class_dice , on_step=False, on_epoch=True)
    #     self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
    #     return metrics['val_dice']
        

    # def validation_epoch_end(self, outputs):
        
    #     dic_score = torch.mean(torch.stack(outputs))  # get val accuracy from current epoch
        
    #     if self.best_dice < dic_score: 
    #         self.save_model_pth()
    #         self.best_dice = dic_score

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, mask = self.step(batch)

        # log test metrics
        # dic_score = self.dice_coef(mask,preds).item()
        metrics = self.metrics[f"test_metrics"](preds,mask)
        self.log("test/total_score", metrics['test_comp_metric'], on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.metrics['train_metrics'].reset()
        self.metrics['val_metrics'].reset()
        self.metrics['test_metrics'].reset()
        

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_kwargs = dict(
                    params=self.parameters(), lr=self.hparams.configure.lr, weight_decay=self.hparams.configure.weight_decay
                )
        if self.hparams.configure.optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(**optimizer_kwargs)
        elif self.hparams.configure.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(**optimizer_kwargs)
        elif self.hparams.configure.optimizer == "Adam":
            optimizer = torch.optim.Adam(**optimizer_kwargs)
        elif self.hparams.configure.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(**optimizer_kwargs)
        elif self.hparams.configure.optimizer == "Adamax":
            optimizer = torch.optim.Adamax(**optimizer_kwargs)
        elif self.hparams.configure.optimizer == "SGD":
            optimizer = torch.optim.SGD(**optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        if self.hparams.configure.scheduler is not None:
            if self.hparams.configure.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.configure.T_max, eta_min=self.hparams.configure.min_lr
                )
            elif self.hparams.configure.scheduler == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.hparams.configure.T_0, eta_min=self.hparams.configure.min_lr
                )
            elif self.hparams.configure.scheduler == "ExponentialLR":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            elif self.hparams.configure.scheduler == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
            else:
                raise ValueError(f"Unknown scheduler: {self.hparams.configure.scheduler}")

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return {"optimizer": optimizer}



        return {'optimizer':optim, 'scheduler': scheduler}
