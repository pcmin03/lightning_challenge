from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
# from src.models.components.segmentation import Seg
import numpy as np 
import segmentation_models_pytorch as smp

from pathlib import Path
from torchmetrics import MetricCollection
from torchmetrics import Dice,JaccardIndex
from ..utils.metrics import DiceMetric,IOUMetric,CompetitionMetric
# form ..utils.loss import hddistloss
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
import sys

import pretty_errors
from monai import losses

CLASS_NAME = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

class HUMAPModule(LightningModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        configure: torch.optim
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False)

        self.net = net
        # loss function
        self.sigmoid = torch.nn.Sigmoid()
        
        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()



        self.best_dice = 0 
        
        self.metrics = self._init_metrics()
        self._init_loss_fn()

    def _init_loss_fn(self):
        self.JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
        self.DiceLoss    = smp.losses.DiceLoss(mode='multiclass')
        self.BCELoss     = smp.losses.SoftBCEWithLogitsLoss(reduction='mean')
        self.CELoss      = smp.losses.SoftCrossEntropyLoss(reduction='none')
        self.LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
        self.TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
    
    def _init_metrics(self):
        

        train_metrics = MetricCollection({"train_dice": Dice(),"train_jacc":JaccardIndex(len(self.classname)+1)})
        val_metrics = MetricCollection({"val_dice": Dice(),"val_jacc":JaccardIndex(len(self.classname)+1)})
        test_metrics = MetricCollection({"test_comp_metric": Dice(),"test_jacc":JaccardIndex(len(self.classname)+1)})

        return torch.nn.ModuleDict(
            {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )
    def criterion(self,y_pred, y_true):
        
        bceloss = self.DiceLoss(y_pred, y_true)
        
        loss = bceloss.mean()
        
        return loss

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def forward(self, x: torch.Tensor):
        x = self.sigmoid(x)
        return self.net(x)


    def step(self, batch: Any):
        x, y = batch
        
        logits = self.forward(x)
        
        loss = self.criterion(logits, y)
        preds = self.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, mask = self.step(batch)

        # log train metrics
        
        metrics = self.metrics[f"train_metrics"](preds,mask)
        
        # class_dice = {f'train/dic_score_cl{num}':i.item() for num,i in enumerate(metrics['train_dice'])}
        class_dice = {f'train/dic_score':metrics['train_dice'],f'train/jacc_score':metrics['train_jacc']}
        
        self.log_dict(class_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        
        return {"loss": loss,"dic_score":metrics['train_dice']}
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, mask = self.step(batch)
        
        metrics = self.metrics[f"val_metrics"](preds,mask)
        
        class_dice = {f'val/dic_score':metrics['val_dice'],f'val/jacc_score':metrics['val_jacc']}
        self.log_dict(class_dice , on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        
        return metrics['val_dice']
    
    def on_epoch_end(self,): 
        self.metrics['train_metrics'].reset()
        self.metrics['val_metrics'].reset()
        self.metrics['test_metrics'].reset()
        
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
