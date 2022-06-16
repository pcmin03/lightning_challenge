from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

# from src.models.components.segmentation import Seg

import segmentation_models_pytorch as smp

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
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.sigmoid = torch.nn.Sigmoid()
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
        self.DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
        self.BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
        self.LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
        self.TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

    def dice_coef(self,y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred>thr).to(torch.float32)
        inter = (y_true*y_pred).sum(dim=dim)
        den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
        dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
        return dice

    def iou_coef(self,y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred>thr).to(torch.float32)
        inter = (y_true*y_pred).sum(dim=dim)
        union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
        iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
        return iou

    def criterion(self,y_pred, y_true):
        
        loss = 0.5*self.DiceLoss(y_pred, y_true) + 0.5*self.BCELoss(y_pred, y_true)
        return loss
        

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        
        logits = self.forward(x)
        
        loss = self.criterion(logits, y)
        preds = self.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, mask = self.step(batch)

        # log train metrics
        dic_score = self.dice_coef(mask,preds).item()
        iou_score = self.iou_coef(mask,preds).item()
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/dic_score", dic_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou_score", iou_score, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss,"dic_score":dic_score,"iou_score":iou_score}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, mask = self.step(batch)
        
        # self.logger.experiment.add_image('val/sample_img',batch[0][:4],0)
        # self.log.log_image(key="samples", images=batch[0])

        # log val metrics
        dic_score = self.dice_coef(mask,preds).item()
        iou_score = self.iou_coef(mask,preds).item()
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/dic_score", dic_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou_score", iou_score, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss,"dic_score":dic_score,"iou_score":iou_score}

    def validation_epoch_end(self, outputs: List[Any]):
        pass
        # acc = self.val_acc.compute()  # get val accuracy from current epoch
        # self.val_acc_best.update(acc)
        # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, mask = self.step(batch)

        self.log()

        # log test metrics
        dic_score = self.dice_coef(mask,preds).item()
        iou_score = self.iou_coef(mask,preds).item()
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/dic_score", dic_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou_score", iou_score, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss,"dic_score":dic_score,"iou_score":iou_score}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
