from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from .components.tiger_dataset import SigleCells
import numpy as np 

from sklearn.model_selection import train_test_split
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class TIGERDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.63462753, 0.44200307, 0.66190372), (0.1932225 , 0.20969465, 0.15634316))]
        # )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    # def prepare_data(self):
    #     """Download data if needed.

    #     This method is called only from a single GPU.
    #     Do not use it to assign state (self.x = y).
    #     """
    #     MNIST(self.hparams.data_dir, train=True, download=True)
    #     MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        base_path = Path('/nfs3/team/pathos/TIGER/wsirois/roi-level-annotations/tissue-cells/single_cells')

        tp_gt = np.load(base_path/'tp_gt_v2.npy')
        fp_predict = np.load(base_path/'fp_prdict_v2.npy')
        fp_score = np.load(base_path/'fp_score.npy')

        # ranked = np.argsort(fp_score*100) < 5000
        test_set = fp_predict[(fp_score > 0.2) &(fp_score < 0.5)]
        tp_gt = np.repeat(tp_gt,3,axis=0)
        fp_predict = fp_predict[fp_score < 0.3]
        pretrained_means = np.array([161.83001964, 112.71078204, 168.78544866])/255.
        pretrained_stds = np.array([49.27173631, 53.47213493, 39.86750637])/255.

        train_transforms = A.Compose([
                                A.Resize(32,32),
                                A.OneOf([
                                    A.MotionBlur(p=.2),
                                    A.MedianBlur(blur_limit=3, p=0.1),
                                    A.Blur(blur_limit=3, p=0.1),
                                            ], p=0.5),
                                A.OneOf([
                                    A.CLAHE(clip_limit=2),
                                    A.IAASharpen(p=0.5),
                                    A.IAAEmboss(p=0.0),
                                    A.ToGray(p=0.5),

                                ], p=0.0),

                                A.OneOf([
                                    A.ColorJitter(),
                                    A.HueSaturationValue(),

                                ], p=0.3),
                                              
                                A.RandomRotate90(),
                                A.HorizontalFlip(),
                                A.Normalize(mean = pretrained_means, 
                                                        std = pretrained_stds),
                                ToTensorV2()
                            ])

        test_transforms = A.Compose([
                                A.Resize(32,32),
                                A.Normalize(mean = pretrained_means, 
                                                        std = pretrained_stds),
                                ToTensorV2()
                            ])
        
        classes = np.concatenate([np.ones(len(tp_gt)),np.zeros(len(fp_predict))])
        images = np.concatenate([tp_gt,fp_predict])

        x_train,x_valid,y_train,y_valid= train_test_split(images,classes,test_size=0.2,random_state=2022)

        test_classes = np.zeros(len(test_set))
        test_images = test_set
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = SigleCells(x_train,y_train,train_transforms,type='train')
            # self.data_train = SigleCells(x_valid,y_valid,test_transforms)
            self.data_val = SigleCells(x_valid,y_valid,test_transforms,type='valid')
            self.data_test = SigleCells(test_images,test_classes,test_transforms,type='valid')
            # dataset = ConcatDataset(datasets=[trainset, testset])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
