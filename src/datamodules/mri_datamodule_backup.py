from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from .components.mri_dataset import BuildDataset
import numpy as np 

from sklearn.model_selection import train_test_split
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2,ToTensor

import pandas as pd
import cv2

class MRIDataModule(LightningDataModule):
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
        file_name : str = '25d_train_test_fold.csv',
        fold: int = 0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        add_channel: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.63462753, 0.44200307, 0.66190372), (0.1932225 , 0.20969465, 0.15634316))]
        # )
        self.data_dir = data_dir
        self.fold = fold
        self.file_name = file_name
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.add_channel = add_channel
        
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
        
        df = pd.read_csv(self.data_dir + self.file_name)
        test_df = df[df['test'] == 1]
        not_test_df = df[df['test'] != 1]
        train_df = not_test_df[not_test_df['fold'] != self.fold]
        valid_df = not_test_df[not_test_df['fold'] == self.fold]
        

        data_transforms = {
            "train": A.Compose([
                # A.Resize(320,320),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                ], p=0.5),
                A.CoarseDropout(max_holes=8, max_height=160//20, max_width=192//20,
                                min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
                A.RandomSizedCrop((224,224),320, 384,p=0.25),
                A.Cutout(10,10,p=0.25),
                # A.Cutout(10,10,p=0.25),
                ToTensorV2()], p=1.0),
                
            
            "valid": A.Compose([
                # A.Resize(320,320),
                ToTensorV2(),
                ], p=1.0)
        }
        
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = BuildDataset(train_df,True,transforms=data_transforms['train'],add_channel=self.add_channel)
            self.data_val = BuildDataset(valid_df,True,transforms=data_transforms['valid'],add_channel=self.add_channel)
            self.data_test = BuildDataset(test_df,True,transforms=data_transforms['valid'],add_channel=self.add_channel)
            
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
