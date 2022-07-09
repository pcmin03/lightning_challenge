from typing import Optional, Tuple
import torch

from pytorch_lightning import LightningDataModule
# from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from .components.mri_3d_dataset import NiftiDataset
import numpy as np 

from sklearn.model_selection import train_test_split
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2,ToTensor

import pandas as pd
import cv2
from monai.data import load_decathlon_datalist

from monai.data import CacheDataset, DataLoader
from monai.data import CSVDataset,Dataset
from monai import transforms
#     Compose,
#     Activations,
#     AsDiscrete,
#     Activationsd,
#     AsDiscreted,
#     KeepLargestConnectedComponentd,
#     Invertd,
#     LoadImage,
#     Transposed,
#     LoadImaged,
#     AddChanneld,
#     CastToTyped,
#     Lambdad,
#     Resized,
#     EnsureTyped,
#     SpatialPadd,
#     EnsureChannelFirstd,
#     RandFlipd
# )


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
        batch_size: int = 64,
        fold: int = 0,
        file_name : str = 'dataset_3d_fold_0.json',
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
        self.data_dir = data_dir
        self.fold = fold
        self.file_name = file_name
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 3

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        
        # df = pd.read_csv(self.data_dir)
        # test_df = df[df['test'] == 1].reset_index(drop=True)
        # not_test_df = df[df['test'] != 1]
        # train_df = not_test_df[not_test_df['fold'] != self.fold].reset_index(drop=True)
        # valid_df = not_test_df[not_test_df['fold'] == self.fold].reset_index(drop=True)
        data_transforms = {'train' : transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", "mask"]),
                    # EnsureChannelFirstd(keys=["image", "mask"]),
                    # Spacingd(keys=["image", "mask"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
                    # transforms.RandSpatialCropd(
                    #     keys=("image", "mask"),
                    #     roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    #     random_size=False,
                    # ),
                    transforms.Lambdad(keys="image", func=lambda x: x / 255.),
                    # transforms.RandFlipd(keys=("image", "mask"), prob=args.RandFlipd_prob, spatial_axis=[0]),
                    # transforms.RandFlipd(keys=("image", "mask"), prob=args.RandFlipd_prob, spatial_axis=[1]),
                    # RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[2]),
                    # transforms.RandAffined(
                    #     keys=("image", "mask"),
                    #     prob=0.5,
                    #     rotate_range=np.pi / 12,
                    #     translate_range=(args.roi_x*0.0625, args.roi_y*0.0625),
                    #     scale_range=(0.1, 0.1),
                    #     mode="nearest",
                    #     padding_mode="reflection",
                    # ),
                    transforms.OneOf(
                        [
                            transforms.RandGridDistortiond(keys=("image", "mask"), prob=0.3, distort_limit=(-0.05, 0.05), mode="nearest", padding_mode="reflection"),
                            transforms.RandCoarseDropoutd(
                                keys=("image", "mask"),
                                holes=5,
                                max_holes=8,
                                spatial_size=(1, 1, 1),
                                max_spatial_size=(12, 12, 12),
                                fill_value=0.0,
                                prob=0.3,
                            ),
                        ]
                    ),
                    # transforms.RandScaleIntensityd(keys="image", factors=(-0.2, 0.2), prob=args.RandScaleIntensityd_prob),
                    # transforms.RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=args.RandShiftIntensityd_prob),
                    transforms.EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
                ]
            ),
                    'valid' : transforms.Compose(
                        [
                            transforms.LoadImaged(keys=["image", "mask"]),
                            transforms.Lambdad(keys="image", func=lambda x: x / 255.),
                            # transforms.AddChanneld(keys=["image", "mask"]),
                            # transforms.Orientationd(keys=["image", "mask"], axcodes="RAS"),
                            # transforms.Spacingd(
                            #     keys=["image", "mask"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                            # ),
                            # transforms.ScaleIntensityRanged(
                            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                            # ),
                            # transforms.CropForegroundd(keys=["image", "mask"], source_key="image"),
                            transforms.EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
                        ]
                    )}

        # spatial_size = (224, 224, 80)
        # data_transforms = {
        #     'train': Compose([
        #         LoadImaged(keys=["image", "mask"]),
        #         # AddChanneld(keys=["image","mask"]),
        #         # Resized(keys=["image"], spatial_size=spatial_size, mode="nearest"),
        #         # Transposed(keys="image", indices=[0, 2, 3, 1]), # c, w, h, d
                
        #         Lambdad(keys="image", func=lambda x: x / x.max()),
        #         RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[0]),
        #         RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[1]),
        #         EnsureTyped(keys=["image",'mask'], dtype=torch.float32),
        #         #monai.transforms.ResizeWithPadOrCrop(keys=["image", "mask"], spatial_size=spatial_size),
        #         ]),

        #     'valid' : Compose([
        #         LoadImaged(keys=["image", "mask"]),
        #         # AddChanneld(keys=["image","mask"]),
        #         # Resized(keys=["image"], spatial_size=spatial_size, mode="nearest"),
        #         # Transposed(keys="image", indices=[0, 2, 3, 1]), # c, w, h, d
        #         Lambdad(keys="image", func=lambda x: x / x.max()),
        #         EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
        #         #monai.transforms.ResizeWithPadOrCrop(keys=["image"], spatial_size=spatial_size),
        #         ])}
        
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            
            train_datalist = load_decathlon_datalist(Path(self.data_dir)/self.file_name, True, "train", base_dir=self.data_dir)
            valid_datalist = load_decathlon_datalist(Path(self.data_dir)/self.file_name, True, "val", base_dir=self.data_dir)
            test_datalist = load_decathlon_datalist(Path(self.data_dir)/self.file_name, True, "test", base_dir=self.data_dir)

            self.data_train = Dataset(data=train_datalist, transform=data_transforms['train'])
            self.data_val = Dataset(data=valid_datalist, transform=data_transforms['valid'])
            self.data_test = Dataset(data=test_datalist, transform=data_transforms['valid'])
            
            # self.data_train = CSVDataset(image_files=train_df['image'].to_list(),seg_file=train_df['mask'].to_list(), transform=data_transforms['train'])
            # self.data_val = CSVDataset(image_files=valid_df['image'].to_list(),seg_file=valid_df['mask'].to_list(), transform=data_transforms['valid'])
            # self.data_test = CSVDataset(image_files=test_df['image'].to_list(),seg_file=test_df['mask'].to_list(), transform=data_transforms['valid'])
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
