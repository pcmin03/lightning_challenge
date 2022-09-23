from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from torchvision.transforms import transforms

import torch 

class BuildDataset(Dataset):
    def __init__(self, df, label=True, transforms=None,add_channel=True):
        self.df         = df
        self.label      = label
        self.img_paths  = df['img_path'].tolist()
        self.bin_paths  = df['binary_mask'].tolist()
        self.mul_paths  = df['mclass_mask'].tolist()
        self.transforms = transforms
        self.add_channel = add_channel
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        img  = cv2.imread(self.img_paths[index])[...,::-1]
        msk  = cv2.imread(self.mul_paths[index],cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img,(1920, 1920),cv2.INTER_LINEAR).astype(np.uint8)
        msk = cv2.resize(msk,(1920, 1920),cv2.INTER_LINEAR).astype(np.uint8)

        data = self.transforms(image=img,mask=msk)
        img  = data['image']
        msk  = data['mask'].to(torch.int64)
    
        return img, msk
        