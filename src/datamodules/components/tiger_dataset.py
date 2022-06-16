from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from torchvision.transforms import transforms
import kornia as K

class SigleCells(Dataset):
    def __init__(self, dataset,classes, transform=None,type='valid'):
        self.dataset = dataset
        self.classes = classes
        self.transform = transform
        self.type = type

    def __getitem__(self, index):
        x = self.dataset[index]
        y = self.classes[index] 

        if self.transform:
            x = self.transform(image=x)["image"]        
        # if self.type == 'train':
        #     x =  K.filters.sobel(x.unsqueeze(dim=0)/255.) * 4
        #     x = x[0]
        return x, y

    def __len__(self):
        return len(self.dataset)