import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ThumbnailsDataset(Dataset):
    def __init__(self, imageDir, maskDir, transform=None):
        self.imageDir = imageDir
        self.maskDir = maskDir
        self.transform = transform

    def __len__(self):
        return len(self.os.listdir(self.imageDir))

    def __getitem__(self, index):
        #TODO: AMIR
        pass
        

#TODO: implement this function  


def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_maks_dir, batch_size, train_transforms, val_transforms, num_workers, pin_memory):
    """returns train and validation dataloaders

    Args:
        train_img_dir: path to the directory of the training images
        train_mask_dir: path to the directory of the training masks
        val_img_dir: path to the directory of the validation images
        val_maks_dir: path to the directory of the validation masks
        batch_size: batch size of the dataloaders
        train_transforms: a sequence of transformations to apply on the training set
        val_transforms: a sequence of transformations to apply on the validation set
        num_workers: num workers for the data loading 
        pin_memory 
    """
    pass