import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

import matplotlib.pyplot as plt

class ThumbnailsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.os.listdir(self.imageDir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_mask.png"))
        
        image = read_image(img_path)
        mask = read_image(mask_path, mode=ImageReadMode.GRAY)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
        

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


def test():
    data = ThumbnailsDataset("otsuExamples/data", "otsuExamples/segData")
    img, mask = data[0]
    print(f'image shape is {img.shape} mask shape is {mask.shape}')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()
if __name__=='__main__':
    test()