import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
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
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", "_mask.png"))  # TODO: check again the suffix in gipDeep

        image = read_image(img_path)
        mask = read_image(mask_path, mode=ImageReadMode.GRAY)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def test():
    data = ThumbnailsDataset("otsuExamples/data", "otsuExamples/segData")
    img, mask = data[0]
    print(f'image shape is {img.shape} mask shape is {mask.shape}')
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    test()
