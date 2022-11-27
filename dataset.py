import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np

from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

import matplotlib.pyplot as plt
import albumentations as A


IMAGE_HEIGHT=512
IMAGE_WIDTH=512

class ThumbnailsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, indices, eval_mode,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.indices = indices
        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[self.indices[idx]])
        mask_path = os.path.join(self.mask_dir, self.images[self.indices[idx]].replace("_thumb.jpg",  "_SegMap.png"))  # TODO: check again the suffix in gipDeep
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        height, width, _ = image.shape
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            if self.eval_mode:
                # Resizing just in case we're in validation, since using center crop
                resizing =A.Compose([A.Resize(max(IMAGE_HEIGHT,height),max(IMAGE_WIDTH,width))])
                resizing_aug = resizing(image=image,mask=mask)
                image = resizing_aug["image"]
                mask = resizing_aug["mask"]
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
