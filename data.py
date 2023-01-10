import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision.utils import save_image
import ntpath

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

class ThumbnailsDataset(Dataset):
    def __init__(self, image_dirs:list, mask_dirs:list, indices,transform=list, visualize_aug = True):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.images = []
        self.masks = []
        for img_dir, mask_dir in zip(image_dirs,mask_dirs):
            # There is some thumbs without mask, no need for them
            thumbs_names = [ image.replace("_thumb.jpg","") for image in os.listdir(img_dir) if image.endswith(".jpg")]
            masks_names =  [ mask.replace("_SegMap.png","") for mask in os.listdir(mask_dir) if mask.endswith(".png")]
            
            valid_img_names = [name for name in thumbs_names if name in masks_names]
            
            self.images += [os.path.join(img_dir, valid_img_names+ "_thumb.jpg") for valid_img_names in valid_img_names]
            self.masks += [os.path.join(mask_dir, valid_img_names + "_SegMap.png") for valid_img_names in valid_img_names]
            
        self.indices = indices
        self.visualize_aug = visualize_aug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = self.images[self.indices[idx]]
        mask_path = self.masks[self.indices[idx]]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("1"), dtype=np.float32)

        if self.visualize_aug:
            orig_image = Image.fromarray(image)
            if not(os.path.isdir('./augmented_imgs')):
                os.mkdir('./augmented_imgs')
            orig_image.save(os.path.join('./augmented_imgs', ntpath.basename(img_path)))

        if self.transform is not None:
            # a list of tranforms, to support custom transforms
            if isinstance(self.transform, list):
                for t in self.transform:
                    augmentations = t(image=image, mask=mask)
                    image = augmentations["image"]
                    
                    mask = augmentations["mask"]
                colors = np.array([[0, 0, 0], [255, 255, 255], [0, 0, 255]])
                mask_rgb = colors[mask.int()]
            #one transform
            else:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                
                colors = np.array([[0, 0, 0], [255, 255, 255], [0, 0, 255]])
                mask = augmentations["mask"]
                mask_rgb = colors[mask.int()]
        if self.visualize_aug:
            aug_img_name = ntpath.basename(img_path).replace("_thumb", "_aug")
            save_image(image, os.path.join('./augmented_imgs', aug_img_name))
            cv2.imwrite(os.path.join('./augmented_imgs', 'seg_' + aug_img_name), mask_rgb)

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
