import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import ntpath

IMAGE_HEIGHT=512
IMAGE_WIDTH=512

class ThumbnailsDataset(Dataset):
    def __init__(self, image_dirs:list, mask_dirs:list, indices,transform=None, visualize_aug = False):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.transform = transform
        self.images = []
        self.masks = []
        for img_dir, mask_dir in zip(image_dirs,mask_dirs):
            # There is some thumbs without mask, no need for them
            images_set = set([ image.replace("_thumb.jpg","") for image in os.listdir(img_dir) if image.endswith(".jpg")])
            masks_set =  set([ mask.replace("_SegMap.png","") for mask in os.listdir(mask_dir) if mask.endswith(".png")])
            files_names = images_set.intersection(masks_set)
            self.images += [os.path.join(img_dir, image+ "_thumb.jpg") for image in files_names]
            self.masks += [os.path.join(mask_dir, mask + "_SegMap.png") for mask in files_names]
        self.indices = indices
        self.visualize_aug = visualize_aug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = self.images[self.indices[idx]]
        mask_path = self.masks[self.indices[idx]]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.visualize_aug:
            orig_image = Image.fromarray(image)
            if not(os.path.isdir('./augmented_imgs')):
                os.mkdir('./augmented_imgs')
            orig_image.save(os.path.join('./augmented_imgs', ntpath.basename(img_path)))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            if self.visualize_aug:
                aug_img_name = ntpath.basename(img_path).replace("_thumb", "_aug")
                save_image(image, os.path.join('./augmented_imgs', aug_img_name))

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
