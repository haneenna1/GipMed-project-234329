import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

IMAGE_HEIGHT=512
IMAGE_WIDTH=512

class ThumbnailsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, indices,transform=None, visualize_aug = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [path for path in os.listdir(image_dir) if path.endswith(".jpg")]
        # print(self.images)
        self.indices = indices
        self.visualize_aug = visualize_aug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_name = self.images[self.indices[idx]]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace("_thumb.jpg",  "_SegMap.png"))  # TODO: check again the suffix in gipDeep
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.visualize_aug:
            orig_image = Image.fromarray(image)
            if not(os.path.isdir('./augmented_imgs')):
                os.mkdir('./augmented_imgs')
            orig_image.save(os.path.join('./augmented_imgs', img_name))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            if self.visualize_aug:
                aug_img_name = img_name.replace("_thumb", "_aug")
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
