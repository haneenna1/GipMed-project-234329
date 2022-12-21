from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import numpy as np
import torch
import os
from torchvision.utils import save_image
from PIL import Image
import albumentations as A
from utils import clear_folder
Image.MAX_IMAGE_PIXELS = None



from albumentations.pytorch import ToTensorV2

class CropTissueRoi(object):

    def __call__(self, image, mask):
        
        validation_rois = masks_to_boxes(mask.unsqueeze(0))
        
        x_min = torch.min(validation_rois[:, 0])
        y_min = torch.min(validation_rois[:, 1])
        x_max = torch.max(validation_rois[:, 2])
        y_max = torch.max(validation_rois[:, 3])
        
        top = y_min.int().item()
        left = x_min.int().item()
        width = (x_max - x_min).int().item()
        height = (y_max - y_min).int().item()
        
        cropped_img = TF.crop(image, top=top, left=left, width=width, height=height)
        cropped_masks = TF.crop(mask, top=top, left=left, width=width, height=height)

        return {'image': cropped_img, 'mask': cropped_masks}
    
    
    
from torchvision.io import read_image
if __name__ == '__main__':

    val_transform_for_sliding_window = A.Compose([
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
    ])
    Cr = CropTissueRoi()

    counter = 1

    HEROHE_imgs_path = os.path.join("/mnt/gipmed_new/Data/Breast/HEROHE/SegData/Thumbs")
    HEROHE_masks_path = os.path.join("/mnt/gipmed_new/Data/Breast/HEROHE/SegData/SegMaps")

    clear_folder("./crop_validations")

    for index, img in enumerate( os.listdir(HEROHE_imgs_path)):
        if index >= counter:
            break
        image = np.array(Image.open(os.path.join(HEROHE_imgs_path,img)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(HEROHE_masks_path,img.replace("_thumb.jpg","_SegMap.png"))).convert("1"), dtype=np.float32)
        aug = val_transform_for_sliding_window(image=image, mask=mask)
        aug_image = aug['image']
        aug_mask = aug['mask']
    
        cropped_img = Cr(aug_image, aug_mask)['image']
        cropped_mask = Cr(aug_image, aug_mask)['mask']
        save_image(aug_image, os.path.join(f"./crop_validations", f"orig_img_{index}.jpg"))
        save_image(aug_mask, os.path.join(f"./crop_validations", f"orig_mask{index}.png"))
        save_image(cropped_img, os.path.join(f"./crop_validations", f"cropped_img{index}.jpg"))
        save_image(cropped_mask, os.path.join(f"./crop_validations", f"cropped_mask{index}.png"))

    