from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import numpy as np
import torch
import os
from torchvision.utils import save_image
from PIL import Image
import albumentations as A


from albumentations.pytorch import ToTensorV2

class CropTissueRoi(object):

    def __call__(self, image, mask):
        
        validation_rois = masks_to_boxes(mask.unsqueeze(0))
        print(f'validation_rois = {validation_rois}')
        print(f'shape of mask in CropTissueRoi = {mask.unsqueeze(0).shape}')
        
        x_min = torch.min(validation_rois[:, 0])
        y_min = torch.min(validation_rois[:, 1])
        x_max = torch.max(validation_rois[:, 2])
        y_max = torch.max(validation_rois[:, 3])
        
        top = y_min.int().item()
        left = x_min.int().item()
        width = (x_max - x_min).int().item()
        height = (y_max - y_min).int().item()
        
        print(f'top = {top} left ={left} width = {width} height ={height}')
        
        cropped_img = TF.crop(image, top=top, left=left, width=width, height=height)
        cropped_masks = TF.crop(mask, top=top, left=left, width=width, height=height)

        return {'image': cropped_img, 'mask': cropped_masks}
    
    
    
from torchvision.io import read_image
if __name__ == '__main__':
    img_path = os.path.join("/mnt/gipmed_new/Data/Breast/TCGA/SegData/Thumbs",
                            'TCGA-3C-AAAU-01A-01-TS1.2F52DD63-7476-4E85-B7C6-E06092DB6CC1_thumb.jpg')
    mask_path = os.path.join("/mnt/gipmed_new/Data/Breast/TCGA/SegData/SegMaps",
                            'TCGA-3C-AAAU-01A-01-TS1.2F52DD63-7476-4E85-B7C6-E06092DB6CC1_SegMap.png')
    
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("1"), dtype=np.float32)


    val_transform_for_sliding_window = A.Compose([
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
    ])
    aug = val_transform_for_sliding_window(image=image, mask=mask)

    aug_image = aug['image']
    aug_mask = aug['mask']
    
    Cr = CropTissueRoi()
    cropped_img = Cr(aug_image, aug_mask)['image']
    cropped_mask = Cr(aug_image, aug_mask)['mask']
    
    save_image(aug_image, os.path.join('./crop_validations', 'orig_img.jpg'))
    save_image(aug_mask, os.path.join('./crop_validations', 'orig_mask.png'))
    save_image(cropped_img, os.path.join('./crop_validations', 'cropped_img.jpg'))
    save_image(cropped_mask, os.path.join('./crop_validations', 'cropped_mask.png'))
    