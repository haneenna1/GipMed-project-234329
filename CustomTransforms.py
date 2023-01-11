from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import random
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
        # if empty mask
        if mask.eq(0).all(): 
            return {'image': image, 'mask': mask}
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
    
    

class AddAnnotation(object):
    def __init__(self, p = 0.4, mode = 'train'):
        self.p = p
        self.mode = mode
        
    def __call__(self, image, mask):
        # Determine the size of the output image
        enable = (1 if random.uniform(0, 1) < self.p else 0)
        if enable == 0:
            return {'image': image, 'mask': mask}
        
        annotation_pth, annotation_seg_path = (('Markings/markings/','Markings/markings_segmaps/') if self.mode=='train' 
                                    else ('Markings/val_markings/','Markings/val_markings_segmaps/') )
        
        annotations = os.listdir(annotation_pth)
        annotations_seg = os.listdir(annotation_seg_path)
        
        chosen_idx = random.randint(0, len(annotations)-1) 
        
        annotation_pth = os.path.join(annotation_pth, (annotations[chosen_idx]))
        annotation_seg_pth = os.path.join(annotation_seg_path, (annotations_seg[chosen_idx]))
        
        image2 = np.array(Image.open(annotation_pth).convert("RGB"))
        mask2 = np.array(Image.open(annotation_seg_pth).convert("1"), dtype=np.uint8)
        
        rows = image.shape[0]
        cols = image.shape[1]
        # Create the output image and mask with the correct size
        image_out = image
        mask_out = mask
        
        # Random resize the second image to fit within the bounds of the first image
        scale_factor = np.random.uniform(0.2, 0.8)
        rand_size = (int(rows * scale_factor), int(cols * scale_factor))  
        # Resize the image
        image2 = cv2.resize(image2, rand_size)
        mask2 = cv2.resize(mask2, rand_size, interpolation=cv2.INTER_NEAREST)
        
        x_offset = random.randint(0, cols - image2.shape[1])
        y_offset = random.randint(0, rows - image2.shape[0])
        
        image_out[y_offset:y_offset+image2.shape[0],
                x_offset:x_offset+image2.shape[1], :][mask2 > 0]  = image2[mask2 > 0]
        mask_out[y_offset:y_offset+mask2.shape[0],
                x_offset:x_offset+mask2.shape[1]][mask2 > 0] = 2
            
        return {'image': image_out, 'mask': mask_out}

class Pad16Mul(object):
        
    def __call__(self, image, mask): 
        width = max(image.shape[0], image.shape[1])
        if width%16 != 0 : 
            width = (( width // 16) + 1) * 16
            
        padded = A.PadIfNeeded(1600, 1600)(image = image, mask = mask)
        
        image, mask = padded['image'], padded['mask']  
        return {'image': image, 'mask': mask}
