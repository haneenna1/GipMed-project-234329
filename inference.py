import torch
from model import Unet
from data import ThumbnailsDataset
from torch.utils.data import DataLoader
import torchvision.utils
from main import DEVICE
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from utils import save_predictions_as_imgs,REAL_PATH,load_checkpoint,clear_folder
from main import NUM_WORKERS,PIN_MEMPRY
from data import IMAGE_HEIGHT,IMAGE_WIDTH
import os
from random import choices
from main import val_transform_for_sliding_window
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T

def inference(image_dir, mask_dir = None, num_images=10, out_folder = "inference_output" ):
    clear_folder(REAL_PATH(out_folder))
    # Subset of the indexes
    full_dir_indices = range(len([path for path in os.listdir(image_dir) if path.endswith(".jpg")]))
    chosen_dir_indices = choices(full_dir_indices, k = num_images)
    # passing transfomrs without any augmentation
    dataset = ThumbnailsDataset(image_dir= image_dir, mask_dir=mask_dir, indices=chosen_dir_indices, transform=val_transform_for_sliding_window)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS,pin_memory=PIN_MEMPRY) 
    model = Unet(in_channels=3, out_channels=1,).to(DEVICE)
    # Hardcoded results of the trained models
    load_checkpoint(model,"Results/TCGA_3000_with_sliding_window/my_checkpoint.pth.tar")
    
    if not os.path.exists(REAL_PATH(out_folder)):
        os.mkdir(out_folder)
    model.eval()

    with torch.no_grad():
        for index, (image,mask) in enumerate(dataloader):
            image = image.to(DEVICE)

            image_path = f"{REAL_PATH(out_folder)}/img_{index}.jpg"
            mask_path = f"{REAL_PATH(out_folder)}/img_{index}_mask.jpg"
            inferred_mask_path = f"{REAL_PATH(out_folder)}/img_{index}_infer.jpg"
            
            # define sliding window size and batch size for windows inference
            pred = sliding_window_inference(inputs=image, roi_size=(IMAGE_HEIGHT,IMAGE_WIDTH), sw_batch_size=1, predictor=model, overlap=0,progress=True)

            torchvision.utils.save_image(image, image_path)
            torchvision.utils.save_image(mask, mask_path)
            torchvision.utils.save_image(pred, inferred_mask_path)

            img = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
            inferred_mask = cv2.imread(inferred_mask_path)
            inferrred_mask_image = np.zeros((inferred_mask.shape[0], inferred_mask.shape[1],3), dtype=np.uint8)
            non_black_index = np.where(inferred_mask > 120) # TODO:maybe should apply the sigmoid and then do comparison > 0.5?
            # CV2 uses the BGR color so inferrred_mask_image would be green, you may print and check
            inferrred_mask_image[non_black_index[0], non_black_index[1], :] = (0, 255, 0)
            # overlaying our inferred mask on the original image
            first_segmentation_overlay = cv2.addWeighted(img, 0.5, inferrred_mask_image, 0.5, 0)
            orig_mask_image = np.zeros((mask.shape[0] ,mask.shape[1],3), dtype=np.uint8)
            non_black_index = np.where(mask > 120)
            # orig_mask_image would be blue
            orig_mask_image[non_black_index[0], non_black_index[1], :] = (255, 0, 0)
            # overlaying the original mask on the first overlayed result
            sec_segmentation_overlay = cv2.addWeighted(first_segmentation_overlay, 0.65, orig_mask_image, 0.35, 0)
            overlay_path = f"{REAL_PATH(out_folder)}/img_{index}_overlay.jpg"
            cv2.imwrite(overlay_path,sec_segmentation_overlay)

thumbnails_dir = "/mnt/gipmed_new/Data/Breast/ABCTB_TIF/SegData/Thumbs"
masks_dir = "/mnt/gipmed_new/Data/Breast/ABCTB_TIF/SegData/SegMaps"

if __name__ == "__main__": 
    # the sharp Green indicates our prediction
    # the sharp Blue indicates the original mask
    # the light blue indicates both masks
    inference(thumbnails_dir,masks_dir)
