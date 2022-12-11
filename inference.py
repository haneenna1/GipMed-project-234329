import torch
from model import Unet
from data import ThumbnailsDataset
from torch.utils.data import DataLoader
import torchvision.utils
from utils import REAL_PATH,load_checkpoint,clear_folder
import os
from random import choices
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Dice, JaccardIndex

from main import NUM_WORKERS,PIN_MEMPRY,DEVICE

class Inferer:

    def __init__(self, prev_checkpoint, out_folder = "inference_output") -> None:
        clear_folder(REAL_PATH(out_folder))
        self.out_folder = out_folder
    
        self.model = Unet(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint(self.model, prev_checkpoint)

        self.inference_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        )

        self.accuracies_list = []
        self.dice_scores_list = []
        self.jacard_index_list = []

    def infer(self,image_dir ,mask_dir, num_images=10, visulaize =True)->None:
        self.num_iamges = num_images
        # Subset of the indexes
        full_dir_indices = range(len([path for path in os.listdir(image_dir) if path.endswith(".jpg")]))
        chosen_dir_indices = choices(full_dir_indices, k = num_images)
        # passing transfomrs without any augmentation
        dataset = ThumbnailsDataset(image_dir= image_dir, mask_dir=mask_dir, indices=chosen_dir_indices, transform=self.inference_transforms)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS,pin_memory=PIN_MEMPRY) 
        
        self.model.eval()
        with torch.no_grad():
            for index, (image,mask) in enumerate(dataloader):
                image = image.to(DEVICE)
                mask = mask.to(DEVICE)

                image_path = f"{REAL_PATH(self.out_folder)}/img_{index}.jpg"
                mask_path = f"{REAL_PATH(self.out_folder)}/img_{index}_mask.jpg"
                inferred_mask_path = f"{REAL_PATH(self.out_folder)}/img_{index}_infer.jpg"
                pred_scores = self.model.sliding_window_inference(image)
                # saving images
                torchvision.utils.save_image(image, image_path)
                torchvision.utils.save_image(mask, mask_path)
                torchvision.utils.save_image(pred_scores, inferred_mask_path)
                if visulaize:
                    self.visulaize_shade(image_path, mask_path, inferred_mask_path, index)
                    self.visulaize_sharp(image_path, mask_path, inferred_mask_path, index)
                self.calculate_metrics(pred_scores,mask.unsqueeze(1))
            print(f'------------ Inference ------------ ')
            print(f'------------ ACCURACY:{torch.mean(torch.FloatTensor(self.accuracies_list))} ------------ ')
            print(f'------------ DICE_SCORE:{torch.mean(torch.FloatTensor(self.dice_scores_list))}------------ ')
            print(f'------------ JACARD_INDEX:{torch.mean(torch.FloatTensor(self.jacard_index_list))}------------ ')

    
    # Getting an overlayed image with a shade of colors [Green --- Blue]
    # the sharp Blue indicates the original mask, and the sharp Green 
    # indicates the inferred mask. the light blue regions would indicates both
    def visulaize_shade(self, image_path, mask_path, inferred_mask_path, index)->None:

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        inferred_mask = cv2.imread(inferred_mask_path)
        
        # Coloring the inferred mask with green
        inferrred_mask_image = np.zeros((inferred_mask.shape[0], inferred_mask.shape[1],3), dtype=np.uint8)
        non_black_index = np.where(np.all((inferred_mask >= 127),axis=-1))
        # CV2 uses the BGR color so inferrred_mask_image would be green, you may print and check
        inferrred_mask_image[non_black_index] = (0, 255, 0)
        # To check out the colored mask
        # cv2.imwrite( f"{REAL_PATH(self.out_folder)}/img_{index}_inferrred_mask_image_non_black.jpg",inferrred_mask_image)
        # overlaying our inferred mask on the original image
        first_segmentation_overlay = cv2.addWeighted(img, 0.5, inferrred_mask_image, 0.5, 0)

        # Coloring the original mask with Blue
        orig_mask_image = np.zeros((mask.shape[0] ,mask.shape[1],3), dtype=np.uint8)
        non_black_index = np.where(np.all((mask >= 127),axis=-1))
        orig_mask_image[non_black_index] = (255, 0, 0)
        # To check out the colored mask
        # cv2.imwrite(f"{REAL_PATH(self.out_folder)}/img_{index}_mask_image_non_black.jpg",orig_mask_image,)
        # overlaying the original mask on the first overlayed result
        sec_segmentation_overlay = cv2.addWeighted(first_segmentation_overlay, 0.65, orig_mask_image, 0.35, 0)
        # save the final overlayed image
        shade_visulaize_apth = f"{REAL_PATH(self.out_folder)}/img_{index}_shade_visulaize.jpg"
        cv2.imwrite(shade_visulaize_apth,sec_segmentation_overlay)

    def visulaize_sharp(self, image_path, mask_path, inferred_mask_path, index)->None:
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        inferred_mask = cv2.imread(inferred_mask_path)
        
        # Coloring the inferred mask with green
        both_non_black_index = np.where(np.all(((inferred_mask >= 127) & (mask >= 127)),axis=-1))
        infer_non_black_index = np.where(np.all(((inferred_mask >= 127) & (mask < 127)),axis=-1))
        mask_non_black_index = np.where(np.all(((mask >= 127 )& (inferred_mask<127) ),axis=-1))
        img[both_non_black_index] = (0, 0, 255)
        img[infer_non_black_index] = (0, 255, 0)
        img[mask_non_black_index] = (255, 0, 0)
        sharp_visulaize_path = f"{REAL_PATH(self.out_folder)}/img_{index}_sharp_visulaize.jpg"
        cv2.imwrite(sharp_visulaize_path,img)

    
    def calculate_metrics(self, per_pixel_score_predictions, masks_batch)->None:
        jaccard = JaccardIndex('binary').to(DEVICE)
        dice_metric = Dice().to(DEVICE)
        accuracy_metric = BinaryAccuracy().to(DEVICE)
        pred_masks = self.model.predict_labels(per_pixel_score_predictions) 
        self.dice_scores_list.append(dice_metric(pred_masks, masks_batch.int()))
        self.accuracies_list.append(accuracy_metric(pred_masks, masks_batch))
        self.jacard_index_list.append(jaccard(pred_masks,masks_batch))


if __name__ == "__main__": 
    thumbnails_dir = "/mnt/gipmed_new/Data/Breast/ABCTB_TIF/SegData/Thumbs"
    masks_dir = "/mnt/gipmed_new/Data/Breast/ABCTB_TIF/SegData/SegMaps"
    prev_checkpoint = "my_checkpoint.pth.tar"
    unet_inferer = Inferer(prev_checkpoint, out_folder="ABCB_TIF_infer")
    unet_inferer.infer(thumbnails_dir, masks_dir, num_images=10, visulaize=True)
