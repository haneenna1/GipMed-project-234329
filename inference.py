import torch
from torch.utils.data import DataLoader
import torchvision.utils
import os
from random import choices
import numpy as np
import cv2
import argparse

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics import Dice
from utils import save_012_mask_as_img

from Unet import Unet
from data import ThumbnailsDataset
from utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'******************** device you are using is : {DEVICE}')
class Inferer:

    def __init__(self, checkpoint_name = None, model = Unet(in_channels=3, out_channels=3).to(DEVICE), out_folder = "inference_output") -> None:
        if not(os.path.isdir(REAL_PATH(out_folder))):
                os.makedirs(REAL_PATH(out_folder))
        clear_folder(REAL_PATH(out_folder))
        self.out_folder = out_folder
    
        self.model = model
        
        if checkpoint_name != None:
            load_checkpoint(self.model, checkpoint_name= checkpoint_name)


        self.accuracies_list = []
        self.dice_scores_list = []
        self.jacard_index_list = []

    def infer(self, dataloader, visulaize =False)->None:
        
        self.model.eval()
        with torch.no_grad():
            for index, (image,mask) in enumerate(dataloader):
                image = image.to(DEVICE)
                mask = mask.to(DEVICE)

                per_pixel_pred_scores = self.model.sliding_window_validation(image, mask)
                inferred_mask = self.model.predict_labels_from_scores(per_pixel_pred_scores)

                image_path = f"{REAL_PATH(self.out_folder)}/img_{index}.jpg"
                mask_path = f"{REAL_PATH(self.out_folder)}/img_{index}_mask.jpg"
                inferred_mask_path_jpg = f"{REAL_PATH(self.out_folder)}/img_{index}_infer.jpg"
                # png suffix for saving the image pixes as is without any changes
                inferred_mask_path_png = f"{REAL_PATH(self.out_folder)}/img_{index}_infer.png"

                # saving images
                torchvision.utils.save_image(image, image_path)
                torchvision.utils.save_image(mask, mask_path)
                save_012_mask_as_img(inferred_mask.squeeze(1), inferred_mask_path_jpg)

                inferred_mask = inferred_mask.cpu()
                inferred_mask_numpy = PIL.Image.fromarray(np.array(inferred_mask, dtype = np.uint8).reshape(inferred_mask.shape[2],inferred_mask.shape[3]))
                inferred_mask_numpy.save(inferred_mask_path_png)

                # save_012_mask_as_img(inferred_mask.squeeze(1), inferred_mask_path_jpg)

                # stam = np.array(PIL.Image.open(inferred_mask_path).convert("L"), dtype=np.float32)
                # print(np.count_nonzero(stam == 2))
                # print(stam.shape)
                
                if visulaize:
                    # self.visulaize_shade(image_path, mask_path, inferred_mask_path, index)
                    self.visulaize_sharp(image_path, mask_path, inferred_mask_path_png, index)
               
                self.calculate_metrics(per_pixel_pred_scores,mask.unsqueeze(1))
            
            with open(os.path.join(self.out_folder, 'metrics_results'), 'w') as f:
                f.write(f'------------ ACCURACY:{torch.mean(torch.FloatTensor(self.accuracies_list))} ------------ \n')
                f.write(f'------------ DICE_SCORE:{torch.mean(torch.FloatTensor(self.dice_scores_list))}------------ \n')
                f.write(f'------------ JACARD_INDEX:{torch.mean(torch.FloatTensor(self.jacard_index_list))}------------ \n')

    
    # Getting an overlayed image with a shade of colors [Green --- Blue]
    # the sharp Blue indicates the original mask, and the sharp Green 
    # indicates the inferred mask. the light blue regions would indicates both
    def visulaize_shade(self, image_path, mask_path, inferred_mask_path, index)->None:
        img = np.array(PIL.Image.open(image_path).convert("RGB"))
        mask = np.array(PIL.Image.open(mask_path).convert("1"), dtype=np.float32)
        
        inferred_mask = np.array(PIL.Image.open(inferred_mask_path).convert("1"), dtype=np.float32)
        
        # Coloring the inferred mask with green
        inferrred_mask_colored = np.zeros((inferred_mask.shape[0], inferred_mask.shape[1],3), dtype=np.uint8)
        non_black_index = np.where(inferred_mask == 1)
    
        # CV2 uses the BGR color so inferrred_mask_image would be green, you may print and check
        inferrred_mask_colored[non_black_index[0], non_black_index[1], :] = (0, 255, 0)
        first_segmentation_overlay = cv2.addWeighted(img, 0.5, inferrred_mask_colored, 0.5, 0)

        # Coloring the original mask with Blue
        orig_mask_image = np.zeros((mask.shape[0] ,mask.shape[1],3), dtype=np.uint8)
        non_black_index = np.where(mask == 1)
        orig_mask_image[non_black_index[0], non_black_index[1], :] = (255, 0, 0)
        sec_segmentation_overlay = cv2.addWeighted(first_segmentation_overlay, 0.65, orig_mask_image, 0.35, 0)
        
        # save the final overlayed image
        shade_visulaize_apth = f"{REAL_PATH(self.out_folder)}/img_{index}_shade_visulaize.jpg"
        
        cv2.imwrite(shade_visulaize_apth,sec_segmentation_overlay)

    def visulaize_sharp(self, image_path, mask_path, inferred_mask_path, index)->None:
        
        img = np.array(PIL.Image.open(image_path).convert("RGB"))
        mask = np.array(PIL.Image.open(mask_path).convert("1"), dtype=np.float32)        
        inferred_mask = np.array(PIL.Image.open(inferred_mask_path).convert("L"), dtype=np.float32)
        # print(stam.shape)
        # amir = np.array(stam)
        # print(np.count_nonzero(amir == 2))
        # print(inferred_mask.shape)
        # inferred_mask=inferred_mask.cpu()
        # amir = np.array(inferred_mask)
        # print(np.count_nonzero(amir == 2))
        # print(inferred_mask.shape)
        # inferred_mask = inferred_mask.squeeze(dim=0)
        # inferred_mask = inferred_mask.squeeze(dim=1)
        # inferred_mask = inferred_mask.squeeze(dim=1)
        # inferred_mask = amir.reshape(amir.shape[2], amir.shape[3])


        # .squeeze(1)
        # print(inferred_mask.shape)

        # inferred_mask = inferred_mask.cpu()
        bad_pathologist_artifatcs = np.where((inferred_mask == 2))
        print(bad_pathologist_artifatcs)
        both_non_black_index = np.where((inferred_mask == 1) & (mask == 1))
        infer_non_black_index = np.where((inferred_mask == 1) & (mask == 0))
        mask_non_black_index = np.where((inferred_mask == 0) & (mask == 1))

        img[bad_pathologist_artifatcs[0], bad_pathologist_artifatcs[1], :] = (0, 128, 255)
        img[both_non_black_index[0], both_non_black_index[1], :] = (0, 0, 255)
        img[infer_non_black_index[0], infer_non_black_index[1], :] = (0, 255, 0)
        img[mask_non_black_index[0], mask_non_black_index[1], : ] = (255, 0, 0)
        
        sharp_visulaize_path = f"{REAL_PATH(self.out_folder)}/img_{index}_sharp_visulaize.jpg"
        # save_012_mask_as_img(inferred_mask, inferred_mask_path)
        
        # #add legend
        # cv2.rectangle(img, (20, 20), (500, 250), (200, 200, 200), -1)
        
        # position = (60, 60)
        # text = "Predicted by Unet only"
        # color = (0, 255, 0)

        # cv2.rectangle(img, (position[0]-40, position[1]-30), (position[0], position[1] + 20), color, -1)
        # cv2.putText(img, text, position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

        # position = (60, 120)
        # text = "Predictedf by OTSU only"
        # color = (255, 0, 0)
        
        # cv2.rectangle(img, (position[0]-40, position[1]-30), (position[0], position[1] + 20), color, -1)
        # cv2.putText(img, text, position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)        

        # position = (60, 180)
        # text = "Predicted by Both"
        # color = (0, 0, 255)
        
        # cv2.rectangle(img, (position[0]-40, position[1]-30), (position[0], position[1] + 20), color, -1)
        # cv2.putText(img, text, position, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        
        
        
        cv2.imwrite(sharp_visulaize_path,img)

    
    def calculate_metrics(self, per_pixel_score_predictions, masks_batch)->None:
        jaccard =  MulticlassJaccardIndex(num_classes = 3, average ='weighted').to(DEVICE)
        dice_metric = Dice(num_classes=3, multiclass=True).to(DEVICE)
        accuracy_metric = MulticlassAccuracy(num_classes=3, average ='weighted').to(DEVICE)
        pred_masks = self.model.predict_labels_from_scores(per_pixel_score_predictions) 
        self.dice_scores_list.append(dice_metric(pred_masks, masks_batch.int()))
        self.accuracies_list.append(accuracy_metric(pred_masks, masks_batch.int()))
        self.jacard_index_list.append(jaccard(pred_masks,masks_batch.int()))


if __name__ == "__main__": 

    ###########################################################################
    #  usage: inference.py [-h] [--checkpoint-name CHECKPOINT_NAME]
    #  [--model-type MODEL_TYPE] [--out-dir OUT_DIR] 
    #  [--datasets DATASETS [DATASETS ...]] [--data-size DATA_SIZE] 
    #  [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--visulaize]
    #
    ###########################################################################

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--test-dataset', type=str)
    parser.add_argument('--data-size', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--visualize', action='store_true', default=True)
    
    args = parser.parse_args()

    print(args)
    
    inference_transforms = [
            A.Compose([
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])
        ]
    
    test_thumbnails_dir, test_masks_dir = get_datasets_paths([args.test_dataset])
    dataloader, _ = get_data_loaders(test_thumbnails_dir, test_masks_dir, inference_transforms, val_transforms= None, validation_ratio=0, data_size = args.data_size, num_workers= args.num_workers, train_batch_size= args.batch_size, shuffle=False) 
    
    out_folder = os.path.join("test_inference/", args.model_name, args.test_dataset)
    # REAL_PATH(out_folder)


    unet_inferer = Inferer(args.model_name, out_folder=out_folder)
    unet_inferer.infer(dataloader, visulaize=args.visualize)
