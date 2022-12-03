import torch
from model import Unet
from data import ThumbnailsDataset
from torch.utils.data import DataLoader
import torchvision.utils
from main import DEVICE
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from utils import save_predictions_as_imgs,REAL_PATH,load_checkpoint
from main import NUM_WORKERS,PIN_MEMPRY
from data import IMAGE_HEIGHT,IMAGE_WIDTH
import os
from random import choices
from main import val_transform_for_sliding_window

def inference(image_dir, mask_dir = None, num_images=20):
    full_dir_indices = range(len([path for path in os.listdir(image_dir) if path.endswith(".jpg")]))
    # we want a subset of the data
    chosen_dir_indices = choices(full_dir_indices, k = num_images)
    dataset = ThumbnailsDataset(image_dir= image_dir, mask_dir=mask_dir, indices=chosen_dir_indices, transform=val_transform_for_sliding_window)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS,pin_memory=PIN_MEMPRY)
    model = Unet(in_channels=3, out_channels=1,).to(DEVICE)
    load_checkpoint("Results/TCGA_3000_with_sliding_window/my_checkpoint.pth.tar")
    model.eval()
    with torch.no_grad():
        for index, (image,mask) in enumerate(dataloader):
            images = image.to(DEVICE)
            # define sliding window size and batch size for windows inference
            pred = sliding_window_inference(inputs=images, roi_size=(IMAGE_HEIGHT,IMAGE_WIDTH), sw_batch_size=1, predictor=model, overlap=0)
            out_folder = "inference_output"
            if not os.path.exists(REAL_PATH(out_folder)):
                os.mkdir(out_folder)
            torchvision.utils.save_image(images,f"{REAL_PATH(out_folder)}/img_{index}.jpg")
            torchvision.utils.save_image(pred,f"{REAL_PATH(out_folder)}/img_infer_{index}.jpg")


thumbnails_dir = "/mnt/gipmed_new/Data/Breast/ABCTB_TIF/SegData/Thumbs"
masks_dir = "/mnt/gipmed_new/Data/Breast/ABCTB_TIF/SegData/SegMaps"

if __name__ == "__main__": 
    inference(thumbnails_dir,masks_dir)
