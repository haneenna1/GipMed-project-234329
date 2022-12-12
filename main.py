import torch
from torch import nn
import torch.optim as optim
from model import Unet
from trainer import Trainer
from cmath import inf
# For reading huge images
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Setting the device for the Training

import albumentations as A
from albumentations.pytorch import ToTensorV2
import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 75
BATCH_SIZE = 10
PIN_MEMPRY = True
NUM_WORKERS = 8 # <= cpus
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512  
MANUAL_SEED = 42

def main():
    print(f'******************** device you are using is : {DEVICE}')
    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    img_dir = "/mnt/gipmed_new/Data/Breast/TCGA/SegData/Thumbs"
    # check other path than mnt/
    mask_dir = "/mnt/gipmed_new/Data/Breast/TCGA/SegData/SegMaps"
    
    
    train_transform = A.Compose(
        [
            A.PadIfNeeded(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.CropNonEmptyMaskIfExists(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            # A.RandomCrop(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            A.ColorJitter(),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # should passed when using the centerCrop without sliding window
    val_transform = A.Compose(
        [
            A.PadIfNeeded(1024, 1024),
            A.CropNonEmptyMaskIfExists(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform_for_sliding_window = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    default_datasets = [
        "ABCTB_TIF",
        "Carmel",
        "Covilha",
        "Haemek",   
        "HEROHE",
        "Ipatimup",
        "Sheba",
        "TCGA",
        "TMA"
    ]

    image_dirs, mask_dirs = utils.get_datasets_paths(default_datasets)
    train_dl, val_dl = utils.get_data_loaders(image_dirs, mask_dirs, train_transform, val_transform_for_sliding_window,
                                                inf, BATCH_SIZE, NUM_WORKERS, PIN_MEMPRY)

    trainer = Trainer(model, optimizer,loss_fn , sliding_window_validation=True, device = DEVICE)
    trainer.fit(train_dl, val_dl,  NUM_EPOCHS, early_stopping = 8)
  
if __name__ == "__main__": 
    main()


