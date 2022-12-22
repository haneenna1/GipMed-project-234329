import torch
from torch import nn
import torch.optim as optim
from Unet import Unet
from trainer import Trainer
from cmath import inf
# For reading huge images
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Setting the device for the Training

import albumentations as A
from albumentations.pytorch import ToTensorV2
import utils
import CustomTransforms as C

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 10
PIN_MEMPRY = True
NUM_WORKERS = 10 # <= cpus
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512  
MANUAL_SEED = 42

def main():
    print(f'******************** device you are using is : {DEVICE}')
    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    
    train_transform = [ 
        C.CropTissueRoi(), # we could use this for the train data too! espicially useful for training on sparse images like HEROHE
        A.Compose([
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
        ]), 
    ]
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

    val_transform_for_sliding_window = [
        A.Compose([
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        , 
        C.CropTissueRoi(),
    ]

    default_datasets = [
        "TCGA",
        "ABCTB_TIF",
        "Carmel",
        "Haemek",   
        "Covilha",
        "HEROHE",
        "Ipatimup",
        "Sheba",
        # "TMA"
    ]
    # ===== Change here  =====
    datasets = ["TCGA",  "ABCTB_TIF", "Haemek", "Ipatimup" ]
    model_name = 'TCG_ABC_emk_Iptmp'
    # ==========  ===========

    image_dirs, mask_dirs = utils.get_datasets_paths(datasets)
    train_dl, val_dl = utils.get_data_loaders(image_dirs, mask_dirs, train_transform, val_transform_for_sliding_window, 
                                              train_batch_size=BATCH_SIZE, num_workers= NUM_WORKERS, pin_memory= PIN_MEMPRY)

    trainer = Trainer(model, model_name,optimizer,loss_fn , sliding_window_validation=True, device = DEVICE)
    trainer.fit(train_dl, val_dl,  NUM_EPOCHS)
  
if __name__ == "__main__": 
    main()


