import torch
from torch import nn
import torch.optim as optim
from model import Unet
from trainer import Trainer
# Setting the device for the Training
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import IMAGE_HEIGHT,IMAGE_WIDTH
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
BATCH_SIZE = 8
PIN_MEMPRY = True
NUM_WORKERS = 2 # <= cpus
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
    hyper_paramas = {'lr':LEARNING_RATE, 'num_epochs':NUM_EPOCHS,
                       'batch_size':BATCH_SIZE, 'pin_memory':PIN_MEMPRY, 'num_workers':NUM_WORKERS }
    
    
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
    val_transform = A.Compose(
        [
            A.PadIfNeeded(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.CropNonEmptyMaskIfExists(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train= Trainer(model, optimizer, loss_fn, img_dir, mask_dir, hyper_paramas, num_imgs=10, train_transform= train_transform, val_transform= val_transform, load_model=False)
    train()
  
if __name__ == "__main__": 
    main()



