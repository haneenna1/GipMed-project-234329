import torch
from torch import nn
import torch.optim as optim
from model import Unet
from train import Train
# Setting the device for the Training
import torchvision.transforms as T
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 16
PIN_MEMPRY = True
NUM_WORKERS = 2 # <= cpus

IMAGE_HEIGHT = 160  # 1280 originally for carvna
IMAGE_WIDTH = 240  # 1918 originally for carvana

def main():
    model = Unet(in_channels=3, out_channels=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    img_dir = "dummy_data/carvana_train_images"
    mask_dir = "dummy_data/carvana_train_masks"
    hyper_paramas = {'lr':LEARNING_RATE, 'num_epochs':NUM_EPOCHS,
                       'batch_size':BATCH_SIZE, 'pin_memory':PIN_MEMPRY, 'num_workers':NUM_WORKERS }
    
    
    train_transform = T.Compose(
        [
            T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
            # T.Rotate(limit=35, p=1.0),
            # T.HorizontalFlip(p=0.5),
            # T.VerticalFlip(p=0.1),
            T.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
            T.PILToTensor(),
        ],
    )

    val_transform = T.Compose(
        [
            T.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
            T.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
            T.PILToTensor(),
        ],
    )
    train= Train(model, optimizer, loss_fn, img_dir, mask_dir, hyper_paramas, train_transform= train_transform, val_transform= val_transform, load_model=False)
    train()
    

if __name__ == "__main__": 
    main()