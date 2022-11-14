import torch
from torch import nn
import torch.optim as optim
from model import Unet
from train import Train

# Setting the device for the Training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 16
PIN_MEMPRY = True
NUM_WORKERS = 2 # <= cpus

def main():
    model = Unet(in_channels=3, out_channels=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    img_dir = ""
    mask_dir = ""
    train_transform = []
    val_transform = []
    hyper_paramas = {"lr":LEARNING_RATE, "num_epochs":NUM_EPOCHS,
                       "batch_size":BATCH_SIZE, "pin_memory":PIN_MEMPRY, "num_workers":NUM_WORKERS }
    
    trainer = Train(model, optimizer, loss_fn, img_dir, mask_dir, train_transform, val_transform, hyper_paramas)
    trainer.train()
    

if __name__ == "__main__": 
    main()