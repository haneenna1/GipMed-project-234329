import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Unet

from dataset import ThumbnailsDataset, get_loaders    
from utils import save_checkpoint, load_checkpoint

LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 16
PIN_MEMPRY = True
NUM_WORKERS = 2

# Setting the device for the Training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_epoch(model, loader, optimizer, scaler, loss_fn):
    loop = tqdm(loader)

    for batch_index, (data, labels) in enumerate(loop): # each iteration is one epoch 
        data = data.to(device = DEVICE)
        labels = labels.to(device = DEVICE)   # check if need to unsqueeze

        #forward
        with torch.cuda.amp.autocast(): 
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm 
        loop.set_postfix(loss = loss.item())
        
def main(): 
    train_transform = []
    val_transforms = []
    model = Unet(in_channels=3, out_channels=2).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR,
        VAL_IMG_DIR, 
        VAL_MAKS_DIR,
        BATCH_SIZE, 
        train_transforms, 
        val_transforms, 
        NUM_WORKERS,
        PIN_MEMPRY
    )

    for epoch in range(NUM_EPOCHS):
        train_epoch(model, train_loader, optimizer, scaler, loss_fn)

        # save model every two epochs
        if(epoch %2 == 0 ): 
            checkpoint = {model_state: model.state_dict(), optimizer: optimizer.state_dict()} 
            save_checkpoint(checkpoint, file_name = f'unet_checkpoint_epoch{epoch}')
        
        # check accuuracy

        # print some examples to a file 
        
if __name__=="__main__": 
    main()