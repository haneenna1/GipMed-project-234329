import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Unet

from dataset import ThumbnailsDataset, get_loaders    
from utils import save_checkpoint, load_checkpoint

class Train:
    def __init__(self, model, optimizer, loss_fn, img_dir, mask_dir,
                 train_transform, val_transform, hyper_paramas, load_model):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        if load_model:
            load_checkpoint(self.model)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        self.loss_fn = loss_fn
        self.hyper_params = hyper_paramas
        self.dataset = ThumbnailsDataset(ImageDir, maskDir)
        self.train_loader, self.val_loader = get_loaders(img_dir, mask_dir, batch_size, 
                                                         train_transforms, val_transforms, num_workers, pin_memory)
        
    def train_epoch(self):
        loop = tqdm(self.train_loader)

        for batch_index, (data, targets) in enumerate(loop): # each iteration is one epoch 
            data = data.to(self.device)
            targets = labels.to(self.device)   # check if need to unsqueeze

            #forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #update tqdm 
            loop.set_postfix(loss = loss.item())
            
    def train(self): 
        for epoch in range(self.hyper_params.num_epochs):
            self.train_epoch()
            
            #after each epoch, check accuracy on validation set 
            check_accuracy(val_loader, model, device=DEVICE)
            
            # print some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )
            # save model every two epochs
            if(epoch %2 == 0 ): 
                checkpoint = {model: model.state_dict(), optimizer: optimizer.state_dict()} 
                save_checkpoint(checkpoint)