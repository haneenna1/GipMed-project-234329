from execnet import MultiChannel
import torch
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
# from utils import gettingDataFolders
from utils import save_checkpoint, load_checkpoint, get_data_loaders, check_accuracy, save_predictions_as_imgs

from utils import BATCH_SIZE
class Trainer:
    def __init__(self, model, optimizer, loss_fn, img_dir, mask_dir, hyper_paramas, num_imgs = 30, 
                 train_transform = None, val_transform = None,  load_model = False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        if load_model:
            load_checkpoint(self.model)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.hyper_params = hyper_paramas
        # img_dirs, mask_dirs = gettingDataFolders()
        self.train_loader, self.val_loader = get_data_loaders(img_dir, mask_dir, hyper_paramas["batch_size"],
                                                         hyper_paramas["num_workers"],train_transform, val_transform,
                                                         num_imgs, hyper_paramas["pin_memory"])
        self.writer = SummaryWriter(comment="FIRST_TRAINING_UNET_OVERFITTING")

    def train_epoch(self,epoch):
        loop = tqdm(self.train_loader)

        losses = []
        for cur_batch,(data, targets) in enumerate(loop):  # each iteration is one epoch
            data = data.to(self.device)
            targets = targets.unsqueeze(1).to(self.device)  # check if need to unsqueeze

            # forward
            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)


            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update tqdm
            loop.set_postfix(loss=loss.item())

            losses.append(loss.item())

            global_batch_counter = BATCH_SIZE*epoch + cur_batch
            self.writer.add_scalar('Loss/train', loss, global_batch_counter)


    def __call__(self):
        for epoch in range(self.hyper_params['num_epochs']):
            print(f'------------ epoch #{epoch} ------------ ')
            self.train_epoch(epoch)

            # after each epoch, check accuracy on validation set
            check_accuracy(self.val_loader, self.model,epoch, self.writer, device=self.device)
            
            # print some examples to a folder
            save_predictions_as_imgs(
                self.val_loader, self.model, device=self.device
            )

            # save model every two epochs
            if (epoch % 2 == 0):
                checkpoint = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                save_checkpoint(checkpoint)
        self.writer.close()


