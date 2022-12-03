
import torch
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
# from utils import gettingDataFolders
from utils import save_checkpoint, load_checkpoint, save_predictions_as_imgs
from train_results_classes import BatchResult, EpochResult, FitResult

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device = None, load_model = False):
        self.model = model
        if load_model:
            load_checkpoint(self.model)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        if self.device:
            model.to(self.device)
        

    def train_batch(self, batch) -> BatchResult:
        batch_imgs, batch_masks = batch
        batch_imgs = batch_imgs.to(self.device)
        batch_masks = batch_masks.unsqueeze(1).to(self.device)  # check if need to unsqueeze

        # forward
        per_pixel_score_predictions = self.model(batch_imgs)
        batch_loss = self.loss_fn(per_pixel_score_predictions, batch_masks)


        # accuracy and dice
        pred_masks = torch.sigmoid(per_pixel_score_predictions)
        pred_masks = (pred_masks > 0.5).float()
        num_correct = (pred_masks == batch_masks).sum()
        num_pixels = torch.numel(pred_masks)
        dice_score = (2 * (pred_masks * batch_masks).sum()) / ((pred_masks + batch_masks).sum() + 1e-8)

        # backward
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return BatchResult(batch_loss, num_correct, num_pixels, dice_score)
    
    def test_batch(self, batch) -> BatchResult:
        batch_imgs, batch_masks = batch
        batch_imgs = batch_imgs.to(self.device)
        batch_masks = batch_masks.unsqueeze(1).to(self.device)  # check if need to unsqueeze

            # forward
        with torch.no_grad():
            per_pixel_score_predictions = self.model(batch_imgs)
            batch_loss = self.loss_fn(per_pixel_score_predictions, batch_masks)


            # accuracy and dice
            pred_masks = torch.sigmoid(per_pixel_score_predictions)
            pred_masks = (pred_masks > 0.5).float()
            num_correct = (pred_masks == batch_masks).sum()
            num_pixels = torch.numel(pred_masks)
            dice_score = (2 * (pred_masks * batch_masks).sum()) / ((pred_masks + batch_masks).sum() + 1e-8)

        return BatchResult(batch_loss, num_correct, num_pixels, dice_score)


    def train_epoch(self, dl_train) -> EpochResult:
        epoch_loop = tqdm(dl_train)

        num_correct = 0
        num_pixels = 0
        dice_score = 0
        losses = []
        for batch in epoch_loop:  # each iteration is one epoch
            
            batch_res = self.train_batch(batch)
            losses.append(batch_res.loss)
            # update tqdm
            epoch_loop.set_postfix(loss=batch_res.loss.item(), dice = batch_res.dice_score.item())
            num_correct += batch_res.num_correct
            num_pixels += batch_res.num_correct
            dice_score += batch_res.dice_score

        accuracy = num_correct / num_pixels
        dice_score = dice_score/len(dl_train)
        return EpochResult(losses, accuracy, dice_score)


    
    def validation_epoch(self, validation_dl):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        losses = []
        for batch in validation_dl:  # each iteration is one epoch
            
            batch_res = self.test_batch(batch)
            losses.append(batch_res.loss)
            # update tqdm
            num_correct += batch_res.num_correct
            num_pixels += batch_res.num_correct
            dice_score += batch_res.dice_score

        accuracy = num_correct / num_pixels
        dice_score = dice_score/len(validation_dl)

        return EpochResult(losses, accuracy, dice_score)


    def fit(
        self,
        dl_train,
        dl_validation,
        num_epochs: int,
        early_stopping: int = None,
        print_every: int = 1,
        **kw,
    )  -> FitResult:
        writer = SummaryWriter('runs/unet_TCGA_experiment')
        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        best_dice_score = None

        for epoch in range(num_epochs):
            actual_num_epochs += 1
            print(f'------------ epoch #{epoch} ------------ ')

            train_epoch_result = self.train_epoch(dl_train)
            val_epoch_result = self.validation_epoch(dl_validation)
            
            train_acc.append(train_epoch_result.accuracy)            
            train_loss.append(torch.mean(torch.FloatTensor(train_epoch_result.losses)).item())
            test_acc.append(val_epoch_result.accuracy)
            test_loss.append(torch.mean(torch.FloatTensor(val_epoch_result.losses)).item())

            writer.add_scalar('Loss/train', torch.mean(torch.FloatTensor(train_epoch_result.losses)).item(), epoch)
            writer.add_scalar('Loss/validation', torch.mean(torch.FloatTensor(val_epoch_result.losses)).item(), epoch)
            writer.add_scalar('Accuracy/train', train_epoch_result.accuracy.item(), epoch)
            writer.add_scalar('Accuracy/validation', val_epoch_result.accuracy.item(),  epoch)
            writer.add_scalar('Dice_Score/train',train_epoch_result.dice_score.item(), epoch)
            writer.add_scalar('Dice_Score/validation', val_epoch_result.dice_score.item(), epoch)


            # early stopping 
            if best_dice_score is None or val_epoch_result.dice_score > best_dice_score:
                # ====== YOUR CODE: ======
                best_dice_score = val_epoch_result.dice_score
                epochs_without_improvement = 0
                checkpoint = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                save_checkpoint(checkpoint)
                # print some examples to a folder
                save_predictions_as_imgs(
                    dl_validation, self.model, device=self.device
                )
            else:
                epochs_without_improvement+= 1
                if (early_stopping is not None) and epochs_without_improvement > early_stopping:
                    break

            writer.flush()
        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)



