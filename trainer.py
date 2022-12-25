
from operator import mod
import torch
import torch.utils.tensorboard as tb
import os
import sys
from tqdm import tqdm
# from utils import save_checkpoint, load_checkpoint, save_validations
import utils
from train_results_classes import BatchResult, EpochResult, FitResult

from torchmetrics.classification import BinaryAccuracy
from torchmetrics import Dice


class Trainer:
    def __init__(
        self,
        model, 
        model_name, #give a name for the model, helps for automatiaclly naming the output folders of the run (e.g tensorboard logs, visualizations ..)
        optimizer, 
        loss_fn, 
        sliding_window_validation = True, 
        accuracy_metric = BinaryAccuracy(), 
        dice_metric = Dice(), 
        device = None, 
        load_model = None
    ):
        self.model = model
        self.model_name = model_name
        

        self.optimizer = optimizer
        if load_model is not None:
            utils.load_checkpoint(self.model, self.optimizer, load_model)
        self.loss_fn = loss_fn
        self.dice_metric = dice_metric
        self.accuracy_metric = accuracy_metric
        self.device = device
        if sliding_window_validation == True:
            self.validation_method = model.sliding_window_validation
        else:
            self.validation_method = model
        
        self.train_cur_batch = 0 
        self.val_cur_batch = 0 
        
        tesnorboard_logs_pth = os.path.join('runs/', model_name)
        self.writer = tb.SummaryWriter(tesnorboard_logs_pth)
        
        if self.device:
            model.to(self.device)
            self.dice_metric = dice_metric.to(self.device)
            self.accuracy_metric = accuracy_metric.to(self.device)
        
    def train_batch(self, batch) -> BatchResult:
        batch_imgs, batch_masks = batch
        batch_imgs = batch_imgs.to(self.device)
        batch_masks = batch_masks.unsqueeze(1).to(self.device)  # check if need to unsqueeze

        # forward
        per_pixel_score_predictions = self.model(batch_imgs)  #per pixel un normalized score

        batch_loss = self.loss_fn(per_pixel_score_predictions, batch_masks)
        # print(f'train_batch_loss = {batch_loss}')

        # accuracy and dice
        pred_masks = self.model.predict_labels(per_pixel_score_predictions) # per pixel classification
        dice_score = self.dice_metric(pred_masks, batch_masks.int())
        pixel_accuracy = self.accuracy_metric(pred_masks, batch_masks)

        # backward
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return BatchResult(batch_loss, pixel_accuracy, dice_score)
    
    def test_batch(self, batch) -> BatchResult:
        batch_imgs, batch_masks = batch
        batch_imgs = batch_imgs.to(self.device)
        batch_masks = batch_masks.unsqueeze(1).to(self.device)  # check if need to unsqueeze

            # forward
        with torch.no_grad():

            per_pixel_score_predictions = self.validation_method(batch_imgs)  #per pixel un normalized score
            batch_loss = self.loss_fn(per_pixel_score_predictions, batch_masks)

            # accuracy and dice
            pred_masks = self.model.predict_labels(per_pixel_score_predictions) # per pixel classification
            dice_score = self.dice_metric(pred_masks, batch_masks.int())
            pixel_accuracy = self.accuracy_metric(pred_masks, batch_masks)

        return BatchResult(batch_loss, pixel_accuracy, dice_score)


    def train_epoch(self, dl_train) -> EpochResult:
        epoch_loop = tqdm(dl_train)

        losses = []
        dice_scores = []
        accuracies = []
        for batch in epoch_loop:


            batch_res = self.train_batch(batch)
            losses.append(batch_res.loss)
            accuracies.append(batch_res.pixel_accuracy)
            dice_scores.append(batch_res.dice_score)
            # update tqdm
            epoch_loop.set_postfix(loss=batch_res.loss.item(), dice = batch_res.dice_score.item(), pixel_accuracy = batch_res.pixel_accuracy.item())
            # self.writer.add_scalar('Loss/train_Batch',batch_res.loss , self.train_cur_batch)
            # self.writer.add_scalar('Accuracy/train_Batch', batch_res.pixel_accuracy, self.train_cur_batch)
            # self.writer.add_scalar('Dice_Score/train_Batch',batch_res.dice_score, self.train_cur_batch)

            self.train_cur_batch += 1

        mean_loss = torch.mean(torch.FloatTensor(losses))
        mean_acc = torch.mean(torch.FloatTensor(accuracies))
        mean_dice = torch.mean(torch.FloatTensor(dice_scores))
        return EpochResult(mean_loss, mean_acc, mean_dice)


    
    def validation_epoch(self, validation_dl):
        losses = []
        dice_scores = []
        accuracies = []
        for batch in validation_dl:  # each iteration is one epoch

            batch_res = self.test_batch(batch)
            losses.append(batch_res.loss)
            accuracies.append(batch_res.pixel_accuracy)
            dice_scores.append(batch_res.dice_score)


            # self.writer.add_scalar('Loss/Validation_Batch',batch_res.loss , self.val_cur_batch)
            # self.writer.add_scalar('Accuracy/Validation_Batch', batch_res.pixel_accuracy, self.val_cur_batch)
            # self.writer.add_scalar('Dice_Score/Validation_Batch',batch_res.dice_score, self.val_cur_batch)

            self.val_cur_batch += 1

        mean_loss = torch.mean(torch.FloatTensor(losses))
        mean_acc = torch.mean(torch.FloatTensor(accuracies))
        mean_dice = torch.mean(torch.FloatTensor(dice_scores))
        return EpochResult(mean_loss, mean_acc, mean_dice)

    def fit(
        self,
        dl_train,
        dl_validation,
        num_epochs: int,
        early_stopping = None,
        save_checkpoint:bool = True, 
        print_every: int = 1,
        **kw,
    )  -> FitResult:
        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        best_dice_score = None

        for epoch in range(num_epochs):
            actual_num_epochs += 1
            print(f'------------ epoch #{epoch} ------------ ')

            train_epoch_result = self.train_epoch(dl_train)
            train_acc.append(train_epoch_result.pixel_accuracy)            
            train_loss.append(train_epoch_result.loss.item())
            
            if(dl_validation): #if we want to validate during fitting
                val_epoch_result = self.validation_epoch(dl_validation)
                test_acc.append(val_epoch_result.pixel_accuracy)
                test_loss.append(val_epoch_result.loss.item())

                self.writer.add_scalars('Loss',  {'train':train_epoch_result.loss.item(),'validation': val_epoch_result.loss.item()}, epoch)
                self.writer.add_scalars('Accuracy', {'train': train_epoch_result.pixel_accuracy.item(), 'validation':val_epoch_result.pixel_accuracy.item() },  epoch)
                self.writer.add_scalars('Dice_Score', {'train':train_epoch_result.dice_score.item(), 'validation': val_epoch_result.dice_score.item() },  epoch)
            
            
            else:
                self.writer.add_scalars('Loss',  {'train':train_epoch_result.loss.item()}, epoch)
                self.writer.add_scalars('Accuracy', {'train': train_epoch_result.pixel_accuracy.item()},  epoch)
                self.writer.add_scalars('Dice_Score', {'train':train_epoch_result.dice_score.item()},  epoch)
                
            self.writer.flush()
            if early_stopping is None:
                continue

            # early stopping 
            if best_dice_score is None or val_epoch_result.dice_score > best_dice_score:
                # ====== YOUR CODE: ======
                best_dice_score = val_epoch_result.dice_score
                epochs_without_improvement = 0
                
                if save_checkpoint != None:
                    model_checkpoint = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                    checkpoint_name = self.model_name
                    utils.save_checkpoint(model_checkpoint, checkpoint_name)
                # print some examples to a folder
                utils.save_validations(
                    dl_validation, self.model, inference_method=self.validation_method, device=self.device, 
                )
            else:
                epochs_without_improvement+= 1
                if (early_stopping is not None) and epochs_without_improvement > early_stopping:
                    break

        self.writer.close()
        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)




