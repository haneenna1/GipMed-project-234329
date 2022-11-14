import os

import numpy as np
import torch
import torchvision
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from dataset import ThumbnailsDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Loading checkpoint")
    check_point = torch.load(filename)
    model.load_state_dict(checkpoint["model"])


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()  # set the model to be in eval mode not train mode, for parts that behave differently in train/val

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()  # set the model back to the training mode


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def get_loaders(img_dir, mask_dir, batch_size, train_transforms, val_transforms, num_workers, pin_memory):
    """
    returns train and validation data loaders
    Args:
    imageDir: path to the directory of the images
    maskDir: path to the directory of the masks
    batch_size: batch size of the data loaders
    train_transforms: a sequence of transformations to apply on the training set
    val_transforms: a sequence of transformations to apply on the validation set
    num_workers: num workers for the data loading
    pin_memory
    """
    VALIDATION_RATIO = 0.2
    RANDOM_SEED = 42
    imagesNum = len(os.listdir(img_dir))
    validationNum = int(imagesNum * VALIDATION_RATIO)
    indices = list(range(imagesNum))

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[validationNum:], indices[:validationNum]

    trainDataset = ThumbnailsDataset(img_dir, mask_dir, train_indices, train_transforms)
    validationDataset = ThumbnailsDataset(img_dir, mask_dir, val_indices, val_transforms)

    trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory)
    validationLoader = DataLoader(dataset=validationDataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return trainLoader, validationLoader
