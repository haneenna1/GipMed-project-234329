import os

import numpy as np
import torch
import torchvision
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset import ThumbnailsDataset
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
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


# def gettingDataFolders() -> tuple[list, list]:
#     image_dirs = []
#     mask_dirs = []
#     basePath = "/mnt/gipmed_new/Data/Breast/"
#     datasetsNames = ["ABCTB_TIF", "Carmel", "Covilha", "Haemek",
#                      "HEROHE", "Ipatimup", "Sheba", "TCGA", "TMA"]
#     for dataset in datasetsNames:
#         # should check the big sizes of TCGA, HEROHE, IPATIMUP, COVILHA
#         if dataset in ["ABCTB_TIF", "Covilha", "HEROHE", "Ipatimup", "TCGA"]:
#             image_dirs.append(os.path.join(basePath, dataset, "SegData", "Thumbs"))
#             mask_dirs.append(os.path.join(basePath, dataset, "SegData", "SegMaps"))
#         if dataset == "Carmel":
#             for counter in range(1, 12):
#                 if counter <= 8:
#                     folder = "1-8"
#                 else:
#                     folder = "9-11"

#                 image_dirs.append(os.path.join(basePath, dataset, folder,
#                                                "Batch_" + str(counter), "CARMEL" + str(counter),
#                                                "SegData", "Thumbs"))
#                 mask_dirs.append(os.path.join(basePath, dataset, folder,
#                                               "Batch_" + str(counter), "CARMEL" + str(counter),
#                                               "SegData", "SegMaps"))
#             for counter in range(1, 5):
#                 image_dirs.append(os.path.join(basePath, dataset, "BENIGN",
#                                                "Batch_" + str(counter), "BENIGN" + str(counter),
#                                                "SegData", "Thumbs"))
#                 mask_dirs.append(os.path.join(basePath, dataset, "BENIGN",
#                                               "Batch_" + str(counter), "BENIGN" + str(counter),
#                                               "SegData", "SegMaps"))
#         if dataset == "Sheba":
#             for counter in range(2, 7):
#                 image_dirs.append(os.path.join(basePath, dataset,
#                                                "Batch_" + str(counter), "SHEBA" + str(counter),
#                                                "SegData", "Thumbs"))
#                 mask_dirs.append(os.path.join(basePath, dataset,
#                                               "Batch_" + str(counter), "SHEBA" + str(counter),
#                                               "SegData", "SegMaps"))
#         if dataset == "Haemek":
#             for counter in range(1, 2):
#                 image_dirs.append(os.path.join(basePath, dataset,
#                                                "Batch_" + str(counter), "HAEMK" + str(counter),
#                                                "SegData", "Thumbs"))
#                 mask_dirs.append(os.path.join(basePath, dataset,
#                                               "Batch_" + str(counter), "HAEMK" + str(counter),
#                                               "SegData", "SegMaps"))

#         if dataset == "TMA":  # other data in this dataset don't have masks
#             image_dirs.append(os.path.join(basePath, dataset, "bliss_data/01-011/HE/TMA_HE_01-011",
#                                            "SegData", "Thumbs"))
#             mask_dirs.append(os.path.join(basePath, dataset, "bliss_data/01-011/HE/TMA_HE_01-011",
#                                           "SegData", "SegMaps"))
#         return image_dirs, mask_dirs


def get_data_loaders(img_dirs, mask_dirs, batch_size = 3, train_transforms = None, val_transforms = None, num_workers = 2,
                pin_memory = False):
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
    MANUAL_SEED = 42

    train_set_list = []
    validation_set_list = []

    for img_dir, mask_dir in zip(img_dirs,mask_dirs):
        dataset = ThumbnailsDataset(img_dir, mask_dir)
        print(len(dataset))
        train_set, validation_set = random_split(dataset, [1-VALIDATION_RATIO, VALIDATION_RATIO], generator=torch.Generator().manual_seed(MANUAL_SEED))
        train_set.transform = train_transforms
        validation_set.transform = val_transforms

        train_set_list.append(train_set)
        validation_set_list.append(validation_set)

    concatened_train_set = ConcatDataset(train_set_list)
    concatenedValidationDataset = ConcatDataset(validation_set_list)

    trainLoader = DataLoader(dataset=concatened_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory)
    validationLoader = DataLoader(dataset=concatenedValidationDataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return trainLoader, validationLoader


def test():
    train_loader, val_loader = get_data_loaders(['/mnt/gipmed_new/Data/Breast/TCGA/thumbs/'], ['/mnt/gipmed_new/Data/Breast/TCGA/SegData/Thumbs/'],)
    print("train_size = ", len(train_loader))
    print("validation_size = ", len(val_loader))
    
     # Display image and label.
    train_imgs, train_masks= next(iter(train_loader))
    print(f"Feature batch shape: {train_imgs.size()}")
    print(f"Labels batch shape: {train_masks.size()}")
    img = train_imgs[0].squeeze()
    label = train_masks[0]
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")
    
if __name__ == '__main__':
    test()