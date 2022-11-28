import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset
from dataset import ThumbnailsDataset
import matplotlib.pyplot as plt
import sklearn.model_selection
from PIL.Image import Image

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
        if not(os.path.isdir(folder)):
            os.mkdir(folder)
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()



def save_data_set(loader, folder="train_set"):
    for idx, (x, y) in enumerate(loader):
        torchvision.utils.save_image(x, f"{folder}/img_sample_{idx}.jpg")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/img_sample_mask{idx}.png")

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

def get_data_loaders(img_dir, mask_dir, batch_size = 3, num_workers = 2, train_transforms = None, val_transforms = None,
                 num_imgs = 30,pin_memory = False):
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

    # train_set_list = []
    # validation_set_list = []
    # for img_dir, mask_dir in zip(img_dirs,mask_dirs):
    full_dir_indices = range(len(os.listdir(img_dir)))
    # we want a subset of the data
    chosen_dir_indices = random.choices(full_dir_indices, k = num_imgs)

    train_indices, val_indices = sklearn.model_selection.train_test_split(chosen_dir_indices, test_size = VALIDATION_RATIO, random_state = MANUAL_SEED )
    train_set = ThumbnailsDataset(img_dir, mask_dir, train_indices,False, transform= train_transforms)
    validation_set =  ThumbnailsDataset(img_dir, mask_dir, val_indices,True, transform=val_transforms)
    print(f'length of train  :{len(train_set)} lengt of validation: {len(validation_set)}')
        # train_set_list.append(train_set)
        # validation_set_list.append(validation_set)

    # concatened_train_set = ConcatDataset(train_set_list)
    # if (concatened_train_set.transform) :
    #     print('train transform not none after edit')
    # else :
    #     print('no train transform')
    # concatenedValidationDataset = ConcatDataset(validation_set_list)

    # trainLoader = DataLoader(dataset=concatened_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                          pin_memory=pin_memory)
    # validationLoader = DataLoader(dataset=concatenedValidationDataset, batch_size=batch_size, shuffle=True,
    #                               num_workers=num_workers,
    #                               pin_memory=pin_memory)
    
    trainLoader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)
    validationLoader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)
    
    save_data_set(trainLoader, 'train_set')
    save_data_set(validationLoader, 'validation_set')
    return trainLoader, validationLoader

# For extracting the images sizes
def extractingPhotosProperties(img_dirs):
    image_sizes = []
    # img_dirs, _ = gettingDataFolders()
    for img_dir in img_dirs:
        images = os.listdir(img_dir)
        for img in images:
            if not img.endswith(".jpg"):
                continue
            imagePath = os.path.join(img_dir,img)
            image = Image.open(imagePath)
            image_sizes.append(image.size)
    min_width = min([tuple[0] for tuple in image_sizes])
    min_height = min([tuple[1] for tuple in image_sizes])
    max_width = max([tuple[0] for tuple in image_sizes])
    max_height = max([tuple[1] for tuple in image_sizes])
    print("the min (width,height) of an image is : "+str(min_width)+","+str(min_height))
    print("the max (width,height) of an image is : "+str(max_width)+","+str(max_height))

def test():
    img_dir = "/mnt/gipmed_new/Data/Breast/TCGA/SegData/Thumbs"
    mask_dir = "/mnt/gipmed_new/Data/Breast/TCGA/SegData/SegMaps"

    # train_loader, val_loader = get_data_loaders(img_dir, mask_dir)
    # print("train_size = ", len(train_loader))
    # print("validation_size = ", len(val_loader))
    
     # Display image and label.
    # train_imgs, train_masks= next(iter(train_loader))
    # print(f"Feature batch shape: {train_imgs.size()}")
    # print(f"Labels batch shape: {train_masks.size()}")
    # img = train_imgs[0].squeeze()
    # label = train_masks[0]
    # plt.imshow(img)
    # plt.show()
    # print(f"Label: {label}")
    
if __name__ == '__main__':
    test() 