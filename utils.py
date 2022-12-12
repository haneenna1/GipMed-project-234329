import os
import random
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from data import ThumbnailsDataset
import sklearn.model_selection
from PIL.Image import Image
from cmath import inf
BATCH_SIZE = 10


def REAL_PATH(path):
    #changing to hardcoded path
    return os.path.join("/home/amir.bishara/workspace/project/final_repo/GipMed-project-234329", path)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])

def save_layered_predictions(img_path, maks_path, index,  mode = 'ground',folder = 'layered_preds'):
    img = cv2.imread(img_path)
    mask = cv2.imread(maks_path)
    layered_img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    dst_path = f'{REAL_PATH(folder)}/img_{index}_{mode}_layered.jpg'
    cv2.imwrite(dst_path, layered_img)


def save_predictions_as_imgs(loader, model, inference_method, num_batches_for_save = 10, pred_folder="saved_predictions", layers_folder = 'layered_preds' ,device="cuda"):
    clear_folder(pred_folder)
    clear_folder(layers_folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx == num_batches_for_save:
            break
        x = x.to(device=device)
        with torch.no_grad():
            per_pixel_score_predictions = inference_method(x) # TODO: check it??
            pred_masks = model.predict_labels(per_pixel_score_predictions).float()

        img_path = f"{REAL_PATH(pred_folder)}/img_{idx}.jpg"
        predicted_mask_path = f"{REAL_PATH(pred_folder)}/img_{idx}_predicted_mask.jpg"
        ground_mask_path =  f"{REAL_PATH(pred_folder)}/img_{idx}_ground_mask.jpg"

        torchvision.utils.save_image(x, img_path)
        torchvision.utils.save_image(pred_masks, predicted_mask_path)
        torchvision.utils.save_image(y.unsqueeze(1), ground_mask_path)

        save_layered_predictions(img_path, predicted_mask_path, idx,  mode = 'pred', folder = layers_folder)
        save_layered_predictions(img_path, ground_mask_path, idx, folder = layers_folder)

    model.train()


def save_data_set(loader, folder_name="train_set"):
    if not(os.path.isdir(f'./{folder_name}')):
        os.mkdir(f'./{folder_name}')
    clear_folder(folder_name)
    for idx, (x, y) in enumerate(loader):
        torchvision.utils.save_image(x, f"{REAL_PATH(folder_name)}/img_sample_{idx}.jpg")
        torchvision.utils.save_image(y.unsqueeze(1), f"{REAL_PATH(folder_name)}/img_sample_mask{idx}.png")


def get_data_loaders(img_dirs:list, mask_dirs:list, train_transforms = None, val_transforms = None, num_imgs = 100, batch_size = 3, num_workers = 2,
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
    VALIDATION_RATIO = 0.15
    MANUAL_SEED = 42

    total_images = 0
    for img_dir, mask_dir in zip(img_dirs,mask_dirs):
        images_set = set([ image.replace("_thumb.jpg","") for image in os.listdir(img_dir) if image.endswith(".jpg")])
        masks_set =  set([ mask.replace("_SegMap.png","") for mask in os.listdir(mask_dir) if mask.endswith(".png")])
        total_images += len(images_set.intersection(masks_set))
    full_dir_indices = range(total_images)

    # We want a subset of the data based on the num_images param
    if num_imgs == inf:
        chosen_dir_indices = random.choices(full_dir_indices, k= total_images)
    else:
        chosen_dir_indices = random.choices(full_dir_indices, k= num_imgs)

    train_indices, val_indices = sklearn.model_selection.train_test_split(chosen_dir_indices, test_size = VALIDATION_RATIO, random_state = MANUAL_SEED )

    train_set = ThumbnailsDataset(img_dirs, mask_dirs, train_indices, transform= train_transforms)
    validation_set =  ThumbnailsDataset(img_dirs, mask_dirs, val_indices, transform=val_transforms)
    
    print(f'length of train  :{len(train_set)} lengt of validation: {len(validation_set)}')
    trainLoader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)
    validationLoader = DataLoader(dataset=validation_set, batch_size=1 , shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)
    
    # save_data_set(trainLoader, 'train_set')
    # save_data_set(validationLoader, 'validation_set')
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


def clear_folder(dir):
    if os.path.exists(REAL_PATH(dir)):
        files = os.listdir(REAL_PATH(dir))
        for f in files:
            f_path = os.path.join(dir, f)
            os.remove(f_path)
    else:
        os.mkdir(REAL_PATH(dir))

def get_datasets_paths(datasets:list):
    datasetsNames = ["ABCTB_TIF", "Carmel", "Covilha", "Haemek",
                     "HEROHE", "Ipatimup", "Sheba", "TCGA", "TMA"]
    assert len(set(datasets).intersection(set(datasetsNames))) == min(len(datasetsNames),len(datasets))
    image_dirs = []
    mask_dirs = []
    basePath = "/mnt/gipmed_new/Data/Breast/"

    for dataset in datasets:

        if dataset in ["ABCTB_TIF", "Covilha", "HEROHE", "Ipatimup", "TCGA"]:
            image_dirs.append(os.path.join(basePath, dataset, "SegData", "Thumbs"))
            mask_dirs.append(os.path.join(basePath, dataset, "SegData", "SegMaps"))

        if dataset == "Carmel":
            for counter in range(1, 12):
                if counter <= 8:
                    folder = "1-8"
                else:
                    folder = "9-11"

                image_dirs.append(os.path.join(basePath, dataset, folder,
                                               "Batch_" + str(counter), "CARMEL" + str(counter),
                                               "SegData", "Thumbs"))
                mask_dirs.append(os.path.join(basePath, dataset, folder,
                                              "Batch_" + str(counter), "CARMEL" + str(counter),
                                              "SegData", "SegMaps"))
            for counter in range(1, 5):
                image_dirs.append(os.path.join(basePath, dataset, "BENIGN",
                                               "Batch_" + str(counter), "BENIGN" + str(counter),
                                               "SegData", "Thumbs"))
                mask_dirs.append(os.path.join(basePath, dataset, "BENIGN",
                                              "Batch_" + str(counter), "BENIGN" + str(counter),
                                              "SegData", "SegMaps"))
        if dataset == "Sheba":
            for counter in range(2, 7):
                image_dirs.append(os.path.join(basePath, dataset,
                                               "Batch_" + str(counter), "SHEBA" + str(counter),
                                               "SegData", "Thumbs"))
                mask_dirs.append(os.path.join(basePath, dataset,
                                              "Batch_" + str(counter), "SHEBA" + str(counter),
                                              "SegData", "SegMaps"))
        if dataset == "Haemek":
            for counter in range(1, 2):
                image_dirs.append(os.path.join(basePath, dataset,
                                               "Batch_" + str(counter), "HAEMEK" + str(counter),
                                               "SegData", "Thumbs"))
                mask_dirs.append(os.path.join(basePath, dataset,
                                              "Batch_" + str(counter), "HAEMEK" + str(counter),
                                              "SegData", "SegMaps"))

        if dataset == "TMA":  # other data in this dataset don't have masks
            image_dirs.append(os.path.join(basePath, dataset, "bliss_data/01-011/HE/TMA_HE_01-011",
                                           "SegData", "Thumbs"))
            mask_dirs.append(os.path.join(basePath, dataset, "bliss_data/01-011/HE/TMA_HE_01-011",
                                          "SegData", "SegMaps"))
    return image_dirs, mask_dirs