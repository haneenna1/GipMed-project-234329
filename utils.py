import os
import random
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from data import ThumbnailsDataset
import sklearn.model_selection
import PIL
from PIL.Image import Image
import numpy as np
from matplotlib import pyplot as plt

BATCH_SIZE = 10
MANUAL_SEED = 42


# def REAL_PATH(path):
#     #changing to hardcoded path
#     return os.path.join("/home/amir.bishara/workspace/project/final_repo/GipMed-project-234329", path)

def save_012_mask_as_img(mask, path):
    mask = mask.squeeze(0)
    colors = np.array([[0, 0, 0], [255, 255, 255], [0, 255, 255]])
    mask_rgb = colors[mask.cpu().int()]
    cv2.imwrite(path, mask_rgb)

def save_checkpoint(state, checkpoint="my_checkpoint"):# you give inly the name of the checkpoint. this function cares for the exact path + extension  
    if not(os.path.isdir('model_checkpoints/')):
        os.mkdir('models_checkpoint/')
    check_point_path = os.path.join("model_checkpoints/", checkpoint) +".pth.tar"
    print("=> Saving checkpoint at ",check_point_path )
    torch.save(state, check_point_path)


def load_checkpoint(model, optimizer = None, checkpoint_name="my_checkpoint"): # you give inly the name of the checkpoint. this function cares for the exact path + extension  
    check_point_path = os.path.join("model_checkpoints/", checkpoint_name) +".pth.tar"
    print("=> Loading checkpoint from ", check_point_path)
    checkpoint = torch.load(check_point_path)
    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

def save_validations(loader, model, inference_method, num_batches_for_save = 10, out_folder="saved_predictions", device="cuda"):
    clear_folder(out_folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx == num_batches_for_save:
            break
        x = x.to(device=device)
        with torch.no_grad():
            per_pixel_score_predictions = inference_method(x) # TODO: check it??
            pred_masks = model.predict_labels_from_scores(per_pixel_score_predictions).float()

        img_path = f"{REAL_PATH(out_folder)}/img_{idx}.jpg"
        predicted_mask_path = f"{REAL_PATH(out_folder)}/img_{idx}_predicted_mask.jpg"
        ground_mask_path =  f"{REAL_PATH(out_folder)}/img_{idx}_ground_mask.jpg"

        torchvision.utils.save_image(x, img_path)
        save_012_mask_as_img(pred_masks.squeeze(1), predicted_mask_path)
        save_012_mask_as_img(y, ground_mask_path)
        # visulaize_sharp(out_folder, img_path, ground_mask_path, predicted_mask_path, idx)
        

    model.train()


def save_data_set(loader, folder_name="train_set"):
    if not(os.path.isdir(f'./{folder_name}')):
        os.mkdir(f'./{folder_name}')
    clear_folder(folder_name)
    for idx, (x, y) in enumerate(loader):
        torchvision.utils.save_image(x, f"{REAL_PATH(folder_name)}/img_sample_{idx}.jpg")
        save_012_mask_as_img(y, f"{REAL_PATH(folder_name)}/img_sample_mask{idx}.jpg")
    

def get_data_loaders(img_dirs:list, mask_dirs:list, train_transforms = None, val_transforms = None, data_size = None,  validation_ratio = 0.15, train_batch_size = 3, validation_batch_size = 1, num_workers = 2,
                 pin_memory = False, shuffle = True, save_datasets = False):
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
   
    total_images = 0
    for img_dir, mask_dir in zip(img_dirs,mask_dirs):
        images_set = set([ image.replace("_thumb.jpg","") for image in os.listdir(img_dir) if image.endswith(".jpg")])
        masks_set =  set([ mask.replace("_SegMap.png","") for mask in os.listdir(mask_dir) if mask.endswith(".png")])
        total_images += len(images_set.intersection(masks_set))
        
    full_dir_indices = range(total_images)

    #train on all data
    if data_size is None:
        chosen_dir_indices = full_dir_indices
    # We want a subset of the dataset for the train/validation.
    else:
        random.seed(MANUAL_SEED)
        chosen_dir_indices = random.choices(full_dir_indices, k = data_size)

    
    
    train_indices = chosen_dir_indices
    
    val_indices, validationLoader = [], None
    if validation_ratio > 0 :
        train_indices, val_indices = sklearn.model_selection.train_test_split(chosen_dir_indices, test_size = validation_ratio, random_state = MANUAL_SEED )
        validation_set =  ThumbnailsDataset(img_dirs, mask_dirs, val_indices, transform=val_transforms)
        validationLoader = DataLoader(dataset=validation_set, batch_size=validation_batch_size , num_workers=num_workers,
                                pin_memory=pin_memory, shuffle = shuffle)
    
    train_set = ThumbnailsDataset(img_dirs, mask_dirs, train_indices, transform= train_transforms)
    trainLoader = DataLoader(dataset=train_set, batch_size=train_batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle = shuffle)
    
    print(f"Creating dataloaders for the datasets: {img_dirs}. Train size: {len(train_indices)} images, Validation size: {len(val_indices)} images")
    
    if save_datasets: 
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
                     "HEROHE", "Ipatimup", "Sheba", "TCGA", "TMA", "Markings"]
    # assert len(set(datasets).intersection(set(datasetsNames))) == min(len(datasetsNames),len(datasets))
    image_dirs = []
    mask_dirs = []
    
    # faster than using the mnt directory. workso only if on gipdeep10 
    basePath = "/data/Breast/"

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
                image_dirs.append(os.path.join(basePath, dataset, "Benign",
                                               "Batch_" + str(counter), "BENIGN" + str(counter),
                                               "SegData", "Thumbs"))
                mask_dirs.append(os.path.join(basePath, dataset, "Benign",
                                              "Batch_" + str(counter), "BENIGN" + str(counter),
                                              "SegData", "SegMaps"))
        if dataset.startswith("Carmel") and dataset!='Carmel':
            batch_num = int(dataset[6:])
            if batch_num <= 8:
                folder = "1-8"
            else:
                folder = "9-11"
                
            image_dirs.append(os.path.join(basePath, 'Carmel', folder,
                                                "Batch_" + str(batch_num), "CARMEL" + str(batch_num),
                                                "SegData", "Thumbs"))
            mask_dirs.append(os.path.join(basePath, 'Carmel', folder,
                                            "Batch_" + str(batch_num), "CARMEL" + str(batch_num),
                                            "SegData", "SegMaps"))
            if batch_num in range(1, 5):
                image_dirs.append(os.path.join(basePath, 'Carmel', "Benign",
                                                "Batch_" + str(batch_num), "BENIGN" + str(batch_num),
                                                "SegData", "Thumbs"))
                mask_dirs.append(os.path.join(basePath, 'Carmel', "Benign",
                                                "Batch_" + str(batch_num), "BENIGN" + str(batch_num),
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
        if dataset == "Markings": # the dataset that contains only markings(annotations) with balck masks
            image_dirs.append("/home/haneenna/GipMed-project-234329/Markings/dup_markings")
            mask_dirs.append("/home/haneenna/GipMed-project-234329/Markings/dup_markings_segmaps")
            
    return image_dirs, mask_dirs




########### These helped for building the markings dataset, don't use 

def create_empty_masks_for_markings(): 
    markings_dir = 'dup_markings'
    masks_dir =  'dup_markings_segmaps'
    
    imgs = [ f for f in os.listdir(markings_dir) if f.endswith(".jpg") ]
    
    for img_pth in imgs: 
        img_fl_pth = os.path.join('/home/haneenna/GipMed-project-234329/Markings', markings_dir, img_pth )
        segmap_fl_pth = os.path.join('/home/haneenna/GipMed-project-234329/Markings', masks_dir , img_pth ).replace("_thumb.jpg", "_SegMap.png")
        
        print(img_fl_pth)
        print(segmap_fl_pth)
        
        image = np.array(PIL.Image.open(img_fl_pth).convert("1"))
        np_mask = np.zeros_like(image)
        
        mask_img = PIL.Image.fromarray(np_mask)
        mask_img.save(segmap_fl_pth)

def duplicate(num_duplicate = 30): 
    orig_markings_dir = '/home/haneenna/GipMed-project-234329/Markings/original_markings'
    imgs = [ f for f in os.listdir(orig_markings_dir) if f.endswith(".jpg") ]
    orig_markings_seg_dir = '/home/haneenna/GipMed-project-234329/Markings/original_segmaps'
    segs = [ f for f in os.listdir(orig_markings_seg_dir) if f.endswith(".png") ]
    
    
    dup_markings_dir = '/home/haneenna/GipMed-project-234329/Markings/dup_markings'
    dup_segmaps_dir = '/home/haneenna/GipMed-project-234329/Markings/dup_markings_segmaps'
    
    clear_folder(dup_markings_dir)
    clear_folder(dup_segmaps_dir)
    
    for img_pth, seg_pth in zip(imgs, segs): 
        for i in range(1, num_duplicate+1):
            img_fl_pth = os.path.join(orig_markings_dir, img_pth )
            seg_fl_pth = os.path.join(orig_markings_seg_dir, seg_pth )

            dup_img_fl_pth = os.path.join(dup_markings_dir, img_pth ).replace("_thumb.jpg", f"_{i}_thumb.jpg")
            dup_seg_fl_pth = os.path.join(dup_segmaps_dir, seg_pth ).replace("_SegMap.png", f"_{i}_SegMap.png")
          
            image = PIL.Image.open(img_fl_pth)
            seg = PIL.Image.open(seg_fl_pth)
            
            image.save(dup_img_fl_pth)
            seg.save(dup_seg_fl_pth)
        
        
# def rename():
#     markings_dir = '/home/haneenna/GipMed-project-234329/Markings/original_markings'
#     imgs = [ f for f in os.listdir(markings_dir) if f.endswith(".jpg") ]
#     markings_seg_dir = '/home/haneenna/GipMed-project-234329/Markings/original_segmaps/'
#     segs = [ f for f in os.listdir(markings_seg_dir) if f.endswith(".png") ]
    
#     for img_pth, seg_pth in zip(imgs, segs): 
#         img_fl_pth = os.path.join(markings_dir, img_pth )
#         seg_fl_pth = os.path.join(markings_seg_dir, seg_pth )
       
#         image = PIL.Image.open(img_fl_pth)
#         seg = PIL.Image.open(seg_fl_pth)
        
#         image.save(img_fl_pth.replace("_0_thumb.jpg", f"_thumb.jpg"))
#         seg.save(seg_fl_pth.replace("_0_SegMap.png", f"_SegMap.png"))


# Building histogram of the (hue, saturation, and value) to do more analysis on the photos
def mean_color_hist(images):
    hist = np.zeros((3,256), dtype=np.float32)
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist[i] += cv2.calcHist([img], [i], None, [256], [0,256]).ravel()
    return hist / len(images)

def plot_mean_hist(mean_hist, save_path ):
    plt.figure()
    color = ['blue', 'green', 'red']
    bins = range(256)
    for i in range(3):
                plt.bar(bins, mean_hist[i], color=color[i], label=["Hue", "Saturation", "Value"][i], width=1, alpha=0.5)
    plt.xlim([0,256])
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.title("HSV histogram")
    plt.savefig(save_path)
    plt.show()

def build_HSV_histogram(images_path, save_path):
    images = [cv2.imread(path) for path in images_path]
    mean_hist = mean_color_hist(images)
    plot_mean_hist(mean_hist, save_path)

def calculate_contrast(image):
    mean, std_dev = cv2.meanStdDev(image)
    return std_dev[0][0]

# Building histogram of contrast (different between brightest and darkest pixel) to do more analysis on the photos

def plot_contrast_hist(contrasts, save_path ):
    plt.figure()
    plt.hist(contrasts, bins=20, label='Contrast')
    plt.xlabel("Contrast Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.title("Contrast Histogram")
    plt.savefig(save_path)
    plt.show()

def build_contrast_histogram(images_path, save_path):
    images = [cv2.imread(path) for path in images_path]
    mean_hist = [calculate_contrast(image) for image in images]
    plot_contrast_hist(mean_hist, save_path)

if __name__ == '__main__':
    build_HSV_histogram(['./GREEN/'+e for e in os.listdir("./GREEN")], 'my_green_contrast_histogram.png')