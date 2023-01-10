import torch
from torch import nn
import torch.optim as optim

import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
import CustomTransforms as C
from trainer import Trainer

from Unet import Unet
from FusionNet import FusionNet
# For reading huge images
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Setting the device for the Training



LEARNING_RATE = 1e-4
MANUAL_SEED = 42

MODELS = {
    'unet': Unet, 
    'fusionnet': FusionNet   
}

def experiment(
    model_name, 
    model_type, 
    datasets, 
    data_size = None, 
    num_epochs = 50, 
    early_stopping = 5,
    batch_size = 5,
    input_size = 1024, 
    num_workers = 10,
    pin_memory = True,
    load_model = False,
    
):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'******************** device you are using is : {device}')
    model = MODELS[model_type](in_channels=3, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    
    train_transform = [ 
        # C.CropTissueRoi(), # we could use this for the train data too! espicially useful for training on sparse images like HEROHE
        A.Compose([
            A.PadIfNeeded(input_size, input_size),
            A.CropNonEmptyMaskIfExists(height=input_size,width=input_size),
            A.ColorJitter(),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
        ]),
        C.AddAnnotation(),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
    # should passed when using the centerCrop without sliding window
    val_transform = A.Compose(
        [
            A.PadIfNeeded(1024, 1024),
            A.CropNonEmptyMaskIfExists(height=input_size,width=input_size),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform_for_sliding_window = [
        A.PadIfNeeded(input_size, input_size),
        A.Compose([
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
        , 
        C.CropTissueRoi(),
    ]
    

    image_dirs, mask_dirs = utils.get_datasets_paths(datasets)
    train_dl, val_dl = utils.get_data_loaders(image_dirs, mask_dirs, train_transform, val_transform_for_sliding_window,  data_size=data_size, 
                                              train_batch_size=batch_size, num_workers= num_workers, pin_memory= pin_memory)

    trainer = Trainer(model, model_name,optimizer,loss_fn , sliding_window_validation=True, device = device, load_model=load_model)
    trainer.fit(train_dl, val_dl,  num_epochs=num_epochs, early_stopping=early_stopping)
  
  
  
  

# call the experiment like this : 
# sbatch train_exp.sh experiment.py --model-name dmdm --model-type unet --datasets TCGA --data-size 10 --pin-memory
  
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--model-type', type=str)
    parser.add_argument('--datasets', nargs = '+')
    parser.add_argument('--data-size', type=int)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--early-stopping', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--pin-memory', action='store_true', default=True)
    parser.add_argument('--load-model', type=str, default=None)
    
    args = parser.parse_args()
    print(args)
    experiment(**vars(args))


