import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.segmentation import FCN_ResNet101_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'******************** device you are using is : {device}')

# Load the model
model = torchvision.models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT).to(device=device)

# Set the model to evaluation mode
model.eval()

def segment_images(image_directory, seg_directory):

    for file in os.listdir(image_directory):
        # Load the image and convert it to grayscale
        image = cv2.imread(os.path.join(image_directory, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding method to the grayscale image
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(mask)
        
    
        # Save the segmented image
        cv2.imwrite(os.path.join(seg_directory, file).replace('_thumb.jpg', '_SegMap.png'), np.invert(mask))
if __name__ == '__main__':
    segment_images("Markings/original_markings/", "Markings/original_segmaps/")
