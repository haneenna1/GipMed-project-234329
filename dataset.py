import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ThumbnailsDataset(Dataset):
    def __init__(self, imageDir, maskDir, transform=None):
        self.imageDir = imageDir
        self.maskDir = maskDir
        self.transform = transform

    def __len__(self):
        return len(self.os.listdir(self.imageDir))

    def __getitem__(self, index):
        #TODO: AMIR

