import io
import os
import glob
from PIL import Image
import torch.utils.data as data


class DataReader(data.Dataset):
    
    def __init__(self, root, transform):
        self.images_path = glob.glob(root + "/*/*.png")
        self.transform = transform
    
    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images_path)