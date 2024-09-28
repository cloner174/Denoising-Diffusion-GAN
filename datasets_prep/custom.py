from torch.utils.data import Dataset
from PIL import Image
import glob
import os

class DatasetCustom(Dataset):
    
    def __init__(self, data_dir, class_ ='train', transform = None):
        
        self.class_ = class_
        self.transform = transform
        self.data_dir = data_dir
        self.images_all = None
        
        self.__prepeare__()
        
    
    def __prepeare__(self):
        
        data_path = os.path.join(self.data_dir, self.class_)
        
        if not os.path.isdir(data_path):
            raise FileNotFoundError("The class_ param, should be one of [train, val, test]!")
        
        self.images_all =  glob.glob(data_path + "/*/*.jpg")
        
    
    def __getitem__(self, index):
        
        image_path = self.images_all[index]
        image = Image.open(image_path)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        return image, 0
    
    def __len__(self):
        return len(self.images_all)
    



class PositivePatchDataset(Dataset):
    
    def __init__(self, patch_dir):
        self.patch_paths = [
            os.path.join(patch_dir, f) for f in os.listdir(patch_dir)
            if f.endswith('.npy') and 'label_1' in f
        ]

    
    def __len__(self):
        return len(self.patch_paths)

    
    def __getitem__(self, idx):
        patch = np.load(self.patch_paths[idx])
        # Normalize to [-1, 1] for Tanh activation
        patch = (patch - 0.5) * 2.0
        patch = patch.astype(np.float32)
        patch = np.expand_dims(patch, axis=0)
        return torch.tensor(patch)


#cloner174
