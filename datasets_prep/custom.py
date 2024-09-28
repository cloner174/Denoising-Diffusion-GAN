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
        
        return image, 'Dumm'
    
    def __len__(self):
        return len(self.images_all)


class PositivePatchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.slice_info = []  # List of tuples: (npy_file_path, slice_index)
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Please check your data_dir path and try again. The current path is: {self.data_dir}")
        
        # Get all positive .npy files (label_1)
        npy_files = glob.glob(os.path.join(self.data_dir, '*/*label_1.npy'))
        
        if not npy_files:
            raise FileNotFoundError("No positive patches found in the specified directory.")
        
        # Build the slice_info list
        for npy_file_path in npy_files:
            patch = np.load(npy_file_path)
            num_slices = patch.shape[0]
            for slice_index in range(num_slices):
                self.slice_info.append((npy_file_path, slice_index))
    
    def __len__(self):
        return len(self.slice_info)
    
    def __getitem__(self, index):
        npy_file_path, slice_index = self.slice_info[index]
        patch = np.load(npy_file_path)
        
        # Get the specified slice
        if slice_index < 0 or slice_index >= patch.shape[0]:
            raise IndexError(f"Slice index {slice_index} out of bounds for patch with shape {patch.shape}")
        
        image_2d = patch[slice_index, :, :]
        
        # Normalize the image
        image_2d = image_2d - np.min(image_2d)
        max_value = np.max(image_2d)
        if max_value != 0:
            image_2d = image_2d / max_value
        else:
            image_2d = np.zeros_like(image_2d)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_2d).float().unsqueeze(0)  # Add channel dimension
        
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, 'Dummy'  # You can replace 'Dummy' with an actual label if needed


#cloner174
