from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np
import os
import nibabel as nib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from additionals.utilities import load_slice_info, save_slice_info




class Luna16Dataset(Dataset):
    
    def __init__(self, data_dir, mask_dir, transform=None, bound_exp_lim = 5, path_to_slices_info = None):
        
        self.transform = transform
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.bound_exp_lim = bound_exp_lim
        
        if path_to_slices_info is not None:
            self.path_to_slice_info = path_to_slices_info
            self.slice_info = load_slice_info(path_to_slices_info)
        else:
            self.slice_info = []  # List of tuples: ( .nii.gz file path, slice index)
            self._prepare_dataset()
            save_slice_info(self.slice_info)
    
    def _prepare_dataset(self):
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Please check your data_dir path and try again. The current path is: {self.data_dir}")
        # Get all .nii.gz files
        nii_files =  [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith(".nii.gz")]
        if not nii_files:
            raise FileNotFoundError("No patches found in the specified directory.")
        # Build the slice_info list
        for nii_file_path in nii_files:
            #patch = nib.load(nii_file_path) همه ی آرایه ها 256 هستند -->> 256*256*256
            nii_file_name = os.path.split(nii_file_path)[-1]
            mask_path = os.path.join( self.mask_dir, nii_file_name)
            mask = nib.load(mask_path).get_fdata()
            handled_indexes = self.__handle_edges__(np.nonzero(mask))
            if handled_indexes is None:
                continue
            else:
                Dx, Dy, Dz = handled_indexes
            
            for i in range(len(Dx)):
                self.slice_info.append((nii_file_path, 'x', Dx[i]))
            
            for i in range(len(Dy)):
                self.slice_info.append((nii_file_path, 'y', Dy[i]))
            
            for i in range(len(Dz)):
                self.slice_info.append((nii_file_path, 'z', Dz[i]))
        
    
    def __handle_edges__(self, indxes):
        data_shape = (256, 256, 256)
        if len(indxes) < 3 or len(data_shape) < 3:
            return None
        
        min_bound_x = min(indxes[0])
        min_bound_y = min(indxes[1])
        min_bound_z = min(indxes[2])
        max_bound_x = max(indxes[0]) + 1 if max(indxes[0]) + 1 < data_shape[0] else max(indxes[0])
        max_bound_y = max(indxes[1]) + 1 if max(indxes[1]) + 1 < data_shape[1] else max(indxes[1])
        max_bound_z = max(indxes[2]) + 1 if max(indxes[2]) + 1 < data_shape[2] else max(indxes[2])
        if min_bound_x > self.bound_exp_lim :
            min_bound_x -= self.bound_exp_lim
        
        if min_bound_y > self.bound_exp_lim :
            min_bound_y -= self.bound_exp_lim
        
        if min_bound_z > self.bound_exp_lim :
            min_bound_z -= self.bound_exp_lim
        
        if max_bound_x + self.bound_exp_lim < data_shape[0] :
            max_bound_x += self.bound_exp_lim
        
        if max_bound_y + self.bound_exp_lim < data_shape[1] :
            max_bound_y += self.bound_exp_lim
        
        if max_bound_z + self.bound_exp_lim < data_shape[2] :
            max_bound_z += self.bound_exp_lim
        
        Dx = range(min_bound_x,max_bound_x)
        Dy = range(min_bound_y,max_bound_y)
        Dz = range(min_bound_z,max_bound_z)
        return Dx, Dy, Dz
    
    def __len__(self):
        return len(self.slice_info)
    
    def __getitem__(self, index):
        nii_file_path, _where_, slice_index = self.slice_info[index]
        patch = nib.load(nii_file_path).get_fdata()
        # Get the specified slice
        if slice_index < 0 or slice_index >= 256:
            raise IndexError(f"Slice index {slice_index} out of bounds for patch with shape {patch.shape}")
        
        if _where_ == 'x':
            image_2d = patch[ slice_index , : , : ]
        
        elif _where_ == 'y':
            image_2d = patch[ : , slice_index, : ]
        
        elif _where_ == 'z':
            image_2d = patch[ : , : , slice_index ]
        
        image_2d = Image.fromarray(image_2d.astype(np.uint8))
        if self.transform is not None:
            image_2d = self.transform(image_2d)
        
        return image_2d, 1  # 'Dummy' label! برای مدل جن نیاز ی به لیبل نیست



class PositivePatchDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, limited_slices = False):
        
        self.transform = transform
        self.data_dir = data_dir
        self.limited_slices = limited_slices
        
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
            #patch = np.load(npy_file_path) همه ی آرایه ها 64 هستند -->> 64*64*64
            num_slices = 64
            num_skip = 8 if self.limited_slices else 1
            for slice_index in range(0, num_slices, num_skip):
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
        max_value = (np.max(image_2d) - np.min(image_2d))
        if max_value != 0:
            image_2d = image_2d / max_value
        else:
            image_2d = np.zeros_like(image_2d)
        
        image_2d = (image_2d * 255).astype(np.uint8)
        image_2d = Image.fromarray(image_2d)
        
        if self.transform is not None:
            image_2d = self.transform(image_2d)
        
        return image_2d, 1  # 'Dummy' label! برای مدل جن نیاز ی به لیبل نیست




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


#cloner174