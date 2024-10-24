import os
import numpy as np
from PIL import Image
import argparse
import torchvision.transforms as transforms


def _data_transforms_luna16(image_size = 64 , _3d = False):
    """Get data transforms for luna16."""
    to_do = [transforms.ToTensor()]
    norm = transforms.Normalize( (0.5,), (0.5,) ) if not _3d else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    to_do_train = to_do.extend([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ])
    to_do_train.append(norm)
    to_do_val = to_do.extend([norm])
    train_transform =  transforms.Compose(to_do_train)
    
    valid_transform = transforms.Compose(to_do_val)
    
    return train_transform, valid_transform



def npy_to_image(npy_dir, image_dir, image_format='png', normalize=False):
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    for filename in os.listdir(npy_dir):
        
        image_name = os.path.join(image_dir, os.path.splitext(filename)[0])
        if filename.endswith('.npy'):
            
            npy_path = os.path.join(npy_dir, filename)
            
            # Load .npy
            image_array_main = np.load(npy_path)
            if len(image_array_main.shape) != 3:
                raise ValueError("Use Simple Convert Function Mannualyy!")
            
            for ii in range(0, image_array_main.shape[0], 8):
                
                image_array = image_array_main[ii, :, :]
                if normalize:
                    image_array = image_array - np.min(image_array)
                    max_value = (np.max(image_array) - np.min(image_array))
                    if max_value != 0:
                        image_array = image_array / max_value
                    else:
                        image_array = np.zeros_like(image_array)
                
                image_array = (image_array * 255).astype(np.uint8)
                image_array = Image.fromarray(image_array)
                image_path = image_name + '_' + str(ii) + '.' + image_format
                
                image_array.save(image_path)
            
            print(f"Saved {image_name}")
    
    print("Conversion complete.")



def simple_convert(name, image_array, save_path, image_format='png', normalize=False):
    if normalize:
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
    
    image_array = image_array.astype(np.uint8)
    if image_array.ndim == 2:
        image = Image.fromarray(image_array, mode='L')
    elif image_array.ndim == 3:
        image = Image.fromarray(image_array)
    else:
        raise ValueError(f"Unsupported array shape: {image_array.shape}")
    filename = os.path.split(name)[-1]
    image_name = filename.replace('.npy', f'.{image_format}')
    image_path = os.path.join(save_path, image_name)
    image.save(image_path)
    return True




def nii_to_png_simple(nii_file_path, 
                      _where_, 
                      slice_index, 
                      only_z = True,
                      save_dir = './real_images', 
                      do_resize_to: tuple|None = (128,128)):
    try:
        import nibabel as nib
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please Install NiBabel Lib using: pip install nibabel")
    
    patch = nib.load(nii_file_path).get_fdata()
    
    if slice_index < 0 or slice_index >= 256:
        raise IndexError(f"Slice index {slice_index} out of bounds for patch with shape {patch.shape}")
    
    if only_z:
        if _where_ == 'z':
            image_2d = patch[ : , : , slice_index ]
        else:
            return
    else:
        if _where_ == 'x':
            image_2d = patch[ slice_index , : , : ]
        elif _where_ == 'y':
            image_2d = patch[ : , slice_index, : ]
        elif _where_ == 'z':
            image_2d = patch[ : , : , slice_index ]
    
    image_2d = Image.fromarray(image_2d.astype(np.uint8))
    if do_resize_to is not None:
        image_2d = image_2d.resize(do_resize_to)
    
    temp_name = os.path.split(nii_file_path)[-1].split('.nii.gz')[0]
    temp_name += f'_{_where_}_{slice_index}.png'
    image_2d.save(os.path.join(save_dir, temp_name))
    return



def nii_to_png(slices_info, save_dir = './real_images', only_z = True, lim = None, do_transform_for = 'none'):#can be 'train' or 'val' or 'none'
    os.makedirs(save_dir, exist_ok=True)
    
    if do_transform_for == 'train':
        transform, _ = _data_transforms_luna16()
    elif do_transform_for == 'val':
        _, transform = _data_transforms_luna16()
    else:
        transform = None
    if lim is not None:
            lim = lim if isinstance(lim, int) else 1000
    for any_ in slices_info :
        if lim is not None:
            if len( os.listdir(save_dir) ) > lim:
                return
        nii_file_path, _where_, slice = any_
        nii_to_png_simple(nii_file_path, _where_, slice, only_z, save_dir , transform )
    
    return





def nii_to_npy_simple(nii_file_path, 
                      _where_, 
                      slice_index, 
                      only_z = True,
                      save_dir = './real_images', 
                      transform = None):
    try:
        import nibabel as nib
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please Install NiBabel Lib using: pip install nibabel")
    
    patch = nib.load(nii_file_path).get_fdata()
    
    if slice_index < 0 or slice_index >= 256:
        raise IndexError(f"Slice index {slice_index} out of bounds for patch with shape {patch.shape}")
    
    if only_z:
        if _where_ == 'z':
            image_2d = patch[ : , : ,  slice_index ]
        else:   
            return
    else:
        if _where_ == 'x':
            image_2d = patch[ slice_index , : , : ]
        elif _where_ == 'y':
            image_2d = patch[ : , slice_index, : ]
        elif _where_ == 'z':
            image_2d = patch[ : , : , slice_index ]
    
    image_2d = Image.fromarray(image_2d.astype(np.uint8))
    if transform is not None:
        try:
            image_2d = image_2d.reshape(transform)
        except:
            pass
    
    temp_name = os.path.split(nii_file_path)[-1].split('.nii.gz')[0]
    temp_name += f'_{_where_}_{slice_index}.npy'
    np.save(os.path.join(save_dir, temp_name), image_2d)
    return



def nii_to_npy_3d(slices_info, bounders, _where_, save_dir = './real_numpy_3d'):
    
    try:
        import nibabel as nib
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please Install NiBabel Lib using: pip install nibabel")
    
    os.makedirs(save_dir, exist_ok=True)
    patch_out = []
    patch_name = []
    to_add = 0
    for i in range( len(slices_info) - 1 ):
        i += to_add
        j = i + 1
        if i > len(slices_info) or j > len(slices_info):
            break
        temp_slice = []
        temp_name = slices_info[i][0]
        patch_in = nib.load(slices_info[i][0]).get_fdata()
        while slices_info[i][0] == slices_info[j][0]:
            if len(temp_slice) == 0 and slices_info[i][1] == _where_:
                temp_slice.append(slices_info[i][-1])
            if slices_info[j][1] == _where_ :
                temp_slice.append(slices_info[j][-1])
            to_add = j
            j += 1
        while len(temp_slice) >= bounders + 1 :
            temp_slices_new = temp_slice[:bounders+1]
            if _where_ == 'x':
                image_3d = patch_in[ temp_slices_new[0] : temp_slices_new[-1] , : , : ]
            elif _where_ == 'y':
                image_3d = patch_in[ : , temp_slices_new[0] : temp_slices_new[-1] , : ]
            elif _where_ == 'z':
                image_3d = patch_in[ : , : , temp_slices_new[0] : temp_slices_new[-1] ]
            patch_out.append( image_3d )
            patch_name.append( temp_name )
            for _ in range(bounders):
                temp_slice.pop(0)
    
    for i in range(len(patch_out)):
        name_ = os.path.split(patch_name[i])[-1].split('.nii.gz')[0] 
        name_ += f'_{_where_}_{i}.npy'
        np.save(os.path.join(save_dir, name_), patch_out[i])
    
    return 


def nii_to_npy(slices_info, 
               save_dir = './real_numpy', 
               only_z = True, 
               lim = None, 
               do_transform_for = 'none'):#can be 'train' or 'val' or 'none'
    
    os.makedirs(save_dir, exist_ok=True)
    
    if do_transform_for == 'train':
        transform, _ = _data_transforms_luna16()
    elif do_transform_for == 'val':
        _, transform = _data_transforms_luna16()
    else:
        transform = None
    if lim is not None:
            lim = lim if isinstance(lim, int) else 1000
    
    for any_ in slices_info :
        if lim is not None:
            if len( os.listdir(save_dir) ) > lim:
                return
        nii_file_path, _where_, slice = any_
        nii_to_npy_simple(nii_file_path, _where_, slice, only_z, save_dir , transform )
    
    return







if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--npy_dir', type=str, default='./real_images_dir',
                        help='Path to npy files for converting to png')
    
    parser.add_argument('--save_dir', type=str, default='./real_images',
                        help='Path to where the converted images will be saved!')
    
    parser.add_argument('--image_format', type=str, default='png')
    
    parser.add_argument('--do_normalize' , default=True)
    
    args = parser.parse_args()
    
    npy_to_image(args.npy_dir, args.save_dir, args.image_format, args.do_normalize)

#cloner174