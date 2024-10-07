import os
import numpy as np
from PIL import Image
import argparse


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