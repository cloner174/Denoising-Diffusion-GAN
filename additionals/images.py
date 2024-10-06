import os
import numpy as np
from PIL import Image
import argparse


def npy_to_image(npy_dir, image_dir, image_format='png', normalize=False):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for filename in os.listdir(npy_dir):
        if filename.endswith('.npy'):
            npy_path = os.path.join(npy_dir, filename)
            # Load the .npy file
            image_array = np.load(npy_path)
            # Handle the shape and data type
            if normalize:
                # If the array is not in [0, 255], adjust accordingly
                image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
            image_array = image_array.astype(np.uint8)
            # Handle grayscale images
            if image_array.ndim == 2:
                image = Image.fromarray(image_array, mode='L')
            elif image_array.ndim == 3:
                image = Image.fromarray(image_array)
            else:
                raise ValueError(f"Unsupported array shape: {image_array.shape}")
            # Save the image
            image_name = filename.replace('.npy', f'.{image_format}')
            image_path = os.path.join(image_dir, image_name)
            image.save(image_path)
            print(f"Saved {image_path}")
    
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
    
    parser.add_argument('--save_dir', type=int, default=16,
                        help='Path to where the converted images will be saved!')
    
    parser.add_argument('--image_format', type=str, default='png')
    
    parser.add_argument('--do_normalize' , default=True)
    
    args = parser.parse_args()
    
    npy_to_image(args.npy_dir, args.save_dir, args.image_format, args.do_normalize)

#cloner174