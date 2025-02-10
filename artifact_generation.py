import os
import h5py
import numpy as np
from scipy.ndimage import affine_transform

def apply_displacement(image):
    shifts = np.random.randint(-5, 5, size=3)  
    return np.roll(image, shift=shifts, axis=(0, 1, 2))

def apply_rotation(image):
    rotation_matrix = np.array([[np.cos(np.pi / 18), -np.sin(np.pi / 18), 0],
                                 [np.sin(np.pi / 18), np.cos(np.pi / 18), 0],
                                 [0, 0, 1]])
    return affine_transform(image, rotation_matrix, mode='nearest')

def apply_warping(image):
    coords = np.meshgrid(np.arange(image.shape[0]),  
                         np.arange(image.shape[1]),  
                         np.arange(image.shape[2]),  
                         indexing='ij')

    coords[0] = np.clip(coords[0] + 5 * np.sin(coords[1] / 10), 0, image.shape[0] - 1).astype(int)  
    coords[1] = np.clip(coords[1] + 5 * np.sin(coords[2] / 10), 0, image.shape[1] - 1).astype(int)  
    coords[2] = np.clip(coords[2] + 5 * np.sin(coords[0] / 10), 0, image.shape[2] - 1).astype(int)  

    return image[coords[0], coords[1], coords[2]]
    
def add_gaussian_noise(image, mean=0.0, std=0.005):
    noise = np.random.normal(mean, std, image.shape)
    return image + noise

def add_periodic_ghosting(image):
    ghost_image = image.copy()
    ghost_image[::16] = 0
    return ghost_image

def add_k_space_motion(image):
    k_space = np.fft.fftn(image)
    motion = np.random.uniform(0.98, 1.02, k_space.shape)
    return np.abs(np.fft.ifftn(k_space * motion))

def apply_motion_artifacts(image):
    image = apply_displacement(image)
    image = apply_rotation(image)
    image = apply_warping(image)
    image = add_gaussian_noise(image)
    image = add_periodic_ghosting(image)
    return add_k_space_motion(image)

def process_and_save_artifact_data(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for file_name in os.listdir(input_path):
        if not file_name.endswith(".h5"):
            continue
        
        file_path = os.path.join(input_path, file_name)
        with h5py.File(file_path, 'r') as hf:
            images = hf['image'][:]  
            labels = hf['label'][:]  

        artifact_images = np.stack([apply_motion_artifacts(images[i]) for i in range(images.shape[0])], axis=0)

        output_file = os.path.join(output_path, file_name.replace(".h5", "_artifact.h5"))

        with h5py.File(output_file, 'w') as hf:
            hf.create_dataset('image', data=artifact_images.astype(np.float32), compression="gzip")
            hf.create_dataset('label', data=labels.astype(np.uint8), compression="gzip")

        print(f"Saved {output_file}")

input_path = r'url'
output_path = r'url'
process_and_save_artifact_data(input_path, output_path)
