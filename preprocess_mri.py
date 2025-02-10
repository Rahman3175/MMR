import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import h5py

def image_resampling(image, new_shape, order=3):
    """
    Resample an image to a new shape.
    """
    image_tensor = torch.tensor(image, dtype=torch.float32, device='cuda')
    new_shape = torch.Size([1, 1, *new_shape])
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    resampled_image = F.interpolate(image_tensor, size=new_shape[2:], mode='trilinear', align_corners=False)
    return resampled_image.squeeze().cpu().numpy()

def intensity_normalization(image):
    """
    Normalize the image intensities to the range [0, 1].
    """
    image_tensor = torch.tensor(image, dtype=torch.float32, device='cuda')
    min_value = torch.min(image_tensor)
    max_value = torch.max(image_tensor)
    if min_value == max_value:
        return torch.ones_like(image_tensor).cpu().numpy()  
    else:
        normalized_image = (image_tensor - min_value) / (max_value - min_value)
        return normalized_image.cpu().numpy()

# Define paths
data_path = r'url'
output_path = r'url'

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load and preprocess data
def preprocess_data(data_path, output_path):
    # List patient folders
    patient_folders = os.listdir(data_path)
    
    for patient_folder in patient_folders:
        patient_path = os.path.join(data_path, patient_folder)
        if os.path.isdir(patient_path):
            # Load MRI modalities
            seg_img = nib.load(os.path.join(patient_path, f'{patient_folder}-seg.nii.gz')).get_fdata().astype(np.uint8)
            t1ce_img = nib.load(os.path.join(patient_path, f'{patient_folder}-t1c.nii.gz')).get_fdata()
            t1_img = nib.load(os.path.join(patient_path, f'{patient_folder}-t1n.nii.gz')).get_fdata()
            t2f_img = nib.load(os.path.join(patient_path, f'{patient_folder}-t2f.nii.gz')).get_fdata()
            t2w_img = nib.load(os.path.join(patient_path, f'{patient_folder}-t2w.nii.gz')).get_fdata()
            
            # Resample MRI modalities and segmentation mask
            resized_t1ce_img = image_resampling(t1ce_img, new_shape=(128, 128, 128))
            resized_t1_img = image_resampling(t1_img, new_shape=(128, 128, 128))
            resized_t2f_img = image_resampling(t2f_img, new_shape=(128, 128, 128))
            resized_t2w_img = image_resampling(t2w_img, new_shape=(128, 128, 128))
            resized_seg_img = image_resampling(seg_img, new_shape=(128, 128, 128), order=0)

            # Normalize the modalities
            normalized_t1ce_img = intensity_normalization(resized_t1ce_img)
            normalized_t1_img = intensity_normalization(resized_t1_img)
            normalized_t2f_img = intensity_normalization(resized_t2f_img)
            normalized_t2w_img = intensity_normalization(resized_t2w_img)

            # Combine the four modalities into a single 4D array (C x H x W x D)
            combined_image = np.stack([normalized_t1ce_img, normalized_t1_img, normalized_t2f_img, normalized_t2w_img], axis=0)

            # Define the output file name
            output_file = os.path.join(output_path, f'{patient_folder}.h5')

            # Save the data in .h5 format
            with h5py.File(output_file, 'w') as hf:
                hf.create_dataset('image', data=combined_image.astype(np.float32), compression="gzip")
                hf.create_dataset('label', data=resized_seg_img.astype(np.uint8), compression="gzip")
                
            print(f"Saved {output_file}")

# Perform preprocessing
preprocess_data(data_path, output_path)
