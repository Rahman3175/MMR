import os
import h5py
import torch
import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import laplace
from skimage.feature import canny
import math
import pandas as pd
from train_diffusion import TransformerDiffusionModel  # Import trained diffusion model

# Paths
artifact_affected_path = r'url'
artifact_free_path = r'url'
original_sample_path = r'url'  
detailed_metrics_path = r'url'
summary_metrics_path = r'url'
model_path = r'url'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_path):
    model = TransformerDiffusionModel(image_size=128, patch_size=8, dim=512, depth=6, heads=8, mlp_dim=1024).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def calculate_noise_level(image1, image2):
    return np.var(image1 - image2)

def calculate_sharpness(image):
    return np.var(laplace(image))

def calculate_fsim(image1, image2):
    # Placeholder for FSIM implementation
    return np.mean((image1 * image2) / (np.abs(image1) + np.abs(image2) + 1e-6))

# Initialize dictionaries to store results
artifact_affected_metrics = {
    'Noise Level': [],
    'Sharpness': [],
    'PSNR': [],
    'MSE': [],
    'SSIM': [],
    'CNR': [],
    'FSIM': []
}

artifact_free_metrics = {
    'Noise Level': [],
    'Sharpness': [],
    'PSNR': [],
    'MSE': [],
    'SSIM': [],
    'CNR': [],
    'FSIM': []
}
# Open text file for detailed metrics
print("Evaluation process is starting...")
with open(detailed_metrics_path, 'w') as detailed_file:
    detailed_file.write("Detailed Per-Sample Metrics\n")
    detailed_file.write("=" * 50 + "\n")

    # Get the list of all .h5 files in the artifact-affected directory
    all_files = [file for file in os.listdir(artifact_affected_path) if file.endswith('.h5')]

    # Process only the first 2250 files
    selected_files = all_files[:2250]

    for file_name in selected_files:  # Loop through the selected files
        print(f"Processing file: {file_name}")  # Added print statement to track progress
        # Define file paths
        artifact_affected_file = os.path.join(artifact_affected_path, file_name)
        artifact_free_file = os.path.join(artifact_free_path, file_name)
        original_file_base = file_name
        original_sample_file = os.path.join(original_sample_path, original_file_base)

        # Skip if any necessary file is missing
        if not os.path.exists(artifact_affected_file):
            print(f"Missing artifact-affected file: {artifact_affected_file}")
            continue
        if not os.path.exists(artifact_free_file):
            print(f"Missing artifact-free file: {artifact_free_file}")
            continue
        if not os.path.exists(original_sample_file):
            print(f"Missing original file: {original_sample_file}")
            continue

        try:
            with h5py.File(artifact_affected_file, 'r') as affected_h5, \
                 h5py.File(artifact_free_file, 'r') as free_h5, \
                 h5py.File(original_sample_file, 'r') as original_h5:

                # Load artifact-affected images
                affected_images = affected_h5['image'][:]  # Updated key

                # Load artifact-free images
                artifact_free_images = free_h5['artifact_free_images'][:]  # Updated key

                # Load original images
                original_images = original_h5['image'][:]  # Updated key

                # Normalize and evaluate each modality
                for i in range(affected_images.shape[0]):  # Assuming the first dimension indexes modalities
                    # Normalize images individually with clipping for affected_image
                    affected_image = np.clip(affected_images[i], 0, 1) / np.max(affected_images[i])  # Clip negative values
                    artifact_free_image = artifact_free_images[i] / np.max(artifact_free_images[i])
                    original_image = original_images[i] / np.max(original_images[i])

                    # Debug: Log dynamic ranges
                    print(f"File: {file_name}, Modality Index: {i}")
                    print(f"  Clipped Affected Image Range: {np.min(affected_image)} to {np.max(affected_image)}")
                    print(f"  Artifact-Free Image Range: {np.min(artifact_free_image)} to {np.max(artifact_free_image)}")
                    print(f"  Original Image Range: {np.min(original_image)} to {np.max(original_image)}")

                    # Calculate metrics
                    mse_value_affected = calculate_mse(affected_image, original_image)
                    psnr_value_affected = calculate_psnr(affected_image, original_image)
                    noise_level_affected = calculate_noise_level(affected_image, original_image)
                    sharpness_affected = calculate_sharpness(affected_image)
                    ssim_value_affected = ssim(
                        affected_image, original_image,
                        data_range=affected_image.max() - affected_image.min()
                    )
                    fsim_value_affected = calculate_fsim(affected_image, original_image)

                    mse_value_free = calculate_mse(artifact_free_image, original_image)
                    psnr_value_free = calculate_psnr(artifact_free_image, original_image)
                    noise_level_free = calculate_noise_level(artifact_free_image, original_image)
                    sharpness_free = calculate_sharpness(artifact_free_image)
                    ssim_value_free = ssim(
                        artifact_free_image, original_image,
                        data_range=artifact_free_image.max() - artifact_free_image.min()
                    )
                    fsim_value_free = calculate_fsim(artifact_free_image, original_image)

                    # Append metrics
                    artifact_affected_metrics['Noise Level'].append(noise_level_affected)
                    artifact_affected_metrics['Sharpness'].append(sharpness_affected)
                    artifact_affected_metrics['PSNR'].append(psnr_value_affected)
                    artifact_affected_metrics['MSE'].append(mse_value_affected)
                    artifact_affected_metrics['SSIM'].append(ssim_value_affected)
                    artifact_affected_metrics['FSIM'].append(fsim_value_affected)

                    artifact_free_metrics['Noise Level'].append(noise_level_free)
                    artifact_free_metrics['Sharpness'].append(sharpness_free)
                    artifact_free_metrics['PSNR'].append(psnr_value_free)
                    artifact_free_metrics['MSE'].append(mse_value_free)
                    artifact_free_metrics['SSIM'].append(ssim_value_free)
                    artifact_free_metrics['FSIM'].append(fsim_value_free)

                    # Write metrics to file
                    detailed_file.write(f"File: {file_name}, Modality Index: {i}\n")
                    detailed_file.write(f"  Artifact-Affected:\n")
                    detailed_file.write(f"    MSE: {mse_value_affected:.4f}, PSNR: {psnr_value_affected:.4f}, Noise Level: {noise_level_affected:.4f}, "
                                        f"Sharpness: {sharpness_affected:.4f}, SSIM: {ssim_value_affected:.4f}, FSIM: {fsim_value_affected:.4f}\n")
                    detailed_file.write(f"  Artifact-Free:\n")
                    detailed_file.write(f"    MSE: {mse_value_free:.4f}, PSNR: {psnr_value_free:.4f}, Noise Level: {noise_level_free:.4f}, "
                                        f"Sharpness: {sharpness_free:.4f}, SSIM: {ssim_value_free:.4f}, FSIM: {fsim_value_free:.4f}\n")
                    detailed_file.write("-" * 50 + "\n")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# Filter metrics dynamically based on non-empty lists
valid_metrics = [metric for metric in artifact_affected_metrics if artifact_affected_metrics[metric] or artifact_free_metrics[metric]]

# Debug: Print valid metrics to ensure only populated ones are used
print("Valid Metrics for Summary:", valid_metrics)

# Compute summary statistics 
summary_data = {
    'Metric': valid_metrics,
    'Artifact-Affected (Avg)': [
        np.mean(artifact_affected_metrics[metric]) if artifact_affected_metrics[metric] else None
        for metric in valid_metrics
    ],
    'Artifact-Affected (Max)': [
        np.max(artifact_affected_metrics[metric]) if artifact_affected_metrics[metric] else None
        for metric in valid_metrics
    ],
    'Artifact-Free (Avg)': [
        np.mean(artifact_free_metrics[metric]) if artifact_free_metrics[metric] else None
        for metric in valid_metrics
    ],
    'Artifact-Free (Max)': [
        np.max(artifact_free_metrics[metric]) if artifact_free_metrics[metric] else None
        for metric in valid_metrics
    ],
}

# Convert to DataFrame and save
df = pd.DataFrame(summary_data)
df.to_csv(summary_metrics_path, index=False)

# Display the summary table
print("Summary of Metrics:")
print(df)