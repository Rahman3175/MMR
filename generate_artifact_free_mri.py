import os
import h5py
import torch
import numpy as np

# Import the trained Transformer Diffusion Model from the training script
from train_diffusion import TransformerDiffusionModel  

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Function to load the trained model
def load_trained_model(model_path):
    """
    Load the trained Transformer Diffusion Model.
    """
    model = TransformerDiffusionModel(image_size=128, patch_size=8, dim=512, depth=6, heads=8, mlp_dim=1024).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

# Function to generate artifact-free MRI images
def generate_artifact_free_images(model, input_path, output_path):
    """
    Uses the trained diffusion model to generate artifact-free MRI images.
    """
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists

    for file_name in os.listdir(input_path):
        if file_name.endswith(".h5"):
            input_file_path = os.path.join(input_path, file_name)

            # Load artifact-affected images
            with h5py.File(input_file_path, 'r') as hf:
                artifact_images = hf['artifact_images'][:]  # Shape: [4, H, W, D]
                artifact_images = np.clip(artifact_images, 0, 1)  

            # Convert to PyTorch tensor and move to GPU
            artifact_images_tensor = torch.tensor(artifact_images, dtype=torch.float32).unsqueeze(0).to(device)  

            # Generate artifact-free images using the trained model
            with torch.no_grad():
                artifact_free_images = model(artifact_images_tensor).squeeze(0).cpu().numpy()  

            # Save the artifact-free images to .h5 format
            output_file_path = os.path.join(output_path, file_name)
            with h5py.File(output_file_path, 'w') as hf_out:
                hf_out.create_dataset('artifact_free_images', data=artifact_free_images, compression="gzip")

            print(f"Generated artifact-free images for {file_name} and saved to {output_file_path}")

# Define paths
model_path = r'url'
input_path = r'url'
output_path = r'url'

# Load the trained Transformer Diffusion Model
diffuse_model = load_trained_model(model_path)

# Generate artifact-free MRI images
generate_artifact_free_images(diffuse_model, input_path, output_path)
