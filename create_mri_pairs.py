import os
import h5py
import numpy as np

# Paths
synthetic_data_path = r"url"
clean_data_path = r"url"
output_path = r"url"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

def get_h5_keys(file_path):
    """Retrieve available keys from an HDF5 file."""
    with h5py.File(file_path, 'r') as hf:
        return list(hf.keys())

def load_h5_data(file_path):
    """Load data from an HDF5 file using available keys."""
    with h5py.File(file_path, 'r') as hf:
        keys = list(hf.keys())
        if len(keys) == 1:  # If there's only one dataset key
            return hf[keys[0]][:], keys[0]
        elif "artifact_images" in keys:
            return hf["artifact_images"][:], "artifact_images"
        elif "clean_images" in keys:
            return hf["clean_images"][:], "clean_images"
        else:
            raise KeyError(f"No expected key found in {file_path}. Available keys: {keys}")

def save_image_pair(artifact_data, clean_data, output_file):
    """Save artifact and clean image pair into an HDF5 file."""
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('artifact_images', data=artifact_data.astype(np.float32), compression="gzip")
        hf.create_dataset('clean_images', data=clean_data.astype(np.float32), compression="gzip")

def generate_image_pairs(synthetic_path, clean_path, output_path):
    """Generate artifact + clean image pairs."""
    synthetic_files = sorted([f for f in os.listdir(synthetic_path) if f.endswith(".h5")])
    clean_files = sorted([f for f in os.listdir(clean_path) if f.endswith(".h5")])

    clean_files_set = set(clean_files)  # Store clean file names for quick lookup

    for synthetic_file in synthetic_files:
        synthetic_file_path = os.path.join(synthetic_path, synthetic_file)

        # Adjust filename replacement (_artifact instead of _motion)
        clean_file_name = synthetic_file.replace("_artifact", "")
        clean_file_path = os.path.join(clean_path, clean_file_name)

        # Debugging print
        print(f"Checking: Synthetic File = {synthetic_file}, Expected Clean File = {clean_file_name}")

        if clean_file_name in clean_files_set:
            # Load synthetic (artifact-affected) data
            artifact_data, artifact_key = load_h5_data(synthetic_file_path)
            
            # Load clean (artifact-free) data
            clean_data, clean_key = load_h5_data(clean_file_path)

            # Save paired data
            output_file = os.path.join(output_path, clean_file_name)  # Save using clean file name
            save_image_pair(artifact_data, clean_data, output_file)
            print(f"Saved image pair: {output_file} (Keys used: {artifact_key}, {clean_key})")
        else:
            print(f"Clean file not found for {synthetic_file} (Expected: {clean_file_name})")

# Generate image pairs
generate_image_pairs(synthetic_data_path, clean_data_path, output_path)
