import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange, repeat
from torchvision.transforms import Compose, ToTensor, Resize
from kornia.losses import SSIMLoss  
from torchmetrics import StructuralSimilarityIndexMeasure  
from einops import rearrange

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset Class with Validation Support
class MRIImagePairDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_names = [f for f in os.listdir(data_path) if f.endswith(".h5")]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_names[idx])
        with h5py.File(file_path, 'r') as hf:
            artifact_images = hf['artifact_images'][:]
            artifact_images = np.clip(artifact_images, 0, 1)
            clean_image = hf['clean_images'][:]

        # Convert to PyTorch tensors
        artifact_images = torch.tensor(artifact_images, dtype=torch.float32)
        clean_image = torch.tensor(clean_image, dtype=torch.float32)

        return artifact_images, clean_image

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=4):
        super().__init__()

        # Store parameters
        self.image_size = image_size
        self.patch_size = patch_size

        # Ensure image size is divisible by patch size
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."

        # Compute the correct number of patches for a 3D volume
        self.num_patches = (image_size // patch_size) ** 3  

        patch_dim = channels * (patch_size ** 3)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))  

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True),
            num_layers=depth
        )

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim)  
        )

    def forward(self, img):
        p = self.patch_size
        b, c, h, w, d = img.shape

        # Ensure dimensions are divisible by patch size
        assert h % p == 0 and w % p == 0 and d % p == 0, "Image dimensions must be divisible by patch size."

        # Convert image into patches (Corrected for 3D)
        x = rearrange(img, 'b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', 
                      p1=p, p2=p, p3=p)
        x = self.patch_to_embedding(x)

        # Add CLS token to the input
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Check if the number of patches matches the positional embedding size
        if x.shape[1] != self.pos_embedding.shape[1]:
            raise ValueError(f"Position embedding shape mismatch: Expected {self.pos_embedding.shape[1]}, "
                             f"but got {x.shape[1]}. Ensure correct image_size={self.image_size}, "
                             f"patch_size={self.patch_size}, and computed num_patches={self.num_patches}.")

        # Apply position embeddings
        x += self.pos_embedding[:, :x.shape[1], :]  
        x = self.transformer(x)

        return self.mlp_head(x[:, 1:])  # Return all patches except CLS token, ensuring correct shape

class TransformerDiffusionModel(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.vit = ViT(image_size, patch_size, dim, depth, heads, mlp_dim)
        self.ssim_loss = SSIMLoss(window_size=11)  

        # Define noise schedule (alphas and alphas_cumprod)
        self.betas = self._linear_beta_schedule(timesteps).to(device)  
        self.alphas = (1.0 - self.betas).to(device)  
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)  

    def _linear_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, timesteps)  

    def forward(self, x, t):
        # Add noise to input based on timestep t
        noise = torch.randn_like(x).to(device)  
        x_noisy = self.q_sample(x, t, noise)
        # Predict noise using ViT
        predicted_noise = self.vit(x_noisy)
        return predicted_noise

    def q_sample(self, x_start, t, noise):
        # Ensure indices are on the correct device
        t = t.to(device)  # Move t to the same device as model parameters

        # Forward diffusion process: gradually add noise
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])  
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])  
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def p_losses(self, x_start, t):
        # Generate random noise with the same shape as input
        noise = torch.randn_like(x_start).to(device)

        # Forward diffusion: add noise
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict noise using ViT
        predicted_noise = self.forward(x_noisy, t)  

        # Extract useful parameters
        p = self.vit.patch_size
        batch_size = x_start.shape[0]
        num_patches = predicted_noise.shape[1]  

        # Ensure expected number of patches
        expected_patches = (x_start.shape[2] // p) * (x_start.shape[3] // p) * (x_start.shape[4] // p)
        assert num_patches == expected_patches, f"Mismatch: expected {expected_patches} patches, got {num_patches}"

        # Ensure output maps correctly back to (batch, channels, H, W, D)
        channels = 4
        patch_dim = p ** 3
        feature_dim = channels * patch_dim

        # Verify the dimension alignment
        assert predicted_noise.shape[-1] == feature_dim, \
            f"Expected last dimension to be {feature_dim}, but got {predicted_noise.shape[-1]}"

        # Reshape to reconstruct the 3D volume
        predicted_noise = predicted_noise.reshape(batch_size, 
                                                  x_start.shape[2] // p, 
                                                  x_start.shape[3] // p, 
                                                  x_start.shape[4] // p, 
                                                  channels, 
                                                  p, p, p)
        
        # Rearrange back to (batch, channels, H, W, D)
        predicted_noise = rearrange(predicted_noise, 
                                    'b hp wp dp c p1 p2 p3 -> b c (hp p1) (wp p2) (dp p3)')

        # Compute MSE Loss
        mse_loss = torch.mean((predicted_noise - noise) ** 2)

        # Compute SSIM Loss (Perceptual Loss) slice-wise along depth axis (D)
        ssim_loss = 0
        for d in range(x_start.shape[4]):  
            ssim_loss += self.ssim_loss(predicted_noise[:, :, :, :, d], noise[:, :, :, :, d])

        ssim_loss /= x_start.shape[4]  

        # Combine losses
        total_loss = mse_loss + ssim_loss

        return total_loss, mse_loss, ssim_loss

# Define paths
train_path = r'url'
validation_path = r'url'  
save_path = r'url'

def train_diffuse_model(train_path, validation_path, save_path, epochs=400, batch_size=4, learning_rate=1e-4, print_every=10):
    # Prepare datasets and loaders
    train_dataset = MRIImagePairDataset(train_path)
    validation_dataset = MRIImagePairDataset(validation_path)  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)  # No shuffling for validation

    # Initialize model, optimizer
    model = TransformerDiffusionModel(image_size=160, patch_size=8, dim=512, depth=6, heads=8, mlp_dim=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        mse_loss_total = 0
        ssim_loss_total = 0

        for step, (artifact_images, clean_images) in enumerate(train_loader):
            artifact_images, clean_images = artifact_images.to(device), clean_images.to(device)

            # Random timesteps for diffusion
            t = torch.randint(0, model.timesteps, (artifact_images.shape[0],), device=device).long()

            # Forward pass and loss calculation
            total_loss, mse_loss, ssim_loss = model.p_losses(artifact_images, t)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            mse_loss_total += mse_loss.item()
            ssim_loss_total += ssim_loss.item()

            # Print training progress
            if (step + 1) % print_every == 0:
                avg_loss = train_loss / (step + 1)
                avg_mse_loss = mse_loss_total / (step + 1)
                avg_ssim_loss = ssim_loss_total / (step + 1)
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], "
                      f"Avg Train Loss: {avg_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, SSIM Loss: {avg_ssim_loss:.4f}")

        # Average loss for the epoch
        avg_epoch_train_loss = train_loss / len(train_loader)
        avg_epoch_mse_loss = mse_loss_total / len(train_loader)
        avg_epoch_ssim_loss = ssim_loss_total / len(train_loader)

        # **Validation Loop**
        model.eval()
        validation_loss = 0
        validation_mse_loss = 0
        validation_ssim_loss = 0

        with torch.no_grad():
            for artifact_images, clean_images in validation_loader:
                artifact_images, clean_images = artifact_images.to(device), clean_images.to(device)
                t = torch.randint(0, model.timesteps, (artifact_images.shape[0],), device=device).long()
                total_loss, mse_loss, ssim_loss = model.p_losses(artifact_images, t)

                validation_loss += total_loss.item()
                validation_mse_loss += mse_loss.item()
                validation_ssim_loss += ssim_loss.item()

        avg_validation_loss = validation_loss / len(validation_loader)
        avg_validation_mse_loss = validation_mse_loss / len(validation_loader)
        avg_validation_ssim_loss = validation_ssim_loss / len(validation_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Completed, "
              f"Avg Train Loss: {avg_epoch_train_loss:.4f}, MSE Loss: {avg_epoch_mse_loss:.4f}, SSIM Loss: {avg_epoch_ssim_loss:.4f}, "
              f"Validation Loss: {avg_validation_loss:.4f}, Validation MSE: {avg_validation_mse_loss:.4f}, Validation SSIM: {avg_validation_ssim_loss:.4f}")

    # Save final model
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, "transformer_diffusion_model_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Final Model saved to {model_save_path}")

# Train the model with validation dataset
train_diffuse_model(train_path, validation_path, save_path, epochs=400, batch_size=4, learning_rate=1e-4, print_every=10)
