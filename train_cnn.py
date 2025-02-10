import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import time
import torch.cuda as cuda
import torchvision.transforms as transforms

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set cuDNN configurations for optimized performance
cudnn.benchmark = True
cudnn.deterministic = True

# Define Paths
base_path = r'url'
model_save_path = os.path.join(base_path, 'model', 'new_best_model_ukan.pth')
metrics_value_dir = os.path.join(base_path, 'Metrics_value')
predictions_dir = os.path.join(base_path, 'Prediction_mask')

# Create directories if they don't exist
os.makedirs(os.path.join(base_path, 'model'), exist_ok=True)
os.makedirs(metrics_value_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Define SE Block
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se_weight = self.global_pool(x)
        se_weight = self.relu(self.fc1(se_weight))
        se_weight = self.sigmoid(self.fc2(se_weight))
        return x * se_weight

class SE_SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_SpatialAttention, self).__init__()

        # Squeeze-and-Excitation (SE) Block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),  
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),  
            nn.Sigmoid()  
        )

        # Spatial Attention Block
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        # SE Block
        se_weight = self.se(x)  
        x_se = x * se_weight  

        # Spatial Attention Block
        spatial_weight = self.spatial(x_se)  
        x_spatial = x_se * spatial_weight  

        return x_spatial
        
# Define 3D CNN with Skip Connections and Lightweight Attention
class AxialAttention(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(AxialAttention, self).__init__()
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5

        # Query, Key, Value projections for each axis
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W, D = x.shape

        # Process along Height axis
        q_h = self.query(x).view(B, self.heads, C // self.heads, H, W, D)
        k_h = self.key(x).view(B, self.heads, C // self.heads, H, W, D)
        v_h = self.value(x).view(B, self.heads, C // self.heads, H, W, D)

        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        out_h = (attn_h @ v_h).reshape(B, C, H, W, D)

        # Process along Width axis
        q_w = self.query(x).view(B, self.heads, C // self.heads, W, H, D)
        k_w = self.key(x).view(B, self.heads, C // self.heads, W, H, D)
        v_w = self.value(x).view(B, self.heads, C // self.heads, W, H, D)

        attn_w = (q_w @ k_w.transpose(-2, -1)) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        out_w = (attn_w @ v_w).reshape(B, C, H, W, D)

        # Process along Depth axis
        q_d = self.query(x).view(B, self.heads, C // self.heads, D, H, W)
        k_d = self.key(x).view(B, self.heads, C // self.heads, D, H, W)
        v_d = self.value(x).view(B, self.heads, C // self.heads, D, H, W)

        attn_d = (q_d @ k_d.transpose(-2, -1)) * self.scale
        attn_d = attn_d.softmax(dim=-1)
        out_d = (attn_d @ v_d).reshape(B, C, H, W, D)

        # Combine outputs from all axes
        out = out_h + out_w + out_d
        out = self.proj(out)

        return out


class CNN3D_SE_SpatialAttention_Axial(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN3D_SE_SpatialAttention_Axial, self).__init__()

        def CBR_SE_Spatial_Axial(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                SE_SpatialAttention(out_channels), 
                AxialAttention(out_channels)  
            )

        # Encoder Pathway
        self.encoder1 = CBR_SE_Spatial_Axial(in_channels, 64)
        self.encoder2 = CBR_SE_Spatial_Axial(64, 128)
        self.encoder3 = CBR_SE_Spatial_Axial(128, 256)
        self.encoder4 = CBR_SE_Spatial_Axial(256, 512)
        self.encoder5 = CBR_SE_Spatial_Axial(512, 1024)

        self.pool = nn.MaxPool3d(2)

        # Decoder Pathway with Skip Connections
        self.decoder5 = CBR_SE_Spatial_Axial(1024, 512)
        self.decoder4 = CBR_SE_Spatial_Axial(512, 256)
        self.decoder3 = CBR_SE_Spatial_Axial(256, 128)
        self.decoder2 = CBR_SE_Spatial_Axial(128, 64)

        self.upconv5 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        # Final Output Layer
        self.conv_last = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Pathway
        enc1 = self.encoder1(x)  # [B, 64, H, W, D]
        enc2 = self.encoder2(self.pool(enc1))  # [B, 128, H/2, W/2, D/2]
        enc3 = self.encoder3(self.pool(enc2))  # [B, 256, H/4, W/4, D/4]
        enc4 = self.encoder4(self.pool(enc3))  # [B, 512, H/8, W/8, D/8]
        enc5 = self.encoder5(self.pool(enc4))  # [B, 1024, H/16, W/16, D/16]

        # Decoder Pathway with Skip Connections
        dec5 = self.upconv5(enc5)  # [B, 512, H/8, W/8, D/8]
        if dec5.size(2) != enc4.size(2) or dec5.size(3) != enc4.size(3) or dec5.size(4) != enc4.size(4):
            dec5 = F.pad(dec5, [0, enc4.size(4) - dec5.size(4),
                                0, enc4.size(3) - dec5.size(3),
                                0, enc4.size(2) - dec5.size(2)])
        dec5 = torch.cat((dec5, enc4), dim=1)  # [B, 1024, H/8, W/8, D/8]
        dec5 = self.decoder5(dec5)  # [B, 512, H/8, W/8, D/8]

        dec4 = self.upconv4(dec5)  # [B, 256, H/4, W/4, D/4]
        if dec4.size(2) != enc3.size(2) or dec4.size(3) != enc3.size(3) or dec4.size(4) != enc3.size(4):
            dec4 = F.pad(dec4, [0, enc3.size(4) - dec4.size(4),
                                0, enc3.size(3) - dec4.size(3),
                                0, enc3.size(2) - dec4.size(2)])
        dec4 = torch.cat((dec4, enc3), dim=1)  # [B, 512, H/4, W/4, D/4]
        dec4 = self.decoder4(dec4)  # [B, 256, H/4, W/4, D/4]

        dec3 = self.upconv3(dec4)  # [B, 128, H/2, W/2, D/2]
        if dec3.size(2) != enc2.size(2) or dec3.size(3) != enc2.size(3) or dec3.size(4) != enc2.size(4):
            dec3 = F.pad(dec3, [0, enc2.size(4) - dec3.size(4),
                                0, enc2.size(3) - dec3.size(3),
                                0, enc2.size(2) - dec3.size(2)])
        dec3 = torch.cat((dec3, enc2), dim=1)  # [B, 256, H/2, W/2, D/2]
        dec3 = self.decoder3(dec3)  # [B, 128, H/2, W/2, D/2]

        dec2 = self.upconv2(dec3)  # [B, 64, H, W, D]
        if dec2.size(2) != enc1.size(2) or dec2.size(3) != enc1.size(3) or dec2.size(4) != enc1.size(4):
            dec2 = F.pad(dec2, [0, enc1.size(4) - dec2.size(4),
                                0, enc1.size(3) - dec2.size(3),
                                0, enc1.size(2) - dec2.size(2)])
        dec2 = torch.cat((dec2, enc1), dim=1)  # [B, 128, H, W, D]
        dec2 = self.decoder2(dec2)  # [B, 64, H, W, D]

        # Final Output
        return self.conv_last(dec2)  # [B, out_channels, H, W, D]
        
# Dice coefficient calculation for multi-class
def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float().to(device)
    target = target.float().to(device)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection) / (union + 1e-6)
    return dice.mean().item()

def calculate_et_tc_wt_dice(pred, target):
    pred_class = torch.argmax(pred, dim=1)

    et_pred = (pred_class == 3).float()
    et_target = (target == 3).float()
    et_dice = dice_coefficient(et_pred, et_target)

    tc_pred = ((pred_class == 1) | (pred_class == 3)).float()
    tc_target = ((target == 1) | (target == 3)).float()
    tc_dice = dice_coefficient(tc_pred, tc_target)

    wt_pred = ((pred_class == 1) | (pred_class == 2) | (pred_class == 3)).float()
    wt_target = ((target == 1) | (target == 2) | (target == 3)).float()
    wt_dice = dice_coefficient(wt_pred, wt_target)

    return et_dice, tc_dice, wt_dice

# Define Dataset and DataLoader for .h5 format with augmentation
class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        h5_path = self.file_list[idx]
        with h5py.File(h5_path, 'r') as h5_file:
            image = h5_file['image'][:].astype(np.float32)
            mask = h5_file['label'][:].astype(np.uint8)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# Data Augmentation function for multi-channel input
def data_augmentation(image, mask):
    image = image[:, :128, :128, :128]
    mask = mask[:128, :128, :128]

    # Random flips
    if torch.rand(1) > 0.5:
        image = torch.flip(image, [1])
        mask = torch.flip(mask, [0])
    if torch.rand(1) > 0.5:
        image = torch.flip(image, [2])
        mask = torch.flip(mask, [1])
    if torch.rand(1) > 0.5:
        image = torch.flip(image, [3])
        mask = torch.flip(mask, [2])

    # Gaussian noise
    noise = torch.randn_like(image) * 0.01
    image += noise

    # Random rotations
    angle = torch.randint(-10, 10, (1,)).item()
    image = transforms.functional.rotate(image, angle)
    mask = transforms.functional.rotate(mask, angle)

    return image, mask

# Initialize Datasets and DataLoaders
train_dataset = BraTSDataset(
    data_dir=r'url',
    transform=data_augmentation
)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

val_dataset = BraTSDataset(data_dir=r'url')
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

# Evaluation function with ET, TC, WT Dice Scores
def evaluate(model, val_loader):
    model.eval()
    val_loss = 0
    et_dices, tc_dices, wt_dices = [], [], []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

            et, tc, wt = calculate_et_tc_wt_dice(outputs, masks)
            et_dices.append(et)
            tc_dices.append(tc)
            wt_dices.append(wt)

    avg_val_loss = val_loss / len(val_loader)
    avg_et_dice = torch.tensor(et_dices).mean().item()
    avg_tc_dice = torch.tensor(tc_dices).mean().item()
    avg_wt_dice = torch.tensor(wt_dices).mean().item()

    return avg_val_loss, avg_et_dice, avg_tc_dice, avg_wt_dice

# Combined Loss with weighted CrossEntropyLoss and Dice loss
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.7, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        dice = 1 - dice_coefficient(outputs, F.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float())
        return self.ce_weight * ce + self.dice_weight * dice

# Training loop
model = CNN3D_SE_SpatialAttention_Axial(in_channels=4, out_channels=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = amp.GradScaler()

num_epochs = 400
# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Initialize early stopping
early_stopping = EarlyStopping(patience=30, path=model_save_path)

# Training loop with step display every 10 steps
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)

    # Total steps per epoch
    total_steps = len(train_loader)

    for step, (images, masks) in enumerate(train_loader, start=1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Display loss every 10 steps
        if step % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{total_steps}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)

    # Evaluate on validation set
    avg_val_loss, avg_et_dice, avg_tc_dice, avg_wt_dice = evaluate(model, val_loader)

    # Display training and validation metrics after each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] Summary:")
    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"ET Dice: {avg_et_dice:.4f}, TC Dice: {avg_tc_dice:.4f}, WT Dice: {avg_wt_dice:.4f}\n")

    # Save metrics to file
    with open(os.path.join(metrics_value_dir, f'epoch_{epoch+1}_metrics.txt'), 'w') as f:
        f.write(f'Epoch {epoch+1}\n')
        f.write(f'Validation Loss: {avg_val_loss:.4f}\n')
        f.write(f'ET Dice: {avg_et_dice:.4f}, TC Dice: {avg_tc_dice:.4f}, WT Dice: {avg_wt_dice:.4f}\n')

    # Check early stopping
    early_stopping(avg_val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        break  