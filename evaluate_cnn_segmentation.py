import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from medpy.metric.binary import hd95

# Paths
model_path = r'url'
test_data_path = r'url'
metrics_output_path = r'url'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(metrics_output_path, exist_ok=True)

# Load trained model (CNN3D_SE_SpatialAttention_Axial)
class CNN3D_SE_SpatialAttention_Axial(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN3D_SE_SpatialAttention_Axial, self).__init__()

        def CBR_SE_Spatial_Axial(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = CBR_SE_Spatial_Axial(in_channels, 64)
        self.encoder2 = CBR_SE_Spatial_Axial(64, 128)
        self.encoder3 = CBR_SE_Spatial_Axial(128, 256)
        self.encoder4 = CBR_SE_Spatial_Axial(256, 512)
        self.encoder5 = CBR_SE_Spatial_Axial(512, 1024)

        self.pool = nn.MaxPool3d(2)
        self.decoder5 = CBR_SE_Spatial_Axial(1024, 512)
        self.decoder4 = CBR_SE_Spatial_Axial(512, 256)
        self.decoder3 = CBR_SE_Spatial_Axial(256, 128)
        self.decoder2 = CBR_SE_Spatial_Axial(128, 64)

        self.upconv5 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.conv_last = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        dec5 = self.upconv5(enc5)
        dec5 = torch.cat((dec5, enc4), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)

        return self.conv_last(dec2)

# Load model
model = CNN3D_SE_SpatialAttention_Axial(in_channels=4, out_channels=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Compute segmentation metrics
def compute_metrics(pred, label, class_id):
    pred_binary = (pred == class_id).astype(np.uint8)
    label_binary = (label == class_id).astype(np.uint8)

    intersection = np.sum(pred_binary * label_binary)
    union = np.sum(pred_binary + label_binary)
    dice = 2 * intersection / (union + 1e-6)
    sensitivity = intersection / (np.sum(label_binary) + 1e-6)
    specificity = np.sum((pred_binary == 0) & (label_binary == 0)) / (np.sum(label_binary == 0) + 1e-6)
    hd95_value = hd95(pred_binary, label_binary) if np.any(label_binary) and np.any(pred_binary) else np.nan

    return dice * 100, sensitivity * 100, specificity * 100, hd95_value

# Load test sample
def load_test_sample(file_path):
    with h5py.File(file_path, 'r') as f:
        image = f['image'][:]
        label = f['label'][:]
    return torch.tensor(image).float().to(device), torch.tensor(label).long()

# Initialize metrics storage
metrics_summary = []

# Evaluate all files in the test dataset
test_files = [f for f in os.listdir(test_data_path) if f.endswith('.h5')]

for test_file in test_files:
    file_path = os.path.join(test_data_path, test_file)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    image, label = load_test_sample(file_path)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    file_metrics = {"File": test_file}
    for class_id, class_name in enumerate(["ET", "TC", "WT"], start=1):
        dice, sensitivity, specificity, hd95_value = compute_metrics(prediction, label.cpu().numpy(), class_id)
        file_metrics[f"{class_name}_Dice"] = dice
        file_metrics[f"{class_name}_Sensitivity"] = sensitivity
        file_metrics[f"{class_name}_Specificity"] = specificity
        file_metrics[f"{class_name}_HD95"] = hd95_value

    metrics_summary.append(file_metrics)

# Convert results to DataFrame
df_metrics = pd.DataFrame(metrics_summary)

# Compute mean metrics
summary_metrics = df_metrics.mean(numeric_only=True).rename("Mean").to_frame().T

# Save detailed and summary metrics
detailed_metrics_file = os.path.join(metrics_output_path, "detailed_segmentation_metrics.csv")
summary_metrics_file = os.path.join(metrics_output_path, "summary_segmentation_metrics.csv")

df_metrics.to_csv(detailed_metrics_file, index=False)
summary_metrics.to_csv(summary_metrics_file, index=False)

# Display results
print("\nDetailed Metrics per File:")
print(df_metrics)
print("\nSummary of Metrics (Mean Across All Files):")
print(summary_metrics)

print("\nMetrics saved to:")
print(f"- {detailed_metrics_file}")
print(f"- {summary_metrics_file}")
