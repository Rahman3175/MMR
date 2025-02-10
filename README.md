# MMR: MRI Artifact Suppression using Diffusion Models

## 📌 Overview
This project implements a **diffusion-based Transformer model** for **MRI artifact suppression** and **3D CNN-based segmentation**. The framework includes:  
- **Dataset preparation** (Preprocessing, Synthetic Artifact Generation, and Clean MRI Pairing)  
- **Diffusion Model Training** for artifact suppression  
- **3D CNN Training** for brain tumor segmentation  
- **Evaluation Metrics** to measure performance  

---

## 📂 **Dataset Preparation**
This project follows a structured **data preparation pipeline**:
### (a) **Preprocessing**
- **Resampling and Normalization** of MRI scans  
- **Standardization of input size** to **128×128×128**  
- **Saving in `.h5` format** for efficient storage  

### (b) **Synthetic Artifact Generation**
- Introduces **motion, noise, ghosting, and k-space corruption**  
- Saves paired **synthetic+clean MRI** images for training  

### (c) **Synthetic + Clean MRI Image Pairing**
- Creates paired data for supervised training of the **diffusion model**  
- Stored as **HDF5 (`.h5`) format** for efficient access  

---

## 🎯 **Diffuse Model Training**
- Uses a **Transformer-based Vision Model** for artifact suppression  
- Trained using **MSE + SSIM loss**  
- Uses a **linear beta scheduling strategy** over **1000 timesteps**  
- Implemented in **PyTorch**  

**🔹 Training Command:**
```bash
python scripts/train_diffusion.py --data_path "path/to/dataset" --epochs 400 --batch_size 4

3D CNN Model Training
Training Command:
python scripts/train_cnn.py --data_path "path/to/preprocessed_data" --epochs 300 --batch_size 2
Evaluation Metrics
python scripts/evaluate.py --model_path "path/to/model.pth" --test_data "path/to/test_data"
Installation
git clone https://github.com/Rahman3175/MMR.git
cd MMR
pip install -r requirements.txt
Usage
Training the Diffusion Model
python scripts/train_diffusion.py --data_path "path/to/dataset" --epochs 400 --batch_size 4
Training the 3D CNN Model
python scripts/train_cnn.py --data_path "path/to/preprocessed_data" --epochs 300 --batch_size 2
Running Inference
python scripts/infer.py --input "path/to/input_image.h5" --output "path/to/output_image.h5"
Running Evaluation
python scripts/evaluate.py --model_path "path/to/model.pth" --test_data "path/to/test_data"
Directory Structure
MMR/
│── data/                 # Data files and preprocessing
│── models/               # Model architecture
│── scripts/              # Training, inference, and evaluation scripts
│── results/              # Evaluation results
│── README.md             # Project description
│── requirements.txt      # Required dependencies
│── LICENSE               # License file
│── .gitignore            # Ignored files
License
This project is licensed under the MIT License. See LICENSE for details.

Contact
For any issues, please open a GitHub issue or contact me via GitHub Discussions.



