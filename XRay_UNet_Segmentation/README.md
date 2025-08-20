# 🩻 X-ray Image Segmentation using U-Net 🧠  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Colab](https://img.shields.io/badge/Run%20on-Colab-orange?logo=googlecolab)

---

## 📌 Overview  
This project implements a **U-Net based deep learning pipeline** for **binary segmentation** of **X-ray images** using **PyTorch**.  
The model learns to segment regions of interest from X-ray scans by predicting binary masks.  

### **Key Features**
- ⚡ Fully **PyTorch-based** implementation  
- 🛠️ **U-Net** architecture with encoder-decoder + skip connections  
- 📊 Tracks **Dice, IoU, Accuracy, Loss**  
- 🧪 Dataset visualization & predicted overlays  
- 🎨 Real-time augmentations using **Albumentations**  

---

## 📂 Dataset  

This project uses a dataset containing **704 X-ray images** and their corresponding **704 segmentation masks**.  

**Structure:**  
x-ray/
- ├── images/ # Original X-ray images
- └── masks/ # Binary segmentation masks


- Images are grayscale `.png` / `.jpg`
- Masks are binary (0 for background, 1 for ROI)
- Dataset was split into:
  - **Train:** 70%
  - **Validation:** 15%
  - **Test:** 15%

### 📷 Dataset Visualization  

![Dataset Visualization](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/XRay_UNet_Segmentation/Outputs/dataset.png)


---

## ⚙️ Installation & Setup  

### **1. Clone the Repository**
```bash
git clone https://github.com/HussamUmer/Vision4Healthcare.git
cd Vision4Healthcare


```
---
## Run on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/XRay_UNet_Segmentation/Notebook%20File/U_Net_X_Ray.ipynb)

---

## 📈 Training Progress

![Training vs Validation Loss & Dice](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/XRay_UNet_Segmentation/Outputs/download%20(2).png)

---

## 🏆 Results

After **100 epochs**, the model achieved **excellent segmentation performance**:

| Dataset | Loss   | Dice   | IoU    | Accuracy |
|---------|--------|--------|--------|-----------|
| **Train** | 0.0891 | 0.9525 | 0.9107 | 0.9765 |
| **Val**   | 0.0814 | 0.9611 | 0.9264 | 0.9796 |
| **Test**  | 0.0751 | 0.9640 | 0.9311 | 0.9816 |

---

## 🖼️ Predicted vs Ground Truth

Predicted vs Ground Truth by our trained model:
![Predicted vs Ground Truth By Trained Model](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/XRay_UNet_Segmentation/Outputs/Predicted.png)

---

## 🛠️ How to Use
### 1. Inference on Single Image

```
from model import UNet
from inference import predict_mask
import torch

# Load model
model = UNet(in_ch=1, out_ch=1)
model.load_state_dict(torch.load("best_unet.pth", map_location="cpu"))
model.eval()

# Predict mask
mask = predict_mask(model, "path/to/image.png")

```
### 2. Evaluate Model on Test Set

```

python evaluate.py --weights best_unet.pth

```
---

## 📌 Tech Stack
```
- Language: Python 3.10+
- Framework: PyTorch
- Augmentation: Albumentations
- Visualization: Matplotlib, OpenCV
- Training Environment: Google Colab 
```
---

## 📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it.

---
