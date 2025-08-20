# 🩻 X-ray Image Segmentation using U-Net 🧠  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/xray_segmentation_unet.ipynb)  
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch)  
![License](https://img.shields.io/github/license/<your-username>/<your-repo>)  
![Stars](https://img.shields.io/github/stars/<your-username>/<your-repo>?style=social)  
![Forks](https://img.shields.io/github/forks/<your-username>/<your-repo>?style=social)

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
├── images/ # Original X-ray images
└── masks/ # Binary segmentation masks


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
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

