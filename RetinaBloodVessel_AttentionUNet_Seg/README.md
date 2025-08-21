# 👁️ Retina Blood Vessel Segmentation using Attention U-Net 🧠  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Colab](https://img.shields.io/badge/Run%20on-Colab-orange?logo=googlecolab)

---

## 📌 Overview  
This project implements an **Attention U-Net based deep learning pipeline** for **binary segmentation** of **retinal blood vessels** using **PyTorch**.  
Unlike the standard U-Net used in the X-ray segmentation project, this model integrates **attention gates** to better capture **thin, low-contrast blood vessels**, improving segmentation accuracy.

### **Key Features**
- ⚡ Fully **PyTorch-based** implementation  
- 🛠️ **Attention U-Net** architecture with encoder-decoder + attention gates  
- 📊 Tracks **Dice, IoU, Accuracy, AUC, Loss**  
- 🧪 Dataset visualization & predicted overlays  
- 🎨 Real-time augmentations using **Albumentations**

---

## 📂 Dataset  

This project uses a dataset containing **100 retinal fundus images** and their corresponding **100 segmentation masks**.  

**Structure:**  
data/
- ├── train/
      -│ ├── images/ # 80 training images
      -│ └── mask/ # 80 segmentation masks
- └── test/
      - ├── images/ # 20 test images
      - └── mask/ # 20 segmentation masks

  
- Images are `.png`
- Masks are binary (0 for background, 1 for blood vessels)
- Dataset is **already split** into train and test sets.

### 📷 Dataset Visualization  

![Dataset Visualization](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/RetinaBloodVessel_AttentionUNet_Seg/Output/download%20(2).png)

---

## ⚙️ Installation & Setup  

### **1. Clone the Repository**
```bash
git clone https://github.com/HussamUmer/Vision4Healthcare.git
cd Vision4Healthcare
```
---
## Run on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/RetinaBloodVessel_AttentionUNet_Seg/Notebook/RetinaBloodVessel_Segmentation_Attention_UNet.ipynb)

---

## 📈 Training Progress

![Training vs Validation Loss & Dice](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/RetinaBloodVessel_AttentionUNet_Seg/Output/download%20(3).png)

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
![Predicted vs Ground Truth By Trained Model](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/RetinaBloodVessel_AttentionUNet_Seg/Output/download%20(4).png)

---

## 📌 Tech Stack
- Language: Python 3.10+
- Framework: PyTorch
- Model: Attention U-Net
- Loss: Focal-Tversky
- Augmentation: Albumentations
- Visualization: Matplotlib, OpenCV
- Training Environment: Google Colab

---

## 📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it.

---

