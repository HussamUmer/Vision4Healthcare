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

## 📝 Why We Did This Project  
This project focuses on **retinal blood vessel segmentation** using **Attention U-Net**, an advanced variant of U-Net that integrates **attention gates** for better feature extraction.  

The aim is to support **early diagnosis of eye-related diseases** such as:  
- 👁️ **Diabetic Retinopathy (DR)**  
- 🧠 **Hypertension-induced Retinal Damage**  
- 🩸 **Glaucoma and Vascular Abnormalities**  

Key motivations include:  
- 🔍 Identifying **tiny and thin blood vessels** that are often missed in manual inspections.  
- 📈 Enhancing **segmentation accuracy** for small and complex vascular structures.  
- 🧠 Demonstrating how **attention-based deep learning** improves results compared to traditional U-Net.  

This project showcases how **AI in ophthalmology** can **assist doctors**, enable **faster screenings**, and potentially **prevent vision loss** by enabling **early detection**.


---


## 📂 Dataset  

This project uses a dataset containing **100 retinal fundus images** and their corresponding **100 segmentation masks**.  

**Structure:**  
**Dataset Structure:**
- **data/**
  - **train/**
    - **images/** → 80 training images
    - **mask/** → 80 segmentation masks
  - **test/**
    - **images/** → 20 test images
    - **mask/** → 20 segmentation masks



  
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/RetinaBloodVessel_AttentionUNet_Seg/Notebook/RetinalBloodVessel_Segmentation_AttentionUNet.ipynb)

---

## 📈 Training Progress

![Training vs Validation Loss & Dice](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/RetinaBloodVessel_AttentionUNet_Seg/Output/download%20(3).png)

---

## 🏆 Results

After **39 epochs**, the model achieved **excellent segmentation performance**:

| Dataset   | Loss   | Dice   | IoU    | Accuracy | AUC    |
|-----------|--------|--------|--------|-----------|--------|
| **Train** | 0.4865 | 0.8082 | 0.6787 | 0.9658    |   —    |
| **Test**  | 0.4802 | 0.8129 | 0.6850 | 0.9674    | 0.9727 |

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

