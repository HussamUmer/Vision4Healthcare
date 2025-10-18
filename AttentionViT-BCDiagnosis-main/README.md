# Attention-Based Vision Transformer for Breast Cancer Diagnosis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-BreakHis-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-purple)

## 📌 Overview

This repository contains the official implementation of our research paper:

> **Attention-Based Vision Transformers for Enhanced Breast Cancer Diagnosis**  
> *Sana Ullah Khan, Bakht Azam, Hussam Umer*  

In this work, we propose a **self-attention-based Vision Transformer (ViT)** architecture tailored for **multi-class breast cancer histopathology classification** using the **BreakHis dataset**.  
Our model addresses the limitations of traditional CNNs in capturing **long-range dependencies** by leveraging the transformer’s **global context modeling**, achieving:

| Magnification | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| 40X           | 96.25%   | 96.54%    | 96.25% | 96.25%   |
| 100X          | 96.67%   | 96.86%    | 96.67% | 96.66%   |
| 200X          | 96.05%   | 96.11%    | 96.05% | 96.03%   |
| 400X          | 96.25%   | 96.27%    | 96.25% | 96.22%   |
| **Average**   | **96.305%** | **96.305%** | **96.305%** | **96.305%** |

---

## 📂 Dataset

We used the **BreakHis histopathology dataset** containing 7,909 images of benign and malignant tumors across **four magnifications**: 40X, 100X, 200X, and 400X.

**Dataset Preparation:**
- For **each class** at **each magnification level**:
  - **70 images** for training  
  - **30 images** for testing  
- Ensures balanced data for **fair and consistent evaluation**.

---

## 🧠 Methodology

Our proposed approach:

1. **Image Preprocessing**  
   - Color normalization  
   - Data augmentation  
   - Magnification-specific resizing (224×224)

2. **Patch Embedding**  
   - Splits images into patches & embeds them into a sequence.

3. **Self-Attention Mechanism**  
   - Captures both local and global dependencies.

4. **Classification Head**  
   - Multi-Layer Perceptron (MLP) with ReLU activation & dropout.

5. **Evaluation**  
   - Accuracy, Precision, Recall, and F1-Score.


---

## Run notebooks in Google Colab

| Magnification | Colab Link |
|---------------|------------|
| **40X**       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/AttentionViT-BCDiagnosis/blob/main/Notebook%20Files/40x_multiclassification_vit.ipynb) |
| **100X**      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/AttentionViT-BCDiagnosis/blob/main/Notebook%20Files/100x_multiclassification_vit.ipynb) |
| **200X**      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/AttentionViT-BCDiagnosis/blob/main/Notebook%20Files/200x_multiclassification_vit.ipynb) |
| **400X**      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/AttentionViT-BCDiagnosis/blob/main/Notebook%20Files/400x_multiclassification_vit.ipynb) |

---

## 📊 Results

Here are the results across each magnification:

### 📉 Confusion Matrices for All Magnifications

| **40X** | **100X** |
|---------|----------|
| ![40X Confusion Matrix](Results%20and%20Outputs/Confusion%20Matrics/40x.png) | ![100X Confusion Matrix](Results%20and%20Outputs/Confusion%20Matrics/100x.png) |

| **200X** | **400X** |
|----------|----------|
| ![200X Confusion Matrix](Results%20and%20Outputs/Confusion%20Matrics/200x.png) | ![400X Confusion Matrix](Results%20and%20Outputs/Confusion%20Matrics/400x.png) |

### 📊 Performance Metrics

Below are the classification metric graphs (Accuracy, Precision, Recall, F1-Score) for each magnification level.

| **40X** | **100X** |
|---------|----------|
| ![Performance Metrics - 40X](Results%20and%20Outputs/Classification%20Metrics/400x%20Metrics.png) | ![Performance Metrics - 100X](Results%20and%20Outputs/Classification%20Metrics/100x%20Metrics.png) |

| **200X** | **400X** |
|----------|----------|
| ![Performance Metrics - 200X](Results%20and%20Outputs/Classification%20Metrics/200x%20Metrics.png) | ![Performance Metrics - 400X](Results%20and%20Outputs/Classification%20Metrics/400x%20Metrics.png) |

### 📈 Model Comparison

The following graph compares our proposed **Attention-Based ViT** with several state-of-the-art models for breast cancer classification.

![Model Comparison](Results%20and%20Outputs/Comparative%20Graph/msedge_JdxRUbOpg2.png)

**Key Insight:**  
Our proposed model achieves **96.305%** average accuracy across all magnifications, outperforming many existing CNN-based and hybrid architectures.

---

## 📬 Contact
For any questions regarding the code, dataset, or paper:

- **Hussam Umer** – [hussamumer28092000@gmail.com](mailto:hussamumer28092000@gmail.com)

📄 **Note:**  
The related research paper is currently **under review**. 

