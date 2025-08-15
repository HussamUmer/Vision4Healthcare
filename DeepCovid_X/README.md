# 🩺 DeepCOVID-X: Comparative Analysis of CNN Architectures on COVID-19 Radiography

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Colab](https://img.shields.io/badge/Run%20on-Colab-orange?logo=googlecolab)
![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)

---

## 📖 Project Overview
This repository presents a **comparative performance analysis** of four different deep learning architectures for detecting **COVID-19, Viral Pneumonia, and Normal** chest X-ray images using the **COVID-19 Radiography Dataset** (customized to 1,345 images per class).

We fine-tuned:
- 🧠 **Simple CNN** (custom architecture)
- ⚡ **EfficientNet-B0** (ImageNet-pretrained)
- 📱 **MobileNetV2** (ImageNet-pretrained)
- 🏛 **VGG16** (ImageNet-pretrained)

Each model was trained, validated, and tested using the **exact same data splits and preprocessing** to ensure fairness in comparison.

---

## 🚀 Quick Start

### Open in Google Colab

| Model | Colab Link |
|-------|------------|
| 🧠 Simple CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Colab%20Notebooks/CNN_Covid_Radiography.ipynb) |
| ⚡ EfficientNet-B0 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Colab%20Notebooks/EfficientNet_Covid_Radiography.ipynb) |
| 📱 MobileNetV2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Colab%20Notebooks/MobileNet_Covid_Radiography.ipynb) |
| 🏛 VGG16 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Colab%20Notebooks/VGG16_Covid_Radiography.ipynb) |

---

## 📂 Dataset Overview

**Source:** [COVID-19 Radiography Dataset – Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)  
**Custom Setup:** Balanced to 1,345 images per class.

| Class            | Train | Validation | Test | Total |
|------------------|-------|------------|------|-------|
| COVID-19         | 941   | 202        | 202  | 1345  |
| Normal           | 941   | 202        | 202  | 1345  |
| Viral Pneumonia  | 941   | 202        | 202  | 1345  |
| **Total Images** | 2823  | 606        | 606  | 4035  |

**Split Ratios:** 70% Train, 15% Validation, 15% Test  
**Image Size:** 224 × 224 pixels (normalized with ImageNet stats)  
**Augmentation:** Resize + Normalize

---

## 📊 Results & Analysis

### 1️⃣ Confusion Matrices (Test Set)


| **Simple CNN** | **EfficientNet-B0** |
|----------------|----------------------|
| ![Simple CNN CM](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ConfusionMatrices/cnncm.png) | ![EfficientNet-B0 CM](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ConfusionMatrices/efficientnetcm.png) |

| **MobileNetV2** | **VGG16** |
|------------------|------------|
| ![MobileNetV2 CM](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ConfusionMatrices/Mobilenetcm.png) | ![VGG16 CM](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ConfusionMatrices/vgg16cm.png) |


---

### 2️⃣ Overall Classification Metrics

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------|--------------|---------------|------------|--------------|
| 🧠 Simple CNN | 95.87 | 95.92 | 95.87 | 95.87 |
| ⚡ EfficientNet-B0 | 97.69 | 97.17 | 97.69 | 97.23 |
| 📱 MobileNetV2 | 97.03 | 97.13 | 97.03 | 97.04 |
| 🏛 VGG16 | 97.03 | 97.07 | 97.03 | 97.02 |

---

### 3️⃣ Classification Performance Metrics Visuals

| **Simple CNN** | **EfficientNet-B0** |
|----------------|----------------------|
| ![Simple CNN Performance](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ClassificationPerformanceMetrics/cnnclass.png) | ![EfficientNet-B0 Performance](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ClassificationPerformanceMetrics/efficientnetclass.png) |

| **MobileNetV2** | **VGG16** |
|------------------|------------|
| ![MobileNetV2 Performance](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ClassificationPerformanceMetrics/mobilenetclass.png) | ![VGG16 Performance](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/ClassificationPerformanceMetrics/vgg16class.png) |


---

### 4️⃣ Performance Statistics


| Model | Training Time (s) | Peak GPU (MB) | Testing Time (s) | Test GPU (MB) |
|-------|-----------------|---------------|-----------------|---------------|
| 🧠 Simple CNN | 801.39 | 1820.73 | 108.79 | 1820.73 |
| ⚡ EfficientNet-B0 | 235.97 | 2833.55 | 270.72 | 417.83 |
| 📱 MobileNetV2 | 1009.05 | 2503.03 | 107.11 | 391.01 |
| 🏛 VGG16 | 767.30 | 4272.32 | 102.36 | 4272.32 |


---

### 5️⃣ 📊 Performance Comparison

The following plot shows a comparative overview of the performance metrics (accuracy, precision, recall, F1-score, training/testing time, and GPU usage) across all models:

![Performance Comparison](https://github.com/HussamUmer/Vision4Healthcare/blob/main/DeepCovid_X/Results_Graphs/Performance/newplot.png)


---








---

### 6️⃣ Overall Performance Graph

*(Insert comparative accuracy/precision/recall/F1 graph here)*  
![Overall Metrics Graph](path/to/overall_graph.png)

---
### 7️⃣ Other Metrics

#### 📊 t-SNE Visualizations

To check the **t-SNE visualizations** of all models, [click here](https://github.com/HussamUmer/Vision4Healthcare/tree/main/DeepCovid_X/Results_Graphs/tsne).


#### 📈 Confidence Distribution

To check the **Confidence Distribution graphs** of all models, [click here](https://github.com/HussamUmer/Vision4Healthcare/tree/main/DeepCovid_X/Results_Graphs/ConfidenceDistribution).


---

## 📜 Methodology

1. **Imports & Environment Setup** – PyTorch, Torchvision, Matplotlib, Seaborn, NumPy, PIL  
2. **Data Loading & Preprocessing** – Stratified Train/Val/Test split (70/15/15)  
3. **Data Augmentation** – Resize + Normalize (ImageNet stats)  
4. **Model Selection & Customization** – Load pretrained weights, modify classifier layers  
5. **Training & Validation** – Track loss, accuracy, GPU memory, and time  
6. **Testing & Evaluation** – Accuracy, classification report, confusion matrix  
7. **Visualization** – Sample predictions, misclassifications, t-SNE plots, confidence histograms  
8. **Result Comparison** – Metrics and resource usage across models

---

## 📌 Key Insights
- Pretrained models (EfficientNet, MobileNet, VGG16) generally outperform the custom CNN.
- MobileNetV2 offers an excellent trade-off between **accuracy and speed**.
- t-SNE visualizations reveal clear class separation in higher-performing models.
- Misclassifications often occur between **COVID-19 and Viral Pneumonia**.

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements
- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)  
- PyTorch & Torchvision Teams  
- Google Colab for free GPU access


