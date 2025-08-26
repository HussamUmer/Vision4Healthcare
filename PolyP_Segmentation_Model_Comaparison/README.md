# 🩺 Polyp Segmentation with U-Net and TransUNet on Kvasir-SEG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/your-repo/blob/main/notebook.ipynb)

## 📌 Project Overview

This project compares two powerful deep learning architectures — **U-Net** (a classic CNN-based segmentation model) and **TransUNet** (a hybrid CNN + Transformer model) — for **polyp segmentation** on the **Kvasir-SEG dataset**.  

The goal is to evaluate how traditional convolutional models and transformer-augmented models perform on a small, challenging medical imaging dataset, both in terms of **accuracy** and **efficiency**.

---

## 📂 Dataset

We use the **Kvasir-SEG** dataset:  
- **1,000** colonoscopy images with **pixel-level annotations** of polyps.  
- Images are RGB, with varying resolutions.  
- Publicly available via Kaggle: [Kvasir-SEG Dataset](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)  

**Citation (please cite if you use this dataset):**
> Debesh Jha, Pia H. Smedsrud, Michael A. Riegler, Dag Johansen, Thomas de Lange, Pål Halvorsen, Håvard D. Johansen.  
> *Kvasir-SEG: A Segmented Polyp Dataset*.  
> In International Conference on Multimedia Modeling (MMM), 2020.  
> DOI: [10.1007/978-3-030-37734-2_33](https://doi.org/10.1007/978-3-030-37734-2_33)

---

## ⚙️ Methodology

We trained **two models under the same experimental setup** for fairness:

### 1. Preprocessing
- Resize all images and masks to **256×256**.  
- Normalize using ImageNet mean/std.  
- Augmentations: random flips, rotations, elastic transforms, brightness/contrast jitter.  
- Dataset split: **70% train / 15% validation / 15% test**.

### 2. U-Net (Baseline CNN)
- Classic encoder-decoder with skip connections.  
- Encoder: 4 downsampling stages.  
- Loss: **Dice + BCE**.  
- Optimizer: AdamW (lr=1e-3).  
- Early stopping on validation Dice.  

### 3. TransUNet (CNN + Transformer)
- Hybrid model with ResNet/ViT encoder + U-Net style decoder.  
- CNN extracts low-level features → Transformer encoder captures long-range dependencies → Decoder upsamples with skip connections.  
- Same training settings as U-Net for fair comparison.  

### 4. Evaluation
- Metrics: **Dice Coefficient, IoU, Precision, Recall, Boundary F1 (BF-score)**.  
- Efficiency: **Params (M), FLOPs (G), Model Size (MB), CPU Latency (ms/frame)**.  
- Visuals: side-by-side **Image | Ground Truth | U-Net Prediction | TransUNet Prediction**.

---

## 🚀 Running the Project

### 1. Clone the repo
```bash
git clone https://github.com/HussamUmer/Vision4Healthcare.git
cd Vision4Healthcare

