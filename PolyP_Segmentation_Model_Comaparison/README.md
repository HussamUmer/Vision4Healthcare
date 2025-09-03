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

## 📊 Results

We compared **U-Net** and **TransUNet** on the Kvasir-SEG dataset under the same experimental setup.

---

### 🔹 Training & Validation Performance

| Model      | Train Loss ↓ | Train Dice ↑ | Train IoU ↑ | Train BF ↑ | Val Loss ↓ | Val Dice ↑ | Val IoU ↑ | Best Epoch |
|------------|--------------|--------------|-------------|------------|------------|------------|-----------|------------|
| **U-Net**      | 0.1645       | 0.8934       | 0.8227      | 0.6291     | 0.2544     | 0.8393     | 0.7574    | 77 / 100   |
| **TransUNet**  | 0.1997       | 0.7580       | 0.6470      | –          | 0.1959     | 0.7721     | 0.6583    | 10 / 100   |

📈 **Graph:**  
- The following graph shows the **training and validation loss, Dice, and IoU** for both **U-Net** and **TransUNet** across epochs:

![Metrics Graph](https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/Comparative/Metrics.png)

---

### 🔹 Test Performance

| Model      | Test Loss ↓ | Test Dice ↑ | Test IoU ↑ |
|------------|-------------|-------------|------------|
| **U-Net**      | 0.2825      | 0.8256      | 0.7338     |
| **TransUNet**  | 0.1351      | 0.8200      | 0.7342     |

📈 **Graphs:**  
- [U-Net Test Metrics](assets/unet_test_metrics.png)  
- [TransUNet Test Metrics](assets/transunet_test_metrics.png)  

---

### 🔹 Qualitative Comparison

Below are sample predictions showing **Image → Ground Truth → U-Net Prediction → TransUNet Prediction**:

| Image | Ground Truth | U-Net Prediction | TransUNet Prediction |
|-------|--------------|------------------|----------------------|
| ![](assets/sample_img.png) | ![](assets/sample_gt.png) | ![](assets/sample_unet.png) | ![](assets/sample_transunet.png) |

📸 More visual comparisons: [Qualitative Results](assets/qualitative_comparison.png)

---

### 📝 Key Observations
- **U-Net** achieved **higher training Dice (0.8934)** and slightly better **validation Dice (0.8393)**, but had higher test loss.  
- **TransUNet** converged faster (best at epoch 10) and achieved **lower test loss (0.1351)**, while Dice and IoU were very close to U-Net.  
- **Both models generalize well** with nearly identical Dice (~0.82) and IoU (~0.73) on the test set.  
- Visual comparisons show **U-Net** captures boundaries more smoothly in some cases, while **TransUNet** segments harder polyps with lower loss.  

---





---
## 🚀 Running the Project

### 1. Clone the repo
```bash
git clone https://github.com/HussamUmer/Vision4Healthcare.git
cd Vision4Healthcare

