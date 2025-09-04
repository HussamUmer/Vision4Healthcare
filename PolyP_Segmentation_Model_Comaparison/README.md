# 🩺 Polyp Segmentation with U-Net and TransUNet on Kvasir-SEG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/your-repo/blob/main/notebook.ipynb)

## 📌 Project Overview

This project compares two powerful deep learning architectures — **U-Net** (a classic CNN-based segmentation model) and **TransUNet** (a hybrid CNN + Transformer model) — for **polyp segmentation** on the **Kvasir-SEG dataset**.  

The goal is to evaluate how traditional convolutional models and transformer-augmented models perform on a small, challenging medical imaging dataset, both in terms of **accuracy** and **efficiency**.

---

## ❓ Why This Study?

Polyp detection and segmentation are critical for **early diagnosis of colorectal cancer**, one of the leading causes of cancer-related deaths worldwide.  
Accurate polyp segmentation can assist clinicians during colonoscopy by highlighting suspicious regions in real-time, reducing the risk of missed polyps.  

This study was carried out to:  
- Compare a **classical CNN-based approach (U-Net)** with a **hybrid Transformer-based approach (TransUNet)**.  
- Understand how **different architectures generalize** on a small but challenging medical dataset (Kvasir-SEG).  
- Provide an open, reproducible benchmark to guide future research in **medical image segmentation**.


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
>
#### 📂 Dataset Split

| Split | Number of Images |
|-------|------------------|
| **Train** | 699 |
| **Validation** | 151 |
| **Test** | 150 |
| **Total** | 1000 |



#### 🗂️ Dataset Sample

Below is an example from the **Kvasir-SEG dataset**, showing colonoscopy images alongside their corresponding ground-truth polyp masks:

<img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/Dataset/download%20(3).png?raw=1" width="600">

*Kvasir-SEG provides 1,000 annotated colonoscopy images for polyp segmentation.*

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

The following graph illustrates the **training duration and best epoch selection** for both models (U-Net and TransUNet):

![Epochs Graph](https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/Comparative/Epochs.png)

---

### 🔹 Test Performance

| Model      | Test Loss ↓ | Test Dice ↑ | Test IoU ↑ |
|------------|-------------|-------------|------------|
| **U-Net**      | 0.2825      | 0.8256      | 0.7338     |
| **TransUNet**  | 0.1351      | 0.8200      | 0.7342     |

📈 **Graph:**  
- The following graph shows the **comparison of test metrics (Loss, Dice, IoU)** between **U-Net** and **TransUNet**:

![Test Metrics](https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/Test/newplot.png)

---

## 📈 Epoch-wise Metric Comparisons (Train & Val)

### mDice per epoch (higher is better)  
| U-Net | TransUNet |
|---|---|
| <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/UNet/Dice.png?raw=1" width="400"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/TransUNet/Dice.png?raw=1" width="400"> |

*Dice reflects segmentation overlap quality; higher values indicate better mask alignment.*

---

### mIoU per epoch (higher is better)  
| U-Net | TransUNet |
|---|---|
| <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/UNet/IOU.png?raw=1" width="400"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/TransUNet/IOU.png?raw=1" width="400"> |

*IoU complements Dice; trends across epochs highlight model stability and generalization.*

---

### Loss per epoch (lower is better)  
| U-Net | TransUNet |
|---|---|
| <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/UNet/loss.png?raw=1" width="400"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/TransUNet/loss.png?raw=1" width="400"> |

*Loss curves reflect optimization behavior; stable convergence indicates reliable learning.*


---

### 📸 Qualitative Comparison

Below are sample predictions showing **Image → Ground Truth → U-Net Prediction → TransUNet Prediction**:

| U-Net | TransUNet |
|---|---|
| <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/PredictedVsTruth/unet.png?raw=1" width="450"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Output/PredictedVsTruth/transunet.png?raw=1" width="450"> |

*Side-by-side comparison of predicted masks vs. ground truth — U-Net vs. TransUNet.*

---

### 📝 Key Observations
- **U-Net** achieved **higher training Dice (0.8934)** and slightly better **validation Dice (0.8393)**, but had higher test loss.  
- **TransUNet** converged faster (best at epoch 10) and achieved **lower test loss (0.1351)**, while Dice and IoU were very close to U-Net.  
- **Both models generalize well** with nearly identical Dice (~0.82) and IoU (~0.73) on the test set.  
- Visual comparisons show **U-Net** captures boundaries more smoothly in some cases, while **TransUNet** segments harder polyps with lower loss.  

---

## ▶️ Run the Notebooks in Google Colab


### Direct “Open in Colab” badges

<!-- U-Net notebook -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Notebooks/Kvasir_SEG_UNet.ipynb)

<!-- TransUNet notebook -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/PolyP_Segmentation_Model_Comaparison/Notebooks/Kvasir_SEG_TransUNet.ipynb)


---
## 🚀 Running the Project

### 1. Clone the repo
```bash
git clone https://github.com/HussamUmer/Vision4Healthcare.git
cd Vision4Healthcare

