# 📡 UltraSeg-Bench: AbdomenUSMS Comparative Study

[![Open In Colab — U-Net](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GH_USER>/ultraseg-bench/blob/main/notebooks/unet_abdomenus.ipynb)
[![Open In Colab — Attention U-Net](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GH_USER>/ultraseg-bench/blob/main/notebooks/attunet_abdomenus.ipynb)
[![Open In Colab — DeepLabV3+](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GH_USER>/ultraseg-bench/blob/main/notebooks/deeplabv3p_abdomenus.ipynb)

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-ff69b4)

>  A small, carefully controlled **comparative study** of three segmentation baselines — **U-Net**, **Attention U-Net**, and **DeepLabV3+** — on the **AbdomenUSMSBench** (ultrasound) subset of **MedSegBench**. We focus on **reproducibility**, **fairness**, and **clear reporting** (val-selected best checkpoints, identical splits/metrics).

---

## ✨ Why this study?

Ultrasound segmentation is **hard**: speckle noise, low contrast, and small/ambiguous organs make generalization tricky. We wanted a **practical, Colab-friendly baseline** that:
- compares **classic U-Net**, **attention-gated U-Net**, and **DeepLabV3+** under the **same pipeline**,
- uses **val mDice** to select best checkpoints and reports **test** performance fairly,
- is **reproducible** (fixed seeds, deterministic settings where possible).

---

## 📦 Dataset

**AbdomenUSMSBench** (MedSegBench) — abdominal **ultrasound** multi-structure segmentation.  
- Format: images with pixel-wise class labels (background + organs).  
- We use the **256×256** preprocessed NPZ release for Colab friendliness.  
- **Citation / Source:** MedSegBench repository + Zenodo data release (AbdomenUS 256).  
  - Zenodo data file (as used here): `abdomenus_256.npz`  
  - Zenodo record link (download): https://zenodo.org/records/13358372/files/abdomenus_256.npz?download=1  
  - Project page: https://github.com/zekikus/MedSegBench

> ⚠️ Please respect the dataset’s license and terms. This repo **does not** redistribute data; the notebooks **download directly** from the official Zenodo link at runtime.

**Why it’s challenging:** ultrasound artifacts (speckle), low SNR, fuzzy boundaries, and **class imbalance** (background dominance, small/rare organs). These factors penalize models that lack strong context modeling or robust regularization.

---

## 🧪 Methods (all Colab-ready)

- **U-Net** (scratch)  
- **Attention U-Net** (scratch; attention gating on skip connections)  
- **DeepLabV3+** (ImageNet-pretrained ResNet-50 backbone; ASPP multi-scale context)

All three share the **same steps**:
1. Setup (installs, imports), **seeding** (Python/NumPy/Torch), deterministic cuDNN.
2. Manual dataset fetch + **MD5** check to avoid flaky auto-download.
3. Load **train/val/test** splits; count samples.
4. Visualize images, masks, overlays (legendless for simplicity).
5. **NumPy-safe** paired augmentations (flips; optional light affine/intensity).
6. Torch `Dataset`/`DataLoader` (conditional `pin_memory`, `drop_last` for BN-safety).
7. **Losses:** class-weighted CrossEntropy (inverse frequency) + **soft Dice**.  
   **Metrics:** mIoU, mDice (ignore background in metrics).
8. **Training:** AMP (guarded), AdamW, ReduceLROnPlateau, early stopping, **best-on-val-Dice** checkpointing.
9. **Evaluation:** test metrics (optionally with horizontal-flip **TTA**).  
10. **Visualization:** qualitative predictions (overlay).  
11. **Reproducibility:** history saved (`.npz`, `.json`) + plotting cells.

---

## 🚀 Quick start (Colab)

Click a badge at the top, or open:

- U-Net: `notebooks/unet_abdomenus.ipynb`  
- Attention U-Net: `notebooks/attunet_abdomenus.ipynb`  
- DeepLabV3+: `notebooks/deeplabv3p_abdomenus.ipynb`

> Replace `<YOUR_GH_USER>` in the badge links with your GitHub username after you push.

---

## 📊 Results

### 1) Best-Checkpoint Summary (Validation-selected → Tested on held-out test set)

| Model | Val mDice | Val mIoU | Test mDice | Test mIoU | Test Loss |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.5919 | 0.5051 | 0.4650 | 0.3855 | 0.8516 |
| **Attention U-Net** | 0.6227 | 0.5398 | **0.5674** | **0.4906** | 0.8844 |
| DeepLabV3+ | **0.6782** | **0.6019** | 0.5540 | 0.4826 | 0.8677 |

**Reading the table:** We select the **best checkpoint by validation mDice** for each model, then compute **test** metrics with the same evaluation settings. **Attention U-Net** generalizes best on this dataset (highest **test mDice/mIoU**), while **DeepLabV3+** shows stronger **val** performance but a larger generalization gap. **U-Net** (baseline) trails both.

---

### 2) Generalization Gaps (smaller is better)

| Model | Train→Test ΔDice | Val→Test ΔDice | Train→Test ΔIoU | Val→Test ΔIoU |
|---|---:|---:|---:|---:|
| U-Net | 0.6513 → 0.4650 = **0.1863** | 0.5919 → 0.4650 = **0.1269** | 0.5649 → 0.3855 = **0.1794** | 0.5051 → 0.3855 = **0.1196** |
| **Attention U-Net** | 0.6705 → 0.5674 = **0.1031** | 0.6227 → 0.5674 = **0.0553** | 0.5868 → 0.4906 = **0.0962** | 0.5398 → 0.4906 = **0.0492** |
| DeepLabV3+ | 0.8850 → 0.5540 = **0.3310** | 0.6782 → 0.5540 = **0.1242** | 0.8581 → 0.4826 = **0.3755** | 0.6019 → 0.4826 = **0.1193** |

**Interpretation:** **Attention U-Net** overfits the least (smallest gaps), **DeepLabV3+** overfits the most (despite high train/val scores).

---

### 3) Efficiency

| Model | Total Epochs Run | Best Epoch |
|---|---:|---:|
| U-Net | 100 | 79 |
| **Attention U-Net** | 57 | 49 |
| DeepLabV3+ | 31 | 31 |

**Interpretation:** With early stopping/scheduling, **Attention U-Net** converges efficiently; **DeepLabV3+** reaches best quickly but generalizes slightly worse than AttU-Net on test.

---

## 🖼️ Visuals (qualitative)

![Predict vs Truth — Example#1](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/PredictVsTruth/download%20(10).png?raw=1)

![Predict vs Truth — Example#2](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/PredictVsTruth/download%20(11).png?raw=1)

![Predict vs Truth — Example#3](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/PredictVsTruth/download%20(12).png?raw=1)


---

## 🔁 Reproducibility

- Fixed seeds for `random`, `numpy`, `torch` (+ CUDA)  
- cuDNN deterministic + benchmarking disabled  
- History saved (`.npz`, `.json`) for all runs  
- Colab notebooks include **step-by-step**, **commented** cells

---

## 🧱 Repository Structure (suggested)



