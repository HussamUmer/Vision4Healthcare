# 📡 UltraSeg-Bench: AbdomenUSMS Comparative Study

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
Here is a single sample from each set:

>Train Sample:

![Predict vs Truth — Example#1](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/Dataset/download%20(2).png?raw=1)

>Validation Sample:
![Predict vs Truth — Example#2](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/Dataset/download%20(3).png?raw=1)


>Test Sample:
![Predict vs Truth — Example#2](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/Dataset/download%20(4).png?raw=1)

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

Click a badge to open notebooks in colab:

- U-Net:
[![Open In Colab — AbdomenUSMSBench Attention U-Net](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Notebooks/AbdomenUSMSBench_UNet.ipynb)
- Attention U-Net: 
[![Open In Colab — AbdomenUSMSBench U-Net](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Notebooks/AbdomenUSMSBench_AttentionUNet.ipynb)
- DeepLabV3+:
[![Open In Colab — AbdomenUSMSBench DeepLabV3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Notebooks/AbdomenUSMSBench_DeepLabV3.ipynb)



---

## 📊 Results

### 1) Best-Checkpoint Summary (Validation-selected → Tested on held-out test set)

| Model | Val mDice | Val mIoU | Val Loss | Test mDice | Test mIoU | Test Loss |
|---|---:|---:|---:|---:|---:| ---:|
| U-Net | 0.6059 | 0.5075 | 0.6035 |0.4868 | 0.4056 | 0.7915 |
| DeepLabV3+ | 0.6934 | 0.6186 | 0.5665 | 0.5565 | 0.4821 | 0.8345|
| Attention U-Net | 0.6227 | 0.5398 | 0.7008 | 0.5674 | 0.4906 | 0.8844|

**Reading the table:** We select the **best checkpoint by validation mDice** for each model, then compute **test** metrics with the same evaluation settings. **Attention U-Net** generalizes best on this dataset (highest **test mDice/mIoU**), while **DeepLabV3+** shows stronger **val** performance but a larger generalization gap. **U-Net** (baseline) trails both.

---

### 2) Generalization Gaps (smaller is better)

| Model | Train→Test ΔDice | Val→Test ΔDice | Train→Test ΔIoU | Val→Test ΔIoU |
|---|---:|---:|---:|---:|
| U-Net | 0.6927 → 0.4868 = **0.2059** | 0.6059 → 0.4868 = **0.1191** | 0.6087 → 0.4056 = **0.2031** | 0.5075 → 0.4056 = **0.1019** |
| **Attention U-Net** | 0.6705 → 0.5674 = **0.1031** | 0.6227 → 0.5674 = **0.0553** | 0.5868 → 0.4906 = **0.0962** | 0.5398 → 0.4906 = **0.0492** |
| DeepLabV3+ | 0.8531 → 0.5565 = **0.2966** | 0.6934 → 0.5565 = **0.1369** | 0.8102 → 0.4821 = **0.3281** | 0.6186 → 0.4821 = **0.1365** |

**Interpretation**: **Attention U-Net** has the smallest gaps across Dice and IoU, indicating the strongest generalization among the three under my current setup.

---

### 3) Efficiency

| Model | Total Epochs Run | Best Epoch |
|---|---:|---:|
| U-Net | 100 | 99 | 
| **Attention U-Net** | 57 | 49 |
| DeepLabV3+ | 46 | 26 |

**Interpretation:** With early stopping/scheduling, **Attention U-Net** converges efficiently; **DeepLabV3+** reaches best quickly but generalizes slightly worse than AttU-Net on test.

---

## 🖼️ Visuals (qualitative)

Below are representative cases from the test split. For each case we show the same image across models so we can compare boundaries, small structures, and failure modes at a glance. Overlays use a consistent class color palette; background is hidden.

*These visuals are illustrative and complement the quantitative metrics; they highlight where models agree, where boundaries are uncertain, and which organs are most challenging.*

![Predict vs Truth — Example#1](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/PredictVsTruth/download%20(10).png?raw=1)

![Predict vs Truth — Example#2](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/PredictVsTruth/download%20(11).png?raw=1)

![Predict vs Truth — Example#3](https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/PredictVsTruth/download%20(12).png?raw=1)


---

## 📈 Epoch-wise Metric Comparisons (Train & Val)

### mDice per epoch (higher is better)
| U-Net | Attention U-Net | DeepLabV3+ |
|---|---|---|
| <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/UNet/download%20(9).png?raw=1" width="320"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/AttUNet/download%20(13).png?raw=1" width="320"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/DeepLabV3/download%20(6).png?raw=1" width="320"> |

*Shows segmentation quality across training; train–val gaps hint at under/overfitting.*

---

### mIoU per epoch (higher is better)
| U-Net | Attention U-Net | DeepLabV3+ |
|---|---|---|
| <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/UNet/download%20(8).png?raw=1" width="320"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/AttUNet/download%20(3).png?raw=1" width="320"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/DeepLabV3/download%20(5).png?raw=1" width="320"> |

*mIoU complements mDice; consistent trends reinforce conclusions about stability and generalization.*

---

### Loss per epoch (lower is better)
| U-Net | Attention U-Net | DeepLabV3+ |
|---|---|---|
| <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/UNet/download%20(7).png?raw=1" width="320"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/AttUNet/download%20(2).png?raw=1" width="320"> | <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/UltraSeg-Bench%3A%20AbdomenUSMS%20Comparative%20Study/Output/TrainValGraphs/DeepLabV3/download%20(4).png?raw=1" width="320"> |

*Loss curves reflect optimization behavior; widening train–val gaps indicate overfitting, smooth plateaus indicate convergence.*



---

## 🔁 Reproducibility

- Fixed seeds for `random`, `numpy`, `torch` (+ CUDA)  
- cuDNN deterministic + benchmarking disabled  
- History saved (`.npz`, `.json`) for all runs  
- Colab notebooks include **step-by-step**, **commented** cells

---

## 🚧 Limitations & Next Steps

- Ultrasound segmentation remains challenging; small/rare organs and fuzzy boundaries depress Dice/IoU.

- DeepLabV3+ shows strong capacity but needs stronger regularization.

Next steps:

- Try Tversky/Focal-Tversky mixes; class-balanced sampling.

- 512px training or 256→512 fine-tuning (VRAM permitting).

- Pretrained encoders for U-Net/Attention U-Net for a more apples-to-apples backbone comparison.

- Per-class deep-dive and targeted augmentations (elastic, intensity, speckle).

---

## 📚 Citations & Acknowledgments

Dataset: MedSegBench — AbdomenUSMSBench (Abdomen ultrasound multi-structure).
Zenodo (data file used): abdomenus_256.npz — https://zenodo.org/records/13358372/files/abdomenus_256.npz?download=1

### Project: https://github.com/zekikus/MedSegBench

### Architectures:

```

- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015).

- Oktay et al., Attention U-Net (ArXiv 2018).

- Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+) (ECCV 2018).

If you use this repository, please cite the dataset and the original method papers above.

```

---

## 📄 License

This project is released under the MIT License. See LICENSE
 for details.



