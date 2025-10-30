# UNet Family vs DeepLabV3+: Strong, Efficient Baselines on ISIC-2016

> **Status:** 🚧 _Under construction — figures coming soon_  
> **Goal:** Reproduce and compare three classic, high-performing 2D segmentation models (**UNet, UNet++, DeepLabV3+**) on **ISIC-2016** using one unified, research-grade pipeline where **only Step 6 (the model block)** changes.

![Task](https://img.shields.io/badge/Task-Segmentation-blue)
![Dataset](https://img.shields.io/badge/Dataset-ISIC2016-orange)
![Models](https://img.shields.io/badge/Models-UNet%20%7C%20UNet%2B%2B%20%7C%20DeepLabV3%2B-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch-black)
![Pipeline](https://img.shields.io/badge/Pipeline-Plug--and--Play-purple)

---

## ✨ Why this project?

Strong, **transparent baselines** matter. This repo delivers:

- **One plug-and-play pipeline**: identical data handling, losses, metrics, logging & evaluation — **swap only Step 6** to change the model.
- **Apples-to-apples comparisons**: same training and evaluation policy; we report accuracy, loss, speed, params, MACs, and memory.
- **Ready for models**: paste model into Step 6 (forward API: `model(x) → {"logits": (B,1,H,W)}`) and reuse everything else.

---

## 📚 Dataset

**ISIC-2016** (binary dermoscopy lesion segmentation)

- Standardized NPZ (e.g., **256×256**) with predefined **train / val / test** splits  
  _Typical counts: Train ≈ 810, Val ≈ 90, Test ≈ 379._
- Expected MedSegBench-style cache:
  - `MEDSEGBENCH_DIR=/path/to/cache`
  - File: `isic2016_256.npz` (or `..._512.npz` if you choose 512)

> ⚠️ On Colab: keep the dataset cache on **runtime (local)** for speed and store artifacts on **Drive**.

---

## 🧱 Models (Step 6 only)

- **UNet** — classic encoder–decoder with skip connections; fast and stable.
- **UNet++** — nested dense skip pathways for crisper boundaries.
- **DeepLabV3+** — ASPP for multi-scale context + light decoder.

> All implemented to match the pipeline’s forward API: `{"logits": (B,1,H,W)}`

---

## 🧪 Experimental setup (shared across models)

- **Image size:** 256×256 (512×512 optional)
- **Loss:** Dice + BCE (e.g., 0.7 / 0.3)
- **Metrics:** Dice, IoU (fixed threshold); PR/ROC (AUPRC/AUROC) optional
- **Training:** AdamW, cosine LR with warmup, AMP on GPU
- **Logging:** per-epoch CSV, best-by-val-Dice checkpointing, early stopping
- **Reports:** default **test** evaluation; optional train/val for completeness
- **Resources:** Params (M), MACs (G @ 256²), peak VRAM (MB), median latency (ms/img, batch=1)

---

## 📊 Results

### Table 1 — **Test Dice & IoU**
| Model       | Test Dice | Test IoU |
|-------------|-----------|----------|
| UNet        | 0.9144    | 0.8517   |
| UNet++      | **0.9194**| **0.8577** |
| DeepLabV3+  | 0.9158    | 0.8531   |

<h3 align="center">Figure 1 — Test Dice & IoU (bar chart)</h3>
<p align="center">
  <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/SegCompare_ISIC2016_UNetFamily_DeepLabV3Plus/output_figures/graphs/newplot.png" alt="Test Dice & IoU comparison" width="900">
</p>
<p align="center"><i>Figure 1. Test Dice & IoU comparison (bar chart).</i></p>

**Takeaways**
- **UNet++** tops both Dice and IoU in our runs.  
- **DeepLabV3+** is a close second; **UNet** is slightly behind but stable.
---

### Table 2 — **Test Loss**
| Model       | Test Loss |
|-------------|-----------|
| UNet        | 0.1139    |
| UNet++      | **0.1024** |
| DeepLabV3+  | 0.1033    |

_Figure 2. Test loss comparison (bar chart)._  
**Graph:** _[Insert Plotly link]_

---

### Table 3 — **Inference Speed (median, batch=1)**
| Model       | ms / image |
|-------------|------------|
| UNet        | **19.70**  |
| UNet++      | 31.21      |
| DeepLabV3+  | 31.93      |

_Figure 3. Inference latency (lower is better)._  
**Graph:** _[Insert Plotly link]_

---

### Table 4 — **Peak VRAM (MB)**
| Model       | Peak VRAM (MB) |
|-------------|-----------------|
| UNet        | **411.38**      |
| UNet++      | 466.23          |
| DeepLabV3+  | 468.60          |

_Figure 4. Peak VRAM during inference (lower is better)._  
**Graph:** _[Insert Plotly link]_

---

## 🖼️ Qualitative Results — 12-Image Test Grids

> Each grid shows **12 fixed test cases** (ISIC-2016, 256×256). Columns (left → right): **input image**, **ground-truth overlay**, **prediction overlay**, **prediction boundary**. All runs use the same preprocessing, threshold, and evaluation settings.

---

<h3 align="center">DeepLabV3+ — 12-Image Test Grid</h3>

<p align="center">
  <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/SegCompare_ISIC2016_UNetFamily_DeepLabV3Plus/output_figures/12_figure/DeepLabV3%2B_ISIC2016_IMG256_SEED42_2025-10-29_10-00-54_test_grid_4panels_12.png" alt="DeepLabV3+ 12-image qualitative grid on ISIC2016" width="900">
</p>
<p align="center"><i>Figure 1 — DeepLabV3+: consistent lesion localization with strong boundary adherence across diverse appearances.</i></p>

---

<h3 align="center">UNet — 12-Image Test Grid</h3>

<!-- Note: The file name includes "TransUNet" but this figure corresponds to UNet results. -->
<p align="center">
  <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/SegCompare_ISIC2016_UNetFamily_DeepLabV3Plus/output_figures/12_figure/TransUNet_ISIC2016_IMG256_SEED42_2025-10-28_10-42-07_test_grid_4panels_12.png" alt="UNet 12-image qualitative grid on ISIC2016 (file name contains TransUNet)" width="900">
</p>
<p align="center"><i>Figure 2 — UNet: clean masks with low false positives; small structures occasionally under-segmented.</i></p>

---

<h3 align="center">UNet++ — 12-Image Test Grid</h3>

<p align="center">
  <img src="https://github.com/HussamUmer/Vision4Healthcare/blob/main/SegCompare_ISIC2016_UNetFamily_DeepLabV3Plus/output_figures/12_figure/UNet%2B%2B_ISIC2016_IMG256_SEED42_2025-10-28_15-06-31_test_grid_4panels_12.png" alt="UNet++ 12-image qualitative grid on ISIC2016" width="900">
</p>
<p align="center"><i>Figure 3 — UNet++: sharper lesion contours and fewer boundary leaks, reflecting its dense skip design.</i></p>


---

## 🔍 Takeaways

- **UNet++** is the most accurate here (best Dice/IoU).
- **UNet** is the fastest and most memory-efficient at inference.
- **DeepLabV3+** is close to UNet++ in accuracy, with moderate overhead.

> Because all runs share the same pipeline, these trade-offs are easy to interpret: boundary handling vs. speed vs. memory.

---

## 🧩 Pipeline (what’s included)

- **Step 0** Environment + versions dump + artifact paths (Drive) + cache paths (runtime)  
- **Step 1** Dataset fetch/verify (MD5) into `MEDSEGBENCH_DIR`  
- **Step 2** Config snapshot (YAML + echo)  
- **Step 3** Load predefined splits + counts + persist IDs  
- **Step 4** Preprocessing/augmentations (identical across models)  
- **Step 5** Data sanity visuals (overlay + boundary)  
- **Step 6** 🔁 **Model block (swap here): UNet / UNet++ / DeepLabV3+**  
- **Step 7** Loss & metrics (Dice+BCE, Dice/IoU, optional ECE/HD95)  
- **Step 8** Training loop + logging + ckpts + timing/resources  
- **Step 9** Post-training **test** evaluation + qualitative grids  
- **Step 10** Inference speed test (median, batch=1, fixed 50 imgs)  
- **Step 11** Threshold & calibration (PR/ROC, sweeps, ECE)  
- **Step 12** Final summary export (JSON + CSV) for tables/plots

> 🔁 To add **another** model: duplicate a notebook and **replace Step 6** only (keep the same forward API). Everything else just works.

---

### Open in Colab — Notebooks

| Model       | Launch |
|-------------|--------|
| UNet        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/SegCompare_ISIC2016_UNetFamily_DeepLabV3Plus/note%20book/UNETR34_BaseLine.ipynb) |
| UNet++      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/SegCompare_ISIC2016_UNetFamily_DeepLabV3Plus/note%20book/UNET++R34_BaseLine.ipynb) |
| DeepLabV3+  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/SegCompare_ISIC2016_UNetFamily_DeepLabV3Plus/note%20book/dlab_r50_model_BaseLine.ipynb) |

---

### 📦 Run Artifacts (figures, configs, logs, checkpoints)

All files generated during each run — figures, configs, CSV logs, JSON summaries, checkpoints, and qualitative grids — are available here:

| Model        | Artifacts Folder |
|--------------|------------------|
| **UNETR34**  | [Open artifacts](https://drive.google.com/drive/folders/1iZWiMTICBFLs9kOMsfznP6ze58VpR5NI?usp=sharing) |
| **UNETR34++**| [Open artifacts](https://drive.google.com/drive/folders/1hNdwskmpUuQF5vMyo0Vlt3bmx5-Ckha_?usp=sharing) |
| **DeepLabV3**| [Open artifacts](https://drive.google.com/drive/folders/1dN8J0I0Djb1TD-t54ITXoy5OgKqQkeFr?usp=sharing) |

