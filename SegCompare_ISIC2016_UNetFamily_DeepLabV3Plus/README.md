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

_Figure 1. Test Dice & IoU comparison (bar chart)._  
**Graph:** _[Insert Plotly link]_

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

> 🔁 To add **your** model: duplicate a notebook and **replace Step 6** only (keep the same forward API). Everything else just works.

---



