# 🧪🧬 **MagFusion-ViT: Multi-Magnification Fusion with Vision Transformers for Robust Breast Histopathology Classification**


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-vision--transformers-50C878.svg)](https://github.com/huggingface/pytorch-image-models)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#open-in-colab)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Dataset: BreakHis](https://img.shields.io/badge/Dataset-BreakHis-8A2BE2.svg)](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

---

## ❓ **Why this study?**

Histopathology slides are captured at multiple magnifications (e.g., **100×** and **400×**). We compare **DeiT-Small** and **Swin-Tiny** on **single-magnification** vs **mixed-magnification** training to understand:

- 🔁 whether mixing magnifications improves **generalization** and **robustness**  
- 🧱 which transformer backbone is more **scale-tolerant** to nuclei/texture patterns  
- 🔄 how performance shifts **cross-magnification**

We report **macro-F1**, **balanced accuracy**, **per-class F1**, **confusion matrices**, and **latency/throughput**.

---

## 🗃️ **Dataset**

- **Name:** BreakHis — *Breast Cancer Histopathological Database*  
- **Magnifications used:** **100×**, **400×**, and **Mixed (100×+400×)**  
- **Official page & access:** https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/  
- **Classes (8):** adenosis, ductal carcinoma, fibroadenoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma, phyllodes tumor, tubular adenoma

### 📦 **Our splits (per setup)**
- **Train:** 140 images/class  
- **Val:** 30 images/class  
- **Test:** 30 images/class

---

## 🧠 **Methodology (Overview)**

### 🏗️ **Backbones**
- **DeiT-Small** (ImageNet-1k pretrained)  
- **Swin-Tiny**  (ImageNet-1k pretrained)

### 🔧 **Training setups (3)**
1. **100× only**
2. **400× only**
3. **Mixed (100×+400×)**

### ⚙️ **Training protocol (kept identical across models)**
- **Transforms:** RGB → resize **224×224** → normalize (ImageNet mean/std)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.05)
- **Schedule:** cosine decay with **5–10** warmup epochs
- **Runtime:** AMP mixed-precision, grad-clip=1.0, early stopping on **val macro-F1**, max 100 epochs
- **Logging:** train/val loss & acc, **macro-F1**, per-class F1, epoch/total time, peak GPU MB

### 📏 **Evaluation**
- **In-domain:** Train=Test magnification (primary comparison)  
- **Cross-domain robustness:** Evaluate on other magnifications  
- **Metrics:** Accuracy, Balanced Accuracy, **Macro-F1**, Per-class PRF, Confusion Matrix, **Latency/Throughput**, Peak GPU MB  
- **Qualitative:** dataset previews, **t-SNE** (pretrained features), TP/FP/FN grids, attention rollout (DeiT)

---

## 🚀 **Open in Colab**

> Replace `<USER>` and `<REPO>` with your GitHub path. Each notebook mounts Google Drive and asks you to set `DATA_ROOT`, `DATA_ROOT_100X`, `DATA_ROOT_400X`, `DATA_ROOT_MIXED`.

### 🤖 **DeiT-Small runs**
| Run | Notebook | Notes |
|---|---|---|
| **DeiT-100×** | [Open in Colab](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/deit_100x.ipynb) | Train on **100×**; primary test **100×**; robustness: **400×**, **Mixed** |
| **DeiT-400×** | [Open in Colab](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/deit_400x.ipynb) | Train on **400×**; primary test **400×**; robustness: **100×**, **Mixed** |
| **DeiT-Mixed** | [Open in Colab](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/deit_mixed.ipynb) | Train on **Mixed**; primary test **Mixed**; robustness: **100×**, **400×** |

### 🪟 **Swin-Tiny runs**
| Run | Notebook | Notes |
|---|---|---|
| **Swin-Tiny-100×** | [Open in Colab](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/swin_tiny_100x.ipynb) | Train on **100×**; primary test **100×**; robustness: **400×**, **Mixed** |
| **Swin-Tiny-400×** | [Open in Colab](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/swin_tiny_400x.ipynb) | Train on **400×**; primary test **400×**; robustness: **100×**, **Mixed** |
| **Swin-Tiny-Mixed** | [Open in Colab](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/swin_tiny_mixed.ipynb) | Train on **Mixed**; primary test **Mixed**; robustness: **100×**, **400×** |

---

## ⚡ **Quick Start (Colab)**

1. 🔗 Open a notebook above and **Run All**  
2. 💾 Mount Google Drive; set dataset roots for **100×**, **400×**, and **Mixed**  
3. 🏃 Run **Data checks → Training → Final Testing → Reports**  
4. 📦 Artifacts (checkpoints, logs, plots) are saved in a timestamped **`RUN_DIR`** under your Drive

---

## 🔁 **Reproducibility**

- 🎯 Fixed seeds (Python/NumPy/PyTorch) + deterministic cuDNN flags  
- 🧾 Frozen splits saved as path lists; `config.yaml` logs software + hardware  
- 🏅 Best checkpoint = **highest validation macro-F1** (tie-break: val accuracy)

---

## 📊 **Results Summary — MagFusion-ViT (DeiT-Small & Swin-Tiny across 100× / 400× / Mixed)**

# 📊 

Below are **ready-to-paste README tables** built from the numbers you provided.  
Quick takeaways:
- ✅ **In-domain** (train=test) is near-perfect on **400×** for both models (Macro-F1 ≈ **0.992–0.996**).
- 🧪 **Cross-domain** drops are **asymmetric**: training on **400× → testing on 100×** drops more than the reverse.
- 🧩 **Mixed training** improves **robustness**. **Swin-Tiny (Mixed)** generalizes best overall (Macro-F1 ≈ **0.955–0.963** on 100×/400×/Mixed).

---

## 🧷 Primary (In-Domain) Performance — Headline Metrics
*Each row reports the primary evaluation where **Train = Test** magnification. Includes efficiency stats to aid practical comparisons.*

| Model | Train=Test Setup | Macro-F1 (↑) | Accuracy (↑) | Balanced Acc (↑) | Latency (ms/img) (↓) | Throughput (img/s) (↑) | Peak GPU (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| **DeiT-Small** | 100× | **0.979** | 0.979 | 0.979 | 166.581 | 6.003 | 528.254 |
| **DeiT-Small** | 400× | **0.996** | 0.996 | 0.996 | 704.081 | 1.420 | 528.254 |
| **DeiT-Small** | Mixed | **0.920** | 0.921 | 0.921 | 477.284 | 2.095 | 528.004 |
| **Swin-Tiny** | 100× | **0.958** | 0.958 | 0.958 | 303.168 | 3.298 | 1302.659 |
| **Swin-Tiny** | 400× | **0.992** | 0.992 | 0.992 | 269.120 | 3.716 | 1302.659 |
| **Swin-Tiny** | Mixed | **0.963** | 0.963 | 0.963 | 224.129 | 4.462 | 1302.659 |

**What it shows:** how well each model performs on the distribution it was trained on, plus runtime/memory.  
**Observation:** Both models excel on **400×** in-domain; **Swin-Tiny (Mixed)** reaches strong, balanced in-domain performance with the **best latency** among Swin runs.

---

## 🔀 Cross-Domain Robustness — Macro-F1 (↑)
*Rows are the **training** setup; columns are the **testing** setup. Higher is better. These matrices reveal how well each trained model generalizes across magnifications.*

### DeiT-Small
| Train \ Test | 100× | 400× | Mixed |
|---|---:|---:|---:|
| **100×** | **0.979** | 0.518 | 0.769 |
| **400×** | 0.405 | **0.996** | 0.734 |
| **Mixed** | 0.921 | 0.933 | **0.920** |

**Reads like:**  
- Training on **400× → 100×** suffers (0.405), more than **100× → 400×** (0.518).  
- **Mixed** training is consistently high (≈0.92–0.93) across all tests.

### Swin-Tiny
| Train \ Test | 100× | 400× | Mixed |
|---|---:|---:|---:|
| **100×** | **0.958** | 0.337 | 0.681 |
| **400×** | 0.245 | **0.992** | 0.685 |
| **Mixed** | 0.955 | 0.963 | **0.963** |

**Reads like:**  
- **400× → 100×** is harsh (0.245), harsher than **100× → 400×** (0.337).  
- **Mixed** training for Swin is excellent and symmetric (≈0.955–0.963 across all tests), indicating **best robustness** overall.

---

## ⏱️ Efficiency Snapshot — In-Domain Only
*Compare speed/memory where each model is evaluated on its training distribution.*

| Model | Train=Test | Latency (ms/img) (↓) | Throughput (img/s) (↑) | Peak GPU (MB) |
|---|---|---:|---:|---:|
| **DeiT-Small** | 100× | **166.581** | **6.003** | **528.254** |
| **DeiT-Small** | 400× | 704.081 | 1.420 | 528.254 |
| **DeiT-Small** | Mixed | 477.284 | 2.095 | 528.004 |
| **Swin-Tiny** | 100× | 303.168 | 3.298 | 1302.659 |
| **Swin-Tiny** | 400× | 269.120 | 3.716 | 1302.659 |
| **Swin-Tiny** | Mixed | 224.129 | 4.462 | 1302.659 |

**What it shows:** DeiT has **lower VRAM** footprint and is **fastest** on **100×** in-domain; Swin is fastest on **Mixed** and **400×** among its own runs.

---

## 🧩 Optional: Subgroup Summary (Macro-F1, Acc, BalAcc) — Per Test Distribution
*Handy for quick graphing across test sets; combine rows from the three runs of each model.*

### Swin-Tiny — by Test Distribution
| Test Group | Macro-F1 | Acc | BalAcc | Latency (ms/img) | Throughput (img/s) |
|---|---:|---:|---:|---:|---:|
| **100×** (from 100× run) | 0.958 | 0.958 | 0.958 | 303.168 | 3.298 |
| **400×** (from 400× run) | 0.992 | 0.992 | 0.992 | 269.120 | 3.716 |
| **Mixed** (from Mixed run) | 0.963 | 0.963 | 0.963 | 224.129 | 4.462 |

### DeiT-Small — by Test Distribution
| Test Group | Macro-F1 | Acc | BalAcc | Latency (ms/img) | Throughput (img/s) |
|---|---:|---:|---:|---:|---:|
| **100×** (from 100× run) | 0.979 | 0.979 | 0.979 | 166.581 | 6.003 |
| **400×** (from 400× run) | 0.996 | 0.996 | 0.996 | 704.081 | 1.420 |
| **Mixed** (from Mixed run) | 0.920 | 0.921 | 0.921 | 477.284 | 2.095 |

**What it shows:** a compact “best per test distribution” view, useful for **bar charts** (Macro-F1, Acc, BalAcc) and **runtime plots**.

---

### ✅ Notes for plotting
- Use **Macro-F1** as the headline bar; overlay **Acc**/**BalAcc** if needed.
- For **robustness**, heatmap the **Cross-Domain** matrices.
- For **efficiency**, draw **latency vs throughput** scatter per model/setup.



---

## 📚 **Citations (Background)**

- **DeiT:** Touvron et al., *Training data-efficient image transformers & distillation through attention*. ICML 2021.  
- **Swin Transformer:** Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. 2021.  
- **BreakHis:** Spanhol et al., *A Dataset for Breast Cancer Histopathological Image Classification*. IEEE T-BioCAS 2016.

*(Please cite the original papers and the dataset per their licenses.)*

---

## 📜 **License**

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
