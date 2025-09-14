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

## 📊 **Results (Templates)**

> Replace placeholders with your exported numbers (JSON/CSV) and insert your PNG plots.

### 🧷 **Primary (in-domain) — Macro-F1 (↑)**
| Model | Train=Test: 100× | Train=Test: 400× | Train=Test: Mixed |
|---|---:|---:|---:|
| **DeiT-Small** | `0.XXX` | `0.XXX` | `0.XXX` |
| **Swin-Tiny**  | `0.XXX` | `0.XXX` | `0.XXX` |

### 🔀 **Robustness (cross-domain) — Macro-F1 (↑)**
| Train \ Test | **100×** | **400×** | **Mixed** |
|---|---:|---:|---:|
| **DeiT-100×** | **—** | `0.XXX` | `0.XXX` |
| **DeiT-400×** | `0.XXX` | **—** | `0.XXX` |
| **DeiT-Mixed** | `0.XXX` | `0.XXX` | **—** |
| **Swin-100×** | **—** | `0.XXX` | `0.XXX` |
| **Swin-400×** | `0.XXX` | **—** | `0.XXX` |
| **Swin-Mixed** | `0.XXX` | `0.XXX` | **—** |

### 🧩 **Per-class F1 — Example (Mixed test)**
| Class | DeiT-Small | Swin-Tiny |
|---|---:|---:|
| Adenosis | `0.XXX` | `0.XXX` |
| Ductal carcinoma | `0.XXX` | `0.XXX` |
| Fibroadenoma | `0.XXX` | `0.XXX` |
| Lobular carcinoma | `0.XXX` | `0.XXX` |
| Mucinous carcinoma | `0.XXX` | `0.XXX` |
| Papillary carcinoma | `0.XXX` | `0.XXX` |
| Phyllodes tumor | `0.XXX` | `0.XXX` |
| Tubular adenoma | `0.XXX` | `0.XXX` |

> 📎 Include confusion matrices (`cm_*.png`), reliability diagrams (optional), and latency tables exported by notebooks.

---

## 📚 **Citations (Background)**

- **DeiT:** Touvron et al., *Training data-efficient image transformers & distillation through attention*. ICML 2021.  
- **Swin Transformer:** Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. 2021.  
- **BreakHis:** Spanhol et al., *A Dataset for Breast Cancer Histopathological Image Classification*. IEEE T-BioCAS 2016.

*(Please cite the original papers and the dataset per their licenses.)*

---

## 📜 **License**

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
