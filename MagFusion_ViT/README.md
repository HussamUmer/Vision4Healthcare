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

### 🔗 **Prepared splits (Google Drive)**
> These folders contain our **prepared per-setup splits** with the directory layout  
> `training/`, `validation/`, `testing/` (8 classes each), matching the counts above.

- **100× Split:** https://drive.google.com/drive/folders/1m_4qZeVgjgaNhufJP6tRgP5z6Y6GakoC?usp=sharing  
- **400× Split:** https://drive.google.com/drive/folders/1s9xMnK96_QO084fTWo8vqW5jWFv3WNXr?usp=sharing  
- **Mixed (100× + 400×) Split:** https://drive.google.com/drive/folders/1Rv-1K6j8HFEy6kaf3UJXIP4lPF3TE8cg?usp=sharing

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

### 🤖 **DeiT-Small runs**
| Run | Notebook | Notes |
|---|---|---|
| **DeiT-100×** | [Open in Colab](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/MagFusion_ViT/Notebooks/DEIT_100x%20%281%29.ipynb) | Train on **100×**; primary test **100×**; robustness: **400×**, **Mixed** |
| **DeiT-400×** | [Open in Colab](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/MagFusion_ViT/Notebooks/DEIT_400x.ipynb) | Train on **400×**; primary test **400×**; robustness: **100×**, **Mixed** |
| **DeiT-Mixed** | [Open in Colab](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/MagFusion_ViT/Notebooks/DEIT_Mixed.ipynb) | Train on **Mixed**; primary test **Mixed**; robustness: **100×**, **400×** |

### 🪟 **Swin-Tiny runs**
| Run | Notebook | Notes |
|---|---|---|
| **Swin-Tiny-100×** | [Open in Colab](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/MagFusion_ViT/Notebooks/SwinTiny_100x%20%281%29.ipynb) | Train on **100×**; primary test **100×**; robustness: **400×**, **Mixed** |
| **Swin-Tiny-400×** | [Open in Colab](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/MagFusion_ViT/Notebooks/SwinTiny_400x.ipynb) | Train on **400×**; primary test **400×**; robustness: **100×**, **Mixed** |
| **Swin-Tiny-Mixed** | [Open in Colab](https://colab.research.google.com/github/HussamUmer/Vision4Healthcare/blob/main/MagFusion_ViT/Notebooks/SwinTiny_Mixed.ipynb) | Train on **Mixed**; primary test **Mixed**; robustness: **100×**, **400×** |

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

**What matters here:** performance when **train and test magnifications differ**.  
**Key findings (at a glance):**
- **Mixed training** gives the most reliable cross-domain behavior; **Swin-Tiny (Mixed)** is the most consistent (Macro-F1 ≈ **0.955–0.963** to both 100× and 400×).  
- **Directional gap is asymmetric:** going **400× → 100×** is harder than **100× → 400×** for both models.  
- Averaging all off-diagonal entries, **DeiT ≈ 0.713** and **Swin ≈ 0.644**; using a **Mixed recipe** is the simplest way to close the gap.

---

### 🧭 Cross-Domain Macro-F1 Matrices (for reference; diagonals are in-domain)

#### DeiT-Small — Train × Test (Macro-F1 ↑)
| Train \ Test | 100× | 400× | Mixed |
|---|---:|---:|---:|
| **100×** | 0.979 *(in-domain)* | **0.518** | **0.769** |
| **400×** | **0.405** | 0.996 *(in-domain)* | **0.734** |
| **Mixed** | **0.921** | **0.933** | 0.920 *(in-domain)* |

#### Swin-Tiny — Train × Test (Macro-F1 ↑)
| Train \ Test | 100× | 400× | Mixed |
|---|---:|---:|---:|
| **100×** | 0.958 *(in-domain)* | **0.337** | **0.681** |
| **400×** | **0.245** | 0.992 *(in-domain)* | **0.685** |
| **Mixed** | **0.955** | **0.963** | 0.963 *(in-domain)* |

> **Read:** focus on the **bold off-diagonal** cells—those are the cross-domain results.

![In-Domain vs Cross-Domain (use the orange bars)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/Cross_Domain_Robustness/indomain_vs_crossdomain.png)

<sub><b>Figure 1.</b> Averages of **in-domain** (blue, diagonal) vs **cross-domain** (orange, off-diagonal). For this section, focus on **cross-domain (orange)** to gauge robustness.</sub>

---

### 📐 Directional Asymmetry (train → test)
| Model | 100× → 400× | 400× → 100× |
|---|---:|---:|
| **DeiT-Small** | **0.518** | **0.405** |
| **Swin-Tiny**  | **0.337** | **0.245** |

> **Interpretation:** **400× → 100×** consistently underperforms **100× → 400×**, suggesting models trained on high-mag textures struggle to generalize down to lower magnification.

![Directional Generalization Asymmetry](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/Cross_Domain_Robustness/generalization.png)

<sub><b>Figure 2.</b> **Directional gap** between 100×→400× and 400×→100×. Both models struggle more when moving **down** in magnification (400×→100×).</sub>

---

### 🧪 Generalization from **Mixed** Training (cross-domain only)
| Model | Mixed → 100× | Mixed → 400× |
|---|---:|---:|
| **DeiT-Small** | **0.921** | **0.933** |
| **Swin-Tiny**  | **0.955** | **0.963** |

> **Interpretation:** Mixed training substantially reduces domain shift—**Swin-Tiny (Mixed)** is the most robust, with near-symmetric performance to both 100× and 400×.

![Generalization from Mixed Training](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/Cross_Domain_Robustness/generalization_from_mixed.png)
 
<sub><b>Figure 3.</b> Cross-domain generalization from **Mixed training**. Bars show Macro-F1 on 100× and 400× tests (ignore Mixed→Mixed as it’s in-domain). **Swin-Tiny (Mixed)** is strongest and most symmetric.</sub>

---

### 🧮 Cross-Domain Mean (average of all off-diagonal cells)
| Model | Mean Macro-F1 (↑) |
|---|---:|
| **DeiT-Small** | **0.713** |
| **Swin-Tiny**  | **0.644** |

> **Interpretation:** On average across all cross-domain conditions, **DeiT** edges **Swin**—but **Swin-Tiny (Mixed)** is the **best single recipe** if you can only train once and must handle both magnifications at test time.

![Cross-Domain Robustness Heatmaps](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/Cross_Domain_Robustness/crossdomain_robustness.png)

<sub><b>Figure 4.</b> Heatmaps of Train×Test Macro-F1. Emphasize the **off-diagonal** cells. Mixed rows are uniformly high, especially for **Swin-Tiny**.</sub>

---

## ⏱️ Efficiency Snapshot — In-Domain Only

**What this shows (quick read):**
- **DeiT-Small** has the **lowest VRAM footprint (~528 MB)** and is **fastest** when trained/tested on **100×** (≈**166.6 ms/img**, **6.00 img/s**).
- **Swin-Tiny** is **faster** on **Mixed** and **400×** among its own runs (down to **224.1 ms/img**, up to **4.46 img/s**), but uses **more VRAM (~1.3 GB)**.
- If **memory is tight**, DeiT-Small is the practical choice; if you need **speed at higher mags/mixed**, Swin-Tiny wins within its family.

### 📋 Table (Train = Test for each run)

| Model       | Train=Test | Latency (ms/img) ↓ | Throughput (img/s) ↑ | Peak GPU (MB) |
|---|---|---:|---:|---:|
| **DeiT-Small** | 100×  | **166.581** | **6.003** | **528.254** |
| **DeiT-Small** | 400×  | 704.081 | 1.420 | 528.254 |
| **DeiT-Small** | Mixed | 477.284 | 2.095 | 528.004 |
| **Swin-Tiny**  | 100×  | 303.168 | 3.298 | 1302.659 |
| **Swin-Tiny**  | 400×  | 269.120 | 3.716 | 1302.659 |
| **Swin-Tiny**  | Mixed | **224.129** | **4.462** | 1302.659 |

> These are **in-domain** numbers (each model evaluated on the distribution it was trained on).

![Efficiency Frontier — In-Domain Only](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/Efficiency_Snapshot/efficiency_frontier.png)
<sub><b>Figure 5.</b> In-domain efficiency frontier — <b>latency vs throughput</b>; bubble size = peak GPU MB; color = setup; marker = model.</sub>

---

### ⏱️ Latency (ms/image, ↓) — In-Domain Only
| Model | 100× | 400× | Mixed |
|---|---:|---:|---:|
| **DeiT-Small** | **166.581** | 704.081 | 477.284 |
| **Swin-Tiny**  | 303.168 | 269.120 | **224.129** |

> **Read:** lower is better. **DeiT-Small** excels on **100×**; **Swin-Tiny** excels on **Mixed**/**400×**.

![Latency (ms/img) — In-Domain Only](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs//Efficiency_Snapshot/latency.png)  
<sub><b>Figure 6.</b> In-domain efficiency view — <b>latency</b> (ms/img). Lower is better.</sub>

---

### 🚀 Throughput (images/sec, ↑) — In-Domain Only
| Model | 100× | 400× | Mixed |
|---|---:|---:|---:|
| **DeiT-Small** | **6.003** | 1.420 | 2.095 |
| **Swin-Tiny**  | 3.298 | 3.716 | **4.462** |

> **Read:** higher is better. **DeiT-Small** is the fastest on **100×** overall; **Swin-Tiny** is fastest for **Mixed**.

![Throughput (img/s) — In-Domain Only](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs//Efficiency_Snapshot/throughput.png)  
<sub><b>Figure 7.</b> In-domain efficiency view — <b>throughput</b> (img/s). Higher is better.</sub>

---

### 🧠 Peak GPU Memory (MB, ↓) — In-Domain Only
| Model | 100× | 400× | Mixed |
|---|---:|---:|---:|
| **DeiT-Small** | **528.254** | **528.254** | **528.004** |
| **Swin-Tiny**  | 1302.659 | 1302.659 | 1302.659 |

> **Read:** **DeiT-Small** is ≈**2.5×** more memory-efficient than **Swin-Tiny**.

![Peak GPU Memory (MB) — In-Domain Only](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs//Efficiency_Snapshot/Peak_GPU.png)    
<sub><b>Figure 8.</b> In-domain efficiency view — <b>peak GPU memory</b> (MB). Lower is better.</sub>

---

### 🏁 “Best per Family” (quick picks)
| Model | Fastest Setup | Latency ↓ | Throughput ↑ | Peak GPU (MB) |
|---|---|---:|---:|---:|
| **DeiT-Small** | **100×** | **166.581** | **6.003** | **528.254** |
| **Swin-Tiny**  | **Mixed** | **224.129** | **4.462** | 1302.659 |

> **Interpretation:** choose **DeiT-Small** when VRAM is tight or the target is **100×**; choose **Swin-Tiny** when you expect **mixed/400×** test conditions and have more memory.

---

## 🧩 Subgroup Summary (Macro-F1, Acc, BalAcc) — Per Test Distribution
*Handy for quick graphing across test sets; combine rows from the three runs of each model.*

**At a glance**
- For **100× tests**, **DeiT-Small** is both **more accurate** and **faster** (Macro-F1≈0.979; 166.6 ms/img).
- For **400× tests**, **both models** peak (Macro-F1≈0.992–0.996); **Swin-Tiny** is quicker.
- For **Mixed tests**, **Swin-Tiny** leads in both **accuracy** (≈0.963) and **speed** (≈224 ms/img, 4.46 img/s).

---

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

### 1) Latency (ms/img) — In-Domain per Test Distribution
| Test Distribution | DeiT-Small (↓) | Swin-Tiny (↓) | Winner |
|---|---:|---:|---|
| **100×** | **166.581** | 303.168 | **DeiT-Small** |
| **400×** | 704.081 | **269.120** | **Swin-Tiny** |
| **Mixed** | 477.284 | **224.129** | **Swin-Tiny** |

> **Interpretation:** DeiT-Small is fastest on **100×**; Swin-Tiny is faster on **400×** and **Mixed**.

![Latency (ms/img) — In-Domain per Test Distribution](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs//subgroup_summary/latency.png)    
<sub><b>Figure 9.</b> Latency by test distribution (lower is better).</sub>

---

### 2) Subgroup Summary — Macro-F1 / Acc / BalAcc (Per Test Distribution)

**Swin-Tiny**
| Test | Macro-F1 | Acc | BalAcc |
|---|---:|---:|---:|
| **100×** | 0.958 | 0.958 | 0.958 |
| **400×** | 0.992 | 0.992 | 0.992 |
| **Mixed** | 0.963 | 0.963 | 0.963 |

**DeiT-Small**
| Test | Macro-F1 | Acc | BalAcc |
|---|---:|---:|---:|
| **100×** | 0.979 | 0.979 | 0.979 |
| **400×** | 0.996 | 0.996 | 0.996 |
| **Mixed** | 0.920 | 0.921 | 0.921 |

> **Interpretation:** Both models peak at **400×**; **DeiT-Small** leads on **100×**, while **Swin-Tiny** leads on **Mixed**.

![Subgroup Summary — Macro-F1 / Acc / BalAcc (Per Test Distribution)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs//subgroup_summary/summary.png)
<sub><b>Figure 10.</b> Macro-F1, Accuracy, and Balanced Accuracy per test distribution for each model.</sub>


---

### 3) Throughput (img/s) — In-Domain per Test Distribution
| Test Distribution | DeiT-Small (↑) | Swin-Tiny (↑) | Winner |
|---|---:|---:|---|
| **100×** | **6.003** | 3.298 | **DeiT-Small** |
| **400×** | 1.420 | **3.716** | **Swin-Tiny** |
| **Mixed** | 2.095 | **4.462** | **Swin-Tiny** |

> **Interpretation:** Higher is better. DeiT-Small is the fastest on **100×** overall; Swin-Tiny is faster on **400×** and **Mixed**.

![Throughput (img/s) — In-Domain per Test Distribution](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs//subgroup_summary/throughput.png)  
<sub><b>Figure 11.</b> Throughput by test distribution (higher is better).</sub>

---

### 4) Efficiency Frontier Points — In-Domain per Test Distribution  
*(Scatter shows each point; table lists exact coordinates & VRAM.)*
| Model      | Test | Latency (ms/img) ↓ | Throughput (img/s) ↑ | Peak GPU (MB) |
|---|---|---:|---:|---:|
| **DeiT-Small** | 100×  | **166.581** | **6.003** | **528.254** |
| **DeiT-Small** | 400×  | 704.081 | 1.420 | 528.254 |
| **DeiT-Small** | Mixed | 477.284 | 2.095 | 528.004 |
| **Swin-Tiny**  | 100×  | 303.168 | 3.298 | 1302.659 |
| **Swin-Tiny**  | 400×  | 269.120 | 3.716 | 1302.659 |
| **Swin-Tiny**  | Mixed | 224.129 | 4.462 | 1302.659 |

> **Interpretation:** **DeiT-Small (100×)** sits at the **upper-left** (lowest latency, highest throughput). **Swin-Tiny (Mixed)** is the fastest Swin point; **DeiT-Small (400×)** is furthest from the frontier.

![Efficiency Frontier — In-Domain per Test Distribution](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs//subgroup_summary/efficiency.png)  
<sub><b>Figure 12.</b> Efficiency frontier (latency vs throughput); color = test set, marker = model.</sub>

---

## 🧩 In-Domain Confusion Matrices (DeiT-Small & Swin-Tiny)

**What this shows:** Per-setup, in-domain performance (Train = Test). Each matrix is **row-normalized** (values sum to 1 per true class) so off-diagonal intensity reflects misclassification patterns. All plots use the **same class order** and **same color scale** for fair visual comparison.

---

### 🤖 DeiT-Small — In-Domain CMs (Train = Test)

| 100× | 400× | Mixed |
|---|---|---|
| ![DeiT-Small — 100× (in-domain CM)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/confusion_matrices/deitsmall/indomain/100x.png) | ![DeiT-Small — 400× (in-domain CM)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/confusion_matrices/deitsmall/indomain/400x.png) | ![DeiT-Small — Mixed (in-domain CM)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/confusion_matrices/deitsmall/indomain/mixed.png) |

---

### 🤖 Swin-Tiny — In-Domain CMs (Train = Test)

| 100× | 400× | Mixed |
|---|---|---|
| ![Sin-Tiny — 100× (in-domain CM)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/confusion_matrices/swintiny/indomain/100x.png) | ![Swin-Tiny — 400× (in-domain CM)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/confusion_matrices/swintiny/indomain/400x.png) | ![Swin-Tiny — Mixed (in-domain CM)](https://raw.githubusercontent.com/HussamUmer/Vision4Healthcare/main/MagFusion_ViT/Outputs/Graphs/confusion_matrices/swintiny/indomain/mixed.png) |

---

## 📦 Experiment Artifacts & Checkpoints (Google Drive)

Large files (logs, checkpoints, evaluation CSV/JSON, and figures) are hosted on Google Drive:

🔗 **Drive folder:** https://drive.google.com/drive/folders/1qhvplLgcpmJn7D1f0HVC69GvwmzjEhwa?usp=sharing

**Contents (typical):**
- `runs/` — timestamped run directories (e.g., `2025-09-22_deit_100x/`)
  - `config.yaml` (env, hyperparams, data roots)
  - `train_log.csv` / `events.*` (training & validation curves)
  - `best.ckpt` (checkpoint selected by validation macro-F1)
  - `eval/` (JSON/CSV metrics, confusion matrices, latency/throughput)
  - `figures/` (plots used in the README/paper)
- `splits/` — frozen train/val/test path lists for reproducibility

---

## 📚 **Citations (Background)**

- **DeiT:** Touvron et al., *Training data-efficient image transformers & distillation through attention*. ICML 2021.  
- **Swin Transformer:** Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. 2021.  
- **BreakHis:** Spanhol et al., *A Dataset for Breast Cancer Histopathological Image Classification*. IEEE T-BioCAS 2016.

*(Please cite the original papers and the dataset per their licenses.)*

---

## 📜 **License**

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
