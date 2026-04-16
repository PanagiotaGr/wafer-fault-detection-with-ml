# 🔬 Semiconductor Wafer Defect Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-teal?style=flat-square&logo=scikitlearn)
![Dataset](https://img.shields.io/badge/Dataset-WM--811K-lightgray?style=flat-square)

**Classical ML & deep learning for wafer map defect detection**  
*With a research study on few-shot learning under extreme data scarcity and class imbalance*

[Objective](#-objective) · [Dataset](#-dataset) · [Pipeline](#-pipeline) · [Results](#-results) · [Few-Shot Study](#-few-shot-learning-experiments) · [Install](#-installation--usage)

</div>

---

## 🎯 Objective

Automatically recognize defect types in semiconductor wafer maps using both classical and deep learning approaches. This project studies wafer defect classification under:

- **Standard supervised learning** — full dataset, classical ML & CNN baselines
- **Few-shot (limited-data) scenarios** — extreme data scarcity with only 5–20 samples per class

Models explored:

| Category | Models |
|---|---|
| Classical ML | Logistic Regression, SVM, Random Forest |
| Deep Learning | CNN (baseline, weighted loss, focal loss, focal+augmentation) |
| Research | Few-shot learning variants across k=5, 10, 20 samples/class |

---

## 📦 Dataset

- **Source:** [WM-811K](https://www.kaggle.com/qingyi/wm811k-wafer-map) (Kaggle / MIR Lab)
- **Format:** Wafer map images with labeled defect categories
- **File:** `LSWMD.pkl`

> ⚠️ Dataset not included due to size. Run `python download_dataset.py` or fetch it from Kaggle.

**Defect classes used:**

`edge-ring` · `edge-loc` · `center` · `loc` · `scratch` · `random` · `donut` · `near-full`

---

## ⚙️ Pipeline

```
LSWMD.pkl
    │
    ▼
┌─────────────────┐
│  Preprocessing  │  Remove non-defect samples · clean labels
│                 │  Resize → 64×64 · normalize pixel values
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────────────────────────────────┐
│  ML   │ │           Deep Learning               │
│  SVM  │ │  Baseline → Weighted → Focal → Focal  │
│  LR   │ │                               + Aug   │
│  RF   │ └──────────────┬───────────────────────┘
└───┬───┘                │
    │         ┌──────────▼──────────┐
    │         │  Few-Shot Variants  │
    │         │  k = 5 / 10 / 20   │
    │         └──────────┬──────────┘
    └──────────┬──────────┘
               ▼
        ┌─────────────────┐
        │    Evaluation   │
        │  accuracy       │
        │  confusion mat  │
        │  training curve │
        └─────────────────┘
```

---

## 📈 Results

### Classical ML

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 62.7% | Baseline linear model |
| SVM | 65.7% | Kernel-based, best classical linear |
| Random Forest | 78.5% | Strong non-linear baseline |
| **Random Forest (optimized)** | **79.4%** | ✦ Best classical result |

### CNN Deep Learning

| Model | Strategy | Notes |
|---|---|---|
| CNN Baseline | Standard cross-entropy | Competitive start |
| CNN Weighted | Weighted loss | Handles class imbalance |
| CNN Focal | Focal loss | Targets hard examples |
| CNN Focal + Aug | Focal loss + augmentation | Full augmented pipeline |

> 📊 Full confusion matrices and training curves saved in `outputs/figures/`

---

## 🎯 Few-Shot Learning Experiments

Accuracy under extreme data scarcity (k samples per class):

| Samples / class | Baseline | Weighted loss | Focal loss | Focal + Aug |
|---|---|---|---|---|
| k = 5 | 0.20 | 0.13 | 0.07 | 0.20 |
| k = 10 | 0.43 | 0.43 | 0.37 | 0.10 |
| k = 20 | 0.52 | **0.57 ↑** | 0.18 | 0.47 |

---

## 💡 Key Findings

- ✅ **Weighted loss consistently wins** under class imbalance — even in extreme few-shot scenarios
- ❌ **Focal loss underperforms** at very low data regimes (k=5, k=10)
- ❌ **Augmentation can hurt** when samples per class are too limited (k=10 drops to 0.10)
- 📈 **More data matters most** — k: 5→20 yields the largest single accuracy improvement

> **Scientific insight:** In few-shot scenarios, simple approaches (e.g. weighted loss) can outperform architecturally complex solutions. Complexity alone does not equal performance.

---

## 🗂️ Repository Structure

```
wafer-fault-detection-with-ml/
│
├── src/
│   └── load_data.py                        # Data loading utilities
│
├── wafer_pipeline.py                       # Classical ML pipeline
├── wafer_cnn_pipeline.py                   # CNN baseline
├── wafer_cnn_focal_aug.py                  # CNN focal loss + augmentation
├── wafer_fewshot_focal_experiment.py       # Few-shot experiments
├── wafer_full_comparison_experiment.py     # ⭐ Main research experiment
│
├── outputs/
│   ├── figures/                            # Plots, confusion matrices, curves
│   └── results/                            # CSV result summaries
│
├── download_dataset.py
├── semiconductor-wafer-defect-classification.ipynb
├── CITATION.cff
└── README.md
```

---

## 🛠️ Technologies

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| PyTorch | CNN models & training |
| Scikit-learn | Classical ML |
| NumPy / Pandas | Data processing |
| OpenCV | Image preprocessing |
| Matplotlib | Visualization |

---

## 🚀 Installation & Usage

### Setup

```bash
git clone https://github.com/PanagiotaGr/wafer-fault-detection-with-ml.git
cd wafer-fault-detection-with-ml

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run pipelines

```bash
# Classical ML
python wafer_pipeline.py

# CNN baseline
python wafer_cnn_pipeline.py

# CNN with focal loss + augmentation
python wafer_cnn_focal_aug.py
```

### Run the main research experiment

```bash
python wafer_full_comparison_experiment.py
```

### Output files

Results are saved automatically to:

```
outputs/figures/fewshot_accuracy_comparison.png
outputs/results/fewshot_summary.csv
```

---

## 📖 Citation

If you find this work useful, please cite it:

```bibtex
@software{Grosdouli_Wafer_Fault_Detection_2026,
  author  = {Grosdouli, Panagiota},
  title   = {Semiconductor Wafer Defect Classification: A Study on Classical ML and Few-Shot Deep Learning},
  url     = {https://github.com/PanagiotaGr/wafer-fault-detection-with-ml},
  year    = {2026}
}
```

---

## 🙏 Acknowledgments

- [WM-811K / LSWMD dataset](https://www.kaggle.com/qingyi/wm811k-wafer-map)
- Semiconductor defect detection research community

---

## 📄 License

Distributed under the [Apache 2.0 License](LICENSE).
