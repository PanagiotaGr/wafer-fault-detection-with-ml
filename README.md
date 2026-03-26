
#  Semiconductor Wafer Defect Classification

This project explores how **machine learning** and **deep learning** can be applied to classify defect patterns on semiconductor wafers using real wafer map image data.  
The aim is to improve defect detection accuracy and reduce manual inspection in semiconductor manufacturing.

Additionally, this project includes a **research-oriented study on few-shot learning under extreme data scarcity and class imbalance**.

---

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Technologies](#technologies)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Few-Shot Learning Experiments](#few-shot-learning-experiments)
- [Key Findings](#key-findings)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Objective

Automatically recognize defect types in wafer maps using both classical and deep learning approaches.

The project includes:

- Support Vector Machine (SVM)
- Logistic Regression (LR)
- Random Forest Classifier (RF)
- Convolutional Neural Networks (CNNs)

The goal is to study wafer defect classification under both:

- **standard supervised learning**
- **few-shot (limited-data) scenarios**

---

## Dataset

- **Source:** WM-811K Dataset  
- **Original availability:** Kaggle / MIR Lab  
- **Description:** Contains wafer map images with labeled defect categories.

Main defect classes used:

- edge-ring
- edge-loc
- center
- loc
- scratch
- random
- donut
- near-full

>  Dataset is not included due to size restrictions.

---

## Workflow

1. **Data Loading**
   - Load wafer map data from `LSWMD.pkl`

2. **Preprocessing**
   - Remove non-defect samples
   - Clean label formatting
   - Resize wafer maps (64×64)
   - Normalize pixel values

3. **Classical Machine Learning**
   - Logistic Regression
   - SVM
   - Random Forest

4. **Deep Learning**
   - CNN baseline
   - CNN with weighted loss
   - CNN with focal loss
   - CNN with focal loss + augmentation

5. **Few-Shot Experiments**
   - k = 5 samples/class
   - k = 10 samples/class
   - k = 20 samples/class

6. **Evaluation**
   - Accuracy
   - Confusion matrices
   - Class distribution analysis
   - Few-shot comparison plots

---

## Technologies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- PyTorch

---

## Repository Structure

```bash
.
├── src/
│   └── load_data.py
│
├── wafer_pipeline.py
├── wafer_cnn_pipeline.py
├── wafer_cnn_focal_aug.py
├── wafer_fewshot_focal_experiment.py
├── wafer_full_comparison_experiment.py   #  main research experiment
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── .gitignore
└── README.md
````

---

## Installation & Usage

```bash
git clone https://github.com/PanagiotaGr/wafer-fault-detection-with-ml.git
cd wafer-fault-detection-with-ml

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Run pipelines:

```bash
python wafer_pipeline.py
python wafer_cnn_pipeline.py
python wafer_cnn_focal_aug.py
```

Run research experiment:

```bash
python wafer_full_comparison_experiment.py
```

---

## Results

### Classical ML

| Model                     | Accuracy |
| ------------------------- | -------- |
| Logistic Regression       | 62.7%    |
| SVM                       | 65.7%    |
| Random Forest             | 78.5%    |
| Random Forest (optimized) | 79.4%    |

---

## Few-Shot Learning Experiments

| Samples per class | Baseline | Weighted | Focal | Focal + Aug |
| ----------------- | -------- | -------- | ----- | ----------- |
| 5                 | 0.20     | 0.13     | 0.07  | 0.20        |
| 10                | 0.43     | 0.43     | 0.37  | 0.10        |
| 20                | 0.52     | **0.57** | 0.18  | 0.47        |

---

##  Key Findings

* **Weighted loss improves performance under class imbalance**
*  **Focal loss underperforms in extremely low-data regimes**
* **Data augmentation can degrade performance when data is very limited**
*  Increasing samples per class significantly improves accuracy

---

##  Scientific Insight

This study shows that:

> In few-shot learning scenarios, **simple methods (e.g., weighted loss)** can outperform more complex approaches like focal loss.

---

## Outputs

Saved results include:

* confusion matrices
* training curves
* class distributions
* few-shot comparison plots

Example files:

```
outputs/figures/fewshot_accuracy_comparison.png
outputs/results/fewshot_summary.csv
```

---

## Acknowledgments

* WM-811K / LSWMD dataset
* Semiconductor defect detection research community

---

## License

Apache 2.0 License

```

