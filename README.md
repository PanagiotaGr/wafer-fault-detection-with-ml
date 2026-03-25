
# Semiconductor Wafer Defect Classification

This project explores how **machine learning** and **deep learning** can be applied to classify defect patterns on semiconductor wafers using real wafer map image data.  
The aim is to improve defect detection accuracy and reduce manual inspection in semiconductor manufacturing.

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

The goal is to study wafer defect classification under both **standard** and **limited-data** conditions.

---

## Dataset

- **Source:** WM-811K Dataset  
- **Original availability:** Kaggle / MIR Lab  
- **Description:** Contains wafer map images with labeled defect categories.

The main defect classes used in this project are:

- edge-ring
- edge-loc
- center
- loc
- scratch
- random
- donut
- near-full

> Note: The dataset is not included in the repository due to size restrictions.

---

## Workflow

1. **Data Loading**
   - Load wafer map data from `LSWMD.pkl`

2. **Preprocessing**
   - Remove null / non-defect categories
   - Clean label formatting
   - Resize wafer maps into uniform image inputs
   - Normalize images

3. **Classical Machine Learning**
   - Logistic Regression
   - SVM
   - Random Forest

4. **Deep Learning**
   - CNN baseline
   - CNN with weighted loss
   - CNN with Focal Loss and data augmentation

5. **Few-Shot Learning Experiments**
   - 5 samples per class
   - 10 samples per class
   - 20 samples per class

6. **Evaluation**
   - Accuracy
   - Classification reports
   - Confusion matrices
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
- Jupyter Notebook

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
├── download_dataset.py
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── .gitignore
└── README.md
````

### Main Scripts

* `wafer_pipeline.py`
  Classical ML baseline pipeline

* `wafer_cnn_pipeline.py`
  CNN baseline for wafer defect classification

* `wafer_cnn_focal_aug.py`
  CNN with Focal Loss and augmentation for class imbalance handling

* `wafer_fewshot_focal_experiment.py`
  Few-shot deep learning experiments under limited labeled data

---

## Installation & Usage

Clone this repository:

```bash
git clone https://github.com/PanagiotaGr/wafer-fault-detection-with-ml.git
cd wafer-fault-detection-with-ml
```

Create virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run classical ML pipeline:

```bash
python wafer_pipeline.py
```

Run CNN baseline:

```bash
python wafer_cnn_pipeline.py
```

Run CNN with focal loss and augmentation:

```bash
python wafer_cnn_focal_aug.py
```

Run few-shot experiments:

```bash
python wafer_fewshot_focal_experiment.py
```

---

## Results

### Classical ML Results

| Model                     | Accuracy | Notes                            |
| ------------------------- | -------- | -------------------------------- |
| Logistic Regression       | 62.7%    | Fast, but limited accuracy       |
| Support Vector Machine    | 65.7%    | Better generalization            |
| Random Forest             | 78.5%    | Best classical baseline          |
| Random Forest (optimized) | 79.4%    | Best overall classical ML result |

### Deep Learning Results

The project also includes CNN-based models for wafer defect classification:

* CNN baseline
* CNN with Focal Loss
* CNN with augmentation
* Few-shot CNN experiments

Saved outputs include:

* class distribution plots
* training curves
* confusion matrices
* few-shot comparison plots

---

## Few-Shot Learning Experiments

To simulate realistic industrial scenarios with limited labeled data, the project includes few-shot experiments using:

* **5 samples per class**
* **10 samples per class**
* **20 samples per class**

These experiments help evaluate:

* how performance changes when training data is scarce
* which classes are more sensitive to limited data
* whether Focal Loss and augmentation improve robustness

Main output file:

```bash
outputs/results/fewshot_summary.csv
```

Example output figures:

* `fewshot_accuracy_comparison.png`
* `fewshot_5_confusion_matrix.png`
* `fewshot_10_confusion_matrix.png`
* `fewshot_20_confusion_matrix.png`

---

## Acknowledgments

* Dataset provided by the semiconductor wafer defect research community
* WM-811K / LSWMD dataset
* Inspired by research in semiconductor manufacturing defect detection

---

## License

This project is licensed under the Apache 2.0 License.

```

