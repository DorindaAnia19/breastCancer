# Breast Cancer Detection using Support Vector Machine (SVM)

This project builds a **Support Vector Machine (SVM)** model to classify breast cancer tumors as **malignant (M)** or **benign (B)** using the **Breast Cancer Wisconsin Diagnostic Dataset (WDBC)**.  

The notebook walks through the **entire machine learning workflow**, from loading and exploring the dataset to training, tuning, and evaluating the model.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset Description](#dataset-description)  
3. [Project Workflow](#project-workflow)  
4. [Model Development](#model-development)  
5. [Results](#results)  
6. [Insights and Conclusion](#insights-and-conclusion)  
7. [Results Visualization](#results-visualization)  


---

## Project Overview

Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial for effective treatment and improved survival rates.  
This project uses **Support Vector Machine (SVM)** to distinguish between **malignant** and **benign** tumors based on cell nuclei characteristics.

---

## Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset (WDBC)  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  

Each record represents a tumor, characterized by 30 real-valued features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

| Column | Description |
|:--------|:-------------|
| ID | Unique identifier |
| Diagnosis | Target variable — M (Malignant), B (Benign) |
| radius_mean, texture_mean, ... | Mean values for cell nuclei features |
| *_se*, *_worst* | Standard error and worst-case values for each feature |

---

## Project Workflow

### 1. Importing Libraries and Loading the Dataset  
- Import essential libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.  
- Load and label the dataset columns for clarity.

```python
df = pd.read_csv("./breastCancerData/wdbc.data", header=None, names=column_names)
```
### 2. Exploratory Data Analysis (EDA)
I visualized:
- The distribution of diagnoses (Benign vs Malignant)
- Relationships between selected features using pair plots
- Correlations among all features using a heatmap

### 3. Data Preprocessing
Steps performed:
- Removed irrelevant columns (ID)
- Encoded Diagnosis (M → 1, B → 0)
- Split data into training (67%) and testing (33%) sets

### 4. Baseline Model — Support Vector Classifier
I trained an initial SVM model using default parameters and evaluated its performance using accuracy, confusion matrix, and classification report.

---

## Model Development

### 5. Feature Scaling
Support Vector Machines (SVMs) are sensitive to the scale of input features.  
To ensure fair comparison among features, the dataset was standardized using **StandardScaler**.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
---

## Results

| Metric | Benign (0) | Malignant (1) |
|--------|-------------|---------------|
| **Precision** | 0.98 | 0.96 |
| **Recall** | 0.97 | 0.97 |
| **F1-Score** | 0.98 | 0.97 |

**Overall Accuracy:** 97%  
**Macro Average F1-Score:** 0.97  

These results indicate that the SVM model performs exceptionally well at distinguishing between malignant and benign tumors.

---

## Insights and Conclusion

- The **Support Vector Machine (SVM)** achieved **97% accuracy**, showing strong predictive ability.  
- **Feature scaling** and **hyperparameter tuning** (C = 10, γ = 0.01) greatly improved performance.  
- The model demonstrates **high precision and recall** across both tumor classes, minimizing misclassification.  

---

## Results Visualization

Key visualizations generated in the notebook include:

- **Count Plot** — Distribution of benign vs malignant cases  
- **Pair Plot** — Relationships among selected features (radius, texture, perimeter, area)  
- **Heatmap** — Correlation matrix of numerical features  
- **Confusion Matrix** — Model performance visualization    

---

## Dependencies

To run this notebook, install the required Python packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn


