# COMSYS-HAckhathon: AI-Powered Mental Health Classification & Football Player Valuation
[![Hackathon Ranking](https://img.shields.io/badge/Hackathon-Top%2012-brightgreen)](https://comsysconf.org/) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
## 🎯 Project Overview
Welcome to our **COMSYS-HAckhathon** submission! This repository showcases a dual ML pipeline built during the hackathon:
1. **Mental Health Disorder Classification**: A multiclass text classifier that analyzes patient statements to detect disorders like Anxiety, Depression, and more—promoting early mental health intervention.
2. **Football Player Market Value Prediction**: A regression model forecasting player valuations based on performance stats, blending sports analytics with AI.
We ranked **Top 12** out of 150+ teams in the #COMSYS-HAckhathon, earning accolades for innovative use of NLP and regression techniques on real-world datasets. This project demonstrates end-to-end ML workflows, from data preprocessing to model deployment, emphasizing interpretability and scalability.
## 🚀 Key Features
- **Text Classification Pipeline**:
  - Preprocessing: Tokenization, lemmatization, and stopword removal.
  - Vectorization: TF-IDF for sparse text representation.
  - Models: Multinomial Naive Bayes (87% accuracy), SVM, Decision Trees, and MLP Neural Networks.
 
- **Regression Pipeline**:
  - Feature Engineering: PowerTransformer for Gaussian normalization + RobustScaler for outlier handling.
  - Model: Tuned SVR (RBF kernel) with GridSearchCV for hyperparameter optimization.
  - Evaluation: Cross-validated RMSE (~11.48 train / 22.44 test) on skewed market value data.
- **Achievements**:
  - Balanced F1-scores across 5 mental health classes.
  - Top feature importance via linear SVR coefficients (e.g., "Progressive Passes" dominates valuation).
  - Hackathon spotlight: Praised for ethical AI in mental health and practical sports insights.
## 🧮 Theoretical Foundations
### 1. TF-IDF Vectorization (Text Classification)
TF-IDF transforms text into numerical features, weighting terms by importance:
$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right)
$$
Where:
- \(\text{TF}(t, d)\): Term frequency in document \(d\).
- \(|D|\): Total documents.
- \(|\{d \in D : t \in d\}|\): Documents containing term \(t\).
This reduces noise, highlighting disorder-specific keywords (e.g., "panic" for Panic Disorder).
### 2. Multinomial Naive Bayes (Best Classifier)
Probabilistic prediction via Bayes' Theorem, assuming feature independence:
$$
P(c|d) = \frac{P(d|c) \cdot P(c)}{P(d)} \propto P(c) \prod_{i=1}^{n} P(t_i|c)
$$
- \(c\): Class (e.g., "Depression").
- \(d\): Document.
- \(t_i\): Terms in \(d\).
Achieved 87% weighted F1-score, outperforming trees (64%) due to sparsity handling.
### 3. Support Vector Regression (SVR) Objective
For player valuation, SVR minimizes errors while controlling complexity:
$$
\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$
Subject to:
$$
y_i - (w \cdot \phi(x_i) + b) \leq \epsilon + \xi_i, \quad (w \cdot \phi(x_i) + b) - y_i \leq \epsilon + \xi_i^*
$$
- \(\phi\): RBF kernel for non-linearity.
- \(C=1000\), \(\gamma=0.1\): Tuned params for robust predictions on skewed targets.
Feature importance: \(w_j\) coefficients rank stats like "Age" and "Assists" highest.
## 📊 Results Snapshot
### Mental Health Classification Report (Naive Bayes)
| Class | Precision | Recall | F1-Score | Support |
|--------------------------------|-----------|--------|----------|---------|
| Anger/IED | 0.86 | 0.83 | 0.84 | 154 |
| Anxiety Disorder | 0.84 | 0.90 | 0.86 | 153 |
| Depression | 0.77 | 0.88 | 0.82 | 208 |
| Narcissistic Disorder | 0.99 | 0.99 | 0.99 | 158 |
| Panic Disorder | 0.99 | 0.67 | 0.80 | 112 |
| **Macro Avg** | 0.89 | 0.85 | 0.86 | 785 |
| **Weighted Avg** | 0.88 | 0.87 | 0.87 | 785 |
### Player Valuation (SVR RMSE)
- Train: £11.48M
- Test: £22.44M
- R² Score: ~0.75 (strong fit despite market volatility)
## 🛠️ Quick Start
### Prerequisites
- Python 3.8+
- Libraries: `pip install pandas scikit-learn nltk matplotlib seaborn`
### Installation
```bash
git clone https://github.com/sksohel27/COMSYS-HAckhathon.git
cd COMSYS-HAckhathon
pip install -r requirements.txt
```
### Run Classification
```bash
jupyter notebook "Text_classification running-..." # Trains & predicts on train.csv/test.csv (replace with exact filename)
```
### Run Regression
```bash
jupyter notebook "Value Prediction using svr..." # Predicts values, saves to Value_prediction_grid.csv (replace with exact filename)
```
### Datasets
- **Note**: Due to hackathon privacy rules and data sharing restrictions, the original datasets (`train.csv` and `test.csv` for both tasks) are not included in this repository.
- **Mental Health**: Simulated with anonymized text data for multiclass classification. For similar public datasets, check Kaggle's "Mental Health in Tech" or "Depression Detection" datasets.
- **Football**: Based on player performance stats for regression. Replicate with public sources like Kaggle's "European Soccer Database" or Transfermarkt exports (ensure compliance with usage terms).
- To run the code, generate or download compatible CSV files matching the expected schema (e.g., 'Text'/'label' for classification; stats columns + 'Value at beginning of 2023/24 season' for regression).
## 🤝 Contributing
We welcome PRs! Fork, branch, and submit. Focus on ethical AI enhancements (e.g., bias audits).
## 📄 License
MIT License – Free to use, modify, and distribute.
## 🌟 Acknowledgments
- #COMSYS-HAckhathon organizers for the inspiring challenge.
- Scikit-learn & NLTK teams for robust tools.
- Our team: sksohel27 – Lead ML Engineer.
---
*Built with ❤️ during #COMSYS-HAckhathon. Let's AI for good! 🚀*
[Star this repo](https://github.com/sksohel27/COMSYS-HAckhathon) if it sparks ideas!
