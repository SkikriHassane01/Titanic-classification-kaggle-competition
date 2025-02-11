# Titanic Classification - Kaggle Competition

## Overview

This repository contains my work on the Titanic classification problem from Kaggle. The objective of this competition is to predict whether a passenger survived or not based on different features like age, fare, and class.

I created two versions of the notebook:

1. **Manual Model Selection:** In this version, I performed feature engineering and data preprocessing, then evaluated different models using Stratified K-Fold cross-validation.
2. **LazyClassifier Approach:** In this version, I used the LazyClassifier library to quickly compare multiple models and identify the best-performing one.

## Approach

### Version 1: Manual Model Selection

- Implemented a preprocessing pipeline to clean and transform the dataset.

- Applied feature engineering techniques to extract meaningful information.

- Evaluated different machine learning models:

  - **Logistic Regression**
  - **Random Forest Classifier**
  - **Decision Tree Classifier**
  - **K-Nearest Neighbors**
  - **Linear Discriminant Analysis**
  - **Gaussian Naive Bayes**
  - **Support Vector Machine (SVM)**

- Used **Stratified K-Fold Cross-Validation** to evaluate model performance.

- Identified **RandomForestClassifier** as the best model with the best accuracy

- Performed **GridSearchCV** for hyperparameter optimization, achieving a **test accuracy of 0.8507**.

### Version 2: LazyClassifier Approach

- Used **LazyClassifier** to automate model comparison.

- Evaluated multiple classifiers based on:

  - **Accuracy**
  - **Balanced Accuracy**
  - **ROC AUC Score**
  - **F1 Score**

- The top-performing models were:

  - **RandomForestClassifier** (Accuracy: **0.86**)
  - **XGBClassifier** (Accuracy: **0.85**)
  - **LGBMClassifier** (Accuracy: **0.85**)

- Achieved a final **test accuracy of 0.8603**.

### Model Performance Summary (LazyClassifier)

| Model                  | Accuracy | Balanced Accuracy | ROC AUC | F1 Score |
| ---------------------- | -------- | ----------------- | ------- | -------- |
| RandomForestClassifier | 0.86     | 0.82              | 0.82    | 0.86     |
| XGBClassifier          | 0.85     | 0.82              | 0.82    | 0.85     |
| LGBMClassifier         | 0.85     | 0.82              | 0.82    | 0.85     |
| LogisticRegression     | 0.83     | 0.80              | 0.80    | 0.83     |
| GaussianNB             | 0.82     | 0.80              | 0.80    | 0.82     |

## Conclusion

- **RandomForestClassifier** was the best model in both approaches.
- **LazyClassifier** provided a quick and effective way to compare models
- **GridSearchCV tuning** improved the model performance, achieving **0.8603  test accuracy**.
- Further improvements can be made by fine-tuning hyperparameters and using ensemble methods.

## How to Run

1. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Acknowledgments

- Kaggle Titanic Competition
- Scikit-learn, LazyPredict, XGBoost, LightGBM

