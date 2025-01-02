# Titanic Machine Learning from Disaster

## Overview

This project aims to apply machine learning techniques to the Titanic dataset to predict passenger survival. By exploring and analyzing various features, we will build and evaluate predictive models to understand the factors contributing to survival.

## Objectives

- Strengthen understanding of the machine learning pipeline from data preprocessing to model evaluation.
- Develop critical thinking and problem-solving skills through practical application.

## Data Description

The dataset is divided into two main parts:
- **Training Set**: `train.csv` (contains survival outcomes)
- **Test Set**: `test.csv` (no outcomes provided)

### Features Overview

| Variable | Definition                                      |
|----------|-------------------------------------------------|
| survival | Survival (0 = No, 1 = Yes)                     |
| pclass   | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)      |
| sex      | Sex                                            |
| age      | Age in years                                   |
| sibsp    | # of siblings/spouses aboard the Titanic       |
| parch    | # of parents/children aboard the Titanic       |
| ticket   | Ticket number                                  |
| cabin    | Cabin number                                   |
| embarked | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Tasks

### Task 1: Data Exploration and Visualization
- Load the Titanic dataset.
- Analyze key statistics for each feature.
- Visualize relationships between features.

### Task 2: Data Cleaning and Preprocessing
- Handle missing values.
- Encode categorical variables.
- Normalize/scale numerical features.
- Split the dataset into training and validation sets.

### Task 3: Feature Engineering
- Generate new features.
- Perform feature selection.

### Task 4: Model Selection and Training
- Train at least three different models (e.g., Logistic Regression, Random Forest, Support Vector Machines).
- Use cross-validation for model evaluation.
- Compare models using various metrics (accuracy, precision, recall, F1-score, ROC-AUC).

### Task 5: Model Optimization
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Evaluate optimized models on the validation dataset.

### Task 6: Testing and Submission
- Use the best model to predict outcomes on the test dataset.
- Submit results in a CSV file named `{first_name}_submission.csv`.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
