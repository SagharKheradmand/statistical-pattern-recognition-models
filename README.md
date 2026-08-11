# Statistical Pattern Recognition Models

## Overview

This project contains implementations and experiments developed for the Statistical Pattern Recognition course at Shiraz University.

The project focuses on fundamental supervised learning methods for regression and classification. Several algorithms are implemented and evaluated on real-world tabular datasets, with particular attention to preprocessing, optimization, model evaluation, and comparison of different learning strategies.

The project covers four main tasks:

1. Linear Regression
2. Binary Logistic Regression
3. Multiclass Logistic Regression
4. Ridge Regression

Most of the core learning algorithms are implemented from scratch using NumPy in order to better understand their mathematical foundations and optimization behavior.

---

# Project Tasks

## Task 1 - Linear Regression

The first task investigates simple linear regression using the Auto MPG dataset.

The objective is to predict vehicle fuel efficiency (`mpg`) using vehicle weight as the input feature.

Several approaches are compared:

- Scikit-learn Linear Regression
- Closed-Form Solution
- Batch Gradient Descent
- Stochastic Gradient Descent

---

## Dataset

The local Auto MPG dataset contains:

```text
392 samples
```

The available variables include:

- MPG
- Weight
- Horsepower
- Displacement

For the first experiment:

```text
Input:  Weight
Target: MPG
```

The relationship between vehicle weight and fuel efficiency is strongly negative.

The measured correlation is approximately:

```text
Correlation(weight, mpg) = -0.8322
```

This indicates that heavier vehicles generally have lower fuel efficiency.

---

## Data Splitting

The dataset is divided into training and testing subsets using:

```text
Training: 80%
Testing:  20%
```

with:

```text
random_state = 42
```

For gradient-based methods, the input feature is standardized using the mean and standard deviation calculated from the training data.

The same transformation is then applied to the test set.

---

# Closed-Form Linear Regression

Simple linear regression models the relationship between input `x` and target `y` as:

```text
y_hat = w0 + w1*x
```

The parameters can be estimated analytically using the normal equation.

The closed-form approach provides a direct solution without iterative optimization.

It is useful as a reference for comparing the results obtained using Gradient Descent.

---

# Batch Gradient Descent

Batch Gradient Descent updates the model parameters using the complete training dataset during each optimization step.

The general update rule is:

```text
w := w - alpha * gradient
```

where:

```text
alpha = learning rate
```

The Mean Squared Error objective is minimized during training.

The reported test performance was approximately:

```text
Test MSE  = 16.7196
Test RMSE = 4.0890
```

---

# Stochastic Gradient Descent

Stochastic Gradient Descent updates the parameters using individual training examples rather than the entire dataset.

This produces noisier parameter updates but can converge efficiently.

Different combinations of learning rate and number of epochs were evaluated.

The best reported configuration was:

```text
Learning rate = 0.01
Epochs        = 80
```

with:

```text
Test MSE = 16.1454
```

---

# Regression Evaluation

Regression models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Lower values indicate better prediction performance.

---

# Task 2 - Binary Logistic Regression

## Overview

The second task implements Binary Logistic Regression from scratch using the Bank Marketing dataset.

The objective is to predict whether a bank client subscribes to a term deposit.

The target variable contains two classes:

```text
yes
no
```

which are converted to:

```text
yes -> 1
no  -> 0
```

---

# Bank Marketing Dataset

The dataset contains:

```text
45,211 samples
17 columns
```

The target variable is:

```text
y
```

which indicates whether the customer subscribed to a term deposit.

The dataset contains client information, campaign information, contact information, and previous marketing outcomes.

---

# Class Imbalance

The target distribution is highly imbalanced.

The dataset contains approximately:

```text
No:  39,922
Yes:  5,289
```

Therefore, accuracy alone is not sufficient for evaluating model performance.

Metrics such as precision, recall, and F1-score are also considered.

---

# Data Preprocessing

The preprocessing pipeline includes:

1. Loading the semicolon-separated dataset
2. Converting the target variable to binary values
3. Encoding binary categorical variables
4. One-hot encoding nominal categorical variables
5. Feature selection
6. Train/test splitting
7. Duplicate removal
8. Feature standardization

The split is performed using:

```text
Training: 80%
Testing:  20%
```

with stratification and:

```text
random_state = 42
```

---

# Feature Selection

Features are ranked according to their absolute correlation with the target.

Seven highly correlated features are selected:

```text
duration
poutcome_success
poutcome_unknown
contact_unknown
housing
contact_cellular
month_mar
```

These features are then used to train the binary classifier.

---

# Logistic Regression

Binary Logistic Regression models the probability of the positive class using the sigmoid function:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

where:

```text
z = w^T x + b
```

The model outputs:

```text
P(y = 1 | x)
```

Predicted probabilities are converted into binary labels using a classification threshold.

---

# Binary Cross-Entropy Loss

The model is trained by minimizing Binary Cross-Entropy:

```text
L = -[y log(p) + (1-y) log(1-p)]
```

Gradient Descent is used to update the model parameters.

Different learning rates and epoch counts are tested to analyze convergence behavior.

---

# Binary Classification Results

The best model according to test F1-score used:

```text
Learning rate = 0.1
Epochs        = 600
```

The reported test results were:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.8997 |
| Precision | 0.6370 |
| Recall | 0.3318 |
| F1-Score | 0.4363 |

The test confusion matrix was:

```text
[[7784, 200],
 [ 707, 351]]
```

The relatively high accuracy is influenced by the class imbalance.

Recall and F1-score provide additional information about the model's ability to identify the minority positive class.

---

# Task 3 - Multiclass Logistic Regression

## Overview

The third task investigates multiclass classification using the Wine dataset.

Three different multiclass strategies are implemented and compared:

- One-vs-All Logistic Regression
- One-vs-One Logistic Regression
- Softmax Regression

---

# Wine Dataset

The Wine dataset contains:

```text
178 samples
```

with:

```text
13 input features
```

and one class label.

The features represent chemical measurements of different wine samples.

---

# Wine Data Preprocessing

The preprocessing pipeline includes:

1. Loading the dataset
2. Encoding class labels
3. Stratified train/test splitting
4. Removing duplicate training samples
5. Handling missing values
6. Optional outlier removal
7. Min-Max normalization

The dataset is split using:

```text
Training: 75%
Testing:  25%
```

with:

```text
random_state = 42
```

---

# Outlier Detection

An experiment is performed both with and without outlier removal.

Training outliers are identified using Z-scores with a threshold of:

```text
2.75
```

The experiment reports:

```text
11 training outliers removed
```

The purpose is to study whether removing unusual samples improves multiclass classification performance.

---

# One-vs-All Logistic Regression

One-vs-All trains one binary classifier for each class.

For each classifier:

```text
Positive class = one target class
Negative class = all remaining classes
```

For three classes, three binary classifiers are trained.

During prediction, the class with the highest predicted score is selected.

---

# One-vs-One Logistic Regression

One-vs-One trains a separate classifier for every pair of classes.

For `K` classes, the number of classifiers is:

```text
K(K - 1) / 2
```

Each classifier learns to distinguish between two classes.

During prediction, classifiers vote for their preferred class.

The class receiving the most votes becomes the final prediction.

---

# Softmax Regression

Softmax Regression directly models probabilities across all classes.

For class `k`:

```text
P(y = k | x) =
exp(z_k) / sum_j exp(z_j)
```

Unlike OvA and OvO, Softmax Regression trains a single multiclass model.

The output probabilities sum to:

```text
1
```

across all classes.

---

# Multiclass Results

The reported test accuracies are:

| Outlier Processing | OvA | OvO | Softmax |
| --- | ---: | ---: | ---: |
| With outlier removal | 1.0000 | 1.0000 | 1.0000 |
| Without outlier removal | 0.9778 | 1.0000 | 1.0000 |

The results show excellent classification performance on the Wine dataset.

OvO and Softmax achieved perfect test accuracy in both reported experiments.

OvA achieved perfect accuracy after outlier removal and slightly lower accuracy without outlier removal.

---

# Task 4 - Ridge Regression

## Overview

The fourth task investigates Ridge Regression and the effect of L2 regularization on correlated input features.

The Auto MPG dataset is used again.

This time, three input variables are used:

```text
weight
horsepower
displacement
```

The target remains:

```text
mpg
```

---

# Multicollinearity

The selected vehicle characteristics are strongly related to one another.

Multicollinearity is investigated using the Variance Inflation Factor (VIF).

The reported VIF values are:

| Feature | VIF |
| --- | ---: |
| Weight | 7.9574 |
| Horsepower | 5.2873 |
| Displacement | 10.3105 |

These values indicate notable multicollinearity among the selected predictors.

---

# Ridge Regression

Ridge Regression extends ordinary linear regression by adding an L2 penalty:

```text
L_ridge =
MSE + lambda * sum(w_j^2)
```

where:

```text
lambda
```

controls the strength of regularization.

Increasing `lambda` penalizes large model coefficients and can reduce sensitivity to correlated input variables.

---

# Ridge Regression with SGD

The Ridge Regression model is optimized using mini-batch Stochastic Gradient Descent.

Multiple regularization strengths are evaluated:

```text
lambda = 0
lambda = 0.01
lambda = 0.1
lambda = 1.0
lambda = 5.0
```

---

# Ridge Regression Results

The reported results are:

| Lambda | Train MSE | Test MSE |
| ---: | ---: | ---: |
| 0.00 | 17.9995 | 18.3018 |
| 0.01 | 18.0116 | 18.2602 |
| 0.10 | 18.1644 | 17.9342 |
| 1.00 | 21.5989 | 17.9018 |
| 5.00 | 37.0843 | 28.7433 |

The results demonstrate the effect of regularization strength.

Moderate regularization improves test performance compared with the unregularized model.

However, very strong regularization causes underfitting and increases both training and test error.

Among the tested values:

```text
lambda = 1.0
```

produced the lowest reported test MSE:

```text
17.9018
```

---

# Visual Results

The project contains several saved visualizations.

## Bank Marketing Class Distribution

```text
class_distribution_y.png
```

Shows the imbalance between positive and negative target classes.

## Feature Histograms

```text
hist_duration.png
hist_balance.png
```

These figures show the distributions of selected standardized Bank Marketing features.

## Binary Logistic Regression Loss Curves

```text
loss_curves.png
```

Shows the optimization behavior for different learning-rate and epoch configurations.

## Binary Confusion Matrix

```text
confusion_matrix.png
```

Visualizes the binary classification results on the Bank Marketing test set.

## Multiclass OvR Loss Curves

```text
ovr_loss_curves.png
```

Shows the training loss of the One-vs-All classifiers.

## Multiclass Confusion Matrix

```text
confusion_matrix_multiclass.png
```

Visualizes the classification results on the Wine dataset.

---

# Project Structure

```text
statistical-pattern-recognition-models/
|
|-- 1.ipynb
|-- 2.ipynb
|-- 3.ipynb
|-- 4.ipynb
|
|-- 1.pdf
|-- 2.pdf
|-- 3.pdf
|-- 4.pdf
|
|-- auto_mpg.csv.csv
|-- bank-full.csv
|-- wine_dataset.csv
|
|-- class_distribution_y.png
|-- hist_balance.png
|-- hist_duration.png
|-- loss_curves.png
|-- confusion_matrix.png
|-- ovr_loss_curves.png
|-- confusion_matrix_multiclass.png
|
|-- requirements.txt
|-- README.md
`-- Homework1.pdf
```

---

# File Description

| File | Description |
| --- | --- |
| `1.ipynb` | Linear Regression experiments using the Auto MPG dataset |
| `2.ipynb` | Binary Logistic Regression using the Bank Marketing dataset |
| `3.ipynb` | Multiclass Logistic Regression using the Wine dataset |
| `4.ipynb` | Ridge Regression and multicollinearity analysis |
| `1.pdf` | Report for the Linear Regression task |
| `2.pdf` | Report for the Binary Logistic Regression task |
| `3.pdf` | Report for the Multiclass Classification task |
| `4.pdf` | Report for the Ridge Regression task |
| `auto_mpg.csv.csv` | Auto MPG dataset used for regression experiments |
| `bank-full.csv` | Bank Marketing dataset used for binary classification |
| `wine_dataset.csv` | Wine dataset used for multiclass classification |
| `requirements.txt` | Required Python packages |

---

# Evaluation Metrics

Different metrics are used depending on the learning problem.

### Regression

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error

### Classification

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Multicollinearity

- Correlation
- Variance Inflation Factor

---

# Technologies and Methods

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- statsmodels
- Linear Regression
- Gradient Descent
- Stochastic Gradient Descent
- Logistic Regression
- Binary Cross-Entropy
- One-vs-All Classification
- One-vs-One Classification
- Softmax Regression
- Ridge Regression
- L2 Regularization
- Feature Standardization
- Min-Max Normalization
- Outlier Detection
- Feature Selection
- Multicollinearity Analysis

---

# Key Concepts Demonstrated

This project demonstrates several fundamental concepts in Statistical Pattern Recognition and Machine Learning:

- Supervised learning
- Regression
- Binary classification
- Multiclass classification
- Closed-form parameter estimation
- Gradient-based optimization
- Stochastic Gradient Descent
- Logistic probability modeling
- Sigmoid activation
- Binary Cross-Entropy
- One-vs-All classification
- One-vs-One classification
- Softmax classification
- Data preprocessing
- Feature encoding
- Feature normalization
- Class imbalance
- Outlier analysis
- Model evaluation
- Confusion matrices
- Regularization
- Multicollinearity
- Bias-variance trade-off

---

# Conclusion

This project explores several fundamental supervised learning algorithms used in Statistical Pattern Recognition.

The first task compares analytical and gradient-based approaches for linear regression. The second task implements binary logistic regression from scratch on an imbalanced real-world dataset. The third task extends logistic regression to multiclass problems using One-vs-All, One-vs-One, and Softmax approaches. Finally, Ridge Regression is used to investigate regularization and multicollinearity.

Together, these experiments provide practical experience with data preprocessing, optimization, model evaluation, classification strategies, and regularization while reinforcing the mathematical foundations of statistical machine learning.

---

# Course Information

**Course:** Statistical Pattern Recognition  
**University:** Shiraz University  
**Homework:** 1  
**Year:** 2026

---

# Author

Saghar Kheradmand
