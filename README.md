# Pure Python for Data Science & Machine Learning

A hands-on tutorial project that implements fundamental machine learning and data science algorithms **from scratch using pure Python** — no scikit-learn, no TensorFlow, no PyTorch for the core algorithms. The goal is to build deep understanding of how these algorithms actually work under the hood.

## Who Is This For?

- Students learning ML/DS who want to understand the math and mechanics behind the algorithms
- Developers who want to go beyond calling `model.fit()` and understand what happens inside
- Anyone preparing for ML interviews where algorithmic understanding is tested

## Prerequisites

- Python 3.6+
- Basic understanding of linear algebra (vectors, matrices, dot products)
- Familiarity with calculus (derivatives, chain rule)
- Basic probability and statistics knowledge

## How to Use

1. Clone this repository
2. Follow the chapters **in numerical order** (0 through 19)
3. Each chapter contains Jupyter notebooks (`.ipynb`) for theory and Python scripts (`.py`) for implementations
4. The `X.Kaggle_Practice_Projects/` folder contains end-to-end projects applying each algorithm to real datasets
5. Datasets are stored in `Y.Kaggle_Data/`

## Table of Contents

### Part I: Foundations

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [0. Statistics Supplement](0.Statistics_Supplement/) | Descriptive & inferential statistics | Mean, median, mode, variance, hypothesis testing, odds & log-odds |
| [1. Finding and Reading Data](1.Finding_and_Reading_Data/) | Data I/O | CSV parsing, string-to-float conversion |
| [2. Data Preprocessing](2.Data_Preprocessing/) | Data preparation | Min-max normalization, z-score standardization, feature engineering |
| [3. Resampling Methods](3.Resampling_Methods/) | Train/test strategies | Train/test split, k-fold cross-validation |

### Part II: Evaluation Metrics

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [4. Evaluating Accuracy](4.Evaluating_Accuracy/) | Classification & advanced metrics | Accuracy, precision, recall, F1-score, ROC curve, AUC |
| [5. Confusion Matrix](5.Confusion_Matrix/) | Classification evaluation | Multi-class confusion matrix |
| [6. MAE and RMSE](6.MAE_and_RMSE/) | Regression evaluation | Mean Absolute Error, Root Mean Squared Error, R-squared |
| [7. Baseline Models](7.Baseline_Models/) | Benchmarking | Random prediction, ZeroR algorithm |

### Part III: Linear Models

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [8. Linear Regression](8.Linear_Regression/) | Regression | OLS, covariance, correlation, regularization (Ridge/Lasso) |
| [9. Stochastic Gradient Descent](9.Stochastic_Gradient_Descent/) | Optimization | SGD algorithm, learning rate, convergence |
| [10. Logistic Regression](10.Logistic_Regression/) | Binary classification | Sigmoid function, maximum likelihood, regularization |

### Part IV: Classic ML Algorithms

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [11. Perceptron](11.Perceptron/) | Linear classifier | Step function, perceptron learning rule, linear separability |
| [12. Decision Trees](12.Classification_and_Decision_Tree/) | Tree-based models | CART, Gini impurity, recursive splitting, pruning |
| [13. Naive Bayes](13.Naive_Bayes/) | Probabilistic classifier | Bayes theorem, conditional independence, Gaussian NB |
| [14. K-Nearest Neighbor](14.K_Nearest_Neighbor/) | Instance-based learning | Euclidean distance, choosing k, lazy learning |
| [15. Learning Vector Quantization](15.Learning_Vector_Quantization/) | Prototype-based | Codebook vectors, BMU, competitive learning |

### Part V: Neural Networks & Advanced Topics

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [16. Neural Networks](16.Artificial_Neural_Network_and_Backpropagation/) | Deep learning foundations | Forward/backward propagation, sigmoid, weight updates |
| [17. K-Means Clustering](17.K_Means_Clustering/) | Unsupervised learning | Centroid initialization, cluster assignment, elbow method, K-Means++ |
| [18. PCA](18.Principal_Component_Analysis/) | Dimensionality reduction | Covariance matrix, eigendecomposition, explained variance |
| [19. Support Vector Machine](19.Support_Vector_Machine/) | Maximum margin classifier | Hinge loss, kernel trick, soft margin, SGD-based SVM |

### Part VI: Ensemble Methods

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [Ensemble Algorithms](PlusPlus.Ensemble_Algo/) | Model combination | Bootstrap, bagging, random forests, boosting concepts |

### Practice Projects

| Project | Algorithm | Dataset |
|---------|-----------|---------|
| [case00](X.Kaggle_Practice_Projects/case00_predict_insurance_by_simple_linear_regression.py) | Simple Linear Regression | Insurance costs |
| [case01](X.Kaggle_Practice_Projects/case01_judge_the_wine_quality_by_sgd_method.py) | Linear Regression via SGD | Wine quality |
| [case02](X.Kaggle_Practice_Projects/case02_predict_diabetes_using_logistic_regression.py) | Logistic Regression | Diabetes prediction |
| [case03](X.Kaggle_Practice_Projects/case03_predict_Sonar_data_using_perceptron.py) | Perceptron | Sonar classification |
| [case04](X.Kaggle_Practice_Projects/case04_banknote_auth_using_CART.py) | CART Decision Tree | Banknote authentication |
| [case05](X.Kaggle_Practice_Projects/case05_abalone_age_using_KNN.py) | KNN | Abalone age prediction |
| [case06](X.Kaggle_Practice_Projects/case06_ionosphere_radar_LVQ.py) | LVQ | Ionosphere radar signals |
| [case07](X.Kaggle_Practice_Projects/case07_wheat_classification_using_artificial_neural_network.py) | Neural Network | Wheat seed classification |
| [case08](X.Kaggle_Practice_Projects/case08_using_bagging_to_predict_sonar_data.py) | Bagging | Sonar classification |
| [case09](X.Kaggle_Practice_Projects/case09_using_random_forest_to_predict_sonar_data.py) | Random Forest | Sonar classification |

## Learning Path

```
Statistics & Data Handling (Ch. 0-3)
         |
         v
Evaluation Metrics (Ch. 4-7)
         |
         v
Linear Models (Ch. 8-10)
         |
    +----+----+
    |         |
    v         v
Classic ML    Neural Networks
(Ch. 11-15)   (Ch. 16)
    |         |
    +----+----+
         |
         v
Unsupervised & Advanced (Ch. 17-19)
         |
         v
Ensemble Methods (PlusPlus)
         |
         v
Practice Projects (X.Kaggle_Practice_Projects)
```

## Project Philosophy

- **No black boxes**: Every algorithm is implemented step-by-step so you can see exactly how it works
- **Pure Python first**: Core algorithms use only Python's standard library (`math`, `random`, `csv`)
- **Optional visualization**: Some notebooks use `matplotlib`/`seaborn` for plots, but these are optional and wrapped in try/except blocks
- **Learn by doing**: Each chapter includes working code you can run, modify, and experiment with

## Installation

```bash
git clone https://github.com/your-username/Pure_Python_for_DS_ML.git
cd Pure_Python_for_DS_ML
pip install -r requirements.txt  # optional, only for visualization
jupyter notebook
```

## License

This project is for educational purposes. Feel free to use and modify for learning.

## Author

William
