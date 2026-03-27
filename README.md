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
2. Follow the chapters **in numerical order** (0 through 80)
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

### Part VII: Advanced ML (Industry Track)

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [20. Feature Engineering and Data Leakage](20.Feature_Engineering_and_Data_Leakage/1.Feature_Engineering_and_Data_Leakage.ipynb) | Robust feature pipelines | Leakage prevention, train-only transforms, feature quality checks |
| [21. Model Selection and Hyperparameter Optimization](21.Model_Selection_and_Hyperparameter_Optimization/1.Model_Selection_and_HPO.ipynb) | Reliable model tuning | Nested CV, random search, experiment tracking |
| [22. Gradient Boosting](22.Gradient_Boosting/1.Gradient_Boosting_from_Scratch.ipynb) | Additive tree learning | Residual fitting, shrinkage, regularization |
| [23. Time Series Forecasting](23.Time_Series_Forecasting/1.Time_Series_Forecasting.ipynb) | Temporal ML systems | Walk-forward validation, lag features, forecasting baselines |
| [24. Recommender Systems](24.Recommender_Systems/1.Recommender_Systems.ipynb) | Ranking systems | Matrix factorization, retrieval/ranking pipeline, top-K evaluation |
| [25. ML Engineering and MLOps](25.ML_Engineering_and_MLOps/1.ML_Engineering_and_MLOps.ipynb) | Productionization | Monitoring, deployment strategy, retraining policies |

### Part VIII: Deep Learning (Industry Track)

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [26. Deep Learning Fundamentals](26.Deep_Learning_Fundamentals/1.Deep_Learning_Fundamentals.ipynb) | Optimization and backprop | Autograd, initialization, regularization, MLP training loops |
| [27. CNN and Computer Vision](27.CNN_and_Computer_Vision/1.CNN_and_Computer_Vision.ipynb) | Vision modeling | Convolution, transfer learning, augmentation, slice-based evaluation |
| [28. Sequence Models and Transformers](28.Sequence_Models_and_Transformers/1.Sequence_Models_and_Transformers.ipynb) | Modern sequence learning | Self-attention, transformer blocks, fine-tuning workflow |
| [29. Deep Learning Operations](29.Deep_Learning_Operations/1.Deep_Learning_Operations.ipynb) | Scalable DL training | AdamW, LR schedules, checkpointing, reproducibility |

### Part IX: Advanced Algorithms (Expert Track)

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [30. Random Forest Advanced](30.Random_Forest_Advanced/1.Random_Forest_Advanced.ipynb) | Ensemble tree internals | Bootstrap aggregation, random feature splits, variance reduction |
| [31. AdaBoost Classification](31.AdaBoost_Classification/1.AdaBoost_Classification.ipynb) | Adaptive boosting | Weighted stumps, margin optimization, sample reweighting |
| [32. GMM and EM](32.Gaussian_Mixture_Model_and_EM/1.GMM_and_EM.ipynb) | Probabilistic clustering | Soft assignments, EM updates, likelihood optimization |
| [33. DBSCAN and HAC](33.DBSCAN_and_Hierarchical_Clustering/1.DBSCAN_and_HAC.ipynb) | Structure discovery | Density clustering, outlier handling, agglomerative merging |
| [34. ARIMA and Exponential Smoothing](34.ARIMA_and_Exponential_Smoothing/1.ARIMA_and_Exponential_Smoothing.ipynb) | Classical forecasting | Differencing, autoregression, smoothing baselines |
| [35. Hidden Markov Models](35.Hidden_Markov_Models/1.Hidden_Markov_Models.ipynb) | Sequence probability models | Forward algorithm, Viterbi decoding, latent regimes |
| [36. Autoencoders from Scratch](36.Autoencoders_From_Scratch/1.Autoencoders_From_Scratch.ipynb) | Representation learning | Encoder-decoder training, reconstruction error, anomaly scoring |
| [37. Q-Learning](37.Reinforcement_Learning_Q_Learning/1.Q_Learning.ipynb) | Reinforcement learning | Bellman update, exploration-exploitation, policy extraction |

### Part X: HF + Kaggle Pure Python Master Track

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [38. Tokenization and BPE from Scratch](38.Tokenization_and_BPE_From_Scratch/1.Tokenization_and_BPE_From_Scratch.ipynb) | NLP data pipeline internals | Pre-tokenization, BPE merges, compression analysis, token frequency visualization |
| [39. Attention and Transformer Internals (Advanced)](39.Attention_and_Transformer_Internals_Advanced/1.Attention_and_Transformer_Internals_Advanced.ipynb) | Transformer mechanics | Multi-head attention, causal masking, entropy analysis, attention heatmaps |
| [40. PEFT and LoRA Fine-Tuning](40.PEFT_LoRA_and_FineTuning_Pure_Python/1.PEFT_LoRA_and_FineTuning_Pure_Python.ipynb) | Efficient adaptation | Low-rank adapters, frozen base weights, parameter efficiency, loss curve visualization |
| [41. RAG and Vector Search](41.RAG_and_Vector_Search_Pure_Python/1.RAG_and_Vector_Search_Pure_Python.ipynb) | Retrieval-augmented generation | TF-IDF embeddings, cosine retrieval, grounded answer synthesis, score visualization |
| [42. Kaggle EDA and Visualization Playbook](42.Kaggle_EDA_and_Visualization_Playbook/1.Kaggle_EDA_and_Visualization_Playbook.ipynb) | Competition diagnostics | Distribution checks, missingness analysis, target breakdown, chart-driven hypothesis generation |
| [43. Kaggle Feature Engineering and Leakage (Advanced)](43.Kaggle_Feature_Engineering_and_Leakage_Advanced/1.Kaggle_Feature_Engineering_and_Leakage_Advanced.ipynb) | Robust feature pipelines | Fold-safe target encoding, leakage diagnostics, fold variance analysis |
| [44. Kaggle Ensembling, Stacking, and Blending](44.Kaggle_Ensembling_Stacking_and_Blending/1.Kaggle_Ensembling_Stacking_and_Blending.ipynb) | Leaderboard optimization | OOF stacking, blending strategies, ensemble diversity, comparative visualization |
| [45. Model Monitoring, Drift, and Visualization](45.Model_Monitoring_Drift_and_Visualization/1.Model_Monitoring_Drift_and_Visualization.ipynb) | Production reliability | PSI drift metrics, trend dashboards, alert thresholds, retraining triggers |

### Part XI: Frontier Algorithms Track

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [46. SARSA and DQN Concepts](46.SARSA_and_DQN_Concepts/1.SARSA_and_DQN_Concepts.ipynb) | Reinforcement learning progression | On-policy TD updates, SARSA vs Q-learning, DQN intuition via function approximation |
| [47. BM25 and Ranking from Scratch](47.BM25_and_Ranking_From_Scratch/1.BM25_and_Ranking_From_Scratch.ipynb) | Sparse retrieval | BM25 scoring, TF saturation, length normalization, ranking explanation |
| [48. HNSW-Style ANN Search](48.HNSW_Style_ANN_Search/1.HNSW_Style_ANN_Search.ipynb) | Scalable retrieval systems | Navigable graph search, approximate kNN, recall-latency tradeoff |
| [49. Bayesian Optimization from Scratch](49.Bayesian_Optimization_From_Scratch/1.Bayesian_Optimization_From_Scratch.ipynb) | Efficient hyperparameter tuning | Gaussian Process surrogate, Expected Improvement, sequential optimization |
| [50. Causal Uplift Modeling from Scratch](50.Causal_Uplift_Modeling_From_Scratch/1.Causal_Uplift_Modeling_From_Scratch.ipynb) | Intervention modeling | Treatment effect estimation, T-learner, uplift ranking, policy gain |

### Part XII: Advanced Research-to-Production Concepts

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [51. PPO Concepts and Policy Optimization](51.PPO_Concepts_and_Policy_Optimization/1.PPO_Concepts_and_Policy_Optimization.ipynb) | Policy-gradient stabilization | Clipped surrogate objective, advantage weighting, stable policy updates |
| [52. Contrastive Learning (SimCLR Intuition)](52.Contrastive_Learning_SimCLR_Intuition/1.Contrastive_Learning_SimCLR_Intuition.ipynb) | Self-supervised representations | NT-Xent loss, positive/negative pairs, embedding geometry |
| [53. Graph Neural Network Basics](53.Graph_Neural_Network_Basics/1.Graph_Neural_Network_Basics.ipynb) | Graph representation learning | Message passing, neighborhood aggregation, over-smoothing intuition |
| [54. Calibration and Uncertainty Estimation](54.Calibration_and_Uncertainty_Estimation/1.Calibration_and_Uncertainty_Estimation.ipynb) | Reliable probabilities | Reliability diagrams, ECE, temperature scaling |
| [55. A/B Testing and Sequential Experimentation](55.AB_Testing_and_Sequential_Experimentation/1.AB_Testing_and_Sequential_Experimentation.ipynb) | Experiment decision science | Proportion tests, sequential monitoring, alpha correction |

### Part XIII: Production-Scale Systems and Decision Intelligence

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [56. Distributed Training Concepts](56.Distributed_Training_Concepts/1.Distributed_Training_Concepts.ipynb) | Scalable optimization systems | Data-parallel SGD, sync vs async updates, gradient staleness |
| [57. Feature Store and Offline/Online Consistency](57.Feature_Store_and_Offline_Online_Consistency/1.Feature_Store_and_Offline_Online_Consistency.ipynb) | Feature platform reliability | Point-in-time correctness, train-serving skew checks, feature parity monitoring |
| [58. Online Learning and Concept Drift Adaptation](58.Online_Learning_and_Concept_Drift_Adaptation/1.Online_Learning_and_Concept_Drift_Adaptation.ipynb) | Streaming ML adaptation | Prequential evaluation, drift detection, adaptive updates |
| [59. Recommender Ranking Losses from Scratch](59.Recommender_Ranking_Losses_From_Scratch/1.Recommender_Ranking_Losses_From_Scratch.ipynb) | Ranking objective engineering | Pointwise vs pairwise losses, BPR training, Recall@K comparison |
| [60. Causal Bandits and Thompson Sampling](60.Causal_Bandits_and_Thompson_Sampling/1.Causal_Bandits_and_Thompson_Sampling.ipynb) | Sequential decision optimization | Bayesian exploration, contextual heterogeneity, IPS policy evaluation |

### Part XIV: Evaluation, Compression, and Resilience

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [61. Retrieval Evaluation (MRR, NDCG, Hard Negatives)](61.Retrieval_Evaluation_MRR_NDCG_Hard_Negatives/1.Retrieval_Evaluation_MRR_NDCG_Hard_Negatives.ipynb) | Retrieval benchmarking rigor | Query-level metrics, hard negatives, failure diagnosis |
| [62. Knowledge Distillation and Compression](62.Knowledge_Distillation_and_Compression/1.Knowledge_Distillation_and_Compression.ipynb) | Model size-performance optimization | Teacher-student training, soft targets, compression-quality tradeoff |
| [63. Probabilistic Time Series Forecast Intervals](63.Probabilistic_Time_Series_Forecasting_Intervals/1.Probabilistic_Time_Series_Forecasting_Intervals.ipynb) | Uncertainty-aware forecasting | Quantile intervals, coverage calibration, sharpness analysis |
| [64. Multi-Objective Ranking Optimization](64.Multi_Objective_Ranking_Optimization/1.Multi_Objective_Ranking_Optimization.ipynb) | Ranking policy design | Objective weighting, constraint-aware ranking, frontier analysis |
| [65. MLOps Incident Simulation and Runbook](65.MLOps_Incident_Simulation_and_Runbook/1.MLOps_Incident_Simulation_and_Runbook.ipynb) | Operational reliability engineering | Incident detection, runbook execution, postmortem process |

### Part XV: Frontier Reliability and Governance Systems

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [66. Advanced Anomaly Detection](66.Advanced_Anomaly_Detection/1.Advanced_Anomaly_Detection.ipynb) | Unsupervised risk detection | Isolation-based scoring, thresholding, anomaly ranking precision |
| [67. Survival Analysis from Scratch](67.Survival_Analysis_From_Scratch/1.Survival_Analysis_From_Scratch.ipynb) | Time-to-event modeling | Kaplan-Meier estimation, censoring, log-rank comparison |
| [68. Graph Recommendation Systems](68.Graph_Recommendation_Systems/1.Graph_Recommendation_Systems.ipynb) | Graph-based personalization | Bipartite random walks, personalized PageRank, Hit@K/Recall@K |
| [69. Counterfactual Ranking Evaluation](69.Counterfactual_Ranking_Evaluation/1.Counterfactual_Ranking_Evaluation.ipynb) | Offline policy evaluation | IPS, SNIPS, doubly-robust estimation, estimator bias/variance |
| [70. ML Governance and Compliance Automation](70.ML_Governance_and_Compliance_Automation/1.ML_Governance_and_Compliance_Automation.ipynb) | Policy-as-code operations | Compliance checks, release gates, remediation workflows |

### Part XVI: Causal, Robustness, and Rollout Intelligence

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [71. Causal Discovery from Scratch](71.Causal_Discovery_From_Scratch/1.Causal_Discovery_From_Scratch.ipynb) | Structure learning from observations | Skeleton discovery, partial correlations, v-structure orientation |
| [72. Robust Optimization Under Distribution Shift](72.Robust_Optimization_Under_Distribution_Shift/1.Robust_Optimization_Under_Distribution_Shift.ipynb) | Shift-resistant model training | Group-robust objectives, worst-group accuracy, tradeoff analysis |
| [73. Synthetic Data Generation for ML Pipelines](73.Synthetic_Data_Generation_for_ML_Pipelines/1.Synthetic_Data_Generation_for_ML_Pipelines.ipynb) | Privacy-aware data synthesis | Class-conditional generation, utility validation, leakage-risk checks |
| [74. Advanced Experiment Design (Uplift + Bandits Hybrid)](74.Advanced_Experiment_Design_Uplift_Bandits_Hybrid/1.Advanced_Experiment_Design_Uplift_Bandits_Hybrid.ipynb) | Adaptive experimentation | Uplift targeting, online exploration, hybrid policy optimization |
| [75. Production Rollback Orchestration Simulator](75.Production_Rollback_Orchestration_Simulator/1.Production_Rollback_Orchestration_Simulator.ipynb) | Safe deployment automation | Canary ramps, guardrail triggers, rollback sequencing |

### Part XVII: Advanced Boosting, Drift-Robust Kaggle, and Generative Systems

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| [76. XGBoost-Style Second-Order Tree Boosting](76.XGBoost_Style_Second_Order_Tree_Boosting/1.XGBoost_Style_Second_Order_Tree_Boosting.ipynb) | Modern gradient-boosted tree internals | Second-order split gain, Hessian weighting, leaf regularization |
| [77. CatBoost-Style Ordered Target Encoding](77.CatBoost_Style_Ordered_Target_Encoding/1.CatBoost_Style_Ordered_Target_Encoding.ipynb) | Leakage-safe categorical learning | Ordered statistics, high-cardinality encoding, generalization gap diagnostics |
| [78. HuggingFace-Style Trainer from Scratch](78.HuggingFace_Style_Trainer_From_Scratch/1.HuggingFace_Style_Trainer_From_Scratch.ipynb) | Deep learning training system mechanics | AdamW internals, warmup-cosine scheduling, gradient clipping, early stopping |
| [79. Kaggle Adversarial Validation and Feature Selection](79.Kaggle_Adversarial_Validation_and_Feature_Selection/1.Kaggle_Adversarial_Validation_and_Feature_Selection.ipynb) | Distribution-shift-aware model design | Train-test domain classification, drift feature ranking, robust feature ablation |
| [80. Diffusion Models (DDPM) from Scratch](80.Diffusion_Models_DDPM_From_Scratch/1.Diffusion_Models_DDPM_From_Scratch.ipynb) | Generative deep learning fundamentals | Forward noising schedule, reverse denoising process, sampling diagnostics |

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
Advanced ML Track (Ch. 20-25)
         |
         v
Deep Learning Track (Ch. 26-29)
         |
         v
Advanced Algorithms Track (Ch. 30-37)
         |
         v
HF + Kaggle Master Track (Ch. 38-45)
         |
         v
Frontier Algorithms Track (Ch. 46-50)
         |
         v
Advanced Research-to-Production Track (Ch. 51-55)
         |
         v
Production-Scale Systems Track (Ch. 56-60)
         |
         v
Evaluation & Resilience Track (Ch. 61-65)
         |
         v
Reliability & Governance Track (Ch. 66-70)
         |
         v
Causal & Rollout Intelligence Track (Ch. 71-75)
         |
         v
Advanced Boosting + Generative Systems Track (Ch. 76-80)
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
