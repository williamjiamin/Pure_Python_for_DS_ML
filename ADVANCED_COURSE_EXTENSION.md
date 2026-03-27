# Advanced Course Extension (Data Science + ML + Deep Learning)

This file is your long-form continuation plan after Chapter 19.
Use it as a guided roadmap across multiple sessions.

## Suggested order
1. Chapter 20 - Feature engineering and leakage control.
2. Chapter 21 - Model selection and hyperparameter optimization.
3. Chapter 22 - Gradient boosting from scratch.
4. Chapter 23 - Time series forecasting workflow.
5. Chapter 24 - Recommender systems fundamentals.
6. Chapter 25 - ML engineering and MLOps.
7. Chapter 26 - Deep learning fundamentals and training stability.
8. Chapter 27 - CNN and computer vision.
9. Chapter 28 - Sequence models and transformers.
10. Chapter 29 - Deep learning operations.
11. Chapter 30 - Random forest advanced implementation.
12. Chapter 31 - AdaBoost for binary classification.
13. Chapter 32 - Gaussian mixture model with EM.
14. Chapter 33 - DBSCAN and hierarchical clustering.
15. Chapter 34 - ARIMA and exponential smoothing.
16. Chapter 35 - Hidden Markov models and Viterbi decoding.
17. Chapter 36 - Autoencoders from scratch.
18. Chapter 37 - Q-learning and reinforcement learning basics.
19. Chapter 38 - Tokenization and BPE from scratch.
20. Chapter 39 - Advanced transformer internals and attention visualization.
21. Chapter 40 - LoRA/PEFT fine-tuning in pure Python form.
22. Chapter 41 - RAG and vector retrieval pipeline from scratch.
23. Chapter 42 - Kaggle EDA and visualization playbook.
24. Chapter 43 - Kaggle feature engineering and leakage-safe encoding.
25. Chapter 44 - Kaggle ensembling, stacking, and blending.
26. Chapter 45 - Production drift monitoring and dashboard metrics.
27. Chapter 46 - SARSA and DQN conceptual bridge.
28. Chapter 47 - BM25 ranking from scratch.
29. Chapter 48 - HNSW-style ANN retrieval systems.
30. Chapter 49 - Bayesian optimization for expensive tuning.
31. Chapter 50 - Causal uplift modeling and policy targeting.
32. Chapter 51 - PPO clipping and policy optimization.
33. Chapter 52 - SimCLR-style contrastive learning.
34. Chapter 53 - Graph neural network message passing basics.
35. Chapter 54 - Probability calibration and uncertainty estimation.
36. Chapter 55 - Sequential A/B experimentation and alpha control.
37. Chapter 56 - Distributed training and synchronization tradeoffs.
38. Chapter 57 - Feature store consistency and train-serving skew detection.
39. Chapter 58 - Online learning under concept drift.
40. Chapter 59 - Recommender ranking loss design (pointwise vs BPR).
41. Chapter 60 - Causal bandits with Thompson sampling and IPS evaluation.
42. Chapter 61 - Retrieval evaluation with MRR/NDCG and hard negatives.
43. Chapter 62 - Knowledge distillation and compression tradeoffs.
44. Chapter 63 - Probabilistic forecasting intervals and coverage checks.
45. Chapter 64 - Multi-objective ranking with constraints.
46. Chapter 65 - MLOps incident simulation and runbook execution.
47. Chapter 66 - Advanced anomaly detection with isolation-style methods.
48. Chapter 67 - Survival analysis with censoring-aware estimators.
49. Chapter 68 - Graph recommendation via personalized random walks.
50. Chapter 69 - Counterfactual ranking evaluation (IPS/SNIPS/DR).
51. Chapter 70 - ML governance automation and release policy checks.
52. Chapter 71 - Causal discovery with simplified PC-style algorithm.
53. Chapter 72 - Robust optimization under distribution shift.
54. Chapter 73 - Synthetic data generation and utility/privacy checks.
55. Chapter 74 - Hybrid uplift + bandit experiment design.
56. Chapter 75 - Production rollback orchestration simulation.
57. Chapter 76 - XGBoost-style second-order tree boosting internals.
58. Chapter 77 - CatBoost-style ordered target encoding and leakage control.
59. Chapter 78 - HuggingFace-style trainer internals in pure Python.
60. Chapter 79 - Kaggle adversarial validation and drift-aware feature selection.
61. Chapter 80 - DDPM diffusion fundamentals and reverse sampling mechanics.

## Capstone sequence
1. Capstone A (Tabular ML)
- dataset: `Y.Kaggle_Data/diabetes.csv` or `Y.Kaggle_Data/insurance.csv`
- models: logistic regression, decision tree, gradient boosting
- output: model card + leakage report + deployment checklist

2. Capstone B (Recommendation)
- build matrix factorization baseline
- evaluate top-K metrics
- add cold-start popularity fallback

3. Capstone C (Deep learning)
- train MLP/CNN baseline
- add proper train/val/test protocol
- include optimizer schedule, checkpoint policy, and ablation table

4. Capstone D (Probabilistic + Unsupervised)
- compare KMeans vs GMM vs DBSCAN on mixed-shape synthetic data
- include likelihood/cluster-quality diagnostics
- produce segmentation interpretation report

5. Capstone E (Sequential Decision + Time Series)
- forecast with ARIMA-like model and smoothing baseline
- train Q-learning policy in grid world with reward shaping
- report robustness under parameter sensitivity tests

6. Capstone F (HF-Style NLP in Pure Python)
- train BPE tokenizer and analyze compression tradeoffs
- run attention visual diagnostics and entropy checks
- implement LoRA adaptation toy and summarize parameter efficiency

7. Capstone G (Kaggle-Style Competition Pipeline)
- full EDA report with plots and leakage checks
- fold-safe feature engineering and OOF stacking
- leaderboard-style comparison table and error analysis

8. Capstone H (Monitoring and Lifecycle)
- define drift metrics (PSI, score drift, calibration drift)
- create monthly dashboard and alert criteria
- propose retraining and rollback policy

9. Capstone I (Retrieval and Ranking)
- build BM25 + ANN hybrid retrieval benchmark
- analyze recall-latency tradeoffs under index parameters
- produce ranking explainability report

10. Capstone J (Advanced Optimization + Causal Decisioning)
- run Bayesian optimization for model hyperparameters under fixed budget
- train uplift model and optimize treatment targeting policy
- compare business gain vs treat-all and random targeting baselines

11. Capstone K (Advanced RL + Representation Learning)
- compare Q-learning, SARSA, and PPO-style objective behavior
- train contrastive embeddings and evaluate nearest-neighbor consistency
- summarize stability and sample-efficiency tradeoffs

12. Capstone L (Reliability + Experimentation)
- calibrate classifier probabilities and report ECE improvements
- run sequential A/B simulation with and without correction
- provide final experiment decision framework for production rollout

13. Capstone M (ML Platform and Streaming Adaptation)
- simulate distributed training strategies and compare convergence
- validate offline-online feature parity with skew alerts
- implement online learner with drift-triggered adaptation policy

14. Capstone N (Ranking and Causal Decision Intelligence)
- train pointwise and pairwise recommender objectives and compare Recall@K
- deploy contextual Thompson sampling bandit simulation
- evaluate offline policy value with IPS and recommend rollout policy

15. Capstone O (Evaluation and Compression Excellence)
- benchmark retrieval with Recall@K, MRR, and NDCG under hard negatives
- distill a compact student model and report quality-latency-size tradeoff
- define launch criteria based on robust evaluation metrics

16. Capstone P (Forecast Risk + Incident Resilience)
- produce calibrated forecast intervals and coverage diagnostics
- design multi-objective ranking policy with fairness constraints
- run incident simulation and finalize on-call runbook + postmortem template

17. Capstone Q (Counterfactual and Graph Personalization)
- build graph recommender and benchmark Hit@K / Recall@K
- evaluate alternative ranking policy offline with IPS/SNIPS/DR
- validate offline estimates against controlled online simulation

18. Capstone R (Reliability and Governance Automation)
- deploy anomaly detection scoring pipeline with alert thresholds
- run survival analysis for retention/churn time-to-event understanding
- enforce policy-as-code governance checks for release gating

19. Capstone S (Causal Structure + Shift Robustness)
- discover candidate causal graph on synthetic observational data
- train robust model under domain shift and report worst-group gains
- validate model behavior against intervention-style sanity checks

20. Capstone T (Synthetic Data + Rollout Safety)
- build synthetic dataset and compare real-vs-synthetic utility gap
- design hybrid uplift+bandit experimentation policy
- simulate staged rollout with automated rollback and post-incident summary

21. Capstone U (Tabular SOTA + Leakage-Resilient Categorical Pipeline)
- implement second-order boosting with regularized split gain
- add ordered target encoding for high-cardinality categories
- report robust CV vs holdout gap with strict leakage checks

22. Capstone V (HF-Style Trainer + Diffusion Foundations)
- build pure Python trainer loop with AdamW, warmup-cosine, clipping, and early stopping
- run adversarial validation before final model selection
- train a toy diffusion denoiser and compare real-vs-generated distributions

## Quality bar (industry standard)
- deterministic split and reproducible runs
- proper holdout strategy for data type
- baseline comparisons before complexity
- metric selection tied to business objective
- error analysis by slice, not only global score
- monitoring and retraining plan
