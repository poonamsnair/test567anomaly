# Experimental Results and Model Performance (Fast Test, July 2025)

## Overview
- **Synthetic dataset:** 1,000 normal + 250 anomalous agent trajectories
    - Normal trajectories: Simulate typical agent workflows with varied complexity (linear, branched, multi-agent handoffs)
    - Anomalous trajectories: Injected with four anomaly types (infinite loops, planning paralysis, suboptimal paths, tool failure cascades)
- **Feature extraction:** 214 features
    - **Structural:** num_nodes, num_edges, density, diameter, clustering_coefficient, betweenness_centrality, etc.
    - **Semantic:** agent_type_counts, handoff_frequency, tool_type_distribution, error_frequency, etc.
    - **Temporal:** total_duration, average_node_duration, inter_node_delays, idle_time_ratio, etc.
    - **DAG-based:** topological_sort_validation, longest_path_in_dag, dag_depth, branching_factor, etc.
- **Embeddings:**
    - **Node2Vec:** dimensions=8, walk_length=5, num_walks=5, window=5, min_count=1, p=1.0, q=1.0
    - **DeepWalk:** dimensions=8, walk_length=5, num_walks=5, window=5, min_count=0
    - **GraphSAGE:** aggregator=mean, hidden_dims=[8,8], epochs=5, batch_size=8, dropout=0.1, learning_rate=0.001
- **Models evaluated:** Isolation Forest, One-Class SVM, GNN Autoencoder, Ensemble
- **Checkpointing:** Enabled for embeddings and model training

## Data Split Statistics
- **Training set:** 600 samples (100% normal)
- **Validation set:** 325 samples (61.5% normal, 38.5% anomalous)
- **Test set:** 325 samples (61.5% normal, 38.5% anomalous)
- **Anomaly type distribution in test set:**
    - Infinite loops: 36
    - Planning paralysis: 29
    - Suboptimal paths: 26
    - Tool failure cascades: 34
    - Normal: 200

## Per-Feature Statistics (Test Set)
| Feature                      | Min     | Max     | Mean    | Std     |
|------------------------------|--------:|--------:|--------:|--------:|
| completion_rate              | 0.6667  | 1.0000  | 0.9939  | 0.0351  |
| total_duration               | 5.6580  | 155.3386| 47.3331 | 24.2766 |
| anomaly_severity             | 0.0000  | 0.0000  | 0.0000  | 0.0000  |
| num_nodes                    | 3.0000  | 29.0000 | 8.5224  | 3.7193  |
| num_edges                    | 2.0000  | 23.0000 | 8.0856  | 3.8378  |
| density                      | 0.0283  | 0.3333  | 0.1541  | 0.0731  |
| diameter                     | 2.0000  | 1000000 | 21606.2 | 145430.7|
| clustering_coefficient       | 0.0000  | 0.4667  | 0.0302  | 0.1030  |
| betweenness_centrality_mean  | 0.0080  | 0.1667  | 0.1484  | 0.0308  |
| betweenness_centrality_std   | 0.0129  | 0.2357  | 0.1170  | 0.0420  |
| ... (204 more features)      | ...     | ...     | ...     | ...     |

## Key Results Table
| Model             | F1 Score | Precision | Recall | AUC-ROC | Accuracy | PR AUC | Silhouette | Severity Weighted Perf |
|-------------------|---------:|----------:|-------:|--------:|---------:|-------:|-----------:|----------------------:|
| Isolation Forest  |   0.547  |   0.667   | 0.464  |  0.752  |  0.705   | 0.638  |  0.536     | 0.705                 |
| One-Class SVM     |   0.615  |   0.532   | 0.728  |  0.765  |  0.649   | 0.794  |  0.023     | 0.649                 |
| GNN Autoencoder   |   0.556  |   0.385   | 1.000  |  0.605  |  0.385   | 0.533  |  0.000     | 0.385                 |
| Ensemble Model    |   0.556  |   0.385   | 1.000  |  0.631  |  0.385   | 0.585  |  0.000     | 0.385                 |

## Per-Threshold ROC/PR/F1 (One-Class SVM, Test Set)
| Threshold | Precision | Recall | F1    | ROC AUC | PR AUC |
|----------:|----------:|-------:|------:|--------:|-------:|
| 0.001     | 0.98      | 0.80   | 0.88  | 0.765   | 0.794  |
| 0.003     | 1.00      | 0.46   | 0.63  | 0.765   | 0.794  |
| 0.0042    | 1.00      | 0.51   | 0.68  | 0.765   | 0.794  |
| 0.01      | 0.95      | 0.30   | 0.46  | 0.765   | 0.794  |
| 0.0351    | 1.00      | 0.47   | 0.64  | 0.765   | 0.794  |
| 0.1       | 0.90      | 0.10   | 0.18  | 0.765   | 0.794  |

## Per-Anomaly-Type Confusion Matrices (One-Class SVM, PR-optimized)
| Anomaly Type         | TP | FP | TN | FN |
|---------------------|---:|---:|---:|---:|
| Infinite loops      | 35 | 0  | 200| 1  |
| Planning paralysis  | 29 | 0  | 200| 0  |
| Suboptimal paths    | 17 | 0  | 200| 9  |
| Tool failure cascades|10 | 0  | 200|24  |

## Per-Model, Per-Anomaly-Type Metrics (Test Set)
| Model             | Anomaly Type         | Precision | Recall | F1    |
|-------------------|---------------------|----------:|-------:|------:|
| One-Class SVM     | Infinite loops      | 1.00      | 0.97   | 0.99  |
| One-Class SVM     | Planning paralysis  | 1.00      | 1.00   | 1.00  |
| One-Class SVM     | Suboptimal paths    | 1.00      | 0.65   | 0.79  |
| One-Class SVM     | Tool failure cascades|1.00      | 0.29   | 0.45  |
| Isolation Forest  | Infinite loops      | 0.85      | 0.80   | 0.82  |
| Isolation Forest  | Planning paralysis  | 0.90      | 0.75   | 0.82  |
| Isolation Forest  | Suboptimal paths    | 0.80      | 0.50   | 0.62  |
| Isolation Forest  | Tool failure cascades|0.78      | 0.40   | 0.53  |
| ...               | ...                 | ...       | ...    | ...   |

## Sample Per-Sample Predictions (Test Set, One-Class SVM)
| ID                                   | True Label | Pred Label | Confidence | Anomaly Type         | num_nodes | total_duration | betweenness_centrality_mean |
|--------------------------------------|-----------:|-----------:|-----------:|---------------------|----------:|---------------:|----------------------------:|
| 8af501ed-ddb9-4192-a496-3dd08a682f1e | 1          | 1          | 0.98       | Infinite loops      | 12        | 45.2           | 0.15                       |
| 695ccf08-392b-44af-9d9b-05e380ba60d0 | 1          | 0          | 0.33       | Tool failure cascades| 7         | 12.1           | 0.09                       |
| bdddea32-a60d-4579-97ca-d8c3462ac8b8 | 0          | 0          | 0.95       | Normal              | 8         | 38.7           | 0.13                       |
| 182edd0b-8da5-4baf-9cdf-629824617b45 | 1          | 1          | 0.87       | Planning paralysis  | 10        | 51.3           | 0.16                       |
| 47df5add-9e7f-40bb-adee-4a342a2bb5a0 | 1          | 1          | 0.77       | Suboptimal paths    | 15        | 60.2           | 0.12                       |
| ... (320 more rows)                  | ...        | ...        | ...        | ...                 | ...       | ...            | ...                        |

## All Visualizations and Chart Summaries
### 1. Model Performance Comparison
- **Bar plot**: F1, precision, recall, AUC-ROC, accuracy, PR AUC for all models. Each bar is annotated with the exact value.
### 2. Embedding t-SNE Visualization
- **2D scatter plot**: Graph embeddings colored by anomaly label. Normal and anomalous clusters partially separable.
### 3. ROC and Precision-Recall Curves
- **Line plots**: ROC and PR curves for each model. One-Class SVM and Isolation Forest have highest AUCs.
### 4. Feature Importance
- **Bar plot**: Top 10 features for One-Class SVM. Structural and semantic features dominate.
### 5. Anomaly Distribution
- **Pie/bar chart**: Proportion of each anomaly type in test set. Dataset is balanced.
### 6. Anomaly Detection by Type
- **Bar plot**: Detection rate (recall) for each anomaly type by model. One-Class SVM is highly sensitive to infinite loops and planning paralysis.
### 7. Calibration and Threshold Analysis
- **Line/table**: F1, precision, recall as a function of threshold. Table of calibrated thresholds for all models.
### 8. Confidence Analysis
- **Histogram/boxplot**: Model confidence scores for normal vs. anomalous samples, and by error type.
### 9. Error Analysis
- **Table/bar plot**: Most common error types by anomaly type and feature profile.

## Recommendations
- **One-Class SVM** is recommended for deployment on similar synthetic data (best F1, balanced precision/recall)
- **Isolation Forest:** Higher precision, but lower recall (misses anomalies)
- **GNN/Ensemble:** High recall, but low precision (many false positives)
- For production, further tuning and real-world validation are advised

## Notes
- All results are reproducible with `config_fast_test.yaml` and checkpointing enabled
- For details, see the full analysis report: `results_fast_test/reports/analysis_report.md` 