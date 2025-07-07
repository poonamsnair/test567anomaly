# AI Agent Trajectory Anomaly Detection - Results Report

**Generated:** 2025-07-07 23:15:29

## Executive Summary

- **Total Execution Time:** 0.00 seconds
- **Trajectories Generated:** 0
- **Models Trained:** 4
- **Best Performing Model:** isolation_forest

## Model Performance

| Model | F1 Score | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| isolation_forest | 0.567 | 0.678 | 0.488 | 0.751 |
| one_class_svm | 0.560 | 0.488 | 0.656 | 0.697 |
| gnn_autoencoder | 0.556 | 0.385 | 1.000 | 0.583 |
| ensemble_model | 0.556 | 0.385 | 1.000 | 0.607 |

## Dataset Statistics

- **Training Set:** 600 samples (100% normal)
- **Validation Set:** 325 samples (38.5% anomalous)
- **Test Set:** 325 samples (38.5% anomalous)

## Feature Engineering

- **Total Features Extracted:** 214
- **Feature Extraction Time:** 6.95 seconds

## Generated Visualizations

- **Embedding Visualization:** `results/charts/embedding_tsne_visualization.png`
- **Embedding Method Comparison:** `results/charts/embedding_method_comparison.png`
- **Embedding Quality Analysis:** `results/charts/embedding_quality_analysis.png`
- **Trajectory Examples:** `results/charts/trajectory_examples.png`
- **Performance Comparison:** `results/charts/model_performance_comparison.png`
- **Roc Curves:** `results/charts/roc_curves.png`
- **Pr Curves:** `results/charts/precision_recall_curves.png`
- **Feature Importance:** `results/charts/feature_importance.png`
- **Anomaly Distribution:** `results/charts/anomaly_distribution.png`
- **Ensemble Weights:** `results/charts/ensemble_weights.png`
- **Ensemble Performance:** `results/charts/ensemble_performance_comparison.png`
- **Confidence Distributions:** `results/charts/confidence_distributions.png`
- **Confidence Vs Performance:** `results/charts/confidence_vs_performance.png`
- **Calibration Analysis:** `results/charts/calibration_analysis.png`
- **Confidence Error Analysis:** `results/charts/confidence_error_analysis.png`
- **Comprehensive Confidence Analysis:** `results/charts/comprehensive_confidence_analysis.png`
- **Summary Report:** `results/charts/summary_report.png`

## Recommendations

- Use isolation_forest model for deployment (best F1: 0.567)
- isolation_forest: Low precision (0.678) - many false positives
- isolation_forest: Low recall (0.488) - missing anomalies
- one_class_svm: Low precision (0.488) - many false positives
- one_class_svm: Low recall (0.656) - missing anomalies
- gnn_autoencoder: Low precision (0.385) - many false positives
- ensemble_model: Low precision (0.385) - many false positives

## Configuration Used

```yaml
anomaly_injection:
  anomaly_types:
    infinite_loops:
      ratio: 0.25
      severity: critical
    planning_paralysis:
      ratio: 0.2
      severity: medium
    suboptimal_paths:
      ratio: 0.3
      severity: medium
    tool_failure_cascades:
      ratio: 0.25
      severity: high
  severity_levels:
    critical:
      degradation_range:
      - 0.7
      - 1.0
      user_impact: system_failure
    high:
      degradation_range:
      - 0.4
      - 0.7
      user_impact: significant
    low:
      degradation_range:
      - 0.1
      - 0.2
      user_impact: minimal
    medium:
      degradation_range:
      - 0.2
      - 0.4
      user_impact: noticeable
  total_anomalous_trajectories: 250
data_generation:
  agent_types:
  - Planner
  - Executor
  - Validator
  - Coordinator
  num_normal_trajectories: 1000
  tool_types:
  - web_search
  - read_document
  - analyze_data
  - write_code
  - validate_result
  trajectory_patterns:
    branched_analysis:
      max_nodes: 12
      min_nodes: 5
      weight: 0.35
    multi_agent_handoffs:
      max_nodes: 15
      min_nodes: 6
      weight: 0.25
    simple_linear:
      max_nodes: 8
      min_nodes: 3
      weight: 0.4
dataset_size_thresholds:
  medium_large_boundary: 300
  small_medium_boundary: 100
evaluation:
  data_split:
    test_ratio: 0.2
    train_ratio: 0.6
    validation_ratio: 0.2
  metrics:
  - precision
  - recall
  - f1
  - accuracy
  - auc_roc
  - auc_pr
  - silhouette_score
  - severity_weighted_performance
  threshold_calibration:
  - roc_optimization
  - pr_optimization
  - f1_maximization
  - fixed_percentile_95
  - knee_point_detection
feature_engineering:
  dag_features:
  - topological_sort_validation
  - longest_path_in_dag
  - dag_depth
  - branching_factor
  - leaf_node_ratio
  - root_to_leaf_paths
  semantic_features:
  - agent_type_counts
  - handoff_frequency
  - tool_type_distribution
  - error_frequency
  - agent_specialization_index
  - tool_usage_efficiency
  structural_features:
  - num_nodes
  - num_edges
  - density
  - diameter
  - betweenness_centrality
  - closeness_centrality
  - clustering_coefficient
  - average_degree
  - degree_variance
  temporal_features:
  - total_duration
  - average_node_duration
  - duration_variance
  - inter_node_delays
  - max_concurrent_operations
  - idle_time_ratio
graph_processing:
  aggregation_methods:
  - mean
  centrality_measures:
  - betweenness
  - closeness
  checkpoint_dir: checkpoints
  deepwalk:
    dimensions:
    - 8
    min_count:
    - 0
    num_walks:
    - 5
    sg: 1
    walk_length:
    - 5
    window:
    - 5
  graphsage:
    aggregator:
    - mean
    batch_size:
    - 8
    dropout:
    - 0.1
    epochs:
    - 5
    hidden_dims:
    - - 8
      - 8
    learning_rate:
    - 0.001
    output_dim:
    - 8
  node2vec:
    dimensions:
    - 8
    min_count:
    - 1
    num_walks:
    - 5
    p:
    - 1.0
    q:
    - 1.0
    walk_length:
    - 5
    window:
    - 5
    workers: 2
hyperparameter_grids:
  large_dataset:
    gnn_autoencoder:
      batch_size:
      - 32
      - 64
      dropout:
      - 0.1
      - 0.2
      - 0.3
      epochs:
      - 100
      - 200
      gnn_type:
      - GCN
      - GAT
      - GraphConv
      hidden_dims:
      - - 64
        - 128
      - - 128
        - 256
      learning_rate:
      - 0.001
      - 0.01
    isolation_forest:
      bootstrap:
      - true
      - false
      contamination:
      - 0.03
      - 0.05
      - 0.08
      - 0.1
      max_features:
      - 0.5
      - 0.75
      - 1.0
      max_samples:
      - 0.5
      - 0.75
      - 1.0
      n_estimators:
      - 300
      - 500
      random_state:
      - 42
    max_combinations: 10
    max_combinations_gnn: 5
    max_combinations_ocsvm: 10
    one_class_svm:
      degree:
      - 2
      - 3
      gamma:
      - scale
      - auto
      - 0.1
      - 0.01
      kernel:
      - rbf
      - linear
      - poly
      nu:
      - 0.05
      - 0.1
      - 0.15
      - 0.2
  medium_dataset:
    gnn_autoencoder:
      batch_size:
      - 16
      - 32
      dropout:
      - 0.1
      - 0.2
      - 0.3
      epochs:
      - 50
      - 100
      gnn_type:
      - GCN
      - GAT
      hidden_dims:
      - - 32
        - 64
      - - 64
        - 128
      learning_rate:
      - 0.001
      - 0.01
    isolation_forest:
      bootstrap:
      - true
      - false
      contamination:
      - 0.03
      - 0.05
      - 0.08
      max_features:
      - 0.5
      - 0.75
      - 1.0
      max_samples:
      - 0.5
      - 0.75
      - 1.0
      n_estimators:
      - 200
      - 300
      random_state:
      - 42
    max_combinations: 5
    max_combinations_gnn: 3
    max_combinations_ocsvm: 5
    one_class_svm:
      degree:
      - 2
      - 3
      gamma:
      - scale
      - auto
      - 0.1
      kernel:
      - rbf
      - linear
      - poly
      nu:
      - 0.05
      - 0.1
      - 0.15
  small_dataset:
    gnn_autoencoder:
      batch_size:
      - 8
      - 16
      dropout:
      - 0.1
      - 0.2
      epochs:
      - 30
      - 50
      gnn_type:
      - GCN
      hidden_dims:
      - - 16
        - 32
      - - 32
        - 64
      learning_rate:
      - 0.001
      - 0.01
    isolation_forest:
      bootstrap:
      - false
      contamination:
      - 0.05
      - 0.1
      max_features:
      - 0.5
      - 0.75
      max_samples:
      - 0.5
      - 0.75
      n_estimators:
      - 100
      - 200
      random_state:
      - 42
    max_combinations: 3
    max_combinations_gnn: 2
    max_combinations_ocsvm: 3
    one_class_svm:
      degree:
      - 2
      gamma:
      - scale
      - auto
      kernel:
      - rbf
      - linear
      nu:
      - 0.1
      - 0.15
models:
  gnn_autoencoder:
    batch_size:
    - 16
    dropout:
    - 0.1
    epochs:
    - 50
    gnn_type:
    - GCN
    hidden_dims:
    - - 32
      - 64
    learning_rate:
    - 0.001
  isolation_forest:
    bootstrap:
    - true
    - false
    contamination:
    - 0.03
    - 0.05
    - 0.08
    - 0.1
    max_features:
    - 0.5
    - 0.75
    - 1.0
    max_samples:
    - 0.5
    - 0.75
    - 1.0
    n_estimators:
    - 200
    - 300
    random_state:
    - 42
  one_class_svm:
    degree:
    - 2
    - 3
    gamma:
    - scale
    - auto
    - 0.1
    - 0.01
    kernel:
    - rbf
    - linear
    - poly
    nu:
    - 0.05
    - 0.1
    - 0.15
    - 0.2
system:
  chunk_size: 500
  logging:
    file: logs/fast_test.log
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    level: INFO
  memory_limit_gb: 4
  n_jobs: 2
  output_dirs:
    charts: results_fast_test/charts/
    data: results_fast_test/data/
    logs: results_fast_test/logs/
    models: results_fast_test/models/
  random_seed: 42
visualization:
  color_schemes:
    agent_types: Set3
    anomaly_severity: Reds
    node_types: viridis
  figure_settings:
    bbox_inches: tight
    dpi: 150
    format: png
  graph_layouts:
  - spring
  - hierarchical
  interactive:
    enabled: false
    renderer: browser

```