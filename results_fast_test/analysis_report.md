# AI Agent Trajectory Anomaly Detection - Results Report

**Generated:** 2025-07-06 00:09:15

## Executive Summary

- **Total Execution Time:** 0.00 seconds
- **Trajectories Generated:** 0
- **Models Trained:** 4
- **Best Performing Model:** one_class_svm

## Model Performance

| Model | F1 Score | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| isolation_forest | 0.500 | 0.500 | 0.500 | 0.500 |
| one_class_svm | 0.667 | 0.500 | 1.000 | 0.938 |
| gnn_autoencoder | 0.667 | 0.500 | 1.000 | 0.875 |
| ensemble | 0.000 | 0.000 | 0.000 | 0.500 |

## Dataset Statistics

- **Training Set:** 12 samples (100% normal)
- **Validation Set:** 8 samples (50.0% anomalous)
- **Test Set:** 8 samples (50.0% anomalous)

## Feature Engineering

- **Total Features Extracted:** 8
- **Feature Extraction Time:** 0.43 seconds

## Generated Visualizations

- **Trajectory Examples:** `results_fast_test/charts/trajectory_examples.png`
- **Performance Comparison:** `results_fast_test/charts/model_performance_comparison.png`
- **Roc Curves:** `results_fast_test/charts/roc_curves.png`
- **Pr Curves:** `results_fast_test/charts/precision_recall_curves.png`
- **Feature Importance:** `results_fast_test/charts/feature_importance.png`
- **Anomaly Distribution:** `results_fast_test/charts/anomaly_distribution.png`
- **Summary Report:** `results_fast_test/charts/summary_report.png`

## Recommendations

- Use one_class_svm model for deployment (best F1: 0.667)
- isolation_forest: Low precision (0.500) - many false positives
- isolation_forest: Low recall (0.500) - missing anomalies
- one_class_svm: Low precision (0.500) - many false positives
- gnn_autoencoder: Low precision (0.500) - many false positives
- ensemble: Low precision (0.000) - many false positives
- ensemble: Low recall (0.000) - missing anomalies

## Configuration Used

```yaml
anomaly_injection:
  anomaly_types:
    infinite_loops:
      ratio: 0.3
      severity: critical
    planning_paralysis:
      ratio: 0.2
      severity: medium
    suboptimal_paths:
      ratio: 0.3
      severity: medium
    tool_failure_cascades:
      ratio: 0.2
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
  total_anomalous_trajectories: 8
data_generation:
  agent_types:
  - Planner
  - Executor
  - Validator
  num_normal_trajectories: 20
  tool_types:
  - web_search
  - read_document
  - analyze_data
  - write_code
  trajectory_patterns:
    branched_analysis:
      max_nodes: 12
      min_nodes: 5
      weight: 0.3
    multi_agent_handoffs:
      max_nodes: 15
      min_nodes: 6
      weight: 0.2
    simple_linear:
      max_nodes: 8
      min_nodes: 3
      weight: 0.5
ensemble:
  base_models:
  - isolation_forest
  - one_class_svm
  - gnn_autoencoder
  confidence_weighting: true
  fusion_method: weighted_average
  optimization_method: validation_performance
  weight_constraints:
    max_weight: 1.0
    min_weight: 0.0
    sum_to_one: true
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
  threshold_calibration:
  - roc_optimization
  - pr_optimization
  - f1_maximization
  - fixed_percentile_95
  - knee_point_detection
  - precision_focused
  - fpr_control
feature_engineering:
  dag_features:
  - topological_sort_validation
  - longest_path_in_dag
  - dag_depth
  - branching_factor
  semantic_features:
  - agent_type_counts
  - handoff_frequency
  - tool_type_distribution
  - error_frequency
  structural_features:
  - num_nodes
  - num_edges
  - density
  - diameter
  - betweenness_centrality
  - closeness_centrality
  - clustering_coefficient
  temporal_features:
  - total_duration
  - average_node_duration
  - duration_variance
  - inter_node_delays
graph_processing:
  aggregation_methods:
  - mean
  - max
  centrality_measures:
  - betweenness
  - closeness
  deepwalk:
    dimensions:
    - 32
    - 64
    min_count:
    - 0
    num_walks:
    - 30
    - 60
    sg: 1
    walk_length:
    - 15
    - 30
    window:
    - 5
  graphsage:
    aggregator:
    - mean
    - max
    batch_size:
    - 16
    - 32
    dropout:
    - 0.1
    - 0.3
    epochs:
    - 50
    - 100
    hidden_dims:
    - - 32
      - 64
    - - 64
      - 128
    learning_rate:
    - 0.001
    - 0.01
    output_dim:
    - 32
    - 64
  node2vec:
    dimensions:
    - 32
    - 64
    min_count:
    - 1
    num_walks:
    - 50
    - 100
    p:
    - 1.0
    q:
    - 1.0
    walk_length:
    - 10
    - 20
    window:
    - 5
    workers: 2
models:
  gnn_autoencoder:
    batch_size:
    - 16
    diffusion_steps:
    - 3
    dropout:
    - 0.1
    epochs:
    - 50
    gnn_types:
    - GCN
    hidden_dims:
    - - 32
      - 64
    learning_rate:
    - 0.001
  isolation_forest:
    bootstrap:
    - false
    contamination:
    - 0.1
    - 0.15
    feature_selection:
      enabled: true
      importance_method: comprehensive
      max_features_ratio: 0.8
      min_features: 5
      selection_method: irrelevant_removal
      threshold_percentile: 25.0
    max_features:
    - 0.75
    - 1.0
    max_samples:
    - 0.75
    - 1.0
    n_estimators:
    - 50
    - 100
    random_state: 42
  one_class_svm:
    gamma:
    - scale
    - auto
    kernel:
    - rbf
    - linear
    nu:
    - 0.1
    - 0.15
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