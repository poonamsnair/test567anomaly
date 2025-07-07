# AI Agent Trajectory Anomaly Detection - Results Report

**Generated:** 2025-07-07 10:56:35

## Executive Summary

- **Total Execution Time:** 0.00 seconds
- **Trajectories Generated:** 0
- **Models Trained:** 4
- **Best Performing Model:** isolation_forest

## Model Performance

| Model | F1 Score | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| isolation_forest | 0.571 | 0.400 | 1.000 | 0.214 |
| one_class_svm | 0.571 | 0.400 | 1.000 | 0.500 |
| gnn_autoencoder | 0.250 | 0.250 | 0.250 | 0.464 |
| ensemble_model | 0.462 | 0.333 | 0.750 | 0.250 |

## Dataset Statistics

- **Training Set:** 21 samples (100% normal)
- **Validation Set:** 10 samples (30.0% anomalous)
- **Test Set:** 11 samples (36.4% anomalous)

## Feature Engineering

- **Total Features Extracted:** 226
- **Feature Extraction Time:** 0.67 seconds

## Generated Visualizations

- **Embedding Visualization:** `results/charts/embedding_tsne_visualization.png`
- **Trajectory Examples:** `results/charts/trajectory_examples.png`
- **Performance Comparison:** `results/charts/model_performance_comparison.png`
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

- Use isolation_forest model for deployment (best F1: 0.571)
- isolation_forest: Low precision (0.400) - many false positives
- one_class_svm: Low precision (0.400) - many false positives
- gnn_autoencoder: Low precision (0.250) - many false positives
- gnn_autoencoder: Low recall (0.250) - missing anomalies
- ensemble_model: Low precision (0.333) - many false positives

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
  total_anomalous_trajectories: 7
data_generation:
  agent_types:
  - Planner
  - Executor
  - Validator
  - Coordinator
  num_normal_trajectories: 35
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
    random_state: 42
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